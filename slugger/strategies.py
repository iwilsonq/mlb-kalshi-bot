"""Trading strategies for Slugger MLB bot.

Each strategy provides a probability model; the signal pipeline
(slugger.signal_pipeline) handles market fetching, edge scoring,
Kelly sizing, and signal recording.
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from slugger.calibration import CalibrationLayer
from slugger.config import Config
from slugger.mlb_data import get_team_profile
from slugger.models import (
    HITS_LAMBDA_DEFLATOR, HITS_MIN_PITCHER_IP,
    HIT_PARK_FACTORS,
    LEAGUE_AVG_WHIP, MAX_PITCHER_WHIP_ADJ,
    expected_ab, expected_ks, poisson_ge, shrink_avg,
)
from slugger.signal_pipeline import evaluate_markets
from slugger.tickers import kalshi_team, ks_event_ticker, hit_event_ticker
from slugger.types import (
    BatterProfile, GameContext, GameInfo, MarketClient, MarketSpec,
    ModelResult, PitcherProfile, TradeSignal,
)

log = logging.getLogger(__name__)

# ── Strategy-specific constants (not model math — kept here) ──────────────────
# Phase 0 gates (journal-driven): only trade mid-probability K bands with
# cost-adjusted edge ≥ 20¢. Outside ~25–55% the model is either under- or
# over-confident vs realized outcomes.
_KS_MIN_THRESHOLD   = 6       # skip 5+ and below — low thresholds = longshot noise
_KS_MIN_MODEL_PROB  = 25      # calibrated model prob floor (%)
_KS_MAX_MODEL_PROB  = 55      # calibrated model prob ceiling (%)
_KS_MIN_EDGE_CENTS  = 20      # cost-adjusted edge floor (after EDGE_COST_BUFFER)
_KS_NO_MAX_MODEL_PROB = 10    # buy NO when model says probability is at most this (%)
_KS_NO_MIN_EDGE_CENTS = 5     # minimum edge (market_yes_price - model_prob) to buy NO
# NOTE: NO-side is disabled below (no_side=False). Market-price analysis
# shows actual win rates at 10-30¢ match market pricing (12-29%),
# while our model says 1-8%. The model underpredicts in this range,
# creating phantom NO-side edge. Re-enable only when model calibration
# is accurate enough to reliably identify overpriced YES markets.

# Threshold regex: matches "7+", "over 6.5", "at least 9" in any K-related title
_KS_THRESHOLD_PATTERN = r'(\d+)\s*\+'

# Strategy-specific hits constants — only near-flat strategy; keep but tighten
_HITS_MIN_AB          = 60      # minimum AB before considering a batter
_HITS_MIN_MODEL_PROB  = 18      # avoid thin longshot hit props (Phase2 refine)
_HITS_MIN_EDGE_CENTS  = 10      # cost-adjusted edge floor (journal + cal bias)
_HITS_THRESHOLD_PATTERN = r'(\d+)\s*\+\s*hit'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Strikeout Props
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_pitcher_ks(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Strikeout prop bets — Poisson model via signal pipeline."""
    event_ticker = ks_event_ticker(game_info)
    if not event_ticker:
        return []

    if not pitcher_profile or (
        pitcher_profile.k_per_9 == 0 and pitcher_profile.recent_k_per_start == 0
    ):
        return []

    # ── Identify opposing team and fetch their K rate ──────────────────────
    opp_k_rate = 0.0
    try:
        if pitcher_profile.player_id == game_info.away_pitcher_id:
            opp_abbrev = game_info.home_abbrev
        else:
            opp_abbrev = game_info.away_abbrev
        opp_team = get_team_profile(opp_abbrev)
        opp_k_rate = opp_team.k_rate
        log.debug(
            "Opponent %s K rate: %.1f%% (league avg %.1f%%)",
            opp_abbrev, opp_k_rate * 100, 22.5,
        )
    except Exception as exc:
        log.debug("Could not fetch opponent K rate: %s", exc)

    # ── Compute expected strikeouts (λ) ────────────────────────────────────
    lam = expected_ks(pitcher_profile, opp_k_rate)
    if lam <= 0:
        return []

    # ── In-game adjustment ─────────────────────────────────────────────────
    current_ks = getattr(pitcher_profile, "current_ks", None)
    ip_today = getattr(pitcher_profile, "ip_today", None)
    in_game = current_ks is not None and ip_today is not None

    if in_game:
        expected_ip = pitcher_profile.recent_ip_per_start or 5.5
        ip_remaining = max(0.0, expected_ip - ip_today)
        frac_remaining = ip_remaining / expected_ip if expected_ip > 0 else 0.0
        lam_remaining = lam * frac_remaining
        log.debug(
            "%s  in-game: %dKs/%.1fIP done  ip_remaining=%.1f  "
            "λ_full=%.2f → λ_remaining=%.2f",
            pitcher_profile.name, current_ks, ip_today,
            ip_remaining, lam, lam_remaining,
        )
        lam = lam_remaining
    else:
        log.debug(
            "%s  λ=%.2f  recent_k/start=%.1f  recent_ip/start=%.1f"
            "  whiff=%.3f  opp_k_rate=%.3f",
            pitcher_profile.name, lam,
            pitcher_profile.recent_k_per_start,
            pitcher_profile.recent_ip_per_start,
            pitcher_profile.whiff_rate,
            opp_k_rate,
        )

    # ── Build model closure ────────────────────────────────────────────────
    def ks_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        if threshold is None:
            return None
        if in_game and current_ks is not None:
            if current_ks >= threshold:
                prob_pct = 99
            else:
                remaining_needed = threshold - current_ks
                prob_pct = round(poisson_ge(remaining_needed, lam) * 100)
        else:
            prob_pct = round(poisson_ge(threshold, lam) * 100)

        reason = (
            f"λ={lam:.1f}Ks  P(≥{threshold})={prob_pct}%"
            f"  recent={pitcher_profile.recent_k_per_start:.1f}K/start"
            + (f"  whiff={pitcher_profile.whiff_rate:.2f}" if pitcher_profile.whiff_rate else "")
            + (f"  opp_k={opp_k_rate:.1%}" if opp_k_rate else "")
        )
        return ModelResult(prob_pct=prob_pct, reason=reason)

    # ── Run pipeline ───────────────────────────────────────────────────────
    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="pitcher_ks",
        title_keywords=["strikeout", "k+", " ks"],
        player_name=pitcher_profile.name,
        threshold_pattern=_KS_THRESHOLD_PATTERN,
        min_threshold=_KS_MIN_THRESHOLD,
        min_model_prob=_KS_MIN_MODEL_PROB,
        max_model_prob=_KS_MAX_MODEL_PROB,
        min_edge_cents=_KS_MIN_EDGE_CENTS,
        max_signals=2,
        no_side=False,
        no_max_model_prob=_KS_NO_MAX_MODEL_PROB,
        no_min_edge_cents=_KS_NO_MIN_EDGE_CENTS,
    )
    return evaluate_markets(spec, ks_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Player Hits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



def strategy_player_hits(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Player hits prop — Poisson model via signal pipeline."""
    event_ticker = hit_event_ticker(game_info)
    if not event_ticker or not batter_profile:
        return []

    if batter_profile.ab < _HITS_MIN_AB:
        log.debug(
            "player_hits | %s — only %d AB (need %d) — skipping",
            batter_profile.name, batter_profile.ab, _HITS_MIN_AB,
        )
        return []

    opp_whip = pitcher_profile.whip if pitcher_profile else 0.0
    opp_ip = pitcher_profile.innings_pitched if pitcher_profile else 0.0
    opp_throws = (pitcher_profile.throws if pitcher_profile else "") or ""

    # ── Platoon split selection ────────────────────────────────────────────
    if opp_throws == "L" and batter_profile.vs_lhp_ab >= 30:
        split_h = round(batter_profile.vs_lhp_avg * batter_profile.vs_lhp_ab)
        split_ab = batter_profile.vs_lhp_ab
        platoon_note = "vsL"
    elif opp_throws == "R" and batter_profile.vs_rhp_ab >= 30:
        split_h = round(batter_profile.vs_rhp_avg * batter_profile.vs_rhp_ab)
        split_ab = batter_profile.vs_rhp_ab
        platoon_note = "vsR"
    else:
        split_h = batter_profile.hits
        split_ab = batter_profile.ab
        platoon_note = "overall"

    # ── Compute λ ──────────────────────────────────────────────────────────
    ab_est = expected_ab(batter_profile.batting_order)
    eff_avg = shrink_avg(split_h, split_ab)

    # Blend shrunk average with xBA and recent form
    if batter_profile.xba > 0 and batter_profile.recent_avg > 0:
        blended_avg = 0.40 * eff_avg + 0.30 * batter_profile.xba + 0.30 * batter_profile.recent_avg
    elif batter_profile.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter_profile.xba
    elif batter_profile.recent_avg > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter_profile.recent_avg
    else:
        blended_avg = eff_avg

    lam = blended_avg * ab_est

    pitcher_adj = 1.0
    if opp_whip > 0 and opp_ip >= HITS_MIN_PITCHER_IP:
        raw_whip = opp_whip / LEAGUE_AVG_WHIP
        pitcher_adj = min(1.0 + 0.5 * (raw_whip - 1.0), MAX_PITCHER_WHIP_ADJ)
        lam *= pitcher_adj

    # Hard hit rate adjustment (dampened)
    _LEAGUE_AVG_HHR = 0.370
    hhr_adj = 1.0
    if batter_profile.hard_hit_rate > 0:
        raw_hhr = batter_profile.hard_hit_rate / _LEAGUE_AVG_HHR
        hhr_adj = 1.0 + 0.25 * (raw_hhr - 1.0)
        lam *= hhr_adj

    home_kalshi = kalshi_team(game_info.home_abbrev)
    park_factor = HIT_PARK_FACTORS.get(
        home_kalshi, HIT_PARK_FACTORS.get(game_info.home_abbrev.upper(), 1.0),
    )
    lam *= park_factor

    # Calibration deflation — model over-predicts hit probability
    lam *= HITS_LAMBDA_DEFLATOR

    log.debug(
        "%s (#%d)  split=%s %dH/%dAB  eff_avg=%.3f  xba=%.3f  ab_est=%.1f"
        "  hhr=%.2f  park=%s(×%.2f)  opp_whip=%.2f(%s,%.0fIP)  pitcher_adj=%.2f  λ=%.3f",
        batter_profile.name, batter_profile.batting_order,
        platoon_note, split_h, split_ab, eff_avg,
        batter_profile.xba, ab_est, hhr_adj, home_kalshi, park_factor,
        opp_whip, opp_throws or "?", opp_ip, pitcher_adj, lam,
    )

    if lam <= 0:
        return []

    pitcher_note = (
        f"  opp_{opp_throws}whip={opp_whip:.2f}({opp_ip:.0f}IP)"
        if opp_ip >= HITS_MIN_PITCHER_IP else ""
    )

    # ── Build model closure ────────────────────────────────────────────────
    def hits_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        if threshold is None:
            return None
        prob_pct = round(poisson_ge(threshold, lam) * 100)
        reason = (
            f"{batter_profile.name}"
            f"  {split_h}H/{split_ab}AB({platoon_note})"
            f"  eff_avg={eff_avg:.3f}"
            f"  xba={batter_profile.xba:.3f}"
            f"  park={park_factor:.2f}"
            f"  λ={lam:.2f}"
            f"  P({threshold}+H)={prob_pct}%"
            f"{pitcher_note}"
        )
        return ModelResult(prob_pct=prob_pct, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="player_hits",
        title_keywords=["hit"],
        player_name=batter_profile.name,
        threshold_pattern=_HITS_THRESHOLD_PATTERN,
        min_model_prob=_HITS_MIN_MODEL_PROB,
        min_edge_cents=_HITS_MIN_EDGE_CENTS,
        max_signals=2,
        confidence_fn=lambda e: min(0.4 + e / 100, 0.80),
    )
    return evaluate_markets(spec, hits_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UNIFIED STRATEGY WRAPPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Each wrapper has the uniform StrategyFn signature:
#   (ctx: GameContext, client: MarketClient, config: Config,
#    prior_signals: List[TradeSignal]) -> List[TradeSignal]
#
# Internally they delegate to the per-pitcher or per-batter functions above,
# iterating over the relevant profiles from the GameContext.


def _run_per_pitcher(
    fn,
    ctx: GameContext, client: MarketClient, config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Call a pitcher-level strategy for each pitcher in the context."""
    signals: List[TradeSignal] = []
    for pitcher in [ctx.away_pitcher, ctx.home_pitcher]:
        if pitcher:
            signals.extend(fn(ctx.game, pitcher, None, client, config, calibration=calibration))
    return signals


def _run_per_batter(
    fn,
    ctx: GameContext, client: MarketClient, config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Call a batter-level strategy for each batter vs opposing pitcher."""
    signals: List[TradeSignal] = []
    for batter in ctx.away_batters:
        signals.extend(fn(ctx.game, ctx.home_pitcher, batter, client, config, calibration=calibration))
    for batter in ctx.home_batters:
        signals.extend(fn(ctx.game, ctx.away_pitcher, batter, client, config, calibration=calibration))
    return signals


def _run_pitcher_ks(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_pitcher(strategy_pitcher_ks, ctx, client, config, calibration=calibration)


def _run_player_hits(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_batter(strategy_player_hits, ctx, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Ordered list of (name, strategy_fn) pairs.  process_game() iterates through
# this list in order, feeding each strategy's output into the next as
# prior_signals.
#
# To add a new strategy: define the function, write a wrapper, add it here.
# No changes to game_processor.py needed.
#
# An entry in ENABLED_STRATEGIES that is absent here is inert — the allowlist
# cannot resurrect a strategy the pipeline does not register.

STRATEGY_PIPELINE: List[Tuple[str, Any]] = [
    ("pitcher_ks",    _run_pitcher_ks),
    ("player_hits",   _run_player_hits),
]

# ── Retired strategies: tombstone ─────────────────────────────────────────────
#
# The implementations were deleted rather than left dark, because in every case
# the *model* was the problem, so reviving one means writing a new one — the old
# code was never going to be re-enabled as-is. `git log -- slugger/strategies.py`
# has the originals; what is worth keeping is the evidence for why each died.
#
# ROI figures are settled trades from logs/journal.jsonl as of 2026-08-03.
# Do not rebuild one of these without a model that beats market Brier on a
# walk-forward holdout first.
RETIRED_STRATEGIES: Dict[str, str] = {
    "player_hr":
        "252 trades, −46.0% ROI, 5.6% win rate. Only 24 calibration samples "
        "(below calibration._MIN_SAMPLES=30) so no curve was ever fitted and it "
        "traded uncalibrated the whole time. Model fns kept in models.py.",
    "player_hr_rbis":
        "79 trades, −48.9% ROI. Probability was a 4-bucket step function on "
        "batting average, plus a hardcoded skip for 1+/2+ titles that excluded "
        "most of the market.",
    "game_winner":
        "18 trades, −206.6% ROI, but the sample is not the point: "
        "ticker_suffix=home_abbrev meant it only ever bet home sides, so the "
        "journal never tested the model. Broken instrument, worthless evidence. "
        "game_winner_probability/pythagorean_win_pct kept in models.py.",
    "total_runs":
        "0 trades in three months. total_prob(era) keyed off one pitcher's ERA "
        "and ignored the opposing starter, both offenses, park and weather.",
    "combo":
        "0 trades from 3085 signals. Joint probability was the naive "
        "independence product times a hand-tuned pairwise correlation table — "
        "the same construction bv1.5 removed from pitcher_ks. Needs a real "
        "joint simulation, and singles must be +EV first.",
    "pitcher_er":
        "Never implemented; the body was `return []`.",
}
