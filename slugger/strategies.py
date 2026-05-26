"""Trading strategies for Slugger MLB bot.

Each strategy provides a probability model; the signal pipeline
(slugger.signal_pipeline) handles market fetching, edge scoring,
Kelly sizing, and signal recording.
"""
from __future__ import annotations
import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from slugger.calibration import CalibrationLayer
from slugger.config import Config
from slugger.kalshi_client import market_price
from slugger.journal import record_signal
from slugger.mlb_data import get_team_profile
from slugger.models import (
    HITS_MIN_PITCHER_IP, HR_PARK_FACTORS, HIT_PARK_FACTORS,
    LEAGUE_AVG_WHIP, MAX_PITCHER_WHIP_ADJ, MIN_PITCHER_IP,
    expected_ab, expected_hits_lambda, expected_ks,
    game_winner_probability, hr_prob_poisson, parse_hit_threshold,
    parse_k_threshold, poisson_ge, shrink_avg, shrink_hr_rate, total_prob,
)
from slugger.sizing import kelly_count
from slugger.signal_pipeline import evaluate_markets
from slugger.tickers import (
    kalshi_team, kalshi_date, game_event_ticker, ks_event_ticker,
    hr_event_ticker, total_event_ticker, hit_event_ticker, hrr_event_ticker,
)
from slugger.types import (
    BatterProfile, GameContext, GameInfo, MarketClient, MarketSpec,
    ModelResult, PitcherProfile, TeamProfile, TradeSignal,
)

log = logging.getLogger(__name__)

# ── Strategy-specific constants (not model math — kept here) ──────────────────
_KS_MIN_THRESHOLD   = 6       # skip 4+ and 5+ K markets (unprofitable historically)
_KS_MIN_MODEL_PROB  = 15      # minimum model prob (%) to consider trading YES side
_KS_NO_MAX_MODEL_PROB = 10    # buy NO when model says probability is at most this (%)
_KS_NO_MIN_EDGE_CENTS = 5     # minimum edge (market_yes_price - model_prob) to buy NO

# Threshold regex: matches "7+", "over 6.5", "at least 9" in any K-related title
_KS_THRESHOLD_PATTERN = r'(\d+)\s*\+'

# Strategy-specific HR constants (not model math)
_HR_MIN_MODEL_PROB    = 12     # minimum model probability (%) to even consider trading
_HR_MIN_EDGE_CENTS    = 8      # HR-specific minimum edge (higher than global MIN_EDGE_CENTS)
_HR_MIN_AB            = 80     # minimum AB before considering a batter (filter noise)

# Strategy-specific hits constants (not model math)
_HITS_MIN_AB          = 60      # minimum AB before considering a batter
_HITS_MIN_MODEL_PROB  = 12      # minimum model probability (%) to trade
_HITS_MIN_EDGE_CENTS  = 4       # hits-specific minimum edge in cents
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
        max_signals=2,
        no_side=True,
        no_max_model_prob=_KS_NO_MAX_MODEL_PROB,
        no_min_edge_cents=_KS_NO_MIN_EDGE_CENTS,
    )
    return evaluate_markets(spec, ks_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Game Winner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_game_winner(
    game_info: GameInfo,
    client: MarketClient,
    config: Config,
    home_pitcher: Optional[PitcherProfile] = None,
    away_pitcher: Optional[PitcherProfile] = None,
    home_team: Optional[TeamProfile] = None,
    away_team: Optional[TeamProfile] = None,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Game winner prop — multi-factor model via signal pipeline.

    Uses game_winner_probability() which combines:
      - Both pitchers' quality (xERA preferred over ERA)
      - Team offensive strength (runs/game)
      - Bullpen quality
      - Team record / momentum
      - Home field advantage (~54%)

    Called directly by process_game (not through the per-pitcher loop)
    so it has access to both pitchers and both team profiles.
    """
    event_ticker = game_event_ticker(game_info)
    if not event_ticker:
        return []

    home_abbrev = kalshi_team(game_info.home_abbrev)
    home_prob, away_prob = game_winner_probability(
        home_pitcher, away_pitcher, home_team, away_team,
    )

    # ── Build reason string with all model inputs ──────────────────────────
    def _era_str(p: Optional[PitcherProfile]) -> str:
        if not p:
            return "TBD"
        era = p.xera or p.recent_era or p.era
        label = "xERA" if p.xera else ("rERA" if p.recent_era else "ERA")
        return f"{label}={era:.2f}" if era else "TBD"

    def _team_str(t: Optional[TeamProfile]) -> str:
        parts = []
        if t and t.runs_per_game > 0:
            parts.append(f"{t.runs_per_game:.1f}rpg")
        if t and t.bullpen_era > 0:
            parts.append(f"bp={t.bullpen_era:.2f}")
        if t and (t.wins + t.losses) >= 20:
            parts.append(f"{t.wins}-{t.losses}")
        return " ".join(parts) if parts else "?"

    reason_detail = (
        f"Home({game_info.home_abbrev}): {_era_str(home_pitcher)} {_team_str(home_team)}"
        f"  Away({game_info.away_abbrev}): {_era_str(away_pitcher)} {_team_str(away_team)}"
    )

    def gw_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        reason = f"Home win {home_prob}% | {reason_detail}"
        return ModelResult(prob_pct=home_prob, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="game_winner",
        ticker_suffix=home_abbrev,
        confidence_fn=lambda e: min(0.45 + e / 80, 0.75),
    )
    return evaluate_markets(spec, gw_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Total Runs Over/Under
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_total_runs(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Total runs (over/under) prop — ERA bucket model via signal pipeline."""
    event_ticker = total_event_ticker(game_info)
    if not event_ticker or not pitcher_profile.era:
        return []

    era = pitcher_profile.era
    est_over = total_prob(era)

    def total_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        reason = f"Total runs: ERA {era:.1f} → over {est_over}% vs {price}¢"
        return ModelResult(prob_pct=est_over, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="total_runs",
        title_keywords=["over"],
        confidence_fn=lambda _: 0.5,
    )
    return evaluate_markets(spec, total_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Player Home Runs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HR_THRESHOLD_PATTERN = r'(\d+)\+\s*home\s*run'

def strategy_player_hr(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Player home run prop — Poisson model via signal pipeline."""
    event_ticker = hr_event_ticker(game_info)
    if not event_ticker or not batter_profile:
        return []

    if batter_profile.ab < _HR_MIN_AB:
        log.debug(
            "player_hr | %s — only %d AB (need %d) — skipping",
            batter_profile.name, batter_profile.ab, _HR_MIN_AB,
        )
        return []

    opp_hr_per_9 = pitcher_profile.hr_per_9 if pitcher_profile else 0.0
    opp_ip = pitcher_profile.innings_pitched if pitcher_profile else 0.0
    opp_throws = (pitcher_profile.throws if pitcher_profile else "") or ""

    # ── Platoon split selection ────────────────────────────────────────────
    if opp_throws == "L" and batter_profile.vs_lhp_ab >= 20:
        split_hr, split_ab, platoon_note = batter_profile.vs_lhp_hr, batter_profile.vs_lhp_ab, "vsL"
    elif opp_throws == "R" and batter_profile.vs_rhp_ab >= 20:
        split_hr, split_ab, platoon_note = batter_profile.vs_rhp_hr, batter_profile.vs_rhp_ab, "vsR"
    else:
        split_hr = batter_profile.hr
        split_ab = batter_profile.ab
        platoon_note = "overall" if not opp_throws else f"overall({opp_throws}_split<20AB)"

    # ── Compute λ ──────────────────────────────────────────────────────────
    ab_est = expected_ab(batter_profile.batting_order)
    _, eff_rate, pitcher_adj = hr_prob_poisson(
        hr=split_hr, ab=split_ab,
        opp_hr_per_9=opp_hr_per_9, opp_ip=opp_ip,
        batting_order=batter_profile.batting_order,
    )
    lam = eff_rate * ab_est
    if pitcher_adj != 1.0:
        lam *= pitcher_adj

    home_kalshi = kalshi_team(game_info.home_abbrev)
    park_factor = HR_PARK_FACTORS.get(home_kalshi, HR_PARK_FACTORS.get(game_info.home_abbrev.upper(), 1.0))
    lam *= park_factor

    # ── Barrel rate adjustment (batter, dampened) ────────────────────────
    _LEAGUE_AVG_BARREL = 0.065
    barrel_adj = 1.0
    if batter_profile.barrel_rate > 0:
        raw_barrel = batter_profile.barrel_rate / _LEAGUE_AVG_BARREL
        barrel_adj = 1.0 + 0.5 * (raw_barrel - 1.0)
        lam *= barrel_adj

    # ── Exit velocity adjustment (dampened) ────────────────────────────
    # Average exit velo is the strongest single predictor of HR power.
    # League average EV is ~88.5 mph; each mph above adds HR probability.
    _LEAGUE_AVG_EV = 88.5
    ev_adj = 1.0
    if batter_profile.avg_exit_velo > 0:
        ev_diff = batter_profile.avg_exit_velo - _LEAGUE_AVG_EV
        ev_adj = 1.0 + 0.03 * ev_diff  # ~3% per mph, significant for power
        lam *= ev_adj

    # ── xSLG adjustment (dampened) ─────────────────────────────────────
    # xSLG measures expected power production from contact quality (Statcast).
    # Blends in when available to correct for BABIP luck on SLG.
    _LEAGUE_AVG_XSLG = 0.400
    xslg_adj = 1.0
    if batter_profile.xslg > 0:
        raw_xslg = batter_profile.xslg / _LEAGUE_AVG_XSLG
        xslg_adj = 1.0 + 0.3 * (raw_xslg - 1.0)  # lighter dampening
        lam *= xslg_adj

    # ── Pitcher barrel rate against (dampened) ─────────────────────────
    # Pitchers who allow more barrels give up more HRs.
    _LEAGUE_AVG_BRA = 0.065
    bra_adj = 1.0
    if pitcher_profile and pitcher_profile.barrel_rate_against > 0 and opp_ip >= MIN_PITCHER_IP:
        raw_bra = pitcher_profile.barrel_rate_against / _LEAGUE_AVG_BRA
        bra_adj = 1.0 + 0.3 * (raw_bra - 1.0)
        lam *= bra_adj

    # ── Temperature adjustment ─────────────────────────────────────────
    # Higher temperature = more HRs. ~1.5% more HRs per degree F above 72°F.
    temp_adj = 1.0
    temp_str = game_info.weather.get("temp", "")
    if temp_str:
        try:
            temp_f = float(temp_str.replace("°", "").replace("F", "").strip())
            temp_adj = 1.0 + 0.015 * (temp_f - 72.0) / 10.0
            temp_adj = max(0.85, min(1.15, temp_adj))  # cap at ±15%
        except (ValueError, TypeError):
            pass
    if temp_adj != 1.0:
        lam *= temp_adj

    log.debug(
        "%s (#%d)  split=%s %dHR/%dAB  eff=%.4f  park=%s(×%.2f)"
        "  opp=%s(%.0fIP)  pitcher_adj=%.2f  barrel=%.2f  ev=%.2f"
        "  xslg=%.2f  bra=%.2f  temp=%.2f  λ=%.3f",
        batter_profile.name, batter_profile.batting_order,
        platoon_note, split_hr, split_ab, eff_rate,
        home_kalshi, park_factor, opp_throws or "?", opp_ip, pitcher_adj,
        barrel_adj, ev_adj, xslg_adj, bra_adj, temp_adj, lam,
    )

    if lam <= 0:
        return []

    pitcher_note = (
        f"  opp_{opp_throws}hr/9={opp_hr_per_9:.2f}({opp_ip:.0f}IP)"
        if opp_ip >= MIN_PITCHER_IP else ""
    )

    # ── Build model closure ────────────────────────────────────────────────
    def hr_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        if threshold is None:
            return None
        prob_pct = round(poisson_ge(threshold, lam) * 100)
        reason = (
            f"{batter_profile.name}"
            f"  {split_hr}HR/{split_ab}AB({platoon_note})"
            f"  park={park_factor:.2f}"
            f"  barrel_adj={barrel_adj:.2f}"
            f"  λ={lam:.2f}"
            f"  P({threshold}+HR)={prob_pct}%"
            f"{pitcher_note}"
        )
        return ModelResult(prob_pct=prob_pct, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="player_hr",
        player_name=batter_profile.name,
        threshold_pattern=_HR_THRESHOLD_PATTERN,
        min_model_prob=_HR_MIN_MODEL_PROB,
        min_edge_cents=_HR_MIN_EDGE_CENTS,
        confidence_fn=lambda e: min(0.4 + e / 100, 0.75),
    )
    return evaluate_markets(spec, hr_model, client, config, calibration=calibration)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Hits / Runs / RBIs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_player_hits_runs_rbis(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    """Hits + Runs + RBIs prop — AVG bucket model via signal pipeline."""
    event_ticker = hrr_event_ticker(game_info)
    if not event_ticker or not batter_profile:
        return []

    recent_avg = batter_profile.recent_avg if batter_profile.recent_avg > 0 else 0.220
    if recent_avg >= 0.300:
        est_prob = 55
    elif recent_avg >= 0.270:
        est_prob = 42
    elif recent_avg >= 0.240:
        est_prob = 32
    else:
        est_prob = 22

    def hrr_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        # Skip low thresholds (1+, 2+)
        title_lower = title.lower()
        if "1+" in title_lower or "2+" in title_lower:
            return None
        reason = f"{batter_profile.name} avg={recent_avg:.3f} → est {est_prob}% vs {price}¢"
        return ModelResult(prob_pct=est_prob, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="player_hr_rbis",
        player_name=batter_profile.name,
        confidence_fn=lambda _: 0.45,
    )
    return evaluate_markets(spec, hrr_model, client, config, calibration=calibration)


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
    lam = eff_avg * ab_est

    if batter_profile.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter_profile.xba
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
#  STRATEGY: Pitcher Earned Runs (stub)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_pitcher_er(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: MarketClient,
    config: Config,
) -> List[TradeSignal]:
    """Pitcher earned runs prop (Under). Stub — not yet implemented."""
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Same-Game Combo (Parlay)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Multivariate collection tickers that support MLB combos.
# Both of these contain every MLB event type and allow 2+ legs.
MVE_COLLECTIONS = [
    "KXMVESPORTSMULTIGAMEEXTENDED-R",
    "KXMVECROSSCATEGORY-R",
]

# ── Combo tuning constants ────────────────────────────────────────────────────
_COMBO_MIN_LEG_EDGE       = 3      # each leg must have >= this edge (cents) to be eligible
_COMBO_MIN_LEG_PROB       = 10     # each leg must have >= this model prob (%) to be eligible
_COMBO_MIN_COMBO_EDGE     = 3      # minimum edge on the combo itself (cents)
_COMBO_MIN_COMBO_PROB     = 3      # minimum joint model probability (%) to bother
_COMBO_MAX_LEGS           = 3      # maximum number of legs per combo
_COMBO_MAX_COMBOS_PER_GAME = 2     # don't flood with combo orders
_COMBO_MAX_POSITION_SCALE  = 0.5   # use half normal Kelly for combos (higher variance)

# ── Pairwise correlation adjustments ─────────────────────────────────────────
# When two legs are correlated, the naive product P(A)*P(B) over- or under-
# estimates the true joint probability.  These adjustments are applied per
# pair of legs based on their relationship.
#
# Values > 1.0 = positive correlation (events reinforce each other, naive
#   product UNDERESTIMATES joint prob, so we nudge UP)
# Values < 1.0 = negative correlation (events oppose each other, naive
#   product OVERESTIMATES joint prob, so we nudge DOWN)
# Value = 1.0 = independent (no adjustment needed)
#
# These are empirical estimates from MLB game-level correlations.

# Pitcher dominance + team win: strong positive correlation.
# A pitcher who gets 8+ Ks is likely pitching well, helping his team win.
_CORR_PITCHER_TEAM_WIN     = 1.08

# Batter performance + team win: moderate positive correlation.
# A batter who gets 2+ hits is contributing to offense = team more likely to win.
_CORR_BATTER_TEAM_WIN      = 1.05

# Pitcher Ks + opposing batter hits: negative correlation.
# If the pitcher is dominating (high Ks), opposing batters struggle to get hits.
_CORR_PITCHER_VS_OPP_BATTER = 0.88

# Pitcher Ks + total runs over: moderate negative correlation.
# High-K games tend to be lower-scoring (pitcher dominance suppresses runs).
_CORR_PITCHER_KS_TOTAL_OVER = 0.90

# Same-team batters: weak positive correlation.
# Both benefit from the same game flow (rallies), but individual outcomes
# are mostly independent given the game state.
_CORR_SAME_TEAM_BATTERS     = 1.02

# Default: mild negative adjustment (conservative, like original 0.92).
_CORR_DEFAULT               = 0.95


@dataclass
class ComboLeg:
    """A single leg within a combo, capturing the signal and market info."""
    market_ticker: str
    event_ticker: str
    side: str              # "yes" or "no"
    model_prob_pct: float  # our model's probability for this leg (0-100)
    market_price: int      # what the market charges for this side (cents)
    edge_cents: float      # model_prob - market_price
    strategy: str          # which strategy produced this (e.g. "pitcher_ks")
    label: str             # human-readable label (e.g. "Glasnow 7+ Ks")


def _event_ticker_for_market(market_ticker: str) -> str:
    """Derive event ticker from a market ticker.

    Market tickers follow the pattern: {EVENT_TICKER}-{SUFFIX}
    e.g.  KXMLBKS-26MAY231420HOUCHC-CHCCREA53-7
          -> event = KXMLBKS-26MAY231420HOUCHC
    """
    parts = market_ticker.split("-")
    if len(parts) >= 3:
        # Event ticker is the first two dash-separated segments
        return f"{parts[0]}-{parts[1]}"
    return market_ticker


def _leg_team(leg: ComboLeg) -> Optional[str]:
    """Extract the team abbreviation associated with a combo leg.

    For game_winner legs, the team is in the market ticker suffix (e.g. "-LAD").
    For player props, the team is encoded in the player slug portion
    of the ticker (first 2-3 chars of the slug, e.g. "PHIKSCHWARBER12" → "PHI").

    Returns None if team cannot be determined.
    """
    parts = leg.market_ticker.split("-")
    if leg.strategy == "game_winner" and len(parts) >= 3:
        # Game winner: last segment is the team (e.g. "LAD")
        return parts[-1].upper()
    if len(parts) >= 4:
        # Player prop: slug is parts[2], team prefix is first 2-3 chars
        # e.g. KXMLBKS-26MAY...-CHCCREA53-7 → slug "CHCCREA53" → team "CHC" or "CHW"
        slug = parts[2].upper()
        # Try 3-char then 2-char prefix (most teams are 2-3 chars)
        for n in (3, 2):
            if len(slug) > n:
                return slug[:n]
    return None


def _event_type(leg: ComboLeg) -> str:
    """Extract the event type prefix from a leg's event ticker.

    e.g. "KXMLBKS-26MAY..." → "KXMLBKS"
         "KXMLBGAME-26MAY..." → "KXMLBGAME"
    """
    return leg.event_ticker.split("-")[0] if leg.event_ticker else ""


def _pairwise_correlation(leg_a: ComboLeg, leg_b: ComboLeg) -> float:
    """Compute the correlation adjustment factor for a pair of legs.

    Returns a multiplier to apply to the naive independence product:
      > 1.0 = positive correlation (reinforce each other)
      < 1.0 = negative correlation (conflict with each other)
      = 1.0 = independent

    Uses strategy types and team affiliations to determine the relationship.
    """
    type_a = _event_type(leg_a)
    type_b = _event_type(leg_b)
    team_a = _leg_team(leg_a)
    team_b = _leg_team(leg_b)
    strat_a = leg_a.strategy
    strat_b = leg_b.strategy

    # Determine if teams are the same, opposing, or unknown
    same_team = team_a and team_b and team_a == team_b
    # Can't reliably determine "opposing" without the full game context,
    # but if both teams are known and different, they're on opposite sides
    opposing = team_a and team_b and team_a != team_b

    # ── Pitcher Ks + Game Winner ──────────────────────────────────────────
    if {"pitcher_ks", "game_winner"} == {strat_a, strat_b}:
        # Pitcher Ks and his team winning are positively correlated
        # (dominant pitcher helps team win)
        if same_team:
            return _CORR_PITCHER_TEAM_WIN
        # Pitcher Ks and opposing team winning are negatively correlated
        if opposing:
            return 1.0 / _CORR_PITCHER_TEAM_WIN  # inverse

    # ── Batter Hits/HR + Game Winner ──────────────────────────────────────
    batter_strats = {"player_hits", "player_hr", "player_hr_rbis"}
    if strat_a in batter_strats and strat_b == "game_winner":
        if same_team:
            return _CORR_BATTER_TEAM_WIN
        if opposing:
            return 1.0 / _CORR_BATTER_TEAM_WIN
    if strat_b in batter_strats and strat_a == "game_winner":
        if same_team:
            return _CORR_BATTER_TEAM_WIN
        if opposing:
            return 1.0 / _CORR_BATTER_TEAM_WIN

    # ── Pitcher Ks + Opposing Batter Hits ─────────────────────────────────
    if strat_a == "pitcher_ks" and strat_b in batter_strats and opposing:
        return _CORR_PITCHER_VS_OPP_BATTER
    if strat_b == "pitcher_ks" and strat_a in batter_strats and opposing:
        return _CORR_PITCHER_VS_OPP_BATTER

    # ── Pitcher Ks + Total Runs Over ──────────────────────────────────────
    if {"pitcher_ks", "total_runs"} == {strat_a, strat_b}:
        return _CORR_PITCHER_KS_TOTAL_OVER

    # ── Same-team batters ─────────────────────────────────────────────────
    if strat_a in batter_strats and strat_b in batter_strats and same_team:
        return _CORR_SAME_TEAM_BATTERS

    # ── Default ───────────────────────────────────────────────────────────
    return _CORR_DEFAULT


def _combo_joint_prob(legs: List[ComboLeg]) -> float:
    """Compute joint probability for a combo using pairwise correlation.

    Starts with the product of individual leg probabilities (independence
    assumption), then applies directional correlation adjustments for
    each pair of legs based on their strategy types and team affiliations.

    Positive correlations (e.g. pitcher Ks + team win) nudge the joint
    probability UP from the naive product.  Negative correlations (e.g.
    pitcher Ks + opposing batter hits) nudge it DOWN.

    Returns probability as a fraction (0.0 - 1.0).
    """
    if not legs:
        return 0.0

    # Naive independence product
    prob = 1.0
    for leg in legs:
        prob *= leg.model_prob_pct / 100.0

    # Apply pairwise correlation adjustments
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            corr = _pairwise_correlation(legs[i], legs[j])
            prob *= corr

    return max(0.001, min(0.99, prob))


def _dedupe_legs(legs: List[ComboLeg]) -> List[ComboLeg]:
    """Remove duplicate legs (same market_ticker) keeping highest edge."""
    seen: Dict[str, ComboLeg] = {}
    for leg in legs:
        key = leg.market_ticker
        if key not in seen or leg.edge_cents > seen[key].edge_cents:
            seen[key] = leg
    return list(seen.values())


def _player_slug(market_ticker: str) -> Optional[str]:
    """Extract the player slug from a market ticker.

    Market tickers follow: {EVENT}-{DATE_TEAMS}-{PLAYER_SLUG}-{THRESHOLD}
    e.g.  KXMLBHIT-26MAY231605CLEPHI-PHIKSCHWARBER12-3
          -> player slug = "PHIKSCHWARBER12"

    Returns None for game-level markets that have no player slug.
    """
    parts = market_ticker.split("-")
    if len(parts) >= 4:
        return parts[2]
    return None


def _legs_are_valid_combo(legs: List[ComboLeg]) -> bool:
    """Check that a set of legs forms a valid combo.

    Rules:
      - At least 2 legs
      - No two legs from the exact same market ticker
      - No two legs for the same player within the same event type
        (e.g. Schwarber 2+ hits and 3+ hits are nested thresholds
        and Kalshi rejects the combination)
      - Game-level events (GAME, SPREAD, TOTAL, RFI) limited to 1 per type
    """
    if len(legs) < 2:
        return False

    # No duplicate market tickers
    tickers = [leg.market_ticker for leg in legs]
    if len(set(tickers)) != len(tickers):
        return False

    # No two legs for the same player in the same event type.
    # The key is (event_type, player_slug) — if two legs share both,
    # they are different thresholds for the same player prop and Kalshi
    # will reject the combination.
    player_event_keys: set = set()
    for leg in legs:
        event_type = leg.event_ticker.split("-")[0]
        slug = _player_slug(leg.market_ticker)
        if slug:
            key = (event_type, slug)
            if key in player_event_keys:
                return False
            player_event_keys.add(key)

    # Game-level events: max 1 per event type
    game_level_types = {"KXMLBGAME", "KXMLBSPREAD", "KXMLBTOTAL", "KXMLBRFI"}
    type_counts: Dict[str, int] = {}
    for leg in legs:
        event_type = leg.event_ticker.split("-")[0]
        if event_type in game_level_types:
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            if type_counts[event_type] > 1:
                return False

    return True


def build_combo_legs(
    single_leg_signals: List[TradeSignal],
) -> List[ComboLeg]:
    """Convert single-leg TradeSignals into ComboLeg candidates.

    Filters to legs that meet minimum edge and probability thresholds.
    Each signal carries its calibrated model_prob_pct from the pipeline,
    so no reverse-engineering is needed.
    """
    legs: List[ComboLeg] = []

    for sig in single_leg_signals:
        model_prob = sig.model_prob_pct
        if model_prob <= 0:
            # Fallback for signals without model_prob_pct (shouldn't happen
            # but be defensive for signals from older code paths)
            model_prob = sig.price + sig.edge_cents if sig.side == "yes" else 100 - sig.price + sig.edge_cents

        if model_prob < _COMBO_MIN_LEG_PROB:
            continue
        if sig.edge_cents < _COMBO_MIN_LEG_EDGE:
            continue

        event_ticker = _event_ticker_for_market(sig.ticker)
        label = sig.reason[:50] if sig.reason else sig.ticker.split("-")[-1]

        legs.append(ComboLeg(
            market_ticker=sig.ticker,
            event_ticker=event_ticker,
            side=sig.side,
            model_prob_pct=model_prob,
            market_price=sig.price,
            edge_cents=sig.edge_cents,
            strategy=sig.strategy,
            label=label,
        ))

    return _dedupe_legs(legs)


def generate_combos(
    legs: List[ComboLeg],
    max_legs: int = _COMBO_MAX_LEGS,
) -> List[List[ComboLeg]]:
    """Generate all valid 2-to-max_legs combinations from eligible legs.

    Returns combos sorted by expected joint probability (descending),
    with validation applied to each combination.
    """
    if len(legs) < 2:
        return []

    combos: List[List[ComboLeg]] = []
    for n in range(2, min(max_legs, len(legs)) + 1):
        for combo in itertools.combinations(legs, n):
            combo_list = list(combo)
            if _legs_are_valid_combo(combo_list):
                combos.append(combo_list)

    # Sort by joint probability descending (most likely to hit first)
    combos.sort(key=lambda c: _combo_joint_prob(c), reverse=True)
    return combos


def strategy_combo(
    game_info: GameInfo,
    client: MarketClient,
    config: Config,
    single_leg_signals: Optional[List[TradeSignal]] = None,
    **kwargs,
) -> List[TradeSignal]:
    """Same-game combo (parlay) strategy.

    Builds combo legs from single-leg signals already produced by other
    strategies (game_winner, pitcher_ks, player_hits, etc.).  Each signal
    carries its calibrated model_prob_pct from the pipeline, so no
    re-fetching of markets or re-running of models is needed.

    Generates 2-3 leg combos mixing different prop types, creates the
    combo market on Kalshi via the MVE API, and returns TradeSignals
    if edge exists.

    This strategy is called separately in process_game() rather than
    through the STRATEGIES registry.
    """
    signals: List[TradeSignal] = []

    # ── Build combo legs from single-leg signals ───────────────────────────
    legs = build_combo_legs(single_leg_signals or [])

    if len(legs) < 2:
        log.debug("combo | %s@%s — only %d eligible legs, need 2+",
                  game_info.away_abbrev, game_info.home_abbrev, len(legs))
        return signals

    log.info("  🎰 combo | %d eligible legs for %s@%s",
             len(legs), game_info.away_abbrev, game_info.home_abbrev)
    for leg in legs:
        log.debug("    Leg: %s %s  prob=%d%%  edge=%+d¢  [%s] %s",
                  leg.side, leg.market_ticker, leg.model_prob_pct,
                  leg.edge_cents, leg.strategy, leg.label)

    # ── Generate and score combo candidates ────────────────────────────────
    max_legs = min(_COMBO_MAX_LEGS, getattr(config, "combo_max_legs", _COMBO_MAX_LEGS))
    combos = generate_combos(legs, max_legs=max_legs)
    if not combos:
        return signals

    # Prefer combos that mix different strategy types (e.g. game_winner +
    # pitcher_ks + player_hits) over same-type combos. Count distinct
    # strategies in each combo and use that as the primary sort key.
    def _combo_sort_key(combo: List[ComboLeg]) -> Tuple:
        n_types = len(set(leg.strategy for leg in combo))
        joint = _combo_joint_prob(combo)
        return (n_types, joint)

    combos.sort(key=_combo_sort_key, reverse=True)

    # Cap total combos evaluated to avoid API spam on empty orderbooks.
    # Cross-type combos are sorted first, so we'll always try those.
    _MAX_COMBOS_TO_EVALUATE = 15
    if len(combos) > _MAX_COMBOS_TO_EVALUATE:
        combos = combos[:_MAX_COMBOS_TO_EVALUATE]

    log.debug("  combo | Generated %d candidate combos (evaluating top %d)",
              len(combos), min(len(combos), _MAX_COMBOS_TO_EVALUATE))

    combos_placed = 0
    for combo in combos:
        if combos_placed >= _COMBO_MAX_COMBOS_PER_GAME:
            break

        # ── Compute joint probability and fair price ───────────────────────
        joint_prob = _combo_joint_prob(combo)
        joint_prob_pct = round(joint_prob * 100, 1)

        if joint_prob_pct < _COMBO_MIN_COMBO_PROB:
            log.debug("  combo | Skipping %d-leg combo: joint prob %.1f%% < %d%%",
                      len(combo), joint_prob_pct, _COMBO_MIN_COMBO_PROB)
            continue

        fair_price_cents = round(joint_prob * 100)  # what we think YES is worth

        leg_labels = " + ".join(
            f"{leg.side.upper()} {leg.label}" for leg in combo
        )
        log.info("  🎰 combo | Evaluating: %s", leg_labels)
        log.info("      Joint prob: %.1f%%  fair price: %d¢", joint_prob_pct, fair_price_cents)

        # ── Create the combo market on Kalshi ──────────────────────────────
        mve_legs = [
            {
                "market_ticker": leg.market_ticker,
                "event_ticker":  leg.event_ticker,
                "side":          leg.side,
            }
            for leg in combo
        ]

        result = None
        for coll_ticker in MVE_COLLECTIONS:
            result = client.create_combo_market(
                collection_ticker=coll_ticker,
                legs=mve_legs,
                with_market_payload=True,
            )
            if result:
                break

        if not result:
            log.warning("  combo | Failed to create combo market for: %s", leg_labels)
            continue

        combo_market_ticker = result.get("market_ticker", "")
        combo_event_ticker  = result.get("event_ticker", "")
        market_data         = result.get("market", {})

        if not combo_market_ticker:
            log.warning("  combo | No market ticker returned for: %s", leg_labels)
            continue

        # ── Read the combo market's orderbook ──────────────────────────────
        yes_ask = 0
        yes_bid = 0
        if market_data:
            try:
                yes_ask = int(float(market_data.get("yes_ask_dollars", "0")) * 100)
            except (ValueError, TypeError):
                pass
            try:
                yes_bid = int(float(market_data.get("yes_bid_dollars", "0")) * 100)
            except (ValueError, TypeError):
                pass

        log.info("      Market: %s  bid=%d¢ ask=%d¢",
                 combo_market_ticker, yes_bid, yes_ask)

        # ── Determine trade price and edge ─────────────────────────────────
        # If there's a real ask, check edge against it.
        # If the book is empty (ask=0), post our own limit at fair value
        # minus a small buffer (we want to buy below fair).
        if yes_ask > 0 and yes_ask < 100:
            # Someone is offering — check if it's cheap enough
            edge = fair_price_cents - yes_ask
            trade_price = yes_ask
        else:
            # Empty book — post limit at (fair_price - 1 cent) to attract fills
            trade_price = max(1, fair_price_cents - 1)
            edge = 1  # by construction: we're bidding 1c below fair

        # Record signal for calibration regardless of trade decision
        combo_reason = (
            f"COMBO({len(combo)}): {leg_labels}"
            f"  joint_prob={joint_prob_pct:.1f}%"
            f"  fair={fair_price_cents}¢"
            f"  ask={yes_ask}¢"
        )
        traded = edge >= _COMBO_MIN_COMBO_EDGE and fair_price_cents >= _COMBO_MIN_COMBO_PROB
        record_signal(
            config.log_dir, combo_market_ticker, "combo",
            model_prob_pct=round(joint_prob_pct),
            market_price_cents=yes_ask if yes_ask > 0 else trade_price,
            edge_cents=float(edge),
            traded=traded,
            reason=combo_reason,
        )

        if edge < _COMBO_MIN_COMBO_EDGE:
            log.info("      No edge: %d¢ < %d¢ minimum", edge, _COMBO_MIN_COMBO_EDGE)
            continue

        if fair_price_cents < _COMBO_MIN_COMBO_PROB:
            log.info("      Joint prob too low: %d%% < %d%%", fair_price_cents, _COMBO_MIN_COMBO_PROB)
            continue

        # ── Size the position (reduced Kelly for higher-variance combos) ───
        combo_kelly = config.kelly_fraction * _COMBO_MAX_POSITION_SCALE
        count = kelly_count(
            edge, trade_price,
            combo_kelly,
            config.max_position_usd,
            config.max_contracts_per_trade,
        )

        if count <= 0:
            log.debug("      Kelly sizing returned 0 contracts — skipping")
            continue

        signals.append(TradeSignal(
            ticker=combo_market_ticker,
            action="buy",
            side="yes",
            count=count,
            price=trade_price,
            strategy="combo",
            confidence=min(0.3 + edge / 100, 0.70),
            edge_cents=float(edge),
            reason=combo_reason,
        ))
        combos_placed += 1
        log.info(
            "      ✅ Combo signal: %d contracts @ %d¢  edge=%d¢",
            count, trade_price, edge,
        )

    if not signals:
        log.info("  🎰 combo | No combo signals for %s@%s",
                 game_info.away_abbrev, game_info.home_abbrev)

    return signals


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


def _run_game_winner(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return strategy_game_winner(
        ctx.game, client, config,
        home_pitcher=ctx.home_pitcher,
        away_pitcher=ctx.away_pitcher,
        home_team=ctx.home_team,
        away_team=ctx.away_team,
        calibration=calibration,
    )


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


def _run_total_runs(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_pitcher(strategy_total_runs, ctx, client, config, calibration=calibration)


def _run_player_hr(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_batter(strategy_player_hr, ctx, client, config, calibration=calibration)


def _run_player_hits(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_batter(strategy_player_hits, ctx, client, config, calibration=calibration)


def _run_player_hr_rbis(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return _run_per_batter(strategy_player_hits_runs_rbis, ctx, client, config, calibration=calibration)


def _run_combo(
    ctx: GameContext, client: MarketClient, config: Config,
    prior_signals: List[TradeSignal],
    calibration: Optional[CalibrationLayer] = None,
) -> List[TradeSignal]:
    return strategy_combo(ctx.game, client, config, single_leg_signals=prior_signals)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Ordered list of (name, strategy_fn) pairs.  process_game() iterates through
# this list in order, feeding each strategy's output into the next as
# prior_signals.  Combo runs last so it can see all single-leg signals.
#
# To add a new strategy: define the function, write a wrapper, add it here.
# No changes to game_processor.py needed.

STRATEGY_PIPELINE: List[Tuple[str, Any]] = [
    ("game_winner",   _run_game_winner),
    ("pitcher_ks",    _run_pitcher_ks),
    ("total_runs",    _run_total_runs),
    ("player_hr",     _run_player_hr),
    ("player_hits",   _run_player_hits),
    ("player_hr_rbis", _run_player_hr_rbis),
    ("combo",         _run_combo),
]


# ── Legacy exports (backward compatibility) ───────────────────────────────────
# Kept so any code referencing the old registry or set still works.

STRATEGIES = {
    "pitcher_ks":     strategy_pitcher_ks,
    "player_hr":      strategy_player_hr,
    "player_hits":    strategy_player_hits,
    "total_runs":     strategy_total_runs,
    "player_hr_rbis": strategy_player_hits_runs_rbis,
}

BATTER_STRATEGIES: set = {"player_hr", "player_hr_rbis", "player_hits"}