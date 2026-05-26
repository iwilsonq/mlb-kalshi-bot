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

from slugger.config import Config
from slugger.mlb_data import GameInfo, PitcherProfile, BatterProfile, get_team_profile
from slugger.kalshi_client import KalshiClient, _market_price
from slugger.journal import record_signal
from slugger.sizing import kelly_count
from slugger.signal_pipeline import MarketSpec, ModelResult, evaluate_markets

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A suggested trade from a strategy."""
    ticker: str
    action: str           # "buy"
    side: str             # "yes" or "no"
    count: int            # number of contracts
    price: int            # limit price in cents (1-99)
    strategy: str         # e.g. "game_winner"
    confidence: float     # 0.0-1.0
    edge_cents: float     # expected edge in cents
    reason: str = ""      # human-readable rationale


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KALSHI EVENT TICKER CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Map MLB Stats API abbreviations → Kalshi codes where they differ.
# Most MLB abbreviations already match Kalshi exactly (LAD, SF, NYY, etc.).
# This table only covers the handful that don't.
API_TO_KALSHI = {
    "SFG": "SF",   # Giants: MLB uses "SFG" in some contexts, Kalshi uses "SF"
    "KCR": "KC",   # Royals
    "SDP": "SD",   # Padres
    "TBR": "TB",   # Rays
    "WSN": "WSH",  # Nationals
    # Legacy fallbacks from old name-slice derivation (kept for safety)
    "SAN": "SF",   "SFN": "SF",
    "LAN": "LAD",  "SDN": "SD",
    "SLN": "STL",  "ANA": "LAA",
    "TAM": "TB",   "NEW": "NYY",
}


def _kalshi_date(game: GameInfo) -> Optional[str]:
    """Format game datetime as Kalshi date string (YYMONDD) with ET offset."""
    if not game.game_datetime:
        return None
    from datetime import timezone, timedelta
    dt = __import__("datetime").datetime.fromisoformat(game.game_datetime.replace("Z", "+00:00"))
    et = timezone(timedelta(hours=-4))
    dt_et = dt.astimezone(et)
    return dt_et.strftime("%y%b%d").upper() + dt_et.strftime("%H%M")


def _kalshi_team(abbrev: str) -> str:
    return API_TO_KALSHI.get(abbrev.upper(), abbrev.upper())


def _game_base(game: GameInfo) -> Optional[str]:
    """Build the base Kalshi game event ticker."""
    d = _kalshi_date(game)
    if not d:
        return None
    return f"KXMLBGAME-{d}{_kalshi_team(game.away_abbrev)}{_kalshi_team(game.home_abbrev)}"


def _ks_event(game: GameInfo) -> Optional[str]:
    d = _kalshi_date(game)
    if not d:
        return None
    return f"KXMLBKS-{d}{_kalshi_team(game.away_abbrev)}{_kalshi_team(game.home_abbrev)}"


def _hr_event(game: GameInfo) -> Optional[str]:
    d = _kalshi_date(game)
    if not d:
        return None
    return f"KXMLBHR-{d}{_kalshi_team(game.away_abbrev)}{_kalshi_team(game.home_abbrev)}"


def _total_event(game: GameInfo) -> Optional[str]:
    d = _kalshi_date(game)
    if not d:
        return None
    return f"KXMLBTOTAL-{d}{_kalshi_team(game.away_abbrev)}{_kalshi_team(game.home_abbrev)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROBABILITY MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# League-average constants for normalisation
_LEAGUE_AVG_K_RATE  = 0.225   # ~22.5% of PAs end in strikeout (2024 MLB avg)
_LEAGUE_AVG_WHIFF   = 0.245   # ~24.5% whiff rate on swings (2024 MLB avg)
_DEFAULT_IP         = 5.5     # default expected IP when recent data is missing
_KS_LAMBDA_DEFLATOR = 0.85    # calibration: model over-predicts by ~15-20%, deflate λ
_KS_MIN_THRESHOLD   = 6       # skip 4+ and 5+ K markets (unprofitable historically)
_KS_MIN_MODEL_PROB  = 15      # minimum model prob (%) to consider trading YES side
_KS_NO_MAX_MODEL_PROB = 10    # buy NO when model says probability is at most this (%)
_KS_NO_MIN_EDGE_CENTS = 5     # minimum edge (market_yes_price - model_prob) to buy NO

# Threshold regex: matches "7+", "over 6.5", "at least 9" in any K-related title
_KS_THRESHOLD_PATTERN = r'(\d+)\s*\+'


def _poisson_ge(n: int, lam: float) -> float:
    """P(X >= n) for a Poisson-distributed random variable with mean lam.

    Uses the exact CDF: P(X >= n) = 1 - sum_{k=0}^{n-1} e^{-lam} * lam^k / k!

    Clamped to [0.01, 0.99] to avoid degenerate edge prices.
    """
    if lam <= 0:
        return 0.01
    cumulative = 0.0
    for k in range(n):
        try:
            cumulative += math.exp(-lam) * (lam ** k) / math.factorial(k)
        except (OverflowError, ValueError):
            break
    return max(0.01, min(0.99, 1.0 - cumulative))


def _expected_ks(
    profile: PitcherProfile,
    opp_k_rate: float = 0.0,
) -> float:
    """Estimate the expected number of strikeouts for a pitcher in today's start.

    Combines:
      - Recent K/start (last 5 starts) — weighted 70%
      - Season K/9 × expected IP        — weighted 30%
      - Opponent team K rate adjustment  (dampened — half-weight)
      - Statcast whiff rate adjustment   (dampened — half-weight)
      - Hard ceiling from demonstrated max Ks

    The opponent and whiff adjustments are dampened toward 1.0 to prevent
    the old problem of multiplicative compounding inflating λ beyond what
    the pitcher has ever demonstrated.

    Returns lambda for the Poisson model.
    """
    # ── Base: recent K/start ───────────────────────────────────────────────
    recent_k  = profile.recent_k_per_start   # 0 if not populated
    recent_ip = profile.recent_ip_per_start or _DEFAULT_IP

    # Season rate: K/9 × expected IP
    season_k_per_9 = profile.k_per_9 or 0.0
    season_k = (season_k_per_9 / 9.0) * recent_ip

    if recent_k > 0 and season_k > 0:
        lam = 0.70 * recent_k + 0.30 * season_k
    elif recent_k > 0:
        lam = recent_k
    elif season_k > 0:
        lam = season_k
    else:
        return 0.0

    # ── Opponent K rate adjustment (dampened) ──────────────────────────────
    # Raw multiplier pulled halfway toward 1.0 to prevent over-adjustment.
    # Example: opp_k_rate=0.26, league=0.225 → raw=1.156 → dampened=1.078
    if opp_k_rate > 0:
        raw_opp = opp_k_rate / _LEAGUE_AVG_K_RATE
        lam *= 1.0 + 0.5 * (raw_opp - 1.0)

    # ── Statcast whiff rate adjustment (dampened) ─────────────────────────
    # Same half-weight dampening toward 1.0.
    if profile.whiff_rate > 0:
        raw_whiff = profile.whiff_rate / _LEAGUE_AVG_WHIFF
        lam *= 1.0 + 0.5 * (raw_whiff - 1.0)

    # ── Hard ceiling: cap λ at max Ks observed + 1 ────────────────────────
    # A pitcher who has never exceeded 6 Ks should not have λ > 7.
    # The +1 buffer allows for a reasonable breakout but prevents the model
    # from projecting far beyond demonstrated ability.
    max_k = getattr(profile, "max_k_in_start", 0)
    if max_k > 0:
        ceiling = max_k + 1
        if lam > ceiling:
            log.debug(
                "%s: capping λ from %.1f to %d (max K in any start: %d)",
                profile.name, lam, ceiling, max_k,
            )
            lam = float(ceiling)

    # ── Calibration deflation ──────────────────────────────────────────────
    # Historical calibration shows the model over-predicts by ~15-20%
    # across the 10-50% probability range. Apply a multiplicative correction.
    lam *= _KS_LAMBDA_DEFLATOR

    return max(0.0, lam)


def _parse_k_threshold(title: str) -> Optional[int]:
    """Extract the integer K threshold from a Kalshi market title.

    Handles patterns like:
      "7+ strikeouts"        → 7
      "Pitcher records 8+ Ks" → 8
      "over 6.5 strikeouts"  → 7  (rounds up)
      "at least 9 strikeouts" → 9
    Returns None if no threshold can be parsed.
    """
    t = title.lower()
    # "N+" pattern — most common Kalshi format
    m = re.search(r'(\d+)\s*\+', t)
    if m:
        return int(m.group(1))
    # "over N.5" or "over N" pattern
    m = re.search(r'over\s+(\d+(?:\.\d+)?)', t)
    if m:
        return int(math.ceil(float(m.group(1))))
    # "at least N" pattern
    m = re.search(r'at\s+least\s+(\d+)', t)
    if m:
        return int(m.group(1))
    return None


_AVG_AB_PER_GAME      = 3.9    # MLB average ABs per player per game
_LEAGUE_AVG_HR_PER_9  = 1.1   # league-average HR allowed per 9 IP (2024)
_LEAGUE_AVG_HR_PER_AB = 0.017  # calibrated to ~6.5% per-game HR rate (0.065 / 3.9 AB)
_HR_PRIOR_AB          = 300    # prior weight in AB-equivalents for shrinkage (stronger pull to mean)
_MIN_PITCHER_IP       = 40.0   # minimum IP before trusting a pitcher's HR/9 (~7 starts)
_MAX_PITCHER_HR_ADJ   = 1.5    # cap pitcher HR/9 multiplier (prevent noise amplification)
_HR_MIN_MODEL_PROB    = 12     # minimum model probability (%) to even consider trading
_HR_MIN_EDGE_CENTS    = 8      # HR-specific minimum edge (higher than global MIN_EDGE_CENTS)
_HR_MIN_AB            = 80     # minimum AB before considering a batter (filter noise)

# HR park factors by home team abbreviation (normalized: 1.0 = league average).
# Source: multi-year (2022-2024) HR park factor data.
# A value of 1.15 means 15% more HRs hit in that park than average.
HR_PARK_FACTORS: Dict[str, float] = {
    # Strongly pitcher-friendly
    "SF":  0.82,   # Oracle Park — marine layer + wind + deep CF
    "MIA": 0.85,   # loanDepot park
    "OAK": 0.87,   # Oakland Coliseum
    "NYM": 0.88,   # Citi Field
    "SEA": 0.89,   # T-Mobile Park
    "LAD": 0.90,   # Dodger Stadium
    "PIT": 0.91,   # PNC Park
    "SD":  0.93,   # Petco Park
    "DET": 0.93,   # Comerica Park
    "TB":  0.94,   # Tropicana Field
    # Slightly pitcher-friendly / neutral
    "STL": 0.96,   # Busch Stadium
    "KC":  0.96,   # Kauffman Stadium
    "WSH": 0.96,   # Nationals Park
    "BOS": 0.97,   # Fenway Park
    "CHC": 0.97,   # Wrigley Field
    "TOR": 1.00,
    "ATL": 1.00,   # Truist Park
    "CHW": 1.00,   # Guaranteed Rate Field
    "MIN": 1.02,   # Target Field
    "LAA": 1.02,   # Angel Stadium
    # Slightly hitter-friendly
    "PHI": 1.03,   # Citizens Bank Park
    "MIL": 1.06,   # American Family Field
    "HOU": 1.08,   # Minute Maid Park (Crawford Boxes in LF)
    "TEX": 1.10,   # Globe Life Field
    "BAL": 1.10,   # Camden Yards
    "CLE": 1.05,   # Progressive Field
    "ARI": 1.05,   # Chase Field (altitude helps)
    # Strongly hitter-friendly
    "CIN": 1.14,   # Great American Ballpark
    "NYY": 1.18,   # Yankee Stadium — short right-field porch
    "COL": 1.38,   # Coors Field — altitude
}


def _shrink_hr_rate(hr: int, ab: int) -> float:
    """Bayesian shrinkage of a batter's HR/AB rate toward league average.

    Uses a Beta-Binomial conjugate prior equivalent to observing
    _HR_PRIOR_AB at-bats at the league-average HR rate.  This means:
      - A batter with 0 AB is assigned pure league average (~2.8%)
      - A batter with 150 AB is weighted 50% actual / 50% prior
      - A batter with 500+ AB is mostly driven by actual data

    This prevents 1 HR in 17 AB (5.9%) from being treated as a
    genuine signal vs. the league-average 2.8%.
    """
    prior_hr = _LEAGUE_AVG_HR_PER_AB * _HR_PRIOR_AB
    return (hr + prior_hr) / (ab + _HR_PRIOR_AB)


def _hr_prob_poisson(
    hr: int,
    ab: int,
    opp_hr_per_9: float = 0.0,
    opp_ip: float = 0.0,
) -> tuple:
    """P(batter hits 1+ HR in a game) using a Poisson model with shrinkage.

    Applies Bayesian shrinkage on the batter's HR/AB rate, then adjusts
    for the opposing pitcher's HR/9 only when they have enough innings
    to make that rate meaningful.

    Returns:
        (probability, effective_hr_per_ab, applied_pitcher_adj)
        so callers can log what drove the estimate.
    """
    effective_rate = _shrink_hr_rate(hr, ab)
    lam = effective_rate * _AVG_AB_PER_GAME

    # Only apply pitcher HR/9 adjustment if they have sufficient IP,
    # and cap it so a small stretch of bad luck can't dominate the estimate.
    pitcher_adj = 1.0
    if opp_hr_per_9 > 0 and opp_ip >= _MIN_PITCHER_IP:
        pitcher_adj = min(opp_hr_per_9 / _LEAGUE_AVG_HR_PER_9, _MAX_PITCHER_HR_ADJ)
        lam *= pitcher_adj

    prob = 1.0 - math.exp(-lam) if lam > 0 else 0.0
    return prob, effective_rate, pitcher_adj


def _total_prob(era: float) -> int:
    """Estimate probability of Over based on combined starting ERA.
    Returns estimated % chance of Over 8.5 runs.
    """
    if era >= 6.0:
        return 75
    elif era >= 5.0:
        return 62
    elif era >= 4.5:
        return 52
    elif era >= 4.0:
        return 43
    elif era >= 3.5:
        return 33
    else:
        return 25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Strikeout Props
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_pitcher_ks(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Strikeout prop bets — Poisson model via signal pipeline."""
    event_ticker = _ks_event(game_info)
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
            opp_abbrev, opp_k_rate * 100, _LEAGUE_AVG_K_RATE * 100,
        )
    except Exception as exc:
        log.debug("Could not fetch opponent K rate: %s", exc)

    # ── Compute expected strikeouts (λ) ────────────────────────────────────
    lam = _expected_ks(pitcher_profile, opp_k_rate)
    if lam <= 0:
        return []

    # ── In-game adjustment ─────────────────────────────────────────────────
    current_ks = getattr(pitcher_profile, "current_ks", None)
    ip_today = getattr(pitcher_profile, "ip_today", None)
    in_game = current_ks is not None and ip_today is not None

    if in_game:
        expected_ip = pitcher_profile.recent_ip_per_start or _DEFAULT_IP
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
                prob_pct = round(_poisson_ge(remaining_needed, lam) * 100)
        else:
            prob_pct = round(_poisson_ge(threshold, lam) * 100)

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
    return evaluate_markets(spec, ks_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Game Winner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_game_winner(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Game winner prop — ERA-based heuristic via signal pipeline."""
    # Only run once per game: skip if this pitcher is NOT the home starter.
    if (
        pitcher_profile
        and game_info.home_pitcher_id
        and pitcher_profile.player_id != game_info.home_pitcher_id
    ):
        return []

    event_ticker = _game_base(game_info)
    if not event_ticker or not pitcher_profile.recent_era:
        return []

    home_abbrev = _kalshi_team(game_info.home_abbrev)
    recent_era = pitcher_profile.recent_era

    def gw_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        est = 55 if recent_era < 3.5 else (48 if recent_era > 5.0 else 50)
        reason = f"Home win: recent ERA {recent_era:.1f} → {est}% vs {price}¢"
        return ModelResult(prob_pct=est, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="game_winner",
        ticker_suffix=home_abbrev,
        confidence_fn=lambda _: 0.55,
    )
    return evaluate_markets(spec, gw_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Total Runs Over/Under
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_total_runs(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Total runs (over/under) prop — ERA bucket model via signal pipeline."""
    event_ticker = _total_event(game_info)
    if not event_ticker or not pitcher_profile.era:
        return []

    era = pitcher_profile.era
    est_over = _total_prob(era)

    def total_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        reason = f"Total runs: ERA {era:.1f} → over {est_over}% vs {price}¢"
        return ModelResult(prob_pct=est_over, reason=reason)

    spec = MarketSpec(
        event_ticker=event_ticker,
        strategy_name="total_runs",
        title_keywords=["over"],
        confidence_fn=lambda _: 0.5,
    )
    return evaluate_markets(spec, total_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Player Home Runs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_HR_THRESHOLD_PATTERN = r'(\d+)\+\s*home\s*run'

def strategy_player_hr(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Player home run prop — Poisson model via signal pipeline."""
    event_ticker = _hr_event(game_info)
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
    _, eff_rate, pitcher_adj = _hr_prob_poisson(
        hr=split_hr, ab=split_ab,
        opp_hr_per_9=opp_hr_per_9, opp_ip=opp_ip,
    )
    lam = eff_rate * _AVG_AB_PER_GAME
    if pitcher_adj != 1.0:
        lam *= pitcher_adj

    home_kalshi = _kalshi_team(game_info.home_abbrev)
    park_factor = HR_PARK_FACTORS.get(home_kalshi, HR_PARK_FACTORS.get(game_info.home_abbrev.upper(), 1.0))
    lam *= park_factor

    _LEAGUE_AVG_BARREL = 0.065
    barrel_adj = 1.0
    if batter_profile.barrel_rate > 0:
        raw_barrel = batter_profile.barrel_rate / _LEAGUE_AVG_BARREL
        barrel_adj = 1.0 + 0.5 * (raw_barrel - 1.0)
        lam *= barrel_adj

    log.debug(
        "%s  split=%s %dHR/%dAB  eff=%.4f  park=%s(×%.2f)"
        "  opp=%s(%.0fIP)  pitcher_adj=%.2f  barrel_adj=%.2f  λ=%.3f",
        batter_profile.name, platoon_note, split_hr, split_ab, eff_rate,
        home_kalshi, park_factor, opp_throws or "?", opp_ip, pitcher_adj,
        barrel_adj, lam,
    )

    if lam <= 0:
        return []

    pitcher_note = (
        f"  opp_{opp_throws}hr/9={opp_hr_per_9:.2f}({opp_ip:.0f}IP)"
        if opp_ip >= _MIN_PITCHER_IP else ""
    )

    # ── Build model closure ────────────────────────────────────────────────
    def hr_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        if threshold is None:
            return None
        prob_pct = round(_poisson_ge(threshold, lam) * 100)
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
    return evaluate_markets(spec, hr_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Hits / Runs / RBIs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_player_hits_runs_rbis(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Hits + Runs + RBIs prop — AVG bucket model via signal pipeline."""
    d = _kalshi_date(game_info)
    event_ticker = f"KXMLBHRR-{d}{_kalshi_team(game_info.away_abbrev)}{_kalshi_team(game_info.home_abbrev)}" if d else None
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
    return evaluate_markets(spec, hrr_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Player Hits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Hits model constants ──────────────────────────────────────────────────────

_LEAGUE_AVG_H_PER_AB  = 0.243   # 2024 MLB batting average
_HITS_PRIOR_AB        = 250     # prior weight for Bayesian shrinkage on AVG
_HITS_MIN_AB          = 60      # minimum AB before considering a batter
_HITS_MIN_MODEL_PROB  = 12      # minimum model probability (%) to trade
_HITS_MIN_EDGE_CENTS  = 4       # hits-specific minimum edge in cents
_HITS_MIN_PITCHER_IP  = 30.0    # minimum IP to trust pitcher WHIP/BAA
_LEAGUE_AVG_WHIP      = 1.28    # 2024 MLB league-average WHIP
_MAX_PITCHER_WHIP_ADJ = 1.35    # cap pitcher WHIP multiplier

# Hit park factors by home team abbreviation (normalized: 1.0 = league average).
# Source: multi-year (2022-2024) hit park factor data.
# A value of 1.05 means 5% more hits in that park than average.
# These differ from HR park factors — e.g. Fenway inflates hits (doubles off the
# wall) but not HR; Coors inflates both.
HIT_PARK_FACTORS: Dict[str, float] = {
    # Pitcher-friendly for hits
    "OAK": 0.93,   # Oakland Coliseum — large foul territory
    "SEA": 0.94,   # T-Mobile Park
    "MIA": 0.95,   # loanDepot park
    "SF":  0.95,   # Oracle Park — marine layer suppresses all contact
    "TB":  0.96,   # Tropicana Field
    "SD":  0.96,   # Petco Park
    "NYM": 0.97,   # Citi Field
    "DET": 0.97,   # Comerica Park
    "PIT": 0.97,   # PNC Park
    "LAD": 0.98,   # Dodger Stadium
    # Neutral
    "STL": 0.99,
    "KC":  0.99,
    "WSH": 1.00,
    "ATL": 1.00,
    "TOR": 1.00,
    "CHW": 1.00,
    "MIN": 1.00,
    "LAA": 1.01,
    "CLE": 1.01,
    "PHI": 1.01,
    "MIL": 1.01,
    "HOU": 1.02,
    "BAL": 1.02,
    # Hitter-friendly for hits
    "TEX": 1.03,   # Globe Life Field
    "ARI": 1.03,   # Chase Field — altitude + dry air
    "CHC": 1.04,   # Wrigley Field — wind out = hits galore
    "CIN": 1.04,   # Great American Ballpark
    "NYY": 1.04,   # Yankee Stadium — short porches = doubles too
    "BOS": 1.06,   # Fenway Park — Green Monster = lots of doubles
    "COL": 1.12,   # Coors Field — altitude king
}


def _hit_event(game: GameInfo) -> Optional[str]:
    """Build Kalshi hit event ticker."""
    d = _kalshi_date(game)
    if not d:
        return None
    return f"KXMLBHIT-{d}{_kalshi_team(game.away_abbrev)}{_kalshi_team(game.home_abbrev)}"


def _shrink_avg(hits: int, ab: int) -> float:
    """Bayesian shrinkage of a batter's batting average toward league average.

    Uses a Beta-Binomial conjugate prior equivalent to observing
    _HITS_PRIOR_AB at-bats at the league-average batting average.  This means:
      - A batter with 0 AB is assigned pure league average (~.243)
      - A batter with 250 AB is weighted ~50/50 actual vs prior
      - A batter with 500+ AB is mostly driven by actual data

    This prevents a .400 hitter in 50 AB from dominating the estimate.
    """
    prior_hits = _LEAGUE_AVG_H_PER_AB * _HITS_PRIOR_AB
    return (hits + prior_hits) / (ab + _HITS_PRIOR_AB)


def _parse_hit_threshold(title: str) -> Optional[int]:
    """Extract the integer hit threshold from a Kalshi market title.

    Handles patterns like:
      "2+ hits"        → 2
      "3+ hits?"       → 3
    Returns None if no threshold can be parsed.
    """
    t = title.lower()
    m = re.search(r'(\d+)\s*\+\s*hit', t)
    if m:
        return int(m.group(1))
    m = re.search(r'over\s+(\d+(?:\.\d+)?)\s*hit', t)
    if m:
        return int(math.ceil(float(m.group(1))))
    m = re.search(r'at\s+least\s+(\d+)\s*hit', t)
    if m:
        return int(m.group(1))
    return None


_HITS_THRESHOLD_PATTERN = r'(\d+)\s*\+\s*hit'

def strategy_player_hits(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
    config: Config,
) -> List[TradeSignal]:
    """Player hits prop — Poisson model via signal pipeline."""
    event_ticker = _hit_event(game_info)
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
    eff_avg = _shrink_avg(split_h, split_ab)
    lam = eff_avg * _AVG_AB_PER_GAME

    if batter_profile.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter_profile.xba
        lam = blended_avg * _AVG_AB_PER_GAME

    pitcher_adj = 1.0
    if opp_whip > 0 and opp_ip >= _HITS_MIN_PITCHER_IP:
        raw_whip = opp_whip / _LEAGUE_AVG_WHIP
        pitcher_adj = min(1.0 + 0.5 * (raw_whip - 1.0), _MAX_PITCHER_WHIP_ADJ)
        lam *= pitcher_adj

    home_kalshi = _kalshi_team(game_info.home_abbrev)
    park_factor = HIT_PARK_FACTORS.get(
        home_kalshi, HIT_PARK_FACTORS.get(game_info.home_abbrev.upper(), 1.0),
    )
    lam *= park_factor

    log.debug(
        "%s  split=%s %dH/%dAB  eff_avg=%.3f  xba=%.3f"
        "  park=%s(×%.2f)  opp_whip=%.2f(%s,%.0fIP)  pitcher_adj=%.2f  λ=%.3f",
        batter_profile.name, platoon_note, split_h, split_ab, eff_avg,
        batter_profile.xba, home_kalshi, park_factor,
        opp_whip, opp_throws or "?", opp_ip, pitcher_adj, lam,
    )

    if lam <= 0:
        return []

    pitcher_note = (
        f"  opp_{opp_throws}whip={opp_whip:.2f}({opp_ip:.0f}IP)"
        if opp_ip >= _HITS_MIN_PITCHER_IP else ""
    )

    # ── Build model closure ────────────────────────────────────────────────
    def hits_model(title: str, threshold: Optional[int], price: int) -> Optional[ModelResult]:
        if threshold is None:
            return None
        prob_pct = round(_poisson_ge(threshold, lam) * 100)
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
    return evaluate_markets(spec, hits_model, client, config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRATEGY: Pitcher Earned Runs (stub)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def strategy_pitcher_er(
    game_info: GameInfo,
    pitcher_profile: PitcherProfile,
    batter_profile: Optional[BatterProfile],
    client: KalshiClient,
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
_COMBO_CORRELATION_PENALTY = 0.92  # multiply joint prob by this per extra leg (dampener)
_COMBO_MAX_COMBOS_PER_GAME = 2     # don't flood with combo orders
_COMBO_MAX_POSITION_SCALE  = 0.5   # use half normal Kelly for combos (higher variance)


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


def _combo_joint_prob(legs: List[ComboLeg]) -> float:
    """Compute joint probability for a combo.

    Starts with the product of individual leg probabilities (independence
    assumption), then applies a correlation penalty for each leg beyond
    the first.  Same-game legs are correlated (e.g. a pitcher who Ks a lot
    tends to suppress hits on the other side), so the naive product
    overstates the true joint probability.

    Returns probability as a fraction (0.0 - 1.0).
    """
    if not legs:
        return 0.0
    prob = 1.0
    for leg in legs:
        prob *= leg.model_prob_pct / 100.0
    # Apply dampener for each leg beyond the first
    extra_legs = len(legs) - 1
    prob *= _COMBO_CORRELATION_PENALTY ** extra_legs
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
    game_info: GameInfo,
    single_leg_signals: List[TradeSignal],
    client: KalshiClient,
    config: Config,
) -> List[ComboLeg]:
    """Convert single-leg TradeSignals into ComboLeg candidates.

    Filters to legs that meet minimum edge and probability thresholds.
    Each signal already has a ticker, price, edge, and probability
    embedded from the single-leg strategy that produced it.
    """
    legs: List[ComboLeg] = []

    for sig in single_leg_signals:
        # Reconstruct model probability from edge + price
        if sig.side == "yes":
            model_prob = sig.price + sig.edge_cents
        else:
            # For NO signals: price is the NO cost, edge is how overpriced YES is
            model_prob = (100 - sig.price) - sig.edge_cents
            model_prob = 100 - model_prob  # convert to NO prob

        if model_prob < _COMBO_MIN_LEG_PROB:
            continue
        if sig.edge_cents < _COMBO_MIN_LEG_EDGE:
            continue

        event_ticker = _event_ticker_for_market(sig.ticker)

        # Build a short human-readable label
        # Ticker like KXMLBKS-26MAY231420HOUCHC-CHCCREA53-7
        # -> "Rea 7+ Ks" (extract from reason or ticker suffix)
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


# ── Direct leg sourcing for combos ────────────────────────────────────────────
# These functions scan markets and build ComboLeg candidates using the same
# probability models as the single-leg strategies, but with relaxed edge
# thresholds.  A leg that isn't worth trading solo (e.g. game winner at 2c
# edge) can still add value inside a combo.

_COMBO_LEG_MIN_PROB = 20   # legs below 20% prob produce very low joint probs


def _source_game_winner_leg(
    game_info: GameInfo,
    away_pitcher: Optional[PitcherProfile],
    home_pitcher: Optional[PitcherProfile],
    client: KalshiClient,
    config: Config,
) -> List[ComboLeg]:
    """Source a game-winner leg by picking the favoured team.

    Uses both pitchers' recent ERA to estimate each side's win probability,
    then returns the side (home or away) with the higher model probability.
    """
    legs: List[ComboLeg] = []
    event_ticker = _game_base(game_info)
    if not event_ticker:
        return legs

    try:
        markets = client.get_event_markets(event_ticker, min_liquidity=0)
    except Exception:
        return legs

    if not markets:
        return legs

    # Estimate win probability from pitcher matchup
    home_era = home_pitcher.recent_era if home_pitcher and home_pitcher.recent_era else 4.50
    away_era = away_pitcher.recent_era if away_pitcher and away_pitcher.recent_era else 4.50

    # Better (lower) ERA = higher win probability.  Scale relative to
    # combined ERA with a home-field advantage baseline of 54%.
    total_era = home_era + away_era
    if total_era > 0:
        # Fraction of "badness" belonging to the away pitcher
        away_frac = away_era / total_era
        # away_frac=0.6 means away pitcher is worse → home more likely to win
        home_prob = round(40 + away_frac * 25)  # range ~45-60%
    else:
        home_prob = 54

    # Cap to reasonable range
    home_prob = max(40, min(65, home_prob))
    away_prob = 100 - home_prob

    home_kalshi = _kalshi_team(game_info.home_abbrev)
    away_kalshi = _kalshi_team(game_info.away_abbrev)

    for m in markets:
        price = _market_price(m)
        if price <= 0 or price >= 100:
            continue
        ticker = m.get("ticker", "")

        if ticker.upper().endswith(f"-{home_kalshi}"):
            model_prob = home_prob
            label = f"{game_info.home_team} win"
        elif ticker.upper().endswith(f"-{away_kalshi}"):
            model_prob = away_prob
            label = f"{game_info.away_team} win"
        else:
            continue

        if model_prob < _COMBO_LEG_MIN_PROB:
            continue

        legs.append(ComboLeg(
            market_ticker=ticker,
            event_ticker=event_ticker,
            side="yes",
            model_prob_pct=model_prob,
            market_price=price,
            edge_cents=model_prob - price,
            strategy="game_winner",
            label=label,
        ))

    # Keep only the favoured side (highest model prob)
    if legs:
        legs.sort(key=lambda l: l.model_prob_pct, reverse=True)
        legs = [legs[0]]

    return legs


def _source_pitcher_ks_legs(
    game_info: GameInfo,
    pitchers: List[PitcherProfile],
    client: KalshiClient,
    config: Config,
) -> List[ComboLeg]:
    """Source pitcher strikeout legs using the Poisson model.

    Picks the single best K threshold per pitcher (highest edge).
    """
    legs: List[ComboLeg] = []
    event_ticker = _ks_event(game_info)
    if not event_ticker:
        return legs

    try:
        markets = client.get_event_markets(event_ticker, min_liquidity=0)
    except Exception:
        return legs

    if not markets:
        return legs

    for pitcher in pitchers:
        if not pitcher or (pitcher.k_per_9 == 0 and pitcher.recent_k_per_start == 0):
            continue

        # Get opponent K rate
        opp_k_rate = 0.0
        try:
            if pitcher.player_id == game_info.away_pitcher_id:
                opp_abbrev = game_info.home_abbrev
            else:
                opp_abbrev = game_info.away_abbrev
            opp_team = get_team_profile(opp_abbrev)
            opp_k_rate = opp_team.k_rate
        except Exception:
            pass

        lam = _expected_ks(pitcher, opp_k_rate)
        if lam <= 0:
            continue

        pitcher_last = pitcher.name.split()[-1].lower() if pitcher.name else ""
        best_leg: Optional[ComboLeg] = None

        for m in markets:
            title = m.get("title", "").lower()
            if "strikeout" not in title and "k+" not in title and " ks" not in title:
                continue
            if pitcher_last and pitcher_last not in title:
                continue

            price = _market_price(m)
            if price <= 0 or price >= 100:
                continue

            threshold = _parse_k_threshold(m.get("title", ""))
            if threshold is None or threshold < _KS_MIN_THRESHOLD:
                continue

            prob_pct = round(_poisson_ge(threshold, lam) * 100)
            if prob_pct < _COMBO_LEG_MIN_PROB:
                continue

            edge = prob_pct - price
            ticker = m.get("ticker", "")

            if best_leg is None or edge > best_leg.edge_cents:
                best_leg = ComboLeg(
                    market_ticker=ticker,
                    event_ticker=event_ticker,
                    side="yes",
                    model_prob_pct=prob_pct,
                    market_price=price,
                    edge_cents=edge,
                    strategy="pitcher_ks",
                    label=f"{pitcher.name} {threshold}+ Ks",
                )

        if best_leg:
            legs.append(best_leg)

    return legs


def _source_player_hits_legs(
    game_info: GameInfo,
    batter_pitcher_pairs: List[Tuple],
    client: KalshiClient,
    config: Config,
) -> List[ComboLeg]:
    """Source player hit legs using the Poisson model.

    Picks the single best hit threshold per batter (highest edge).
    Only considers batters with enough ABs and model probability.
    """
    legs: List[ComboLeg] = []
    event_ticker = _hit_event(game_info)
    if not event_ticker:
        return legs

    try:
        markets = client.get_event_markets(event_ticker, min_liquidity=0)
    except Exception:
        return legs

    if not markets:
        return legs

    for batter, opp_pitcher in batter_pitcher_pairs:
        if not batter or batter.ab < _HITS_MIN_AB:
            continue

        opp_whip = opp_pitcher.whip if opp_pitcher else 0.0
        opp_ip = opp_pitcher.innings_pitched if opp_pitcher else 0.0
        opp_throws = (opp_pitcher.throws if opp_pitcher else "") or ""

        # Choose split
        if opp_throws == "L" and batter.vs_lhp_ab >= 30:
            split_h = round(batter.vs_lhp_avg * batter.vs_lhp_ab)
            split_ab = batter.vs_lhp_ab
        elif opp_throws == "R" and batter.vs_rhp_ab >= 30:
            split_h = round(batter.vs_rhp_avg * batter.vs_rhp_ab)
            split_ab = batter.vs_rhp_ab
        else:
            split_h = batter.hits
            split_ab = batter.ab

        eff_avg = _shrink_avg(split_h, split_ab)
        lam = eff_avg * _AVG_AB_PER_GAME

        # xBA blend
        if batter.xba > 0:
            blended_avg = 0.70 * eff_avg + 0.30 * batter.xba
            lam = blended_avg * _AVG_AB_PER_GAME

        # Pitcher WHIP adjustment
        if opp_whip > 0 and opp_ip >= _HITS_MIN_PITCHER_IP:
            raw_whip = opp_whip / _LEAGUE_AVG_WHIP
            pitcher_adj = min(1.0 + 0.5 * (raw_whip - 1.0), _MAX_PITCHER_WHIP_ADJ)
            lam *= pitcher_adj

        # Park factor
        home_kalshi = _kalshi_team(game_info.home_abbrev)
        park_factor = HIT_PARK_FACTORS.get(
            home_kalshi, HIT_PARK_FACTORS.get(game_info.home_abbrev.upper(), 1.0),
        )
        lam *= park_factor

        if lam <= 0:
            continue

        last_name = batter.name.split()[-1].lower()
        best_leg: Optional[ComboLeg] = None

        for m in markets:
            title = m.get("title", "").lower()
            if last_name not in title or "hit" not in title:
                continue

            price = _market_price(m)
            if price <= 0 or price >= 100:
                continue

            threshold = _parse_hit_threshold(m.get("title", ""))
            if threshold is None:
                continue

            prob_pct = round(_poisson_ge(threshold, lam) * 100)
            if prob_pct < _COMBO_LEG_MIN_PROB:
                continue

            edge = prob_pct - price
            ticker = m.get("ticker", "")

            if best_leg is None or edge > best_leg.edge_cents:
                best_leg = ComboLeg(
                    market_ticker=ticker,
                    event_ticker=event_ticker,
                    side="yes",
                    model_prob_pct=prob_pct,
                    market_price=price,
                    edge_cents=edge,
                    strategy="player_hits",
                    label=f"{batter.name} {threshold}+ hits",
                )

        if best_leg:
            legs.append(best_leg)

    return legs


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
    client: KalshiClient,
    config: Config,
    away_pitcher: Optional[PitcherProfile] = None,
    home_pitcher: Optional[PitcherProfile] = None,
    batter_pitcher_pairs: Optional[List[Tuple]] = None,
    single_leg_signals: Optional[List[TradeSignal]] = None,
) -> List[TradeSignal]:
    """Same-game combo (parlay) strategy.

    Sources legs directly from live markets across three prop types:
      - Game winner (favoured team from pitcher ERA matchup)
      - Pitcher strikeouts (Poisson model, best threshold per pitcher)
      - Player hits (Poisson model, best threshold per batter)

    Also incorporates any existing single-leg signals from other strategies.
    Builds 2-3 leg combos mixing different prop types, creates the combo
    market on Kalshi via the MVE API, and returns TradeSignals if edge exists.

    This strategy is called separately in process_game() rather than
    through the STRATEGIES registry.
    """
    signals: List[TradeSignal] = []

    # ── Source legs from each prop type ─────────────────────────────────────
    pitchers = [p for p in [away_pitcher, home_pitcher] if p]

    gw_legs = _source_game_winner_leg(game_info, away_pitcher, home_pitcher, client, config)
    ks_legs = _source_pitcher_ks_legs(game_info, pitchers, client, config) if pitchers else []
    hit_legs = _source_player_hits_legs(game_info, batter_pitcher_pairs or [], client, config)

    # Also pull in any single-leg signal candidates
    signal_legs = build_combo_legs(game_info, single_leg_signals or [], client, config)

    # Merge all legs, deduplicating by market_ticker (keep highest edge)
    all_legs = _dedupe_legs(gw_legs + ks_legs + hit_legs + signal_legs)

    if len(all_legs) < 2:
        log.debug("combo | %s@%s — only %d eligible legs, need 2+",
                  game_info.away_abbrev, game_info.home_abbrev, len(all_legs))
        return signals

    legs = all_legs

    log.info("  🎰 combo | %d eligible legs for %s@%s  (GW:%d KS:%d HIT:%d SIG:%d)",
             len(legs), game_info.away_abbrev, game_info.home_abbrev,
             len(gw_legs), len(ks_legs), len(hit_legs), len(signal_legs))
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
#  STRATEGY REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRATEGIES = {
    "game_winner":    strategy_game_winner,
    "pitcher_ks":     strategy_pitcher_ks,
    "player_hr":      strategy_player_hr,
    "player_hits":    strategy_player_hits,
    "total_runs":     strategy_total_runs,
    "player_hr_rbis": strategy_player_hits_runs_rbis,
}

# Strategies that are called once per *batter* (require a BatterProfile).
# All other strategies are called once per *game* (require a PitcherProfile).
BATTER_STRATEGIES: set = {"player_hr", "player_hr_rbis", "player_hits"}