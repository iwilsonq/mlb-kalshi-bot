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
from slugger.mlb_data import GameInfo, PitcherProfile, BatterProfile, TeamProfile, get_team_profile
from slugger.kalshi_client import KalshiClient, market_price
from slugger.journal import record_signal
from slugger.sizing import kelly_count
from slugger.signal_pipeline import MarketSpec, ModelResult, evaluate_markets
from slugger.tickers import (
    kalshi_team, kalshi_date, game_event_ticker, ks_event_ticker,
    hr_event_ticker, total_event_ticker, hit_event_ticker, hrr_event_ticker,
)

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
#  PROBABILITY MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# League-average constants for normalisation
_LEAGUE_AVG_K_RATE  = 0.225   # ~22.5% of PAs end in strikeout (2024 MLB avg)
_LEAGUE_AVG_WHIFF   = 0.245   # ~24.5% whiff rate on swings (2024 MLB avg)
_LEAGUE_AVG_CHASE   = 0.285   # ~28.5% chase rate (swing at pitches outside zone, 2024)
_LEAGUE_AVG_FB_VELO = 93.5    # mph, average four-seam fastball velocity (2024)
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

    # ── Statcast chase rate adjustment (dampened) ─────────────────────────
    # Chase rate measures how often batters swing at pitches outside the zone.
    # High chase rate = more whiffs on non-competitive pitches = more Ks.
    # This is distinct from whiff rate (which includes in-zone swinging strikes).
    if profile.chase_rate > 0:
        raw_chase = profile.chase_rate / _LEAGUE_AVG_CHASE
        lam *= 1.0 + 0.3 * (raw_chase - 1.0)  # lighter weight than whiff

    # ── Fastball velocity adjustment ──────────────────────────────────────
    # Velocity correlates with K rate: faster = more Ks.  Each mph above
    # average adds ~0.5 K/9.  We use a dampened multiplicative adjustment
    # so extreme values don't dominate.
    if profile.avg_fastball_velo > 0:
        velo_diff = profile.avg_fastball_velo - _LEAGUE_AVG_FB_VELO
        # ~2% adjustment per mph, dampened by half
        lam *= 1.0 + 0.01 * velo_diff

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


_AVG_AB_PER_GAME      = 3.9    # MLB average ABs per player per game (fallback)

# Expected plate appearances by batting order position.
# Source: MLB averages (2022-2024).  PA includes AB + BB + HBP + SF + SH.
# AB is slightly lower (~0.1-0.2 fewer per PA due to walks), but we use PA
# as the opportunity count since walks can still produce runs/RBIs.
_PA_BY_ORDER = {
    1: 4.30,    # Leadoff — most PA
    2: 4.15,
    3: 4.10,
    4: 4.05,    # Cleanup
    5: 3.95,
    6: 3.85,
    7: 3.70,
    8: 3.55,
    9: 3.40,    # 9th hitter — fewest PA
}


def _expected_ab(batting_order: int) -> float:
    """Return expected at-bats per game adjusted for lineup position.

    Uses lineup-position PA estimates when the batting order is known
    (1-9), falls back to league average (3.9) when unknown (0).
    """
    if batting_order < 1 or batting_order > 9:
        return _AVG_AB_PER_GAME
    return _PA_BY_ORDER[batting_order]
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
    batting_order: int = 0,
) -> tuple:
    """P(batter hits 1+ HR in a game) using a Poisson model with shrinkage.

    Applies Bayesian shrinkage on the batter's HR/AB rate, then adjusts
    for the opposing pitcher's HR/9 only when they have enough innings
    to make that rate meaningful.

    Args:
        batting_order: 1-9 lineup position for PA adjustment (0 = use default).

    Returns:
        (probability, effective_hr_per_ab, applied_pitcher_adj)
        so callers can log what drove the estimate.
    """
    effective_rate = _shrink_hr_rate(hr, ab)
    lam = effective_rate * _expected_ab(batting_order)

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


# ── Game winner model constants ───────────────────────────────────────────────
_LEAGUE_AVG_RPG = 4.50     # 2024 MLB average runs per game per team
_LEAGUE_AVG_ERA = 4.10     # 2024 MLB league-average ERA
_HOME_FIELD_ADV = 0.540    # MLB historical home win rate (~54%)
_GW_PITCHING_WEIGHT = 0.40 # weight of pitching component in overall rating
_GW_OFFENSE_WEIGHT  = 0.40 # weight of offensive component in overall rating
_GW_BULLPEN_WEIGHT  = 0.10 # weight of bullpen component in overall rating
_GW_RECORD_WEIGHT   = 0.10 # weight of team record / momentum


def _pitcher_quality(pitcher: Optional[PitcherProfile]) -> float:
    """Rate a pitcher relative to league average.

    Returns a multiplier where 1.0 = league average.  Lower ERA means
    a BETTER pitcher, so we invert: quality = league_avg / pitcher_era.

    Prefers xERA > recent ERA > season ERA as the predictive metric.
    """
    if not pitcher:
        return 1.0

    # Pick best available metric (xERA > recent ERA > season ERA)
    era = pitcher.xera or pitcher.recent_era or pitcher.era
    if not era or era <= 0:
        return 1.0

    return _LEAGUE_AVG_ERA / era


def _game_winner_probability(
    home_pitcher: Optional[PitcherProfile],
    away_pitcher: Optional[PitcherProfile],
    home_team: Optional[TeamProfile] = None,
    away_team: Optional[TeamProfile] = None,
) -> Tuple[int, int]:
    """Estimate home and away win probabilities using a multi-factor model.

    Combines:
      1. Starting pitcher quality (xERA / recent ERA / season ERA)
      2. Team offensive strength (runs/game, OPS)
      3. Bullpen quality (bullpen ERA)
      4. Team record strength (win% from W/L)
      5. Home field advantage (~54% baseline)

    Uses a log5-inspired approach: each team gets a composite rating,
    and the probability is derived from the ratio of ratings adjusted
    for home field advantage.

    Returns:
        (home_prob, away_prob) as integer percentages summing to 100.
    """
    # ── Component 1: Starting pitcher quality ──────────────────────────────
    home_pitch_q = _pitcher_quality(home_pitcher)
    away_pitch_q = _pitcher_quality(away_pitcher)

    # ── Component 2: Offensive strength ────────────────────────────────────
    home_off_q = 1.0
    away_off_q = 1.0
    if home_team and home_team.runs_per_game > 0:
        home_off_q = home_team.runs_per_game / _LEAGUE_AVG_RPG
    if away_team and away_team.runs_per_game > 0:
        away_off_q = away_team.runs_per_game / _LEAGUE_AVG_RPG

    # ── Component 3: Bullpen quality ───────────────────────────────────────
    home_bp_q = 1.0
    away_bp_q = 1.0
    if home_team and home_team.bullpen_era > 0:
        home_bp_q = _LEAGUE_AVG_ERA / home_team.bullpen_era
    if away_team and away_team.bullpen_era > 0:
        away_bp_q = _LEAGUE_AVG_ERA / away_team.bullpen_era

    # ── Component 4: Team record strength ──────────────────────────────────
    home_rec_q = 1.0
    away_rec_q = 1.0
    if home_team and (home_team.wins + home_team.losses) >= 20:
        home_rec_q = (home_team.wins / (home_team.wins + home_team.losses)) / 0.500
    if away_team and (away_team.wins + away_team.losses) >= 20:
        away_rec_q = (away_team.wins / (away_team.wins + away_team.losses)) / 0.500

    # ── Composite rating (weighted) ────────────────────────────────────────
    home_rating = (
        _GW_PITCHING_WEIGHT * home_pitch_q
        + _GW_OFFENSE_WEIGHT * home_off_q
        + _GW_BULLPEN_WEIGHT * home_bp_q
        + _GW_RECORD_WEIGHT  * home_rec_q
    )
    away_rating = (
        _GW_PITCHING_WEIGHT * away_pitch_q
        + _GW_OFFENSE_WEIGHT * away_off_q
        + _GW_BULLPEN_WEIGHT * away_bp_q
        + _GW_RECORD_WEIGHT  * away_rec_q
    )

    # ── Log5 probability with home field advantage ─────────────────────────
    # log5 formula: P(home) = (home_rating * HFA) / (home_rating * HFA + away_rating * (1 - HFA))
    # where HFA = home field advantage expressed as probability (~0.54)
    if home_rating <= 0 or away_rating <= 0:
        return 54, 46

    home_raw = home_rating * _HOME_FIELD_ADV
    away_raw = away_rating * (1.0 - _HOME_FIELD_ADV)
    home_prob = home_raw / (home_raw + away_raw)

    # Clamp to reasonable range (no team is >70% or <30% to win)
    home_prob_pct = round(max(30, min(70, home_prob * 100)))
    return home_prob_pct, 100 - home_prob_pct


def _expected_hits_lambda(
    batter: BatterProfile,
    pitcher: Optional[PitcherProfile],
    home_abbrev: str,
) -> float:
    """Compute expected hits (λ) for a batter in a single game.

    Combines Bayesian-shrunk batting average, xBA Statcast data, opposing
    pitcher WHIP adjustment, and park factor into a single Poisson rate.

    This is the shared model used by both single-leg player_hits strategy
    and combo leg sourcing.

    Args:
        batter:       Batter profile with season stats and Statcast data.
        pitcher:      Opposing pitcher profile (for WHIP adjustment).
        home_abbrev:  Home team abbreviation (for park factor lookup).

    Returns:
        Lambda (expected hits per game) for the Poisson model, or 0 if
        insufficient data.
    """
    if batter.ab < _HITS_MIN_AB:
        return 0.0

    opp_whip = pitcher.whip if pitcher else 0.0
    opp_ip = pitcher.innings_pitched if pitcher else 0.0
    opp_throws = (pitcher.throws if pitcher else "") or ""

    # ── Platoon split selection ────────────────────────────────────────────
    if opp_throws == "L" and batter.vs_lhp_ab >= 30:
        split_h = round(batter.vs_lhp_avg * batter.vs_lhp_ab)
        split_ab = batter.vs_lhp_ab
    elif opp_throws == "R" and batter.vs_rhp_ab >= 30:
        split_h = round(batter.vs_rhp_avg * batter.vs_rhp_ab)
        split_ab = batter.vs_rhp_ab
    else:
        split_h = batter.hits
        split_ab = batter.ab

    ab_est = _expected_ab(batter.batting_order)

    eff_avg = _shrink_avg(split_h, split_ab)
    lam = eff_avg * ab_est

    # xBA blend
    if batter.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter.xba
        lam = blended_avg * ab_est

    # Pitcher WHIP adjustment
    if opp_whip > 0 and opp_ip >= _HITS_MIN_PITCHER_IP:
        raw_whip = opp_whip / _LEAGUE_AVG_WHIP
        pitcher_adj = min(1.0 + 0.5 * (raw_whip - 1.0), _MAX_PITCHER_WHIP_ADJ)
        lam *= pitcher_adj

    # Hard hit rate adjustment (dampened)
    # Hard-hit balls (95+ mph EV) become hits far more often than soft contact.
    # League average hard-hit rate is ~37%.
    _LEAGUE_AVG_HHR = 0.370
    if batter.hard_hit_rate > 0:
        raw_hhr = batter.hard_hit_rate / _LEAGUE_AVG_HHR
        lam *= 1.0 + 0.25 * (raw_hhr - 1.0)  # lighter dampening

    # Park factor
    home_kalshi = kalshi_team(home_abbrev)
    park_factor = HIT_PARK_FACTORS.get(
        home_kalshi, HIT_PARK_FACTORS.get(home_abbrev.upper(), 1.0),
    )
    lam *= park_factor

    return lam


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
    client: KalshiClient,
    config: Config,
    home_pitcher: Optional[PitcherProfile] = None,
    away_pitcher: Optional[PitcherProfile] = None,
    home_team: Optional[TeamProfile] = None,
    away_team: Optional[TeamProfile] = None,
) -> List[TradeSignal]:
    """Game winner prop — multi-factor model via signal pipeline.

    Uses _game_winner_probability() which combines:
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
    home_prob, away_prob = _game_winner_probability(
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
    event_ticker = total_event_ticker(game_info)
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
    ab_est = _expected_ab(batter_profile.batting_order)
    _, eff_rate, pitcher_adj = _hr_prob_poisson(
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
    if pitcher_profile and pitcher_profile.barrel_rate_against > 0 and opp_ip >= _MIN_PITCHER_IP:
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
    ab_est = _expected_ab(batter_profile.batting_order)
    eff_avg = _shrink_avg(split_h, split_ab)
    lam = eff_avg * ab_est

    if batter_profile.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter_profile.xba
        lam = blended_avg * ab_est

    pitcher_adj = 1.0
    if opp_whip > 0 and opp_ip >= _HITS_MIN_PITCHER_IP:
        raw_whip = opp_whip / _LEAGUE_AVG_WHIP
        pitcher_adj = min(1.0 + 0.5 * (raw_whip - 1.0), _MAX_PITCHER_WHIP_ADJ)
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
    home_team: Optional[TeamProfile] = None,
    away_team: Optional[TeamProfile] = None,
) -> List[ComboLeg]:
    """Source a game-winner leg by picking the favoured team.

    Uses _game_winner_probability() to estimate each side's win probability
    from the full multi-factor model, then returns the favoured side.
    """
    legs: List[ComboLeg] = []
    event_ticker = game_event_ticker(game_info)
    if not event_ticker:
        return legs

    try:
        markets = client.get_event_markets(event_ticker, min_liquidity=0)
    except Exception:
        return legs

    if not markets:
        return legs

    home_prob, away_prob = _game_winner_probability(
        home_pitcher, away_pitcher, home_team, away_team,
    )

    home_kalshi = kalshi_team(game_info.home_abbrev)
    away_kalshi = kalshi_team(game_info.away_abbrev)

    for m in markets:
        price = market_price(m)
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
    event_ticker = ks_event_ticker(game_info)
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

            price = market_price(m)
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
    """Source player hit legs using the shared Poisson model.

    Uses _expected_hits_lambda() — the same model as strategy_player_hits.
    Picks the single best hit threshold per batter (highest edge).
    """
    legs: List[ComboLeg] = []
    event_ticker = hit_event_ticker(game_info)
    if not event_ticker:
        return legs

    try:
        markets = client.get_event_markets(event_ticker, min_liquidity=0)
    except Exception:
        return legs

    if not markets:
        return legs

    for batter, opp_pitcher in batter_pitcher_pairs:
        if not batter:
            continue

        lam = _expected_hits_lambda(batter, opp_pitcher, game_info.home_abbrev)
        if lam <= 0:
            continue

        last_name = batter.name.split()[-1].lower()
        best_leg: Optional[ComboLeg] = None

        for m in markets:
            title = m.get("title", "").lower()
            if last_name not in title or "hit" not in title:
                continue

            price = market_price(m)
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
    home_team: Optional[TeamProfile] = None,
    away_team: Optional[TeamProfile] = None,
    batter_pitcher_pairs: Optional[List[Tuple]] = None,
    single_leg_signals: Optional[List[TradeSignal]] = None,
) -> List[TradeSignal]:
    """Same-game combo (parlay) strategy.

    Sources legs directly from live markets across three prop types:
      - Game winner (multi-factor model with both pitchers + team stats)
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

    gw_legs = _source_game_winner_leg(
        game_info, away_pitcher, home_pitcher, client, config,
        home_team=home_team, away_team=away_team,
    )
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
    # game_winner is called directly by process_game (needs both pitchers + teams)
    "pitcher_ks":     strategy_pitcher_ks,
    "player_hr":      strategy_player_hr,
    "player_hits":    strategy_player_hits,
    "total_runs":     strategy_total_runs,
    "player_hr_rbis": strategy_player_hits_runs_rbis,
}

# Strategies that are called once per *batter* (require a BatterProfile).
# All other strategies are called once per *game* (require a PitcherProfile).
BATTER_STRATEGIES: set = {"player_hr", "player_hr_rbis", "player_hits"}