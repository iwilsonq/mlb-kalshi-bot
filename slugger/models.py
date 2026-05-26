"""Probability models for Slugger MLB trading bot.

Pure math — no I/O, no Kalshi client, no Config dependency.  Each model
takes player/team profiles and returns probabilities or Poisson lambdas.

Models:
  - poisson_ge:                P(X >= n) for a Poisson random variable
  - expected_ks:               Pitcher strikeout lambda (Poisson rate)
  - hr_prob_poisson:           P(1+ HR) with Bayesian shrinkage
  - game_winner_probability:   Home/away win probabilities (multi-factor log5)
  - expected_hits_lambda:      Expected hits lambda (Poisson rate)
  - total_prob:                Over/under total runs (ERA bucket model)

Helpers:
  - expected_ab:       Expected at-bats per lineup position
  - shrink_hr_rate:    Bayesian HR rate shrinkage
  - shrink_avg:        Bayesian batting average shrinkage
  - pitcher_quality:   Pitcher rating relative to league average
  - parse_k_threshold: Extract K threshold from market title
  - parse_hit_threshold: Extract hit threshold from market title
"""
from __future__ import annotations

import logging
import math
import re
from typing import Dict, Optional, Tuple

from slugger.tickers import kalshi_team
from slugger.types import BatterProfile, PitcherProfile, TeamProfile

log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Strikeout model ───────────────────────────────────────────────────────────
LEAGUE_AVG_K_RATE  = 0.225   # ~22.5% of PAs end in strikeout (2024 MLB avg)
LEAGUE_AVG_WHIFF   = 0.245   # ~24.5% whiff rate on swings (2024 MLB avg)
LEAGUE_AVG_CHASE   = 0.285   # ~28.5% chase rate (swing at pitches outside zone, 2024)
LEAGUE_AVG_FB_VELO = 93.5    # mph, average four-seam fastball velocity (2024)
DEFAULT_IP         = 5.5     # default expected IP when recent data is missing
KS_LAMBDA_DEFLATOR = 0.85    # calibration: model over-predicts by ~15-20%, deflate lambda

# ── Lineup position ───────────────────────────────────────────────────────────
AVG_AB_PER_GAME = 3.9  # MLB average ABs per player per game (fallback)

# Expected plate appearances by batting order position.
# Source: MLB averages (2022-2024).
PA_BY_ORDER = {
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

# ── Home run model ────────────────────────────────────────────────────────────
LEAGUE_AVG_HR_PER_9  = 1.1    # league-average HR allowed per 9 IP (2024)
LEAGUE_AVG_HR_PER_AB = 0.017  # calibrated to ~6.5% per-game HR rate
HR_PRIOR_AB          = 300    # prior weight in AB-equivalents for shrinkage
MIN_PITCHER_IP       = 40.0   # minimum IP before trusting pitcher HR/9
MAX_PITCHER_HR_ADJ   = 1.5    # cap pitcher HR/9 multiplier

# HR park factors by home team abbreviation (normalized: 1.0 = league average).
# Source: multi-year (2022-2024) HR park factor data.
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

# ── Hits model ────────────────────────────────────────────────────────────────
LEAGUE_AVG_H_PER_AB  = 0.243   # 2024 MLB batting average
HITS_PRIOR_AB        = 250     # prior weight for Bayesian shrinkage on AVG
HITS_MIN_AB          = 60      # minimum AB before considering a batter
HITS_MIN_PITCHER_IP  = 30.0    # minimum IP to trust pitcher WHIP/BAA
LEAGUE_AVG_WHIP      = 1.28    # 2024 MLB league-average WHIP
MAX_PITCHER_WHIP_ADJ = 1.35    # cap pitcher WHIP multiplier

# Hit park factors by home team abbreviation (normalized: 1.0 = league average).
# Source: multi-year (2022-2024) hit park factor data.
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

# ── Game winner model ─────────────────────────────────────────────────────────
LEAGUE_AVG_RPG = 4.50     # 2024 MLB average runs per game per team
LEAGUE_AVG_ERA = 4.10     # 2024 MLB league-average ERA
HOME_FIELD_ADV = 0.540    # MLB historical home win rate (~54%)
GW_PITCHING_WEIGHT = 0.40
GW_OFFENSE_WEIGHT  = 0.40
GW_BULLPEN_WEIGHT  = 0.10
GW_RECORD_WEIGHT   = 0.10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE MATH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def poisson_ge(n: int, lam: float) -> float:
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


def expected_ab(batting_order: int) -> float:
    """Return expected at-bats per game adjusted for lineup position.

    Uses lineup-position PA estimates when the batting order is known
    (1-9), falls back to league average (3.9) when unknown (0).
    """
    if batting_order < 1 or batting_order > 9:
        return AVG_AB_PER_GAME
    return PA_BY_ORDER[batting_order]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STRIKEOUT MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def expected_ks(
    profile: PitcherProfile,
    opp_k_rate: float = 0.0,
) -> float:
    """Estimate the expected number of strikeouts for a pitcher in today's start.

    Combines:
      - Recent K/start (last 5 starts) -- weighted 70%
      - Season K/9 x expected IP        -- weighted 30%
      - Opponent team K rate adjustment  (dampened -- half-weight)
      - Statcast whiff rate adjustment   (dampened -- half-weight)
      - Hard ceiling from demonstrated max Ks

    The opponent and whiff adjustments are dampened toward 1.0 to prevent
    multiplicative compounding inflating lambda beyond what the pitcher
    has ever demonstrated.

    Returns lambda for the Poisson model.
    """
    # Base: recent K/start
    recent_k  = profile.recent_k_per_start   # 0 if not populated
    recent_ip = profile.recent_ip_per_start or DEFAULT_IP

    # Season rate: K/9 x expected IP
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

    # Opponent K rate adjustment (dampened)
    if opp_k_rate > 0:
        raw_opp = opp_k_rate / LEAGUE_AVG_K_RATE
        lam *= 1.0 + 0.5 * (raw_opp - 1.0)

    # Statcast whiff rate adjustment (dampened)
    if profile.whiff_rate > 0:
        raw_whiff = profile.whiff_rate / LEAGUE_AVG_WHIFF
        lam *= 1.0 + 0.5 * (raw_whiff - 1.0)

    # Statcast chase rate adjustment (dampened)
    if profile.chase_rate > 0:
        raw_chase = profile.chase_rate / LEAGUE_AVG_CHASE
        lam *= 1.0 + 0.3 * (raw_chase - 1.0)

    # Fastball velocity adjustment
    if profile.avg_fastball_velo > 0:
        velo_diff = profile.avg_fastball_velo - LEAGUE_AVG_FB_VELO
        lam *= 1.0 + 0.01 * velo_diff

    # Hard ceiling: cap lambda at max Ks observed + 1
    max_k = getattr(profile, "max_k_in_start", 0)
    if max_k > 0:
        ceiling = max_k + 1
        if lam > ceiling:
            log.debug(
                "%s: capping lambda from %.1f to %d (max K in any start: %d)",
                profile.name, lam, ceiling, max_k,
            )
            lam = float(ceiling)

    # Calibration deflation
    lam *= KS_LAMBDA_DEFLATOR

    return max(0.0, lam)


def parse_k_threshold(title: str) -> Optional[int]:
    """Extract the integer K threshold from a Kalshi market title.

    Handles patterns like:
      "7+ strikeouts"        -> 7
      "Pitcher records 8+ Ks" -> 8
      "over 6.5 strikeouts"  -> 7  (rounds up)
      "at least 9 strikeouts" -> 9
    Returns None if no threshold can be parsed.
    """
    t = title.lower()
    m = re.search(r'(\d+)\s*\+', t)
    if m:
        return int(m.group(1))
    m = re.search(r'over\s+(\d+(?:\.\d+)?)', t)
    if m:
        return int(math.ceil(float(m.group(1))))
    m = re.search(r'at\s+least\s+(\d+)', t)
    if m:
        return int(m.group(1))
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HOME RUN MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def shrink_hr_rate(hr: int, ab: int) -> float:
    """Bayesian shrinkage of a batter's HR/AB rate toward league average.

    Uses a Beta-Binomial conjugate prior equivalent to observing
    HR_PRIOR_AB at-bats at the league-average HR rate.  This means:
      - A batter with 0 AB is assigned pure league average (~2.8%)
      - A batter with 150 AB is weighted 50% actual / 50% prior
      - A batter with 500+ AB is mostly driven by actual data
    """
    prior_hr = LEAGUE_AVG_HR_PER_AB * HR_PRIOR_AB
    return (hr + prior_hr) / (ab + HR_PRIOR_AB)


def hr_prob_poisson(
    hr: int,
    ab: int,
    opp_hr_per_9: float = 0.0,
    opp_ip: float = 0.0,
    batting_order: int = 0,
) -> Tuple[float, float, float]:
    """P(batter hits 1+ HR in a game) using a Poisson model with shrinkage.

    Applies Bayesian shrinkage on the batter's HR/AB rate, then adjusts
    for the opposing pitcher's HR/9 only when they have enough innings
    to make that rate meaningful.

    Args:
        batting_order: 1-9 lineup position for PA adjustment (0 = use default).

    Returns:
        (probability, effective_hr_per_ab, applied_pitcher_adj)
    """
    effective_rate = shrink_hr_rate(hr, ab)
    lam = effective_rate * expected_ab(batting_order)

    pitcher_adj = 1.0
    if opp_hr_per_9 > 0 and opp_ip >= MIN_PITCHER_IP:
        pitcher_adj = min(opp_hr_per_9 / LEAGUE_AVG_HR_PER_9, MAX_PITCHER_HR_ADJ)
        lam *= pitcher_adj

    prob = 1.0 - math.exp(-lam) if lam > 0 else 0.0
    return prob, effective_rate, pitcher_adj


def total_prob(era: float) -> int:
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
#  GAME WINNER MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def pitcher_quality(pitcher: Optional[PitcherProfile]) -> float:
    """Rate a pitcher relative to league average.

    Returns a multiplier where 1.0 = league average.  Lower ERA means
    a BETTER pitcher, so we invert: quality = league_avg / pitcher_era.

    Prefers xERA > recent ERA > season ERA as the predictive metric.
    """
    if not pitcher:
        return 1.0
    era = pitcher.xera or pitcher.recent_era or pitcher.era
    if not era or era <= 0:
        return 1.0
    return LEAGUE_AVG_ERA / era


def game_winner_probability(
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
    home_pitch_q = pitcher_quality(home_pitcher)
    away_pitch_q = pitcher_quality(away_pitcher)

    home_off_q = 1.0
    away_off_q = 1.0
    if home_team and home_team.runs_per_game > 0:
        home_off_q = home_team.runs_per_game / LEAGUE_AVG_RPG
    if away_team and away_team.runs_per_game > 0:
        away_off_q = away_team.runs_per_game / LEAGUE_AVG_RPG

    home_bp_q = 1.0
    away_bp_q = 1.0
    if home_team and home_team.bullpen_era > 0:
        home_bp_q = LEAGUE_AVG_ERA / home_team.bullpen_era
    if away_team and away_team.bullpen_era > 0:
        away_bp_q = LEAGUE_AVG_ERA / away_team.bullpen_era

    home_rec_q = 1.0
    away_rec_q = 1.0
    if home_team and (home_team.wins + home_team.losses) >= 20:
        home_rec_q = (home_team.wins / (home_team.wins + home_team.losses)) / 0.500
    if away_team and (away_team.wins + away_team.losses) >= 20:
        away_rec_q = (away_team.wins / (away_team.wins + away_team.losses)) / 0.500

    home_rating = (
        GW_PITCHING_WEIGHT * home_pitch_q
        + GW_OFFENSE_WEIGHT * home_off_q
        + GW_BULLPEN_WEIGHT * home_bp_q
        + GW_RECORD_WEIGHT  * home_rec_q
    )
    away_rating = (
        GW_PITCHING_WEIGHT * away_pitch_q
        + GW_OFFENSE_WEIGHT * away_off_q
        + GW_BULLPEN_WEIGHT * away_bp_q
        + GW_RECORD_WEIGHT  * away_rec_q
    )

    if home_rating <= 0 or away_rating <= 0:
        return 54, 46

    home_raw = home_rating * HOME_FIELD_ADV
    away_raw = away_rating * (1.0 - HOME_FIELD_ADV)
    home_prob = home_raw / (home_raw + away_raw)

    home_prob_pct = round(max(30, min(70, home_prob * 100)))
    return home_prob_pct, 100 - home_prob_pct


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HITS MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def shrink_avg(hits: int, ab: int) -> float:
    """Bayesian shrinkage of a batter's batting average toward league average.

    Uses a Beta-Binomial conjugate prior equivalent to observing
    HITS_PRIOR_AB at-bats at the league-average batting average.
    """
    prior_hits = LEAGUE_AVG_H_PER_AB * HITS_PRIOR_AB
    return (hits + prior_hits) / (ab + HITS_PRIOR_AB)


def parse_hit_threshold(title: str) -> Optional[int]:
    """Extract the integer hit threshold from a Kalshi market title.

    Handles patterns like:
      "2+ hits"        -> 2
      "3+ hits?"       -> 3
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


def expected_hits_lambda(
    batter: BatterProfile,
    pitcher: Optional[PitcherProfile],
    home_abbrev: str,
) -> float:
    """Compute expected hits (lambda) for a batter in a single game.

    Combines Bayesian-shrunk batting average, xBA Statcast data, opposing
    pitcher WHIP adjustment, and park factor into a single Poisson rate.

    Args:
        batter:       Batter profile with season stats and Statcast data.
        pitcher:      Opposing pitcher profile (for WHIP adjustment).
        home_abbrev:  Home team abbreviation (for park factor lookup).

    Returns:
        Lambda (expected hits per game) for the Poisson model, or 0 if
        insufficient data.
    """
    if batter.ab < HITS_MIN_AB:
        return 0.0

    opp_whip = pitcher.whip if pitcher else 0.0
    opp_ip = pitcher.innings_pitched if pitcher else 0.0
    opp_throws = (pitcher.throws if pitcher else "") or ""

    # Platoon split selection
    if opp_throws == "L" and batter.vs_lhp_ab >= 30:
        split_h = round(batter.vs_lhp_avg * batter.vs_lhp_ab)
        split_ab = batter.vs_lhp_ab
    elif opp_throws == "R" and batter.vs_rhp_ab >= 30:
        split_h = round(batter.vs_rhp_avg * batter.vs_rhp_ab)
        split_ab = batter.vs_rhp_ab
    else:
        split_h = batter.hits
        split_ab = batter.ab

    ab_est = expected_ab(batter.batting_order)

    eff_avg = shrink_avg(split_h, split_ab)
    lam = eff_avg * ab_est

    # xBA blend
    if batter.xba > 0:
        blended_avg = 0.70 * eff_avg + 0.30 * batter.xba
        lam = blended_avg * ab_est

    # Pitcher WHIP adjustment
    if opp_whip > 0 and opp_ip >= HITS_MIN_PITCHER_IP:
        raw_whip = opp_whip / LEAGUE_AVG_WHIP
        adj = min(1.0 + 0.5 * (raw_whip - 1.0), MAX_PITCHER_WHIP_ADJ)
        lam *= adj

    # Hard hit rate adjustment (dampened)
    _LEAGUE_AVG_HHR = 0.370
    if batter.hard_hit_rate > 0:
        raw_hhr = batter.hard_hit_rate / _LEAGUE_AVG_HHR
        lam *= 1.0 + 0.25 * (raw_hhr - 1.0)

    # Park factor
    home_kalshi = kalshi_team(home_abbrev)
    park_factor = HIT_PARK_FACTORS.get(
        home_kalshi, HIT_PARK_FACTORS.get(home_abbrev.upper(), 1.0),
    )
    lam *= park_factor

    return lam
