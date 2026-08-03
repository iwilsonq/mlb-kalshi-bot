"""Probability models for Slugger MLB trading bot.

Pure math — no I/O, no Kalshi client, no Config dependency.  Each model
takes player/team profiles and returns probabilities or Poisson lambdas.

Distributions:
  - poisson_ge:    P(X >= n), Poisson — strikeout counts (with negbinom_ge
                   layering measured overdispersion on top)
  - binomial_ge:   P(X >= n), Binomial — hit counts (bounded at-bats)
  - negbinom_ge:   P(X >= n), NB1 with fitted dispersion

Models:
  - expected_ks:              Pitcher strikeout lambda (trained model preferred,
                              fallback_ks_lambda when no artifact on disk)
  - game_winner_probability:  Home/away win probabilities (multi-factor log5;
                              no live strategy — kept with test coverage)

Helpers:
  - expected_ab:                    Expected at-bats per lineup position
  - shrink_avg:                     Bayesian batting average shrinkage
  - pitcher_quality:                Pitcher rating vs league average
  - kalshi_fee_cents_per_contract:  Exact taker fee at a given price
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

# ── Hits model ────────────────────────────────────────────────────────────────
LEAGUE_AVG_H_PER_AB  = 0.243   # 2024 MLB batting average
HITS_PRIOR_AB        = 250     # prior weight for Bayesian shrinkage on AVG
HITS_MIN_PITCHER_IP  = 30.0    # minimum IP to trust pitcher WHIP/BAA
LEAGUE_AVG_WHIP      = 1.28    # 2024 MLB league-average WHIP
MAX_PITCHER_WHIP_ADJ = 1.35    # cap pitcher WHIP multiplier
# Re-derived 2026-08-03 against 7170 real outcomes (mlb-kalshi-bot-6lf): 0.70 is
# empirically optimal even after the binomial fix (Brier 0.17432 vs 0.19311 at
# 1.0; undeflated the model says 77% for 1+ hits when reality is 62%). The
# blended-avg × WHIP × hard-hit × park stack overpredicts per-AB hit probability
# by ~30% and this corrects it. Do not remove without re-running that check.
HITS_LAMBDA_DEFLATOR = 0.70

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
PYTH_EXPONENT  = 1.83     # Pythagorean exponent (PythagPat)
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


def kalshi_fee_cents_per_contract(price_cents: float, fee_rate: float = 0.07) -> float:
    """Kalshi trading fee in cents for one contract at a given price.

    Formula (verified against 724/1042 of our own settled fills exactly):
      fee = ceil_to_cent(fee_rate * P * (1 - P))  per contract

    The fee is symmetric in P, so it is a roughly fixed cost in *notional* but a
    savagely price-dependent cost as a share of *stake*: measured on the journal,
    fee drag was 7.7% of stake at 0-20c entries and 0.6% at 60-80c. Fees were
    $40.24 of a $140.82 total loss — 29% — so this belongs in the edge math,
    not in a flat buffer. A flat 5c buffer is 25% of stake on a 20c contract
    and 6% on an 80c one; the true fee at both is ~1c.

    fee_rate is per-series configurable on Kalshi's side (see
    GET /series/{ticker}/fee_changes); 0.07 is what our MLB fills matched.
    """
    p = min(max(float(price_cents) / 100.0, 0.0), 1.0)
    return math.ceil(fee_rate * p * (1.0 - p) * 100.0)


def negbinom_ge(n: int, lam: float, dispersion: float) -> float:
    """P(X >= n) for a negative binomial with mean lam and variance dispersion*lam.

    Strikeouts per start are overdispersed relative to Poisson. Measured on 649
    holdout starts, conditioning on the model's own predicted lambda so this is
    not just pitcher heterogeneity leaking in:

      pred λ    n   mean K   var K   var/mean
           2   35     0.97    1.50      1.543
           3   72     2.88    3.10      1.077
           4  205     3.91    4.38      1.121
           5  217     5.10    5.30      1.038
           6   94     6.09    7.56      1.243
      weighted                          1.129

    Poisson assumes var/mean == 1, so it gives the tail too little weight and
    understates P(K >= threshold) for the 6+ thresholds pitcher_ks actually
    trades — which is the residual bias left after the retransformation fix.

    Parameterised so var/mean is constant (quasi-Poisson / NB1): p = 1/dispersion
    and r = lam/(dispersion-1). Falls back to Poisson when dispersion <= 1, which
    keeps behaviour identical for a well-specified Poisson fit.

    Note this is the opposite correction to binomial_ge: hits are *under*
    dispersed because at-bats are bounded trials, whereas a start's strikeout
    count mixes over how long the pitcher lasts.
    """
    if lam <= 0:
        return 0.01
    if dispersion <= 1.0 + 1e-9:
        return poisson_ge(n, lam)
    if n <= 0:
        return 0.99

    p = 1.0 / dispersion
    r = lam / (dispersion - 1.0)
    cumulative = 0.0
    for k in range(n):
        try:
            log_pmf = (
                math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                + r * math.log(p) + k * math.log1p(-p)
            )
            cumulative += math.exp(log_pmf)
        except (OverflowError, ValueError):
            break
    return max(0.01, min(0.99, 1.0 - cumulative))


def binomial_ge(n: int, trials: float, p: float) -> float:
    """P(X >= n) for X ~ Binomial(trials, p), with fractional trials supported.

    Hits in a game are binomial, not Poisson: a batter gets a bounded number of
    at-bats and each is one Bernoulli trial. Poisson is overdispersed relative to
    binomial at the same mean — too much mass at zero *and* too much in the tail —
    so using it for hit props biased every threshold in a predictable direction.
    Measured against 7157 real markets with recorded asks and actual outcomes:

      thr   Poisson  Binomial  market   ACTUAL
       1+     58.5%     62.5%   62.6c    62.2%
       2+     23.3%     22.9%   24.6c    24.9%
       3+      7.1%      4.9%    5.5c     5.7%
       4+      2.0%      0.6%    2.1c     1.2%

    Poisson understated 1+ by 4 points and overstated 3+/4+ by 1.3-1.7x. The
    overstatement is what generated phantom edge on longshot props, and is what
    HITS_LAMBDA_DEFLATOR was bolted on to hide.

    `trials` is fractional because expected_ab returns a lineup-weighted average;
    the result blends the two neighbouring integer trial counts.

    Clamped to [0.01, 0.99] to match poisson_ge and avoid degenerate prices.
    """
    if trials <= 0 or p <= 0:
        return 0.01
    p = min(p, 1.0)
    if n <= 0:
        return 0.99

    def _exact(k_trials: int) -> float:
        if k_trials <= 0:
            return 0.0
        if n > k_trials:
            return 0.0
        return sum(
            math.comb(k_trials, k) * (p ** k) * ((1.0 - p) ** (k_trials - k))
            for k in range(n, k_trials + 1)
        )

    lo = int(math.floor(trials))
    frac = trials - lo
    value = (1.0 - frac) * _exact(lo) + frac * _exact(lo + 1)
    return max(0.01, min(0.99, value))


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
    *,
    use_trained: bool = True,
) -> float:
    """Estimate expected strikeouts (Poisson lambda) for today's start.

    Prefer a walk-forward trained log-linear model (slugger.ks_model) when
    logs/ks_model.json exists. Falls back to a reduced hand model without
    multiplicative Statcast stacking.
    """
    recent_k  = profile.recent_k_per_start   # 0 if not populated
    recent_ip = profile.recent_ip_per_start or DEFAULT_IP
    season_k_per_9 = profile.k_per_9 or 0.0
    season_k = (season_k_per_9 / 9.0) * recent_ip

    if use_trained:
        try:
            from slugger.ks_model import get_trained_ks_model
            trained = get_trained_ks_model()
            if trained is not None and trained.n_samples >= 5:
                # 0.0 means "opponent unknown" to callers, and the hand model
                # below reads it that way by skipping the adjustment. The trained
                # model cannot: it has a real positive coefficient on opp_k_rate,
                # so a 0.0 sentinel is indistinguishable from a lineup that never
                # strikes out and costs ~30% of lambda. Training used the league
                # average whenever the opponent could not be resolved
                # (team_k_rate_as_of), so serving must do the same.
                opp = opp_k_rate if opp_k_rate > 0 else LEAGUE_AVG_K_RATE
                lam = trained.predict_lambda(recent_k, season_k, opp)
                max_k = getattr(profile, "max_k_in_start", 0)
                if max_k > 0:
                    lam = min(lam, float(max_k + 1))
                return max(0.0, lam)
        except Exception as exc:
            log.debug("Trained Ks model unavailable: %s", exc)

    lam = fallback_ks_lambda(recent_k, season_k, opp_k_rate)
    if lam <= 0:
        return 0.0

    max_k = getattr(profile, "max_k_in_start", 0)
    if max_k > 0:
        ceiling = max_k + 1
        if lam > ceiling:
            lam = float(ceiling)

    return max(0.0, lam)


def fallback_ks_lambda(
    recent_k: float,
    season_k: float,
    opp_k_rate: float = 0.0,
) -> float:
    """Hand-tuned strikeout lambda: recent/season blend + one dampened opp adj.

    This is the incumbent that the trained model has to beat to be worth
    shipping. Kept as a standalone function so the holdout comparison scores the
    *same* formula the live fallback uses, rather than a copy that can drift.
    """
    if recent_k > 0 and season_k > 0:
        lam = 0.70 * recent_k + 0.30 * season_k
    elif recent_k > 0:
        lam = recent_k
    elif season_k > 0:
        lam = season_k
    else:
        return 0.0

    if opp_k_rate > 0:
        lam *= 1.0 + 0.5 * (opp_k_rate / LEAGUE_AVG_K_RATE - 1.0)

    return max(0.0, lam * KS_LAMBDA_DEFLATOR)




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HOME RUN MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━








# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GAME WINNER MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━




def pythagorean_win_pct(runs_scored: float, runs_allowed: float) -> float:
    """Pythagorean expected win percentage from runs scored/allowed per game.

    Uses the PythagPat exponent of 1.83 (empirically optimal for MLB).
    A team that scores more than it allows has a Pythagorean win% above .500,
    regardless of its actual W-L record.

    Returns 0.500 if either input is zero or invalid.
    """
    if runs_scored <= 0 or runs_allowed <= 0:
        return 0.500
    rs_exp = runs_scored ** PYTH_EXPONENT
    ra_exp = runs_allowed ** PYTH_EXPONENT
    return rs_exp / (rs_exp + ra_exp)


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
        games = home_team.wins + home_team.losses
        # Use Pythagorean win% when run differential is available — it's a
        # better predictor of true quality than actual W-L record.
        if home_team.run_diff != 0 and home_team.runs_per_game > 0:
            runs_allowed = home_team.runs_per_game - (home_team.run_diff / games)
            pyth_pct = pythagorean_win_pct(home_team.runs_per_game, runs_allowed)
            home_rec_q = pyth_pct / 0.500
        else:
            home_rec_q = (home_team.wins / games) / 0.500
    if away_team and (away_team.wins + away_team.losses) >= 20:
        games = away_team.wins + away_team.losses
        if away_team.run_diff != 0 and away_team.runs_per_game > 0:
            runs_allowed = away_team.runs_per_game - (away_team.run_diff / games)
            pyth_pct = pythagorean_win_pct(away_team.runs_per_game, runs_allowed)
            away_rec_q = pyth_pct / 0.500
        else:
            away_rec_q = (away_team.wins / games) / 0.500

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




