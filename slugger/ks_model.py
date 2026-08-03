"""Trained pitcher strikeout lambda model (walk-forward safe).

Training target: **actual strikeouts** from historical starts (game logs),
never model λ as a proxy. Features for start i use only starts j < i
(point-in-time). Holdout evaluates Brier of P(K≥n) vs market-implied
probability when market prices are provided.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from slugger.models import LEAGUE_AVG_K_RATE, poisson_ge

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "logs/ks_model.json"

# Relative shrinkage on standardized slopes. recent_k and season_k are highly
# collinear by construction (season is a superset of recent), so the fit needs
# real regularization to avoid large cancelling coefficients.
RIDGE_ALPHA = 0.10

# Plausible bounds for a starter's expected strikeouts in a single start.
# Guardrail only — a fit that needs this clamp is already suspect.
LAMBDA_MIN = 0.0
LAMBDA_MAX = 14.0


@dataclass
class KsModel:
    """Log-linear Poisson rate model for expected strikeouts."""
    intercept: float = 0.0
    coef: List[float] = field(default_factory=lambda: [0.7, 0.3, 0.0])
    as_of: str = ""
    n_samples: int = 0
    # First date of the walk-forward holdout window. Anything on/after this date
    # was never trained on, so it is the only window ROI may be claimed from.
    holdout_from: Optional[str] = None
    holdout_mae: Optional[float] = None
    holdout_model_brier: Optional[float] = None
    holdout_market_brier: Optional[float] = None
    holdout_beats_market: Optional[bool] = None
    # Brier of the hand-tuned models.fallback_ks_lambda on the same holdout.
    # Beating the market is the bar for having an edge; beating the incumbent is
    # the bar for this artifact being worth shipping at all.
    holdout_incumbent_brier: Optional[float] = None
    holdout_beats_incumbent: Optional[bool] = None

    def features(self, recent_k: float, season_k: float, opp_k_rate: float) -> List[float]:
        return [
            math.log1p(max(0.0, recent_k)),
            math.log1p(max(0.0, season_k)),
            max(0.0, opp_k_rate),
        ]

    def predict_lambda(
        self,
        recent_k: float,
        season_k: float,
        opp_k_rate: float = 0.0,
    ) -> float:
        f = self.features(recent_k, season_k, opp_k_rate)
        z = self.intercept
        for c, x in zip(self.coef, f):
            z += c * x
        # Clamp the exponent before exp() so extreme extrapolation can't overflow
        z = min(z, math.log(LAMBDA_MAX))
        return min(max(LAMBDA_MIN, math.exp(z)), LAMBDA_MAX)

    def prob_ge(self, threshold: int, recent_k: float, season_k: float, opp_k_rate: float = 0.0) -> float:
        lam = self.predict_lambda(recent_k, season_k, opp_k_rate)
        return poisson_ge(threshold, lam)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "intercept": self.intercept,
            "coef": self.coef,
            "as_of": self.as_of,
            "n_samples": self.n_samples,
            "holdout_from": self.holdout_from,
            "holdout_mae": self.holdout_mae,
            "holdout_model_brier": self.holdout_model_brier,
            "holdout_market_brier": self.holdout_market_brier,
            "holdout_beats_market": self.holdout_beats_market,
            "holdout_incumbent_brier": self.holdout_incumbent_brier,
            "holdout_beats_incumbent": self.holdout_beats_incumbent,
        }, indent=2))

    @classmethod
    def load(cls, path: str) -> Optional["KsModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            return cls(
                intercept=float(d.get("intercept", 0)),
                coef=[float(x) for x in d.get("coef", [0.7, 0.3, 0.0])],
                as_of=d.get("as_of", "") or "",
                n_samples=int(d.get("n_samples", 0) or 0),
                holdout_from=d.get("holdout_from"),
                holdout_mae=d.get("holdout_mae"),
                holdout_model_brier=d.get("holdout_model_brier"),
                holdout_market_brier=d.get("holdout_market_brier"),
                holdout_beats_market=d.get("holdout_beats_market"),
                holdout_incumbent_brier=d.get("holdout_incumbent_brier"),
                holdout_beats_incumbent=d.get("holdout_beats_incumbent"),
            )
        except Exception as exc:
            log.warning("Could not load KsModel from %s: %s", path, exc)
            return None


def team_k_rate_as_of(
    team_game_logs: Dict[str, List[dict]],
    team: str,
    as_of: str,
    *,
    min_pa: int = 100,
    default: float = LEAGUE_AVG_K_RATE,
) -> float:
    """Batting K% for `team` using only games strictly before `as_of`.

    Season totals from the MLB API reflect *current* standings, not what was
    known on the date of the start, so opponent strength has to be rebuilt from
    per-game logs or it leaks the future.

    team_game_logs: team key → [{"date", "strikeouts", "plate_appearances"}, ...]
    Falls back to `default` below min_pa, where the estimate is mostly noise.
    """
    if not team:
        return default
    games = team_game_logs.get(team)
    if games is None:
        # Tolerate abbreviation vs full-name key mismatches
        want = team.strip().lower()
        for key, val in team_game_logs.items():
            if key.strip().lower() == want:
                games = val
                break
    if not games:
        return default

    ks = 0
    pa = 0
    for g in games:
        d = (g.get("date") or "")[:10]
        if not d or d >= as_of:
            continue
        pa += int(g.get("plate_appearances", g.get("pa", 0)) or 0)
        ks += int(g.get("strikeouts", g.get("k", 0)) or 0)
    if pa < min_pa:
        return default
    return ks / pa


def _poisson_ridge_fit(
    X: List[List[float]],
    y: List[float],
    *,
    alpha: float = RIDGE_ALPHA,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> Tuple[float, List[float]]:
    """Poisson regression with a log link, ridge-penalised on standardized slopes.

    Fitted by IRLS. This replaces OLS on log(actual_k), which was the wrong
    estimator for a count: exponentiating the fit of a log-transformed response
    recovers the *geometric* mean, not the arithmetic one. On real starts that
    understated λ by 0.57 Ks (predicted 3.95 vs actual 4.52, geometric 3.70 vs
    arithmetic 4.53), and a model biased low can never report YES edge — which is
    why the 20¢ gate admitted zero cells.

    A Poisson likelihood targets E[Y] directly, so no deflator or smearing
    correction is needed.
    """
    n = len(X)
    if n == 0:
        return 0.0, [0.0, 0.0, 0.0]
    p = len(X[0])

    means = [sum(row[j] for row in X) / n for j in range(p)]
    stds: List[float] = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) if var > 1e-18 else 0.0)
    live = [j for j in range(p) if stds[j] > 0.0]

    y_bar = sum(y) / n
    if not live:
        return math.log(max(y_bar, 1e-6)), [0.0] * p

    # Design in z-space with an explicit unpenalised intercept column
    Z = [[1.0] + [(row[j] - means[j]) / stds[j] for j in live] for row in X]
    dim = 1 + len(live)
    beta = [math.log(max(y_bar, 1e-6))] + [0.0] * len(live)

    for _ in range(max_iter):
        AtA = [[0.0] * dim for _ in range(dim)]
        Atz = [0.0] * dim
        sum_w = 0.0
        for i in range(n):
            zi = Z[i]
            eta = sum(b * x for b, x in zip(beta, zi))
            eta = min(max(eta, -20.0), 20.0)
            mu = math.exp(eta)
            w = max(mu, 1e-9)          # Poisson IRLS weight
            work = eta + (y[i] - mu) / w  # working response
            sum_w += w
            for a in range(dim):
                Atz[a] += w * zi[a] * work
                for b in range(dim):
                    AtA[a][b] += w * zi[a] * zi[b]
        # Penalise slopes only; scale by total weight so alpha stays comparable
        for a in range(1, dim):
            AtA[a][a] += alpha * sum_w
        new = _solve(AtA, Atz)
        if not any(new):
            break
        delta = max(abs(a - b) for a, b in zip(new, beta))
        beta = new
        if delta < tol:
            break

    coef = [0.0] * p
    for k, j in enumerate(live):
        coef[j] = beta[1 + k] / stds[j]
    intercept = beta[0] - sum(coef[j] * means[j] for j in range(p))
    return intercept, coef


def samples_from_pitcher_game_logs(
    game_logs: Dict[str, List[dict]],
    *,
    as_of: Optional[str] = None,
    opp_k_rate: float = LEAGUE_AVG_K_RATE,
    min_prior_starts: int = 2,
    team_game_logs: Optional[Dict[str, List[dict]]] = None,
) -> List[dict]:
    """Build training rows from historical starts with point-in-time features.

    For each start on date D, features use only starts with date < D.
    actual_k is the true strikeout count on D (not model output).

    game_logs: pitcher_name → [{"date": "YYYY-MM-DD", "strikeouts": int,
                                "opponent": "ABC"}, ...]

    opp_k_rate is only a fallback. Pass team_game_logs to resolve each start's
    real opponent K% as of that date; without it every row carries the same
    constant, the feature has zero training variance, and the fit correctly
    assigns it a coefficient of 0 — meaning opponent strength is ignored at
    inference even though the live path supplies a real value.
    """
    samples: List[dict] = []
    for name, games in game_logs.items():
        ordered = sorted(
            [g for g in games if g.get("date")],
            key=lambda g: g["date"][:10],
        )
        for i, g in enumerate(ordered):
            d = g["date"][:10]
            if as_of and d >= as_of:
                continue
            prior = ordered[:i]
            if len(prior) < min_prior_starts:
                continue
            actual = int(g.get("strikeouts", g.get("k", 0)) or 0)
            recent = prior[-5:]
            recent_k = sum(int(x.get("strikeouts", x.get("k", 0)) or 0) for x in recent) / len(recent)
            season_k = sum(int(x.get("strikeouts", x.get("k", 0)) or 0) for x in prior) / len(prior)
            row_opp = opp_k_rate
            if team_game_logs is not None:
                row_opp = team_k_rate_as_of(
                    team_game_logs, g.get("opponent") or "", d, default=opp_k_rate,
                )
            samples.append({
                "date": d,
                "pitcher": name,
                "opponent": g.get("opponent") or "",
                "recent_k": recent_k,
                "season_k": season_k,
                "opp_k_rate": row_opp,
                "actual_k": float(actual),
            })
    return samples


def _solve(AtA: List[List[float]], Aty: List[float]) -> List[float]:
    """Gauss-Jordan solve for AtA @ x = Aty. Returns zeros if singular."""
    dim = len(Aty)
    if dim == 0:
        return []
    M = [AtA[i][:] + [Aty[i]] for i in range(dim)]
    for col in range(dim):
        pivot = col
        for r in range(col + 1, dim):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            # Column carries no independent information — leave coefficient at 0
            M[col][col] = 1.0
            M[col][dim] = 0.0
        piv = M[col][col]
        for c in range(col, dim + 1):
            M[col][c] /= piv
        for r in range(dim):
            if r == col:
                continue
            factor = M[r][col]
            for c in range(col, dim + 1):
                M[r][c] -= factor * M[col][c]
    return [M[i][dim] for i in range(dim)]


def _ridge_fit(
    X: List[List[float]],
    y: List[float],
    *,
    alpha: float = RIDGE_ALPHA,
) -> Tuple[float, List[float]]:
    """Ridge regression on standardized features, intercept unpenalized.

    Standardization matters: `log1p(recent_k)` and `log1p(season_k)` are
    strongly collinear, so an unstandardized penalty of 1e-3 leaves the fit
    free to assign huge cancelling weights (e.g. +15.4 / -14.1) that explode
    out of sample. Centering the response and penalizing only the slopes in
    z-space bounds each coefficient by |z'y| / (alpha * n) along the
    near-degenerate direction.

    alpha is relative shrinkage: for orthogonal standardized features the
    fitted slope is roughly ols/(1 + alpha).
    """
    n = len(X)
    if n == 0:
        return 0.0, [0.0, 0.0, 0.0]
    p = len(X[0])

    means = [sum(row[j] for row in X) / n for j in range(p)]
    stds: List[float] = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) if var > 1e-18 else 0.0)

    y_bar = sum(y) / n
    live = [j for j in range(p) if stds[j] > 0.0]
    if not live:
        return y_bar, [0.0] * p

    # Standardized design over the non-constant columns only
    Z = [[(row[j] - means[j]) / stds[j] for j in live] for row in X]
    yc = [v - y_bar for v in y]

    dim = len(live)
    ZtZ = [[0.0] * dim for _ in range(dim)]
    Zty = [0.0] * dim
    for i in range(n):
        zi = Z[i]
        for a in range(dim):
            Zty[a] += zi[a] * yc[i]
            for b in range(dim):
                ZtZ[a][b] += zi[a] * zi[b]
    for i in range(dim):
        ZtZ[i][i] += alpha * n

    b_std = _solve(ZtZ, Zty)

    coef = [0.0] * p
    for k, j in enumerate(live):
        coef[j] = b_std[k] / stds[j]
    intercept = y_bar - sum(coef[j] * means[j] for j in range(p))
    return intercept, coef


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    if not probs or len(probs) != len(outcomes):
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def holdout_brier_vs_incumbent(
    holdout_props: Sequence[dict],
) -> Tuple[Optional[float], int]:
    """Brier of the hand-tuned fallback on the same holdout rows.

    Scores models.fallback_ks_lambda — the exact formula expected_ks uses when
    no trained model is on disk — so "is the trained model an improvement?" can
    be answered instead of assumed.
    """
    from slugger.models import fallback_ks_lambda

    ps: List[float] = []
    ys: List[int] = []
    for row in holdout_props:
        thr = int(row.get("threshold", 0) or 0)
        if thr <= 0 or row.get("market_price_cents") is None:
            continue
        if row.get("synthetic_market"):
            continue
        lam = fallback_ks_lambda(
            float(row.get("recent_k", 0) or 0),
            float(row.get("season_k", 0) or 0),
            float(row.get("opp_k_rate", 0) or 0),
        )
        ps.append(poisson_ge(thr, lam))
        ys.append(1 if float(row.get("actual_k", 0) or 0) >= thr else 0)
    if len(ys) < 5:
        return None, len(ys)
    return brier_score(ps, ys), len(ys)


def holdout_brier_vs_market(
    model: KsModel,
    holdout_props: Sequence[dict],
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    """Compare model P(K≥n) Brier to market_price_cents/100 on holdout props.

    Each prop: recent_k, season_k, opp_k_rate, threshold, actual_k, market_price_cents
    market_price_cents must be the real recorded ask/mid — no synthetic defaults.
    """
    model_ps: List[float] = []
    market_ps: List[float] = []
    ys: List[int] = []
    for row in holdout_props:
        thr = int(row.get("threshold", 0) or 0)
        if thr <= 0:
            continue
        actual = float(row.get("actual_k", 0) or 0)
        mkt = row.get("market_price_cents")
        if mkt is None:
            continue
        # Reject sentinel synthetic prices used only in older code paths
        if row.get("synthetic_market"):
            continue
        p_model = model.prob_ge(
            thr,
            float(row.get("recent_k", 0) or 0),
            float(row.get("season_k", 0) or 0),
            float(row.get("opp_k_rate", 0) or 0),
        )
        p_mkt = min(max(float(mkt) / 100.0, 1e-6), 1.0 - 1e-6)
        y = 1 if actual >= thr else 0
        model_ps.append(p_model)
        market_ps.append(p_mkt)
        ys.append(y)
    if len(ys) < 5:
        return None, None, None
    mb = brier_score(model_ps, ys)
    kb = brier_score(market_ps, ys)
    beats = None
    if mb is not None and kb is not None:
        beats = mb < kb
    return mb, kb, beats


def build_holdout_props_from_signals(
    signals: Sequence[dict],
    game_logs: Dict[str, List[dict]],
    *,
    as_of: Optional[str] = None,
) -> List[dict]:
    """Join pitcher_ks signals (real market prices) to actual game-log Ks + PIT features.

    Only includes rows with a real market_price_cents/ask_cents from signals.jsonl.
    """
    from slugger.calibration import _parse_ks_signal

    # actual K by (name_lower, date)
    ks_by: Dict[Tuple[str, str], int] = {}
    for name, games in game_logs.items():
        for g in games:
            d = (g.get("date") or "")[:10]
            if not d:
                continue
            ks_by[(name.lower(), d)] = int(g.get("strikeouts", g.get("k", 0)) or 0)

    # PIT features keyed by date only within each pitcher (name matching is fuzzy)
    samples = samples_from_pitcher_game_logs(game_logs, as_of=as_of, min_prior_starts=2)
    feat_by_date: Dict[str, List[dict]] = {}
    for s in samples:
        feat_by_date.setdefault(s["date"], []).append(s)

    props: List[dict] = []
    seen: set = set()
    for sig in signals:
        if sig.get("strategy") != "pitcher_ks":
            continue
        px = sig.get("market_price_cents")
        if px is None:
            px = sig.get("ask_cents")
        if px is None or float(px) <= 0:
            continue
        parsed = _parse_ks_signal(sig)
        if parsed is None:
            continue
        name, date, threshold, _prob = parsed
        if not date or (as_of and date >= as_of):
            continue
        dedup = (sig.get("ticker", ""), date, threshold)
        if dedup in seen:
            continue
        seen.add(dedup)

        # Match actual Ks: prefer full-name game log keys containing last name
        last = name.split()[-1].lower() if name else ""
        actual = None
        for (n, d), k in ks_by.items():
            if d != date:
                continue
            if last and last in n.replace(" ", ""):
                actual = k
                break
            if name.lower() in n or n in name.lower():
                actual = k
                break
        if actual is None:
            continue

        # PIT features: any sample on that date with matching pitcher fragment
        feats = None
        for s in feat_by_date.get(date, []):
            pn = (s.get("pitcher") or "").lower()
            if last and last in pn.replace(" ", ""):
                feats = s
                break
        if feats is None and feat_by_date.get(date):
            # fallback: use first PIT sample that day only if unique pitcher day
            day = feat_by_date[date]
            if len(day) == 1:
                feats = day[0]
        if feats is None:
            continue

        props.append({
            "date": date,
            "ticker": sig.get("ticker", ""),
            "threshold": int(threshold),
            "actual_k": float(actual),
            "recent_k": float(feats["recent_k"]),
            "season_k": float(feats["season_k"]),
            "opp_k_rate": float(feats.get("opp_k_rate", 0.225)),
            "market_price_cents": float(px),
            "synthetic_market": False,
        })
    return props


def journal_roi_for_strategy(
    records: Sequence[dict],
    strategy: str = "pitcher_ks",
    *,
    min_edge_cents: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, float]:
    """Settled trade ROI for a strategy, optionally filtered by edge/date gates.

    Returns dict with n, cost_usd, pnl_usd, roi_pct (or empty n=0).
    """
    trades = {}
    for r in records:
        if r.get("type") != "trade":
            continue
        if r.get("strategy") != strategy:
            continue
        d = (r.get("date") or "")[:10]
        if date_from and d and d < date_from:
            continue
        if date_to and d and d >= date_to:
            continue
        if min_edge_cents is not None:
            edge = r.get("edge_cents")
            if edge is None or float(edge) < min_edge_cents:
                continue
        trades[r["ticker"]] = r

    cost = pnl = 0.0
    n = 0
    for r in records:
        if r.get("type") != "settlement":
            continue
        t = trades.get(r.get("ticker", ""))
        if not t:
            continue
        if r.get("market_result") == "void":
            continue
        n += 1
        cost += float(t.get("cost_usd") or 0.0)
        pnl += float(r.get("pnl_usd") or 0.0)
    roi = (pnl / cost * 100.0) if cost > 0 else 0.0
    return {"n": float(n), "cost_usd": cost, "pnl_usd": pnl, "roi_pct": roi}


# Phase-0 trading floors for model-scored ROI (must match strategies.py Ks gates)
_PHASE0_MIN_MODEL_PROB = 25
_PHASE0_MAX_MODEL_PROB = 55
_PHASE0_MIN_EDGE_CENTS = 20
_PHASE0_MIN_THRESHOLD = 6


def model_roi_vs_phase0_baseline(
    model: KsModel,
    cells: Sequence[dict],
    journal_records: Sequence[dict],
    *,
    min_n: int = 10,
    cost_buffer_cents: float = 5.0,
    stake_usd: float = 1.0,
    strategy: str = "pitcher_ks",
    holdout_only: bool = True,
) -> Dict[str, object]:
    """Score trained KsModel ROI on holdout cells vs journal Phase-0 baseline.

    Model path (must call model.prob_ge / predict_lambda):
      - P(K≥threshold) from trained model
      - Trade YES only if threshold≥6, model prob in [25,55], net edge≥20¢
        where net_edge = model_prob_pct - market_price_cents - cost_buffer
      - Settle from actual_k; stake fixed stake_usd per trade at market price

    Baseline path:
      - journal_roi_for_strategy(records, pitcher_ks) on all settled journal trades

    holdout_only (default): drop cells dated before model.holdout_from, so ROI is
    claimed only on starts the model never trained on. Passing the full prop set
    would report an in-sample number and overstate the edge.

    Fail-closed: not_worse_than_baseline is True only when model_n >= min_n
    AND model ROI ≥ baseline ROI. Empty/insufficient → False + insufficient_data.
    """
    baseline = journal_roi_for_strategy(journal_records, strategy)

    scored = list(cells)
    n_in_sample_dropped = 0
    if holdout_only and model.holdout_from:
        kept = [c for c in scored if (c.get("date") or "") >= model.holdout_from]
        n_in_sample_dropped = len(scored) - len(kept)
        scored = kept

    cost = 0.0
    pnl = 0.0
    n = 0
    traded_details: List[dict] = []
    # Which gate rejected each cell. When model_n is 0 this is the only thing
    # that says whether the model never finds edge or the gates are too tight.
    rejected = {
        "threshold_below_min": 0,
        "no_market_price": 0,
        "synthetic_market": 0,
        "prob_below_band": 0,
        "prob_above_band": 0,
        "edge_below_min": 0,
    }

    for cell in scored:
        thr = int(cell.get("threshold", 0) or 0)
        if thr < _PHASE0_MIN_THRESHOLD:
            rejected["threshold_below_min"] += 1
            continue
        mkt = cell.get("market_price_cents")
        if mkt is None or float(mkt) <= 0 or float(mkt) >= 100:
            rejected["no_market_price"] += 1
            continue
        if cell.get("synthetic_market"):
            rejected["synthetic_market"] += 1
            continue
        recent = float(cell.get("recent_k", 0) or 0)
        season = float(cell.get("season_k", 0) or 0)
        opp = float(cell.get("opp_k_rate", 0) or 0)
        actual = float(cell.get("actual_k", 0) or 0)

        # CRITICAL: use trained model probability (not journal edge_cents)
        p_yes = model.prob_ge(thr, recent, season, opp)
        model_pct = p_yes * 100.0
        price = float(mkt)
        net_edge = model_pct - price - float(cost_buffer_cents)

        if model_pct < _PHASE0_MIN_MODEL_PROB:
            rejected["prob_below_band"] += 1
            continue
        if model_pct > _PHASE0_MAX_MODEL_PROB:
            rejected["prob_above_band"] += 1
            continue
        if net_edge < _PHASE0_MIN_EDGE_CENTS:
            rejected["edge_below_min"] += 1
            continue

        # Trade 1 unit sized so cost ≈ stake_usd at market price
        # cost = count * price/100 = stake_usd → count = stake_usd * 100 / price
        # For ROI we use dollar stake equal to cost of 1 contract * scale
        trade_cost = stake_usd
        contracts = trade_cost / (price / 100.0)
        # YES settlement: if win, receive contracts * $1; profit = contracts*(1 - p)
        # loss = -contracts * p  (= -trade_cost)
        won = actual >= thr
        if won:
            trade_pnl = contracts * (1.0 - price / 100.0)
        else:
            trade_pnl = -trade_cost

        n += 1
        cost += trade_cost
        pnl += trade_pnl
        traded_details.append({
            "threshold": thr,
            "model_pct": model_pct,
            "price": price,
            "net_edge": net_edge,
            "won": won,
            "pnl": trade_pnl,
        })

    model_roi = (pnl / cost * 100.0) if cost > 0 else 0.0
    model_stats = {
        "n": float(n),
        "cost_usd": cost,
        "pnl_usd": pnl,
        "roi_pct": model_roi,
    }

    common = {
        "baseline": baseline,
        "model": model_stats,
        "min_n": min_n,
        "n_traded": n,
        "n_cells_scored": len(scored),
        "n_in_sample_dropped": n_in_sample_dropped,
        "holdout_from": model.holdout_from,
        "cost_buffer_cents": cost_buffer_cents,
        "rejected_by_gate": rejected,
    }

    if n < min_n:
        return {
            **common,
            "not_worse_than_baseline": False,
            "status": "insufficient_data",
        }

    not_worse = model_roi >= baseline["roi_pct"] - 1e-9
    return {
        **common,
        "not_worse_than_baseline": not_worse,
        "status": "ok" if not_worse else "worse_than_baseline",
    }


def fit_ks_model(
    samples: Sequence[dict],
    *,
    as_of: Optional[str] = None,
    holdout_frac: float = 0.2,
    holdout_props: Optional[Sequence[dict]] = None,
    ridge_alpha: float = RIDGE_ALPHA,
) -> KsModel:
    """Fit from samples with actual_k (true outcomes). Walk-forward as_of filter."""
    rows = []
    for s in samples:
        if as_of:
            d = (s.get("date") or "")[:10]
            if not d or d >= as_of:
                continue
        actual = float(s.get("actual_k", s.get("strikeouts", -1)))
        if actual < 0:
            continue
        # Reject identity-prior style: require explicit actual_k field
        if "actual_k" not in s and "strikeouts" not in s:
            continue
        rows.append(s)

    if len(rows) < 5:
        return KsModel(intercept=math.log(5.5), coef=[0.0, 0.0, 0.0], as_of=as_of or "", n_samples=len(rows))

    rows = sorted(rows, key=lambda r: r.get("date") or "")
    cut = max(1, int(len(rows) * (1.0 - holdout_frac)))
    if 0 < cut < len(rows):
        # Never split one date across train/test. Several pitchers start on the
        # same day, so an index cut leaves same-day starts in training while the
        # holdout boundary claims they are out of sample. Walk the cut back until
        # the boundary date lives entirely in the holdout.
        boundary = (rows[cut].get("date") or "")
        while cut > 0 and (rows[cut - 1].get("date") or "") == boundary:
            cut -= 1
        if cut == 0:
            # Every row shares one date — no clean walk-forward split exists.
            cut = len(rows)
    train, test = rows[:cut], rows[cut:]

    def pack(data):
        X, y = [], []
        for s in data:
            recent = float(s.get("recent_k", 0) or 0)
            season = float(s.get("season_k", 0) or 0)
            opp = float(s.get("opp_k_rate", 0) or 0)
            actual = float(s.get("actual_k", s.get("strikeouts", 0)) or 0)
            X.append([math.log1p(recent), math.log1p(season), max(0.0, opp)])
            # Raw count: the Poisson fit models E[K] directly. Do not log it —
            # that was the retransformation bias documented on _poisson_ridge_fit.
            y.append(max(0.0, actual))
        return X, y

    X, y = pack(train)
    intercept, coef = _poisson_ridge_fit(X, y, alpha=ridge_alpha)
    model = KsModel(
        intercept=intercept,
        coef=coef,
        as_of=as_of or "",
        n_samples=len(train),
        holdout_from=(test[0].get("date") or None) if test else None,
    )

    if test:
        errs = []
        for s in test:
            pred = model.predict_lambda(
                float(s.get("recent_k", 0) or 0),
                float(s.get("season_k", 0) or 0),
                float(s.get("opp_k_rate", 0) or 0),
            )
            actual = float(s.get("actual_k", s.get("strikeouts", 0)) or 0)
            errs.append(abs(pred - actual))
        model.holdout_mae = sum(errs) / len(errs)

    # Brier vs market only when caller supplies real market holdout props
    # (from signals.jsonl mid/ask). No synthetic market_price_cents fallback.
    if holdout_props:
        # Prefer props whose date is in the holdout (test) window when possible
        test_dates = {s.get("date") for s in test} if test else set()
        scoped = [p for p in holdout_props if p.get("date") in test_dates] if test_dates else list(holdout_props)
        if len(scoped) < 5:
            scoped = list(holdout_props)
        mb, kb, beats = holdout_brier_vs_market(model, scoped)
        model.holdout_model_brier = mb
        model.holdout_market_brier = kb
        model.holdout_beats_market = beats
        ib, _n = holdout_brier_vs_incumbent(scoped)
        model.holdout_incumbent_brier = ib
        if mb is not None and ib is not None:
            model.holdout_beats_incumbent = mb < ib

    return model


def fit_and_save_ks_model(
    signals: Sequence[dict],
    journal_records: Sequence[dict],
    *,
    game_logs: Dict[str, List[dict]],
    team_game_logs: Optional[Dict[str, List[dict]]],
    as_of: str,
    model_path: str,
    cost_buffer_cents: float = 5.0,
    holdout_frac: float = 0.2,
    min_samples: int = 20,
) -> Dict[str, object]:
    """Fit the Ks model from game logs, persist it, and report holdout evidence.

    Pure orchestration with the network fetches injected, so the whole path that
    produces logs/ks_model.json is testable offline. This is the only thing that
    puts a trained model in front of the live bot; if it silently no-ops the bot
    quietly keeps using the hand-tuned fallback in models.expected_ks.

    Returns a report dict with status:
      no_samples        — game logs produced fewer than min_samples rows
      ok                — model fitted and written to model_path
    """
    samples = samples_from_pitcher_game_logs(
        game_logs, as_of=as_of, team_game_logs=team_game_logs or None,
    )
    distinct_opp = len({round(float(s["opp_k_rate"]), 5) for s in samples})

    if len(samples) < min_samples:
        return {
            "status": "no_samples",
            "n_samples": len(samples),
            "min_samples": min_samples,
            "distinct_opp_k_rates": distinct_opp,
        }

    holdout_props = build_holdout_props_from_signals(signals, game_logs, as_of=as_of)
    model = fit_ks_model(
        samples, as_of=as_of, holdout_frac=holdout_frac, holdout_props=holdout_props,
    )
    model.save(model_path)
    clear_ks_model_cache()

    roi = model_roi_vs_phase0_baseline(
        model,
        holdout_props,
        journal_records,
        min_n=10,
        cost_buffer_cents=cost_buffer_cents,
    )
    return {
        "status": "ok",
        "model": model,
        "model_path": model_path,
        "n_samples": len(samples),
        "distinct_opp_k_rates": distinct_opp,
        "n_holdout_props": len(holdout_props),
        "roi": roi,
    }


def format_ks_fit_report(report: Dict[str, object]) -> str:
    """Human-readable summary of fit_and_save_ks_model, for the calibrate CLI."""
    status = report.get("status")
    if status == "no_samples":
        return (
            f"  Not enough game-log samples to fit Ks model "
            f"({report['n_samples']} < {report['min_samples']})"
        )
    m: KsModel = report["model"]  # type: ignore[assignment]
    roi: Dict[str, object] = report["roi"]  # type: ignore[assignment]
    model_stats = roi["model"]  # type: ignore[index]
    baseline = roi["baseline"]  # type: ignore[index]
    lines = [
        f"  {report['n_samples']} point-in-time start samples, "
        f"opponent K% resolved to {report['distinct_opp_k_rates']} distinct values",
        f"  {report['n_holdout_props']} holdout props with real market prices",
        f"Ks model saved to {report['model_path']} n={m.n_samples} "
        f"coef={[round(c, 3) for c in m.coef]} holdout_from={m.holdout_from} "
        f"holdout_mae={m.holdout_mae}",
        f"  BRIER model={m.holdout_model_brier} market={m.holdout_market_brier} "
        f"incumbent={m.holdout_incumbent_brier} "
        f"beats_market={m.holdout_beats_market} "
        f"beats_incumbent={m.holdout_beats_incumbent}",
        f"  MODEL_ROI status={roi.get('status')} "
        f"cells={roi['n_cells_scored']} "  # type: ignore[index]
        f"(dropped_in_sample={roi['n_in_sample_dropped']}) "  # type: ignore[index]
        f"model_n={model_stats['n']:.0f} "  # type: ignore[index]
        f"model_roi={model_stats['roi_pct']:+.1f}% | "  # type: ignore[index]
        f"baseline_n={baseline['n']:.0f} "  # type: ignore[index]
        f"baseline_roi={baseline['roi_pct']:+.1f}% | "  # type: ignore[index]
        f"not_worse={roi['not_worse_than_baseline']}",  # type: ignore[index]
    ]
    gates = roi.get("rejected_by_gate") or {}  # type: ignore[union-attr]
    if gates and not model_stats["n"]:  # type: ignore[index]
        why = ", ".join(f"{k}={v}" for k, v in gates.items() if v)
        lines.append(
            f"  No cell cleared the Phase-0 gates — rejections: {why}. "
            "pitcher_ks would place ~zero trades even if re-enabled."
        )
    if m.holdout_beats_market is not True:
        lines.append(
            "  ⚠️  Model does not beat market Brier on holdout — pitcher_ks has no "
            "demonstrated edge. Do not re-enable on the strength of this fit."
        )
    if m.holdout_beats_incumbent is False:
        lines.append(
            "  ⚠️  Model is also worse than the hand-tuned fallback it replaces. "
            "Writing it to disk makes live pricing worse, not better — delete "
            f"{report['model_path']} to revert to models.fallback_ks_lambda."
        )
    elif m.holdout_beats_incumbent is True:
        lines.append(
            "  ✓ Model beats the hand-tuned fallback, so shipping the artifact "
            "is an improvement even though it does not beat the market."
        )
    return "\n".join(lines)


_LOADED: Optional[KsModel] = None


def get_trained_ks_model(path: Optional[str] = None) -> Optional[KsModel]:
    """Load and memoise the trained model, or None when no artifact exists.

    The path is resolved at call time rather than as a default argument, so
    DEFAULT_MODEL_PATH can actually be overridden — binding it in the signature
    freezes it at import and makes the lookup untestable.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    _LOADED = KsModel.load(path if path is not None else DEFAULT_MODEL_PATH)
    return _LOADED


def clear_ks_model_cache() -> None:
    global _LOADED
    _LOADED = None
