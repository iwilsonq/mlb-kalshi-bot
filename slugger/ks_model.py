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

from slugger.models import poisson_ge

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
            )
        except Exception as exc:
            log.warning("Could not load KsModel from %s: %s", path, exc)
            return None


def samples_from_pitcher_game_logs(
    game_logs: Dict[str, List[dict]],
    *,
    as_of: Optional[str] = None,
    opp_k_rate: float = 0.225,
    min_prior_starts: int = 2,
) -> List[dict]:
    """Build training rows from historical starts with point-in-time features.

    For each start on date D, features use only starts with date < D.
    actual_k is the true strikeout count on D (not model output).

    game_logs: pitcher_name → [{"date": "YYYY-MM-DD", "strikeouts": int}, ...]
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
            samples.append({
                "date": d,
                "pitcher": name,
                "recent_k": recent_k,
                "season_k": season_k,
                "opp_k_rate": opp_k_rate,
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

    for cell in scored:
        thr = int(cell.get("threshold", 0) or 0)
        if thr < _PHASE0_MIN_THRESHOLD:
            continue
        mkt = cell.get("market_price_cents")
        if mkt is None or float(mkt) <= 0 or float(mkt) >= 100:
            continue
        if cell.get("synthetic_market"):
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

        if model_pct < _PHASE0_MIN_MODEL_PROB or model_pct > _PHASE0_MAX_MODEL_PROB:
            continue
        if net_edge < _PHASE0_MIN_EDGE_CENTS:
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
            y.append(math.log(max(actual, 0.5)))
        return X, y

    X, y = pack(train)
    intercept, coef = _ridge_fit(X, y, alpha=ridge_alpha)
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

    return model


_LOADED: Optional[KsModel] = None


def get_trained_ks_model(path: str = DEFAULT_MODEL_PATH) -> Optional[KsModel]:
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    _LOADED = KsModel.load(path)
    return _LOADED


def clear_ks_model_cache() -> None:
    global _LOADED
    _LOADED = None
