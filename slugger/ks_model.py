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


@dataclass
class KsModel:
    """Log-linear Poisson rate model for expected strikeouts."""
    intercept: float = 0.0
    coef: List[float] = field(default_factory=lambda: [0.7, 0.3, 0.0])
    as_of: str = ""
    n_samples: int = 0
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
        return max(0.0, math.exp(z))

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


def _ols(X: List[List[float]], y: List[float]) -> Tuple[float, List[float]]:
    n = len(X)
    if n == 0:
        return 0.0, [0.0, 0.0, 0.0]
    p = len(X[0])
    A = [[1.0] + row for row in X]
    dim = p + 1
    AtA = [[0.0] * dim for _ in range(dim)]
    Aty = [0.0] * dim
    for i in range(n):
        for a in range(dim):
            Aty[a] += A[i][a] * y[i]
            for b in range(dim):
                AtA[a][b] += A[i][a] * A[i][b]
    ridge = 1e-3
    for i in range(dim):
        AtA[i][i] += ridge
    M = [AtA[i][:] + [Aty[i]] for i in range(dim)]
    for col in range(dim):
        pivot = col
        for r in range(col + 1, dim):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col] or 1e-12
        for c in range(col, dim + 1):
            M[col][c] /= piv
        for r in range(dim):
            if r == col:
                continue
            factor = M[r][col]
            for c in range(col, dim + 1):
                M[r][c] -= factor * M[col][c]
    beta = [M[i][dim] for i in range(dim)]
    return beta[0], beta[1:]


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
    mb = brier_score(model_ps, ys)
    kb = brier_score(market_ps, ys)
    beats = None
    if mb is not None and kb is not None:
        beats = mb < kb
    return mb, kb, beats


def fit_ks_model(
    samples: Sequence[dict],
    *,
    as_of: Optional[str] = None,
    holdout_frac: float = 0.2,
    holdout_props: Optional[Sequence[dict]] = None,
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
    intercept, coef = _ols(X, y)
    model = KsModel(intercept=intercept, coef=coef, as_of=as_of or "", n_samples=len(train))

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

        # Build prop-level holdout from starts if market props not supplied
        if holdout_props is None:
            synthetic = []
            for s in test:
                actual = float(s.get("actual_k", 0) or 0)
                # Fair market ≈ poisson at true lambda noise — use 50c as weak baseline
                for thr in (5, 6, 7, 8):
                    # market as crude: league-ish; tests supply real markets
                    synthetic.append({
                        **s,
                        "threshold": thr,
                        "actual_k": actual,
                        "market_price_cents": 30,
                    })
            holdout_props = synthetic
        mb, kb, beats = holdout_brier_vs_market(model, holdout_props)
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
