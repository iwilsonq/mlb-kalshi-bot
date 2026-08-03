"""Trained pitcher strikeout lambda model (walk-forward safe).

Replaces pure hand-tuned multiplications for production when a fitted
coefficient file is present. Fit uses only samples with date < as_of.

Training target: actual strikeouts in a start.
Features (log space):
  log(1 + recent_k_per_start), log(1 + season_k), opp_k_rate

lambda = exp(b0 + b1*f1 + b2*f2 + b3*f3) * optional residual deflator
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "logs/ks_model.json"


@dataclass
class KsModel:
    """Log-linear Poisson rate model for expected strikeouts."""
    intercept: float = 0.0
    # weights for [log1p(recent_k), log1p(season_k), opp_k_rate]
    coef: List[float] = field(default_factory=lambda: [0.7, 0.3, 0.0])
    as_of: str = ""
    n_samples: int = 0
    # holdout metrics when fit with train/test split
    holdout_mae: Optional[float] = None

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

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "intercept": self.intercept,
            "coef": self.coef,
            "as_of": self.as_of,
            "n_samples": self.n_samples,
            "holdout_mae": self.holdout_mae,
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
            )
        except Exception as exc:
            log.warning("Could not load KsModel from %s: %s", path, exc)
            return None


def _ols(X: List[List[float]], y: List[float]) -> Tuple[float, List[float]]:
    """Ordinary least squares via normal equations (small p).

    X rows are feature vectors; returns (intercept, coef).
    """
    n = len(X)
    if n == 0:
        return 0.0, [0.0, 0.0, 0.0]
    p = len(X[0])
    # Augment with intercept column
    A = [[1.0] + row for row in X]
    # AtA and Aty
    dim = p + 1
    AtA = [[0.0] * dim for _ in range(dim)]
    Aty = [0.0] * dim
    for i in range(n):
        for a in range(dim):
            Aty[a] += A[i][a] * y[i]
            for b in range(dim):
                AtA[a][b] += A[i][a] * A[i][b]
    # Ridge for stability
    ridge = 1e-3
    for i in range(dim):
        AtA[i][i] += ridge
    # Gaussian elimination
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


def fit_ks_model(
    samples: Sequence[dict],
    *,
    as_of: Optional[str] = None,
    holdout_frac: float = 0.2,
) -> KsModel:
    """Fit from samples: {recent_k, season_k, opp_k_rate, actual_k, date?}.

    When as_of is set, only samples with date < as_of are used.
    """
    rows = []
    for s in samples:
        if as_of:
            d = (s.get("date") or "")[:10]
            if not d or d >= as_of:
                continue
        actual = float(s.get("actual_k", s.get("strikeouts", 0)) or 0)
        if actual < 0:
            continue
        rows.append(s)

    if len(rows) < 5:
        # Fallback identity-ish: log(lambda) ≈ log(recent) roughly
        return KsModel(intercept=0.0, coef=[1.0, 0.0, 0.0], as_of=as_of or "", n_samples=len(rows))

    # chronological holdout: last holdout_frac
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
            y.append(math.log(max(actual, 0.5)))  # avoid log 0
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

    return model


# Module-level cache for live bot
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
