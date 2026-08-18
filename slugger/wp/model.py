"""In-game home win probability model.

Empirical bucket table with shrinkage:
  fine bucket   (inning, is_top, outs, base_state, score_diff)
  -> shrunk toward coarse bucket (inning, is_top, score_diff)
  -> shrunk toward a 2-parameter logistic in score_diff / sqrt(half_innings
     remaining), fitted with Newton iterations (numpy only, no sklearn).

Extra innings are capped at inning 9 (bucketed with the 9th), score_diff is
clamped to [-8, +8] — both mirror slugger/wp/fetch.py. Artifact serialises to
logs/wp_model.json.

CLI (fit + holdout report from the fetch cache):
    python3 -m slugger.wp.model --train-before 2025-08-01
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "logs/wp_model.json"

INNING_CAP = 9
DIFF_CLAMP = 8

# Shrinkage pseudo-counts. A fine bucket needs ~K_FINE observations before its
# empirical rate dominates the coarse estimate; likewise coarse vs logistic.
K_FINE = 60.0
K_COARSE = 200.0

P_MIN, P_MAX = 0.001, 0.999


def _norm_state(
    inning: int, is_top: bool, outs: int, score_diff: int,
    on1: bool, on2: bool, on3: bool,
) -> Tuple[int, bool, int, int, int]:
    inning = max(1, min(int(inning), INNING_CAP))
    outs = max(0, min(int(outs), 2))
    diff = max(-DIFF_CLAMP, min(DIFF_CLAMP, int(score_diff)))
    base = (1 if on1 else 0) | (2 if on2 else 0) | (4 if on3 else 0)
    return inning, bool(is_top), outs, diff, base


def _half_innings_remaining(inning: int, is_top: bool, outs: int) -> float:
    """Half-innings of baseball left through the end of the 9th (min 1/3).

    Outs count fractionally so WP sharpens within a half-inning too. Extra
    innings look like the 9th, which is the right anchor for sudden-death play.
    """
    full = (INNING_CAP - inning) * 2 + (2 if is_top else 1)
    return max(1.0 / 3.0, full - outs / 3.0)


def _logit_feature(inning: int, is_top: bool, outs: int, diff: int) -> float:
    return diff / math.sqrt(_half_innings_remaining(inning, is_top, outs))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, max_iter: int = 30) -> Tuple[float, float]:
    """Newton/IRLS fit of P(y=1) = sigmoid(b0 + b1 * x). Hand-rolled, numpy only."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    # errstate: Apple Accelerate BLAS emits spurious divide-by-zero warnings on
    # matmul with denormal intermediates; the results are finite and checked.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            mu = _sigmoid(X @ beta)
            grad = X.T @ (y - mu)
            w = np.maximum(mu * (1 - mu), 1e-9)
            H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(2)
            step = np.linalg.solve(H, grad)
            if not np.all(np.isfinite(step)):
                break
            beta = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
    return float(beta[0]), float(beta[1])


def _fine_key(inning: int, is_top: bool, outs: int, base: int, diff: int) -> str:
    return f"{inning}|{'t' if is_top else 'b'}|{outs}|{base}|{diff}"


def _coarse_key(inning: int, is_top: bool, diff: int) -> str:
    return f"{inning}|{'t' if is_top else 'b'}|{diff}"


@dataclass
class WPModel:
    """Predict home win probability from pre-PA game state."""
    # bucket key -> [n, wins]
    fine: Dict[str, List[float]] = field(default_factory=dict)
    coarse: Dict[str, List[float]] = field(default_factory=dict)
    beta0: float = 0.12   # home-field prior when untrained (~0.53 at tie)
    beta1: float = 0.9
    n_train: int = 0
    train_from: str = ""
    train_to: str = ""

    @classmethod
    def fit(cls, rows: Sequence[dict]) -> "WPModel":
        """rows: dicts with inning, is_top, outs, on1/2/3, score_diff, home_win."""
        xs: List[float] = []
        ys: List[float] = []
        fine: Dict[str, List[float]] = {}
        coarse: Dict[str, List[float]] = {}
        dates: List[str] = []
        for r in rows:
            inning, is_top, outs, diff, base = _norm_state(
                r["inning"], r["is_top"], r["outs"], r["score_diff"],
                r.get("on1", False), r.get("on2", False), r.get("on3", False),
            )
            y = 1.0 if r["home_win"] else 0.0
            xs.append(_logit_feature(inning, is_top, outs, diff))
            ys.append(y)
            fk = _fine_key(inning, is_top, outs, base, diff)
            ck = _coarse_key(inning, is_top, diff)
            fine.setdefault(fk, [0.0, 0.0])
            coarse.setdefault(ck, [0.0, 0.0])
            fine[fk][0] += 1.0
            fine[fk][1] += y
            coarse[ck][0] += 1.0
            coarse[ck][1] += y
            d = (r.get("date") or "")[:10]
            if d:
                dates.append(d)

        model = cls(fine=fine, coarse=coarse, n_train=len(xs))
        if dates:
            model.train_from = min(dates)
            model.train_to = max(dates)
        if len(xs) >= 100:
            model.beta0, model.beta1 = _fit_logistic(
                np.asarray(xs), np.asarray(ys),
            )
        return model

    def predict(
        self,
        inning: int,
        is_top: bool,
        outs: int,
        score_diff: int,
        on1: bool = False,
        on2: bool = False,
        on3: bool = False,
    ) -> float:
        inning, is_top, outs, diff, base = _norm_state(
            inning, is_top, outs, score_diff, on1, on2, on3,
        )
        x = _logit_feature(inning, is_top, outs, diff)
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.beta0 + self.beta1 * x))))

        c = self.coarse.get(_coarse_key(inning, is_top, diff))
        if c and c[0] > 0:
            p = (c[1] + K_COARSE * p) / (c[0] + K_COARSE)

        f = self.fine.get(_fine_key(inning, is_top, outs, base, diff))
        if f and f[0] > 0:
            p = (f[1] + K_FINE * p) / (f[0] + K_FINE)

        return min(max(p, P_MIN), P_MAX)

    def get_wp(self, state: dict) -> float:
        """Home win probability from a state dict (recorder gumbo_state shape ok)."""
        return self.predict(
            inning=int(state.get("inning", 1) or 1),
            is_top=bool(state.get("is_top", True)),
            outs=int(state.get("outs", 0) or 0),
            score_diff=int(
                state["score_diff"] if "score_diff" in state
                else (state.get("home_runs", 0) or 0) - (state.get("away_runs", 0) or 0)
            ),
            on1=bool(state.get("on1", state.get("on_first", False))),
            on2=bool(state.get("on2", state.get("on_second", False))),
            on3=bool(state.get("on3", state.get("on_third", False))),
        )

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "fine": self.fine,
            "coarse": self.coarse,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "n_train": self.n_train,
            "train_from": self.train_from,
            "train_to": self.train_to,
        }))

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> Optional["WPModel"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            return cls(
                fine={k: [float(v[0]), float(v[1])] for k, v in d.get("fine", {}).items()},
                coarse={k: [float(v[0]), float(v[1])] for k, v in d.get("coarse", {}).items()},
                beta0=float(d.get("beta0", 0.12)),
                beta1=float(d.get("beta1", 0.9)),
                n_train=int(d.get("n_train", 0) or 0),
                train_from=d.get("train_from", "") or "",
                train_to=d.get("train_to", "") or "",
            )
        except Exception as exc:
            log.warning("Could not load WPModel from %s: %s", path, exc)
            return None


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model: WPModel, rows: Sequence[dict], baseline_p: float = 0.54) -> Dict[str, object]:
    """Brier / log-loss vs a constant-baseline, plus decile calibration."""
    ps = np.array([
        model.predict(
            r["inning"], r["is_top"], r["outs"], r["score_diff"],
            r.get("on1", False), r.get("on2", False), r.get("on3", False),
        )
        for r in rows
    ])
    ys = np.array([1.0 if r["home_win"] else 0.0 for r in rows])
    n = len(ys)
    if n == 0:
        return {"n": 0}
    eps = 1e-9
    brier = float(np.mean((ps - ys) ** 2))
    logloss = float(-np.mean(ys * np.log(ps + eps) + (1 - ys) * np.log(1 - ps + eps)))
    bp = np.clip(baseline_p, eps, 1 - eps)
    base_brier = float(np.mean((bp - ys) ** 2))
    base_logloss = float(-np.mean(ys * np.log(bp) + (1 - ys) * np.log(1 - bp)))
    deciles = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        mask = (ps >= lo) & (ps < hi) if hi < 1.0 else (ps >= lo)
        if mask.sum() == 0:
            continue
        deciles.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": int(mask.sum()),
            "pred": float(ps[mask].mean()),
            "actual": float(ys[mask].mean()),
        })
    return {
        "n": n,
        "brier": brier,
        "logloss": logloss,
        "baseline_brier": base_brier,
        "baseline_logloss": base_logloss,
        "deciles": deciles,
    }


# ─── Module-level convenience loader ─────────────────────────────────────────

_LOADED: Optional[WPModel] = None


def get_wp(state: dict, path: Optional[str] = None) -> float:
    """Home WP from a state dict using the persisted artifact (memoised).

    Falls back to the untrained logistic prior if no artifact exists.
    """
    global _LOADED
    if _LOADED is None:
        _LOADED = WPModel.load(path if path is not None else DEFAULT_MODEL_PATH) or WPModel()
    return _LOADED.get_wp(state)


def clear_wp_model_cache() -> None:
    global _LOADED
    _LOADED = None


if __name__ == "__main__":
    from slugger.wp.fetch import load_cached_rows

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Fit WP model from fetch cache")
    ap.add_argument("--train-before", default="2025-08-01",
                    help="holdout: train on games strictly before this date")
    ap.add_argument("--out", default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()

    rows = load_cached_rows()
    train = [r for r in rows if (r.get("date") or "") < args.train_before]
    test = [r for r in rows if (r.get("date") or "") >= args.train_before]
    print(f"rows={len(rows)} train={len(train)} holdout={len(test)}")

    model = WPModel.fit(train)
    model.save(args.out)
    print(f"saved {args.out}  beta0={model.beta0:.4f} beta1={model.beta1:.4f} "
          f"n_train={model.n_train} range={model.train_from}..{model.train_to}")

    if test:
        rep = evaluate(model, test)
        print(f"HOLDOUT n={rep['n']}  brier={rep['brier']:.5f} "
              f"(baseline {rep['baseline_brier']:.5f})  "
              f"logloss={rep['logloss']:.5f} (baseline {rep['baseline_logloss']:.5f})")
        for d in rep["deciles"]:
            print(f"  {d['bin']}  n={d['n']:>6}  pred={d['pred']:.3f}  actual={d['actual']:.3f}")
