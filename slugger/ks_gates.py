"""Re-derive the pitcher_ks Phase-0 trading gates from walk-forward evidence.

The 25–55% probability band, the 6+ threshold floor and the 20¢ edge floor were
set in Phase 0 from journal ROI bucketed by *market price*, and scored against
the old hand-tuned heuristic. The trained model's probabilities sit lower, so
those gates were never valid for it: on the 2026-08-03 fit, 295 of 449 holdout
cells were rejected for falling below the 25% floor and zero cells traded.

Retiring the strategy on that verdict would be retiring on a measurement taken
with the wrong instrument. This module re-derives the gates using the trained
model's own probabilities, scored strictly out of sample.

Method:
  - Sort every prop with a real recorded market price by date.
  - Step forward in windows. For each window, fit on starts strictly before it
    and score the props inside it. A prop is therefore always priced by a model
    that never saw it, or any later start.
  - Bucket the resulting (model_prob, price, threshold, outcome) rows and report
    realized ROI per bucket, so a gate can be chosen from evidence.

The honest constraint: out-of-sample rows are scarce. Choosing several gate
thresholds by maximising ROI over a few hundred rows will fit noise. Treat the
output as a description of where edge plausibly exists, not as an optimiser, and
prefer wide contiguous regions over isolated high-ROI cells.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from slugger.ks_model import KsModel, fit_ks_model

log = logging.getLogger(__name__)

# A YES contract at `price` cents pays $1 if it settles yes.
# Profit per $1 staked = (100/price - 1) on a win, -1 on a loss.


def walk_forward_scores(
    samples: Sequence[dict],
    props: Sequence[dict],
    *,
    window_days: int = 7,
    min_train: int = 200,
    ridge_alpha: Optional[float] = None,
) -> List[dict]:
    """Score props with models that never saw them.

    Returns rows of {date, threshold, model_prob_pct, price_cents, won}.
    """
    dated_props: List[dict] = []
    for p in props:
        d = (p.get("date") or "")[:10]
        px = p.get("market_price_cents")
        thr = int(p.get("threshold", 0) or 0)
        if not d or px is None or thr <= 0:
            continue
        if p.get("synthetic_market"):
            continue
        px = float(px)
        if px <= 0 or px >= 100:
            continue
        dated_props.append(p)
    dated_props.sort(key=lambda p: p["date"][:10])
    if not dated_props:
        return []

    dates = sorted({p["date"][:10] for p in dated_props})
    scored: List[dict] = []

    # Walk in blocks of `window_days` distinct prop dates
    for i in range(0, len(dates), window_days):
        block = set(dates[i:i + window_days])
        cutoff = min(block)
        train = [s for s in samples if (s.get("date") or "")[:10] < cutoff]
        if len(train) < min_train:
            continue
        kwargs = {"as_of": cutoff, "holdout_frac": 0.0}
        if ridge_alpha is not None:
            kwargs["ridge_alpha"] = ridge_alpha
        model = fit_ks_model(train, **kwargs)  # type: ignore[arg-type]

        for p in dated_props:
            if p["date"][:10] not in block:
                continue
            thr = int(p["threshold"])
            prob = model.prob_ge(
                thr,
                float(p.get("recent_k", 0) or 0),
                float(p.get("season_k", 0) or 0),
                float(p.get("opp_k_rate", 0) or 0),
            ) * 100.0
            scored.append({
                "date": p["date"][:10],
                "threshold": thr,
                "model_prob_pct": prob,
                "price_cents": float(p["market_price_cents"]),
                "won": float(p.get("actual_k", 0) or 0) >= thr,
                "n_train": len(train),
            })
    return scored


def roi_for_rows(rows: Sequence[dict]) -> Dict[str, float]:
    """Realized ROI of buying YES on every row at its recorded price."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "roi_pct": 0.0, "win_pct": 0.0, "avg_price": 0.0,
                "avg_prob": 0.0, "pnl_per_unit": 0.0}
    pnl = 0.0
    wins = 0
    for r in rows:
        px = r["price_cents"] / 100.0
        if r["won"]:
            pnl += (1.0 / px) - 1.0
            wins += 1
        else:
            pnl -= 1.0
    return {
        "n": n,
        "roi_pct": pnl / n * 100.0,
        "win_pct": wins / n * 100.0,
        "avg_price": sum(r["price_cents"] for r in rows) / n,
        "avg_prob": sum(r["model_prob_pct"] for r in rows) / n,
        "pnl_per_unit": pnl / n,
    }


def edge_cents(row: dict, cost_buffer_cents: float = 5.0) -> float:
    return row["model_prob_pct"] - row["price_cents"] - cost_buffer_cents


def roi_by_bucket(
    rows: Sequence[dict],
    key,
    *,
    min_n: int = 10,
) -> List[Tuple[object, Dict[str, float]]]:
    """Group rows by key(row) and report ROI per group with at least min_n rows."""
    groups: Dict[object, List[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for k in sorted(groups, key=lambda x: (x is None, x)):
        stats = roi_for_rows(groups[k])
        if stats["n"] >= min_n:
            out.append((k, stats))
    return out


def evaluate_gate(
    rows: Sequence[dict],
    *,
    min_prob: float,
    max_prob: float,
    min_edge: float,
    min_threshold: int,
    cost_buffer_cents: float = 5.0,
) -> Dict[str, float]:
    """ROI of the subset of rows a given gate configuration would have traded."""
    kept = [
        r for r in rows
        if r["threshold"] >= min_threshold
        and min_prob <= r["model_prob_pct"] <= max_prob
        and edge_cents(r, cost_buffer_cents) >= min_edge
    ]
    stats = roi_for_rows(kept)
    stats.update({
        "min_prob": min_prob, "max_prob": max_prob,
        "min_edge": min_edge, "min_threshold": min_threshold,
    })
    return stats
