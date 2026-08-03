"""Model-vs-market scoring and ROI heatmaps (Phase 1).

Scores probability quality with Brier and log-loss against market-implied
prices (ask or mid), and breaks trade ROI by strategy × price band ×
threshold so live gates can be validated from the journal.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ─── Pure metrics ─────────────────────────────────────────────────────────────

def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """Mean squared error of probability forecasts. Lower is better. 0–1 scale.

    probs: predicted P(event) in [0, 1]
    outcomes: 0/1 realized
    """
    if not probs or len(probs) != len(outcomes):
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def log_loss(probs: Sequence[float], outcomes: Sequence[int], eps: float = 1e-6) -> Optional[float]:
    """Binary cross-entropy. Lower is better.

    probs clipped to [eps, 1-eps] to avoid log(0).
    """
    if not probs or len(probs) != len(outcomes):
        return None
    total = 0.0
    for p, y in zip(probs, outcomes):
        p = min(max(float(p), eps), 1.0 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(probs)


def _clamp_prob_cents(cents: float) -> float:
    """Convert cents (0–100) to probability in (0, 1), clamped."""
    p = float(cents) / 100.0
    return min(max(p, 1e-6), 1.0 - 1e-6)


# ─── Field extraction ─────────────────────────────────────────────────────────

_THRESHOLD_FROM_TICKER = re.compile(r"-(\d+)$")


def parse_threshold_from_ticker(ticker: str) -> Optional[int]:
    """Best-effort threshold from trailing -N on Kalshi prop tickers."""
    m = _THRESHOLD_FROM_TICKER.search(ticker or "")
    if not m:
        return None
    return int(m.group(1))


def price_band(cents: float, width: int = 10) -> str:
    """Bucket price into labels like '20-29'."""
    if cents is None or cents < 0:
        return "unknown"
    c = int(cents)
    if c >= 100:
        return "100"
    lo = (c // width) * width
    hi = lo + width - 1
    return f"{lo}-{hi}"


def _market_implied_cents(row: dict, prefer: str = "mid") -> Optional[float]:
    """Market YES price in cents from a signal or trade row."""
    if prefer == "mid":
        mid = row.get("mid_cents")
        if mid is not None and float(mid) > 0:
            return float(mid)
    ask = row.get("ask_cents") or row.get("market_price_cents") or row.get("price_cents")
    if ask is not None and float(ask) > 0:
        return float(ask)
    mid = row.get("mid_cents")
    if mid is not None and float(mid) > 0:
        return float(mid)
    return None


def _model_prob_cents(row: dict, use_calibrated: bool = True) -> Optional[float]:
    if use_calibrated and row.get("calibrated_prob_pct") is not None:
        return float(row["calibrated_prob_pct"])
    if row.get("model_prob_pct") is not None:
        return float(row["model_prob_pct"])
    return None


def _outcome_yes(settlement: dict) -> Optional[int]:
    """1 if YES won, 0 if NO, None if void/unknown."""
    result = (settlement or {}).get("market_result", "")
    if result == "yes":
        return 1
    if result == "no":
        return 0
    return None


# ─── Scoring tables ───────────────────────────────────────────────────────────

@dataclass
class ProbScoreRow:
    strategy: str
    n: int = 0
    model_brier: Optional[float] = None
    market_brier: Optional[float] = None
    model_logloss: Optional[float] = None
    market_logloss: Optional[float] = None
    brier_edge: Optional[float] = None   # market_brier - model_brier (>0 model better)
    logloss_edge: Optional[float] = None

    @property
    def model_beats_market_brier(self) -> Optional[bool]:
        if self.brier_edge is None:
            return None
        return self.brier_edge > 0

    @property
    def model_beats_market_logloss(self) -> Optional[bool]:
        if self.logloss_edge is None:
            return None
        return self.logloss_edge > 0


@dataclass
class RoiCell:
    strategy: str
    price_band: str
    threshold: str  # "7+" or "n/a"
    n: int = 0
    wins: int = 0
    cost_usd: float = 0.0
    pnl_usd: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.n if self.n else None

    @property
    def roi_pct(self) -> Optional[float]:
        return (self.pnl_usd / self.cost_usd * 100.0) if self.cost_usd > 0 else None


@dataclass
class ScoreReport:
    """Full Phase-1 measurement report."""
    market_price_source: str  # "mid" or "ask"
    prob_scores: List[ProbScoreRow] = field(default_factory=list)
    roi_cells: List[RoiCell] = field(default_factory=list)
    traded_only_scores: List[ProbScoreRow] = field(default_factory=list)
    n_signals_scored: int = 0
    n_trades_scored: int = 0
    notes: List[str] = field(default_factory=list)


def build_settlement_index(journal_records: Iterable[dict]) -> Dict[str, dict]:
    return {r["ticker"]: r for r in journal_records if r.get("type") == "settlement"}


def _score_pairs(
    strategy: str,
    model_ps: List[float],
    market_ps: List[float],
    ys: List[int],
) -> ProbScoreRow:
    row = ProbScoreRow(strategy=strategy, n=len(ys))
    if not ys:
        return row
    row.model_brier = brier_score(model_ps, ys)
    row.market_brier = brier_score(market_ps, ys)
    row.model_logloss = log_loss(model_ps, ys)
    row.market_logloss = log_loss(market_ps, ys)
    if row.model_brier is not None and row.market_brier is not None:
        row.brier_edge = row.market_brier - row.model_brier
    if row.model_logloss is not None and row.market_logloss is not None:
        row.logloss_edge = row.market_logloss - row.model_logloss
    return row


def score_probabilities(
    signals: List[dict],
    settlements: Dict[str, dict],
    *,
    market_source: str = "mid",
    use_calibrated: bool = True,
    traded_only: bool = False,
) -> Tuple[List[ProbScoreRow], int]:
    """Brier/log-loss by strategy for signals with known binary outcomes.

    Uses the latest settlement per ticker.  Only YES/NO markets (voids skipped).
    """
    # Group pairs by strategy
    by_strat: Dict[str, Tuple[List[float], List[float], List[int]]] = defaultdict(
        lambda: ([], [], [])
    )
    overall_m: List[float] = []
    overall_k: List[float] = []
    overall_y: List[int] = []
    n = 0

    # Dedup: one sample per (strategy, ticker) — keep last signal
    latest: Dict[Tuple[str, str], dict] = {}
    for sig in signals:
        strat = sig.get("strategy") or "unknown"
        ticker = sig.get("ticker") or ""
        if not ticker:
            continue
        if traded_only and not sig.get("traded"):
            continue
        latest[(strat, ticker)] = sig

    for (strat, ticker), sig in latest.items():
        sett = settlements.get(ticker)
        if not sett:
            continue
        y = _outcome_yes(sett)
        if y is None:
            continue
        mp = _model_prob_cents(sig, use_calibrated=use_calibrated)
        mk = _market_implied_cents(sig, prefer=market_source)
        if mp is None or mk is None:
            continue
        model_p = _clamp_prob_cents(mp)
        market_p = _clamp_prob_cents(mk)
        bm, bk, by = by_strat[strat]
        bm.append(model_p)
        bk.append(market_p)
        by.append(y)
        overall_m.append(model_p)
        overall_k.append(market_p)
        overall_y.append(y)
        n += 1

    rows = [_score_pairs("overall", overall_m, overall_k, overall_y)]
    for strat in sorted(by_strat.keys()):
        bm, bk, by = by_strat[strat]
        rows.append(_score_pairs(strat, bm, bk, by))
    return rows, n


def roi_heatmap(
    trades: List[dict],
    settlements: Dict[str, dict],
    *,
    price_width: int = 10,
) -> Tuple[List[RoiCell], int]:
    """ROI / win-rate cells: strategy × entry price band × threshold."""
    cells: Dict[Tuple[str, str, str], RoiCell] = {}
    n = 0

    for trade in trades:
        if trade.get("type") and trade.get("type") != "trade":
            continue
        ticker = trade.get("ticker") or ""
        sett = settlements.get(ticker)
        if not sett:
            continue
        # Skip voids for ROI
        if sett.get("market_result") == "void":
            continue
        pnl = float(sett.get("pnl_usd") or 0.0)
        cost = float(trade.get("cost_usd") or 0.0)
        strat = trade.get("strategy") or "unknown"
        # Entry price: prefer ask/fill, else limit
        px = trade.get("fill_price_cents") or trade.get("ask_cents") or trade.get("price_cents") or 0
        band = price_band(float(px), width=price_width)
        thr = parse_threshold_from_ticker(ticker)
        thr_label = f"{thr}+" if thr is not None else "n/a"
        key = (strat, band, thr_label)
        if key not in cells:
            cells[key] = RoiCell(strategy=strat, price_band=band, threshold=thr_label)
        cell = cells[key]
        cell.n += 1
        cell.cost_usd += cost
        cell.pnl_usd += pnl
        if pnl > 0:
            cell.wins += 1
        n += 1

    ordered = sorted(
        cells.values(),
        key=lambda c: (c.strategy, c.price_band, c.threshold),
    )
    return ordered, n


def build_report(
    signals: List[dict],
    journal_records: List[dict],
    *,
    market_source: str = "mid",
    use_calibrated: bool = True,
    price_width: int = 10,
    min_cell_n: int = 1,
) -> ScoreReport:
    """Assemble full score report from signals.jsonl + journal.jsonl."""
    settlements = build_settlement_index(journal_records)
    trades = [r for r in journal_records if r.get("type") == "trade"]

    notes: List[str] = []
    if market_source == "mid":
        notes.append("Market implied uses mid when present, else ask.")
    else:
        notes.append("Market implied uses ask (trade price).")
    notes.append(
        "Brier/log-loss: lower is better. Edge = market_score − model_score "
        "(positive ⇒ model beats market)."
    )
    notes.append(
        "Probability scores use signals with a matching YES/NO settlement "
        "(typically markets we traded)."
    )

    all_scores, n_sig = score_probabilities(
        signals, settlements,
        market_source=market_source,
        use_calibrated=use_calibrated,
        traded_only=False,
    )
    traded_scores, _ = score_probabilities(
        signals, settlements,
        market_source=market_source,
        use_calibrated=use_calibrated,
        traded_only=True,
    )
    cells, n_tr = roi_heatmap(trades, settlements, price_width=price_width)
    if min_cell_n > 1:
        cells = [c for c in cells if c.n >= min_cell_n]

    return ScoreReport(
        market_price_source=market_source,
        prob_scores=all_scores,
        traded_only_scores=traded_scores,
        roi_cells=cells,
        n_signals_scored=n_sig,
        n_trades_scored=n_tr,
        notes=notes,
    )


def format_report(report: ScoreReport, *, min_roi_n: int = 5) -> str:
    """Human-readable multi-section report."""
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("  SLUGGER MODEL vs MARKET REPORT")
    lines.append("=" * 88)
    for note in report.notes:
        lines.append(f"  · {note}")
    lines.append(
        f"  Samples: {report.n_signals_scored} scored signals, "
        f"{report.n_trades_scored} settled trades  |  market={report.market_price_source}"
    )

    def _fmt_score_section(title: str, rows: List[ProbScoreRow]) -> None:
        lines.append("")
        lines.append("-" * 88)
        lines.append(f"  {title}")
        lines.append("-" * 88)
        lines.append(
            f"  {'strategy':<16} {'n':>5}  "
            f"{'Brier_m':>8} {'Brier_k':>8} {'ΔBrier':>8}  "
            f"{'LL_m':>7} {'LL_k':>7} {'ΔLL':>7}  beat?"
        )
        for r in rows:
            if r.n == 0 and r.strategy != "overall":
                continue
            mb = f"{r.model_brier:.4f}" if r.model_brier is not None else "—"
            kb = f"{r.market_brier:.4f}" if r.market_brier is not None else "—"
            db = f"{r.brier_edge:+.4f}" if r.brier_edge is not None else "—"
            ml = f"{r.model_logloss:.4f}" if r.model_logloss is not None else "—"
            kl = f"{r.market_logloss:.4f}" if r.market_logloss is not None else "—"
            dl = f"{r.logloss_edge:+.4f}" if r.logloss_edge is not None else "—"
            beat = ""
            if r.brier_edge is not None:
                beat = "YES" if r.brier_edge > 0 else "no"
            lines.append(
                f"  {r.strategy:<16} {r.n:>5}  "
                f"{mb:>8} {kb:>8} {db:>8}  "
                f"{ml:>7} {kl:>7} {dl:>7}  {beat}"
            )

    _fmt_score_section(
        "PROBABILITY SCORES (all signals with settlement on ticker)",
        report.prob_scores,
    )
    _fmt_score_section(
        "PROBABILITY SCORES (traded=true signals only)",
        report.traded_only_scores,
    )

    lines.append("")
    lines.append("-" * 88)
    lines.append(
        f"  ROI HEATMAP  strategy × price_band × threshold  "
        f"(cells with n≥{min_roi_n} highlighted below full table)"
    )
    lines.append("-" * 88)
    lines.append(
        f"  {'strategy':<16} {'price':>7} {'thr':>5}  "
        f"{'n':>4} {'WR':>7} {'ROI':>8} {'P&L':>9}"
    )

    # Full table
    for c in report.roi_cells:
        wr = f"{c.win_rate:.1%}" if c.win_rate is not None else "—"
        roi = f"{c.roi_pct:+.1f}%" if c.roi_pct is not None else "—"
        lines.append(
            f"  {c.strategy:<16} {c.price_band:>7} {c.threshold:>5}  "
            f"{c.n:>4} {wr:>7} {roi:>8} ${c.pnl_usd:>+8.2f}"
        )

    # Summary: cells that look +EV with enough n
    good = [
        c for c in report.roi_cells
        if c.n >= min_roi_n and c.roi_pct is not None and c.roi_pct > 0
    ]
    bad = [
        c for c in report.roi_cells
        if c.n >= min_roi_n and c.roi_pct is not None and c.roi_pct <= 0
    ]
    lines.append("")
    lines.append("-" * 88)
    lines.append(f"  +EV CELLS (n≥{min_roi_n}, ROI>0): {len(good)}")
    for c in sorted(good, key=lambda x: -(x.roi_pct or 0))[:15]:
        lines.append(
            f"    {c.strategy}  px={c.price_band}  thr={c.threshold}  "
            f"n={c.n}  ROI={c.roi_pct:+.1f}%  P&L=${c.pnl_usd:+.2f}"
        )
    if not good:
        lines.append("    (none)")
    lines.append(f"  −EV CELLS (n≥{min_roi_n}, ROI≤0): {len(bad)}")
    for c in sorted(bad, key=lambda x: (x.roi_pct or 0))[:15]:
        lines.append(
            f"    {c.strategy}  px={c.price_band}  thr={c.threshold}  "
            f"n={c.n}  ROI={c.roi_pct:+.1f}%  P&L=${c.pnl_usd:+.2f}"
        )
    lines.append("=" * 88)
    return "\n".join(lines)
