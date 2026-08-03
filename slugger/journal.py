"""Trade journal for Slugger MLB bot.

Records every placed order and its eventual settlement outcome to a
newline-delimited JSON file (logs/journal.jsonl).  Signal evaluations
go to logs/signals.jsonl.

## Record types (journal.jsonl)

  "trade"      — written at order placement time
  "settlement" — written by settle_pending after Kalshi resolves the market
  "clv"        — mid/bid/ask snapshot ~1h after entry (closing-line value)

## signals.jsonl schema (Phase 1)

Every evaluated market (traded or not) appends one JSON object:

  type                 "signal"
  timestamp            ISO-8601 UTC evaluation time
  date                 YYYY-MM-DD
  ticker               Kalshi market ticker
  strategy             strategy name
  model_prob_pct       raw model probability (0–100); used for calibration fit
  calibrated_prob_pct  calibrated probability used for edge (may equal raw)
  market_price_cents   YES ask used for trading (legacy field name)
  bid_cents            YES bid at evaluation (0 if unknown)
  ask_cents            YES ask at evaluation
  mid_cents            (bid+ask)/2 when both present
  spread_cents         ask − bid
  cost_buffer_cents    EDGE_COST_BUFFER applied to edge
  gross_edge_cents     calibrated_prob − ask (YES) before buffer
  edge_cents           gross_edge (legacy); same as gross for research continuity
  net_edge_cents       gross − cost_buffer (what trade/size uses)
  traded               bool — would/did place after gates
  reason               human-readable model reason
  model_version        slugger.__version__

Existing calibration code only requires model_prob_pct (+ settlements for
outcomes). New fields are additive and optional on historical lines.

## trade schema additions (Phase 1)

  bid/ask/mid/spread_cents, raw/calibrated model probs, gross/net edge,
  cost_buffer_cents, fill_price_cents, fill_count, fill_status, filled_at

## settlement schema additions

  settlement_price_cents — 100 if market_result=yes, 0 if no, null if void

## clv schema

  type, ticker, strategy, placed_at, captured_at, hours_after_entry,
  bid_cents, ask_cents, mid_cents, spread_cents,
  entry_mid_cents, clv_cents (entry_mid − snapshot_mid for YES buyers)
"""
from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import slugger

log = logging.getLogger(__name__)

JOURNAL_FILENAME = "journal.jsonl"
SIGNALS_FILENAME = "signals.jsonl"


# ─── Record dataclasses ───────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Written immediately after a successful order placement."""
    type: str = "trade"
    placed_at: str = ""        # ISO 8601 UTC
    date: str = ""             # YYYY-MM-DD (local date for grouping)
    ticker: str = ""
    strategy: str = ""
    side: str = "yes"
    count: int = 0
    price_cents: int = 0       # limit price
    cost_usd: float = 0.0
    edge_cents: float = 0.0    # net (cost-adjusted) edge at entry
    reason: str = ""
    order_id: str = ""
    model_version: str = ""    # stamped from slugger.__version__
    # Phase 1 measurement
    raw_model_prob_pct: float = 0.0
    model_prob_pct: float = 0.0       # calibrated
    gross_edge_cents: float = 0.0
    cost_buffer_cents: int = 0
    bid_cents: int = 0
    ask_cents: int = 0
    mid_cents: float = 0.0
    spread_cents: int = 0
    fill_price_cents: int = 0
    fill_count: int = 0
    fill_status: str = ""
    filled_at: str = ""


@dataclass
class SettlementRecord:
    """Written by cmd_settle once Kalshi resolves a market we traded."""
    type: str = "settlement"
    settled_at: str = ""       # ISO 8601 UTC from Kalshi
    ticker: str = ""
    market_result: str = ""    # "yes", "no", "void", "scalar"
    revenue_usd: float = 0.0   # total payout in USD (revenue / 100)
    yes_cost_usd: float = 0.0  # what we paid for YES contracts
    fee_usd: float = 0.0
    pnl_usd: float = 0.0       # revenue_usd - yes_cost_usd - fee_usd
    settlement_price_cents: Optional[int] = None  # 100/0 for yes/no binary


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def _journal_path(log_dir: str) -> Path:
    return Path(log_dir) / JOURNAL_FILENAME


def record_trade(
    log_dir: str,
    ticker: str,
    strategy: str,
    side: str,
    count: int,
    price_cents: int,
    cost_usd: float,
    edge_cents: float,
    reason: str,
    order_id: str,
    *,
    raw_model_prob_pct: float = 0.0,
    model_prob_pct: float = 0.0,
    gross_edge_cents: float = 0.0,
    cost_buffer_cents: int = 0,
    bid_cents: int = 0,
    ask_cents: int = 0,
    mid_cents: float = 0.0,
    spread_cents: int = 0,
    fill_price_cents: int = 0,
    fill_count: int = 0,
    fill_status: str = "",
    filled_at: str = "",
) -> None:
    """Append a trade record to the journal."""
    now = datetime.now(timezone.utc)
    rec = TradeRecord(
        placed_at=now.isoformat(),
        date=now.date().isoformat(),
        ticker=ticker,
        strategy=strategy,
        side=side,
        count=count,
        price_cents=price_cents,
        cost_usd=round(cost_usd, 4),
        edge_cents=edge_cents,
        reason=reason,
        order_id=order_id,
        model_version=slugger.__version__,
        raw_model_prob_pct=float(raw_model_prob_pct),
        model_prob_pct=float(model_prob_pct),
        gross_edge_cents=float(gross_edge_cents),
        cost_buffer_cents=int(cost_buffer_cents),
        bid_cents=int(bid_cents),
        ask_cents=int(ask_cents),
        mid_cents=float(mid_cents),
        spread_cents=int(spread_cents),
        fill_price_cents=int(fill_price_cents),
        fill_count=int(fill_count),
        fill_status=fill_status,
        filled_at=filled_at or now.isoformat(),
    )
    _append(log_dir, asdict(rec))
    log.debug("Journal: recorded trade %s %s", strategy, ticker)


def record_settlement(
    log_dir: str,
    ticker: str,
    market_result: str,
    revenue_usd: float,
    yes_cost_usd: float,
    fee_usd: float,
    settled_at: str,
    settlement_price_cents: Optional[int] = None,
) -> None:
    """Append a settlement record to the journal."""
    pnl = round(revenue_usd - yes_cost_usd - fee_usd, 4)
    if settlement_price_cents is None:
        if market_result == "yes":
            settlement_price_cents = 100
        elif market_result == "no":
            settlement_price_cents = 0
    rec = SettlementRecord(
        settled_at=settled_at,
        ticker=ticker,
        market_result=market_result,
        revenue_usd=round(revenue_usd, 4),
        yes_cost_usd=round(yes_cost_usd, 4),
        fee_usd=round(fee_usd, 4),
        pnl_usd=pnl,
        settlement_price_cents=settlement_price_cents,
    )
    _append(log_dir, asdict(rec))
    log.debug("Journal: recorded settlement %s → %s  P&L $%.4f", ticker, market_result, pnl)


def record_clv(
    log_dir: str,
    ticker: str,
    strategy: str,
    placed_at: str,
    bid_cents: int,
    ask_cents: int,
    mid_cents: float,
    spread_cents: int,
    entry_mid_cents: float,
    hours_after_entry: float,
) -> None:
    """Append a 1h (or other horizon) CLV snapshot for a traded ticker."""
    now = datetime.now(timezone.utc)
    # For YES buyers, positive CLV means market mid moved toward YES (price up).
    clv_cents = None
    if entry_mid_cents and mid_cents:
        clv_cents = round(float(mid_cents) - float(entry_mid_cents), 2)
    data = {
        "type": "clv",
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "ticker": ticker,
        "strategy": strategy,
        "placed_at": placed_at,
        "captured_at": now.isoformat(),
        "hours_after_entry": round(hours_after_entry, 3),
        "bid_cents": int(bid_cents),
        "ask_cents": int(ask_cents),
        "mid_cents": float(mid_cents),
        "spread_cents": int(spread_cents),
        "entry_mid_cents": float(entry_mid_cents),
        "clv_cents": clv_cents,
        "model_version": slugger.__version__,
    }
    _append(log_dir, data)
    log.debug("Journal: recorded CLV %s mid=%.1f clv=%s", ticker, mid_cents, clv_cents)


def _signals_path(log_dir: str) -> Path:
    return Path(log_dir) / SIGNALS_FILENAME


def record_signal(
    log_dir: str,
    ticker: str,
    strategy: str,
    model_prob_pct: int,
    market_price_cents: int,
    edge_cents: float,
    traded: bool,
    reason: str = "",
    *,
    calibrated_prob_pct: Optional[float] = None,
    bid_cents: int = 0,
    ask_cents: int = 0,
    mid_cents: float = 0.0,
    spread_cents: int = 0,
    cost_buffer_cents: int = 0,
    fee_cents: float = 0.0,
    gross_edge_cents: Optional[float] = None,
    net_edge_cents: Optional[float] = None,
) -> None:
    """Log every evaluated signal (traded or not) for model calibration.

    model_prob_pct is the *raw* model probability (for calibration fitting).
    calibrated_prob_pct is what edge decisions used (defaults to raw).
    edge_cents remains gross model−ask for backward-compatible research.
    """
    now = datetime.now(timezone.utc)
    cal = float(calibrated_prob_pct) if calibrated_prob_pct is not None else float(model_prob_pct)
    gross = float(gross_edge_cents) if gross_edge_cents is not None else float(edge_cents)
    net = float(net_edge_cents) if net_edge_cents is not None else gross - cost_buffer_cents
    ask = int(ask_cents) if ask_cents else int(market_price_cents)
    data = {
        "type": "signal",
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "ticker": ticker,
        "strategy": strategy,
        "model_prob_pct": int(model_prob_pct),
        "calibrated_prob_pct": cal,
        "market_price_cents": int(market_price_cents),
        "bid_cents": int(bid_cents),
        "ask_cents": ask,
        "mid_cents": float(mid_cents) if mid_cents else (
            (bid_cents + ask) / 2.0 if bid_cents and ask else float(ask or 0)
        ),
        "spread_cents": int(spread_cents) if spread_cents or not bid_cents else max(0, ask - bid_cents),
        "cost_buffer_cents": int(cost_buffer_cents),
        "fee_cents": float(fee_cents),
        "gross_edge_cents": round(gross, 1),
        "edge_cents": round(gross, 1),  # legacy alias = gross
        "net_edge_cents": round(net, 1),
        "traded": traded,
        "reason": reason,
        "model_version": slugger.__version__,
    }
    path = _signals_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data) + "\n")


def load_signals(log_dir: str) -> List[dict]:
    """Return all signal records from signals.jsonl."""
    path = _signals_path(log_dir)
    if not path.exists():
        return []
    records = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Signals line %d is malformed: %s", i, exc)
    return records


def _append(log_dir: str, data: dict) -> None:
    path = _journal_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data) + "\n")


def load_journal(log_dir: str) -> List[dict]:
    """Return all records from the journal as a list of dicts."""
    path = _journal_path(log_dir)
    if not path.exists():
        return []
    records = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Journal line %d is malformed: %s", i, exc)
    return records


# ─── Stats ────────────────────────────────────────────────────────────────────

@dataclass
class StrategyStats:
    strategy: str
    bets: int = 0
    settled: int = 0
    wins: int = 0
    voids: int = 0
    total_cost_usd: float = 0.0
    total_revenue_usd: float = 0.0
    total_fee_usd: float = 0.0
    total_pnl_usd: float = 0.0

    @property
    def losses(self) -> int:
        return self.settled - self.wins - self.voids

    @property
    def win_rate(self) -> Optional[float]:
        decided = self.settled - self.voids
        return self.wins / decided if decided > 0 else None

    @property
    def roi_pct(self) -> Optional[float]:
        return (self.total_pnl_usd / self.total_cost_usd * 100) if self.total_cost_usd > 0 else None

    @property
    def pending(self) -> int:
        return self.bets - self.settled


def get_stats(records: List[dict]) -> Tuple[StrategyStats, Dict[str, StrategyStats]]:
    """Compute overall and per-strategy stats from journal records.

    Returns:
        (overall_stats, {strategy_name: StrategyStats})
    """
    # Build lookup: ticker → trade record
    trades: Dict[str, dict] = {}
    for r in records:
        if r.get("type") == "trade":
            trades[r["ticker"]] = r

    # Build lookup: ticker → settlement record
    settlements: Dict[str, dict] = {}
    for r in records:
        if r.get("type") == "settlement":
            settlements[r["ticker"]] = r

    per_strategy: Dict[str, StrategyStats] = {}
    overall = StrategyStats(strategy="overall")

    for ticker, trade in trades.items():
        strat = trade.get("strategy", "unknown")
        if strat not in per_strategy:
            per_strategy[strat] = StrategyStats(strategy=strat)
        s = per_strategy[strat]

        cost = trade.get("cost_usd", 0.0)
        s.bets += 1
        s.total_cost_usd += cost
        overall.bets += 1
        overall.total_cost_usd += cost

        if ticker in settlements:
            sett = settlements[ticker]
            result = sett.get("market_result", "")
            pnl = sett.get("pnl_usd", 0.0)
            rev = sett.get("revenue_usd", 0.0)
            fee = sett.get("fee_usd", 0.0)

            s.settled += 1
            s.total_revenue_usd += rev
            s.total_fee_usd += fee
            s.total_pnl_usd += pnl
            overall.settled += 1
            overall.total_revenue_usd += rev
            overall.total_fee_usd += fee
            overall.total_pnl_usd += pnl

            if result == "void":
                s.voids += 1
                overall.voids += 1
            elif pnl > 0:
                # Win = positive P&L, regardless of side.
                # YES bets win when market_result=="yes".
                # NO bets win when market_result=="no".
                s.wins += 1
                overall.wins += 1

    overall.strategy = "overall"
    return overall, per_strategy


def format_stats(overall: StrategyStats, per_strategy: Dict[str, StrategyStats]) -> str:
    """Format stats into a human-readable string."""
    lines = []

    def _fmt_row(s: StrategyStats) -> str:
        wr = f"{s.win_rate:.1%}" if s.win_rate is not None else "—"
        roi = f"{s.roi_pct:+.1f}%" if s.roi_pct is not None else "—"
        pending = f" ({s.pending} pending)" if s.pending else ""
        return (
            f"  {s.strategy:<20} "
            f"{s.bets:>4} bets  "
            f"{s.wins:>3}W / {s.losses:>3}L"
            f"{'/' + str(s.voids) + 'V' if s.voids else '':>4}  "
            f"WR {wr:>6}  "
            f"ROI {roi:>7}  "
            f"P&L ${s.total_pnl_usd:>+7.2f}"
            f"{pending}"
        )

    lines.append("=" * 80)
    lines.append("  SLUGGER TRADE STATS")
    lines.append("=" * 80)
    lines.append(_fmt_row(overall))
    lines.append("-" * 80)
    for strat, s in sorted(per_strategy.items()):
        lines.append(_fmt_row(s))
    lines.append("=" * 80)
    return "\n".join(lines)
