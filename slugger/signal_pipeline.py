"""Unified signal pipeline for Slugger MLB trading bot.

Owns the full market → signal flow:
  1. Fetch markets from Kalshi for a given event
  2. Match and filter markets (by player name, title keywords, ticker suffix)
  3. Parse numeric thresholds from market titles
  4. Call the strategy's probability model for each market
  5. Compute edge (model probability - market price)
  6. Size position with Kelly criterion
  7. Record signal for calibration tracking
  8. Build TradeSignal if edge exceeds minimum

Each strategy provides only the probability model (step 4) and configuration.
Everything else is handled here once.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from slugger.calibration import CalibrationLayer
from slugger.config import Config
from slugger.journal import record_signal
from slugger.kalshi_client import KalshiClient, market_price
from slugger.sizing import kelly_count

log = logging.getLogger(__name__)

# Module-level calibration layer — loaded once, shared across all calls.
# Initialized as pass-through (identity) until load_calibration() is called.
_calibration = CalibrationLayer()


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Output from a strategy's probability model for a single market.

    Attributes:
        prob_pct:  Model probability as integer percentage (0-100).
        reason:    Human-readable explanation for logging / journal.
    """
    prob_pct: int
    reason: str


@dataclass
class MarketSpec:
    """Describes how to find and evaluate markets for a strategy.

    Attributes:
        event_ticker:       Kalshi event ticker to query.
        strategy_name:      Name for TradeSignal.strategy and journal recording.
        title_keywords:     Market title must contain at least one of these
                            (case-insensitive).  Empty list = no keyword filter.
        player_name:        If set, market title must contain this player's last
                            name (case-insensitive).
        ticker_suffix:      If set, market ticker must end with this suffix
                            (case-insensitive, e.g. "-LAD" for home team).
        threshold_pattern:  Regex pattern to extract a numeric threshold from the
                            title.  Must have one capture group yielding a number.
                            If None, threshold is not parsed (passed as None to model).
        threshold_ceil:     If True, ceil the parsed threshold (for "over 6.5" → 7).
        min_threshold:      Skip markets with threshold below this value.
        min_model_prob:     Minimum model probability (%) to consider trading YES.
        min_edge_cents:     Minimum edge in cents to trade (overrides config if higher).
        max_signals:        Maximum number of YES signals to return (sorted by edge).
                            0 = unlimited.
        confidence_fn:      Compute TradeSignal.confidence from edge_cents.
                            Default: min(0.5 + edge/100, 0.85).
        no_side:            If True, also evaluate NO-side trades.
        no_max_model_prob:  For NO-side: only buy NO when model YES prob ≤ this (%).
        no_min_edge_cents:  For NO-side: minimum edge to buy NO.
    """
    event_ticker: str
    strategy_name: str
    title_keywords: List[str] = field(default_factory=list)
    player_name: str = ""
    ticker_suffix: str = ""
    threshold_pattern: Optional[str] = None
    threshold_ceil: bool = False
    min_threshold: int = 0
    min_model_prob: int = 0
    min_edge_cents: int = 0
    max_signals: int = 0
    confidence_fn: Optional[Callable[[float], float]] = None
    no_side: bool = False
    no_max_model_prob: int = 10
    no_min_edge_cents: int = 5


# Probability model callable type:
#   (market_title, threshold_or_none, market_price_cents) → ModelResult or None
#   Return None to skip this market entirely.
ModelFn = Callable[[str, Optional[int], int], Optional[ModelResult]]


# ─── Threshold parsing ───────────────────────────────────────────────────────

# Reusable threshold patterns
THRESHOLD_N_PLUS = r'(\d+)\s*\+'            # "7+" or "7 +"
THRESHOLD_OVER = r'over\s+(\d+(?:\.\d+)?)'  # "over 6.5"
THRESHOLD_AT_LEAST = r'at\s+least\s+(\d+)'  # "at least 9"


def parse_threshold(title: str, keyword: str = "") -> Optional[int]:
    """Extract a numeric threshold from a market title.

    Handles:
      "7+ strikeouts"        → 7
      "over 6.5 strikeouts"  → 7  (ceils)
      "at least 9 strikeouts" → 9

    If keyword is given (e.g. "hit", "home run"), uses keyword-aware regex.
    Otherwise falls back to generic N+ / over / at-least patterns.
    """
    t = title.lower()

    if keyword:
        kw = keyword.lower()
        # "N+ keyword" or "N + keyword"
        m = re.search(rf'(\d+)\s*\+\s*{re.escape(kw)}', t)
        if m:
            return int(m.group(1))
        # "over N keyword"
        m = re.search(rf'over\s+(\d+(?:\.\d+)?)\s*{re.escape(kw)}', t)
        if m:
            return int(math.ceil(float(m.group(1))))
        # "at least N keyword"
        m = re.search(rf'at\s+least\s+(\d+)\s*{re.escape(kw)}', t)
        if m:
            return int(m.group(1))

    # Generic patterns (no keyword context)
    m = re.search(THRESHOLD_N_PLUS, t)
    if m:
        return int(m.group(1))
    m = re.search(THRESHOLD_OVER, t)
    if m:
        return int(math.ceil(float(m.group(1))))
    m = re.search(THRESHOLD_AT_LEAST, t)
    if m:
        return int(m.group(1))

    return None


def parse_threshold_regex(title: str, pattern: str, ceil: bool = False) -> Optional[int]:
    """Extract threshold using a custom regex pattern.

    Args:
        title:   Market title string.
        pattern: Regex with one capture group yielding a number.
        ceil:    If True, ceil the parsed value (for half-thresholds like 6.5).

    Returns:
        Integer threshold or None if no match.
    """
    m = re.search(pattern, title.lower())
    if not m:
        return None
    val = float(m.group(1))
    return int(math.ceil(val)) if ceil else int(val)


# ─── Calibration management ──────────────────────────────────────────────────

def load_calibration(path: str) -> None:
    """Load calibration curves from disk into the module-level layer."""
    global _calibration
    _calibration = CalibrationLayer.load(path)


def get_calibration() -> CalibrationLayer:
    """Return the current calibration layer (for inspection/testing)."""
    return _calibration


# ─── Default confidence function ─────────────────────────────────────────────

def _default_confidence(edge_cents: float) -> float:
    return min(0.5 + edge_cents / 100, 0.85)


# ─── Core pipeline ───────────────────────────────────────────────────────────

def evaluate_markets(
    spec: MarketSpec,
    model: ModelFn,
    client: KalshiClient,
    config: Config,
) -> List["TradeSignal"]:
    """Run the full signal pipeline for a strategy.

    Fetches markets for spec.event_ticker, filters/matches them according to
    spec, calls model() for each, and returns TradeSignals for markets with
    positive edge.

    Args:
        spec:   MarketSpec describing what to look for and strategy config.
        model:  Probability model callable. Takes (title, threshold, price)
                and returns ModelResult or None.
        client: KalshiClient for market queries.
        config: Bot configuration.

    Returns:
        List of TradeSignal objects (may be empty).
    """
    from slugger.strategies import TradeSignal  # avoid circular import

    signals: List[TradeSignal] = []

    if not spec.event_ticker:
        return signals

    # ── Fetch markets ──────────────────────────────────────────────────────
    try:
        markets = client.get_event_markets(
            spec.event_ticker,
            min_liquidity=config.min_liquidity_dollars,
        )
    except Exception:
        return signals

    if not markets:
        return signals

    # Effective edge floor: max of config global and strategy-specific
    edge_floor = max(config.min_edge_cents, spec.min_edge_cents)
    confidence_fn = spec.confidence_fn or _default_confidence

    player_last = ""
    if spec.player_name:
        parts = spec.player_name.split()
        player_last = parts[-1].lower() if parts else ""

    # Track all evaluated markets for logging
    evaluated: List[tuple] = []  # (threshold, prob_pct, price, edge)

    # ── YES-side evaluation ────────────────────────────────────────────────
    for m in markets:
        title = m.get("title", "")
        title_lower = title.lower()
        ticker = m.get("ticker", "")

        # ── Filter: title keywords ─────────────────────────────────────────
        if spec.title_keywords:
            if not any(kw.lower() in title_lower for kw in spec.title_keywords):
                continue

        # ── Filter: player name ────────────────────────────────────────────
        if player_last and player_last not in title_lower:
            continue

        # ── Filter: ticker suffix ──────────────────────────────────────────
        if spec.ticker_suffix:
            if not ticker.upper().endswith(f"-{spec.ticker_suffix.upper()}"):
                continue

        # ── Price validation ───────────────────────────────────────────────
        price = market_price(m)
        if price <= 0 or price >= 100:
            continue

        # ── Parse threshold ────────────────────────────────────────────────
        threshold: Optional[int] = None
        if spec.threshold_pattern:
            threshold = parse_threshold_regex(
                title, spec.threshold_pattern, ceil=spec.threshold_ceil,
            )
            if threshold is None:
                log.debug("Could not parse threshold from %r — skipping", title)
                continue
            if threshold < spec.min_threshold:
                log.debug(
                    "Skipping %d+ (below min threshold %d)",
                    threshold, spec.min_threshold,
                )
                continue

        # ── Call probability model ─────────────────────────────────────────
        result = model(title, threshold, price)
        if result is None:
            continue

        raw_prob_pct = result.prob_pct

        # ── Apply calibration ──────────────────────────────────────────────
        # Record raw probability in signals.jsonl (for future recalibration),
        # then use calibrated probability for edge/trade decisions.
        prob_pct = _calibration.calibrate(spec.strategy_name, raw_prob_pct)
        edge = prob_pct - price
        evaluated.append((threshold, prob_pct, price, edge))

        # ── Record signal (raw model prob for calibration data) ────────────
        traded = edge >= edge_floor and prob_pct >= spec.min_model_prob
        record_signal(
            config.log_dir,
            ticker,
            spec.strategy_name,
            model_prob_pct=raw_prob_pct,
            market_price_cents=price,
            edge_cents=float(edge),
            traded=traded,
            reason=result.reason,
        )

        # ── Build TradeSignal if edge is sufficient ────────────────────────
        if traded:
            count = kelly_count(
                edge, price,
                config.kelly_fraction,
                config.max_position_usd,
                config.max_contracts_per_trade,
            )
            if count > 0:
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="buy",
                    side="yes",
                    count=count,
                    price=price,
                    strategy=spec.strategy_name,
                    confidence=confidence_fn(edge),
                    edge_cents=float(edge),
                    reason=result.reason,
                ))

    # ── NO-side evaluation (opt-in) ────────────────────────────────────────
    if spec.no_side:
        yes_tickers: Set[str] = {s.ticker for s in signals}
        no_signals = _evaluate_no_side(
            markets, spec, model, config, edge_floor, confidence_fn, yes_tickers,
        )
        signals.extend(no_signals)

    # ── Cap signals by edge ────────────────────────────────────────────────
    if spec.max_signals and len(signals) > spec.max_signals:
        signals.sort(key=lambda s: s.edge_cents, reverse=True)
        dropped = signals[spec.max_signals:]
        signals = signals[:spec.max_signals]
        log.info(
            "  ✂️ %s | %s | kept top %d of %d signals (dropped: %s)",
            spec.strategy_name,
            spec.player_name or spec.event_ticker,
            spec.max_signals,
            spec.max_signals + len(dropped),
            ", ".join(f"{s.ticker.rsplit('-', 1)[-1]}" for s in dropped),
        )

    # ── Log when no signals found ──────────────────────────────────────────
    if evaluated and not signals:
        best = max(evaluated, key=lambda x: x[3])
        rows = "  ".join(
            f"{thr}+: P={p}% vs {pr}¢ → {e:+d}¢"
            for thr, p, pr, e in sorted(evaluated, key=lambda x: x[0] or 0)
        )
        log.info(
            "  ⬜ %s | %s | no edge ≥%d¢  (best: %s→%+d¢) | %s",
            spec.strategy_name,
            spec.player_name or spec.event_ticker,
            edge_floor,
            f"{best[0]}+" if best[0] is not None else "n/a",
            best[3],
            rows,
        )
    elif not evaluated:
        log.info(
            "  ⬜ %s | %s | no matching markets found",
            spec.strategy_name,
            spec.player_name or spec.event_ticker,
        )

    return signals


def _evaluate_no_side(
    markets: List[dict],
    spec: MarketSpec,
    model: ModelFn,
    config: Config,
    edge_floor: int,
    confidence_fn: Callable[[float], float],
    yes_tickers: Set[str],
) -> List["TradeSignal"]:
    """Evaluate NO-side trades for markets where the model says YES is unlikely.

    Only called when spec.no_side is True.  Buys NO when the model's YES
    probability is very low but the market prices YES higher.
    """
    from slugger.strategies import TradeSignal

    no_signals: List[TradeSignal] = []

    player_last = ""
    if spec.player_name:
        parts = spec.player_name.split()
        player_last = parts[-1].lower() if parts else ""

    for m in markets:
        title = m.get("title", "")
        title_lower = title.lower()
        ticker = m.get("ticker", "")

        # Same filtering as YES side
        if spec.title_keywords:
            if not any(kw.lower() in title_lower for kw in spec.title_keywords):
                continue
        if player_last and player_last not in title_lower:
            continue
        if spec.ticker_suffix:
            if not ticker.upper().endswith(f"-{spec.ticker_suffix.upper()}"):
                continue

        # Parse threshold
        threshold: Optional[int] = None
        if spec.threshold_pattern:
            threshold = parse_threshold_regex(
                title, spec.threshold_pattern, ceil=spec.threshold_ceil,
            )
            if threshold is None or threshold < spec.min_threshold:
                continue

        yes_price = market_price(m)
        if yes_price <= 0 or yes_price >= 100:
            continue

        # Get model's YES probability
        result = model(title, threshold, yes_price)
        if result is None:
            continue

        raw_yes_pct = result.prob_pct
        model_yes_pct = _calibration.calibrate(spec.strategy_name, raw_yes_pct)

        # Only consider NO when model is very confident YES won't happen
        if model_yes_pct > spec.no_max_model_prob:
            continue

        # NO edge = how much the market overprices YES
        no_edge = yes_price - model_yes_pct
        if no_edge < spec.no_min_edge_cents:
            continue

        no_price = 100 - yes_price
        if no_price <= 0 or no_price >= 100:
            continue

        # Don't hedge against our own YES signal
        if ticker in yes_tickers:
            continue

        no_reason = f"[NO] {result.reason}"

        record_signal(
            config.log_dir,
            ticker,
            spec.strategy_name,
            model_prob_pct=model_yes_pct,
            market_price_cents=yes_price,
            edge_cents=float(no_edge),
            traded=True,
            reason=no_reason,
        )

        count = kelly_count(
            no_edge, no_price,
            config.kelly_fraction,
            config.max_position_usd,
            config.max_contracts_per_trade,
        )
        if count > 0:
            no_signals.append(TradeSignal(
                ticker=ticker,
                action="buy",
                side="no",
                count=count,
                price=no_price,
                strategy=spec.strategy_name,
                confidence=confidence_fn(no_edge),
                edge_cents=float(no_edge),
                reason=no_reason,
            ))

    return no_signals
