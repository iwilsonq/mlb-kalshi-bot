"""Portfolio risk units and rolling strategy kill-switches.

- GameFactorBudget: correlated same-game exposure (not just ticker dedup)
- StrategyHealth: rolling window ROI / simple Brier-vs-market auto-disable
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# KXMLBKS-26MAY231420HOUCHC-... → event key from first two segments
_EVENT_RE = re.compile(r"^([A-Z0-9]+)-([A-Z0-9]+)")


def game_factor_key(ticker: str) -> str:
    """Stable same-game risk key from a Kalshi market ticker.

    Uses product + date/teams segment (first two dash parts), e.g.
    KXMLBKS-26MAY231420HOUCHC for all K markets on that game.
    Falls back to full ticker if unparseable.
    """
    if not ticker:
        return ""
    m = _EVENT_RE.match(ticker.upper())
    if not m:
        return ticker
    # Include series so game_winner and Ks for same slate still group by teams+date
    # when middle segment encodes date+teams
    return f"{m.group(1)}-{m.group(2)}"


def game_family_key(ticker: str) -> str:
    """Broader family: strip product prefix, keep date+teams only.

    Maps KXMLBKS-26MAY... and KXMLBHIT-26MAY... for the same game
    into one family when the second segment matches.
    """
    if not ticker:
        return ""
    parts = ticker.upper().split("-")
    if len(parts) >= 2:
        return parts[1]  # date+teams blob
    return ticker


@dataclass
class GameFactorBudget:
    """Caps correlated exposure sharing a game family key."""

    max_signals_per_game: int = 5
    max_exposure_usd: float = 0.0
    # family_key → signals placed / dollars
    signals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    exposure: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def can_place(self, ticker: str, cost_usd: float = 0.0) -> bool:
        key = game_family_key(ticker)
        if self.signals[key] >= self.max_signals_per_game:
            return False
        if self.max_exposure_usd > 0 and self.exposure[key] + cost_usd > self.max_exposure_usd:
            return False
        return True

    def record(self, ticker: str, cost_usd: float) -> None:
        key = game_family_key(ticker)
        self.signals[key] += 1
        self.exposure[key] += cost_usd


@dataclass
class StrategyHealthMonitor:
    """Rolling window of settled trade outcomes → auto-disable strategies.

    A trade is a loss if pnl_usd < 0. Rolling ROI =
    sum(pnl) / sum(cost) over the last window_n settled trades per strategy.
    """

    window_n: int = 50
    min_trades: int = 30
    min_roi_pct: float = -25.0  # disable if rolling ROI below this
    # strategy → deque of (pnl, cost)
    _windows: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    disabled: Set[str] = field(default_factory=set)

    def _window(self, strategy: str) -> Deque[Tuple[float, float]]:
        if strategy not in self._windows:
            self._windows[strategy] = deque(maxlen=self.window_n)
        return self._windows[strategy]

    def observe(self, strategy: str, pnl_usd: float, cost_usd: float) -> None:
        if not strategy:
            return
        w = self._window(strategy)
        w.append((float(pnl_usd), max(0.0, float(cost_usd))))
        self._recompute(strategy)

    def _recompute(self, strategy: str) -> None:
        w = self._window(strategy)
        if len(w) < self.min_trades:
            return
        pnl = sum(x[0] for x in w)
        cost = sum(x[1] for x in w)
        if cost <= 0:
            return
        roi = pnl / cost * 100.0
        if roi < self.min_roi_pct:
            if strategy not in self.disabled:
                log.warning(
                    "⛔ Strategy auto-disabled: %s rolling ROI %.1f%% < %.1f%% (n=%d)",
                    strategy, roi, self.min_roi_pct, len(w),
                )
            self.disabled.add(strategy)

    def is_enabled(self, strategy: str, config_enabled: Iterable[str]) -> bool:
        if strategy not in set(config_enabled):
            return False
        return strategy not in self.disabled

    def load_from_journal(self, records: List[dict]) -> None:
        """Seed windows from historical trades+settlements (chronological)."""
        trades = {r["ticker"]: r for r in records if r.get("type") == "trade"}
        # settlements in file order
        for r in records:
            if r.get("type") != "settlement":
                continue
            t = trades.get(r.get("ticker", ""))
            if not t:
                continue
            self.observe(
                t.get("strategy", "unknown"),
                float(r.get("pnl_usd") or 0.0),
                float(t.get("cost_usd") or 0.0),
            )

    def effective_enabled(self, config_enabled: Iterable[str]) -> Tuple[str, ...]:
        return tuple(s for s in config_enabled if s not in self.disabled)
