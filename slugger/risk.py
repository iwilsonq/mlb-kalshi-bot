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

    Kill-switches (either trips disable):
      1. Rolling ROI = sum(pnl)/sum(cost) over last window_n trades < min_roi_pct
      2. Rolling Brier(model) − Brier(market) < min_brier_edge
         (negative means model worse than market)

    Disabling latches for the life of the process: once a live session trips a
    strategy it stays off until restart, so it cannot flap trade-by-trade.

    Seeding from history is deliberately *not* latching. load_from_journal fills
    the windows and then judges each strategy once from its final window, because
    replaying with per-observation evaluation latches on the worst window that
    ever occurred and the rolling window stops meaning anything.
    """

    window_n: int = 50
    min_trades: int = 30
    min_roi_pct: float = -25.0  # disable if rolling ROI below this
    # Max allowed model Brier deficit vs market (model_brier - market_brier).
    # If average (model_brier_i - market_brier_i) > max_brier_deficit, disable.
    max_brier_deficit: float = 0.02
    # strategy → deque of (pnl, cost)
    _windows: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    # strategy → deque of (model_brier_term, market_brier_term) per settlement
    _brier_windows: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    disabled: Set[str] = field(default_factory=set)
    # strategy → why it was disabled, for logs and reporting
    disabled_reasons: Dict[str, str] = field(default_factory=dict)
    # True while replaying history: record observations but defer judgement
    _seeding: bool = field(default=False, repr=False)

    def _window(self, strategy: str) -> Deque[Tuple[float, float]]:
        if strategy not in self._windows:
            self._windows[strategy] = deque(maxlen=self.window_n)
        return self._windows[strategy]

    def _brier_window(self, strategy: str) -> Deque[Tuple[float, float]]:
        if strategy not in self._brier_windows:
            self._brier_windows[strategy] = deque(maxlen=self.window_n)
        return self._brier_windows[strategy]

    def observe(self, strategy: str, pnl_usd: float, cost_usd: float) -> None:
        if not strategy:
            return
        w = self._window(strategy)
        w.append((float(pnl_usd), max(0.0, float(cost_usd))))
        if not self._seeding:
            self._recompute(strategy)

    def observe_probability(
        self,
        strategy: str,
        model_prob_pct: float,
        market_price_cents: float,
        outcome_yes: int,
    ) -> None:
        """Record one Brier observation for model vs market (outcome 0/1)."""
        if not strategy:
            return
        y = 1.0 if outcome_yes else 0.0
        p_m = min(max(float(model_prob_pct) / 100.0, 1e-6), 1.0 - 1e-6)
        p_k = min(max(float(market_price_cents) / 100.0, 1e-6), 1.0 - 1e-6)
        mb = (p_m - y) ** 2
        kb = (p_k - y) ** 2
        self._brier_window(strategy).append((mb, kb))
        if not self._seeding:
            self._recompute(strategy)

    def health_reason(self, strategy: str) -> Optional[str]:
        """Why the strategy's *current* window is unhealthy, or None if it is fine.

        Pure: reads the windows and returns a verdict without mutating state.
        """
        w = self._window(strategy)
        if len(w) >= self.min_trades:
            pnl = sum(x[0] for x in w)
            cost = sum(x[1] for x in w)
            if cost > 0:
                roi = pnl / cost * 100.0
                if roi < self.min_roi_pct:
                    return (
                        f"rolling ROI {roi:.1f}% < {self.min_roi_pct:.1f}% "
                        f"(n={len(w)})"
                    )

        bw = self._brier_window(strategy)
        if len(bw) >= self.min_trades:
            avg_mb = sum(x[0] for x in bw) / len(bw)
            avg_kb = sum(x[1] for x in bw) / len(bw)
            deficit = avg_mb - avg_kb  # >0 model worse
            if deficit > self.max_brier_deficit:
                return (
                    f"Brier deficit {deficit:.4f} (model {avg_mb:.4f} vs "
                    f"market {avg_kb:.4f}) > {self.max_brier_deficit:.4f} "
                    f"(n={len(bw)})"
                )
        return None

    def _recompute(self, strategy: str) -> None:
        """Judge the current window and latch the strategy off if unhealthy."""
        reason = self.health_reason(strategy)
        if reason is None:
            return
        if strategy not in self.disabled:
            log.warning("⛔ Strategy auto-disabled: %s %s", strategy, reason)
        self.disabled.add(strategy)
        self.disabled_reasons[strategy] = reason

    def is_enabled(self, strategy: str, config_enabled: Iterable[str]) -> bool:
        if strategy not in set(config_enabled):
            return False
        return strategy not in self.disabled

    def load_from_journal(self, records: List[dict]) -> None:
        """Seed windows from historical trades+settlements (chronological).

        Judges each strategy exactly once, from the window state left at the end
        of the replay. Evaluating after every observation instead would latch on
        the worst window in the whole file: pitcher_ks tripped on an early −246%
        window that was 480 trades stale, and player_hits was held off on a
        window it had long since left (final 50-trade window +101.9%). That
        turned a rolling health check into a permanent record of the worst day.
        """
        trades = {r["ticker"]: r for r in records if r.get("type") == "trade"}
        self._seeding = True
        try:
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
                # Brier terms when model + market prices present on trade
                model_p = t.get("model_prob_pct") or t.get("raw_model_prob_pct")
                mkt = t.get("ask_cents") or t.get("price_cents")
                result = r.get("market_result", "")
                if model_p is not None and mkt is not None and result in ("yes", "no"):
                    self.observe_probability(
                        t.get("strategy", "unknown"),
                        float(model_p),
                        float(mkt),
                        1 if result == "yes" else 0,
                    )
        finally:
            self._seeding = False

        for strategy in sorted(set(self._windows) | set(self._brier_windows)):
            self._recompute(strategy)

    def effective_enabled(self, config_enabled: Iterable[str]) -> Tuple[str, ...]:
        return tuple(s for s in config_enabled if s not in self.disabled)
