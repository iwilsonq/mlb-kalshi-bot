"""Position sizing for Slugger MLB trading bot.

Pure math — no I/O, no Kalshi coupling.

Binary YES contracts priced at p dollars pay $1 if YES wins.
With true probability q, the full Kelly fraction of bankroll is:

    f* = (q - p) / (1 - p)     for q > p, else 0

This is *not* edge/price, which overbets longshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def binary_kelly_fraction(q: float, p: float) -> float:
    """Full Kelly bankroll fraction for a binary contract.

    Args:
        q: True (or model) probability of YES in [0, 1].
        p: Contract price in dollars in (0, 1).

    Returns:
        f* in [0, 1]. Zero when there is no edge or inputs are invalid.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if q <= p or q <= 0.0 or q >= 1.0:
        # Allow q==1 slightly for near-certain; clamp below
        if q >= 1.0 and p < 1.0:
            q = 1.0 - 1e-9
        else:
            return 0.0
    return (q - p) / (1.0 - p)


def legacy_edge_over_price_fraction(edge_cents: float, price_cents: int) -> float:
    """Old (incorrect) longshot-heavy fraction used before Phase 4 Kelly fix."""
    if edge_cents <= 0 or price_cents <= 0:
        return 0.0
    return edge_cents / price_cents


def kelly_count(
    edge_cents: float,
    price_cents: int,
    kelly_fraction: float,
    max_position_usd: float,
    max_contracts: int = 5000,
    *,
    model_prob_pct: Optional[float] = None,
    bankroll_usd: Optional[float] = None,
    remaining_daily_usd: Optional[float] = None,
) -> int:
    """Contract count via fractional binary Kelly.

    Prefers calibrated model_prob_pct when provided. Otherwise reconstructs
    q from price + edge (edge should be cost-adjusted net edge in cents).

    Args:
        edge_cents:           Net edge in cents (model% − price − buffer).
        price_cents:          Limit price per contract (1–99).
        kelly_fraction:       Fraction of full Kelly (e.g. 0.25).
        max_position_usd:     Hard cap on dollars risked on this trade.
        max_contracts:        Hard cap on contracts.
        model_prob_pct:       Calibrated model probability 0–100 (preferred).
        bankroll_usd:         Bankroll for f* scaling (defaults to max_position_usd).
        remaining_daily_usd:  Remaining daily risk budget (None = no daily cap).

    Returns:
        Number of contracts (0 if no edge or invalid inputs).
    """
    if edge_cents <= 0 or price_cents <= 0 or price_cents >= 100:
        return 0
    if kelly_fraction <= 0 or max_position_usd <= 0:
        return 0

    p = price_cents / 100.0
    if model_prob_pct is not None and model_prob_pct > 0:
        q = float(model_prob_pct) / 100.0
    else:
        # Reconstruct q from net edge: edge_cents ≈ (q - p) * 100
        q = p + edge_cents / 100.0

    q = min(max(q, 0.0), 1.0)
    f_star = binary_kelly_fraction(q, p)
    if f_star <= 0:
        return 0

    bankroll = float(bankroll_usd) if bankroll_usd is not None and bankroll_usd > 0 else float(max_position_usd)
    dollars = kelly_fraction * f_star * bankroll
    dollars = min(dollars, max_position_usd)
    if remaining_daily_usd is not None:
        if remaining_daily_usd <= 0:
            return 0
        dollars = min(dollars, remaining_daily_usd)

    if dollars <= 0:
        return 0

    count = int((dollars * 100.0) / price_cents)
    return max(0, min(count, max_contracts))


@dataclass
class DailyRiskBudget:
    """Tracks dollar exposure against a daily bankroll fraction cap."""

    bankroll_usd: float
    max_fraction: float
    spent_usd: float = 0.0

    @property
    def cap_usd(self) -> float:
        if self.bankroll_usd <= 0 or self.max_fraction <= 0:
            return 0.0
        return self.bankroll_usd * self.max_fraction

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def can_spend(self, cost_usd: float) -> bool:
        if self.max_fraction <= 0:
            return True  # cap disabled
        return cost_usd <= self.remaining_usd + 1e-9

    def record(self, cost_usd: float) -> None:
        self.spent_usd += max(0.0, cost_usd)


def daily_spent_from_journal(records: list, date_iso: str) -> float:
    """Sum cost_usd of trade records placed on date_iso (YYYY-MM-DD)."""
    total = 0.0
    for r in records:
        if r.get("type") != "trade":
            continue
        if r.get("date") != date_iso:
            continue
        total += float(r.get("cost_usd") or 0.0)
    return total
