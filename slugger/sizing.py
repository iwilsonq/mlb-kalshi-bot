"""Position sizing for Slugger MLB trading bot.

Pure math — no I/O, no Kalshi coupling.
"""
from __future__ import annotations


def kelly_count(
    edge_cents: float,
    price_cents: int,
    kelly_fraction: float,
    max_position_usd: float,
    max_contracts: int = 5000,
) -> int:
    """Calculate contract count using fractional Kelly sizing.

    Args:
        edge_cents:      Expected edge (model_prob - market_price) in cents.
        price_cents:     Limit price per contract in cents (1-99).
        kelly_fraction:  Fraction of full Kelly to use (e.g. 0.25 for quarter-Kelly).
        max_position_usd: Maximum dollar amount to risk on a single position.
        max_contracts:   Hard cap on number of contracts.

    Returns:
        Number of contracts to buy (0 if no edge or invalid inputs).
    """
    if edge_cents <= 0 or price_cents <= 0:
        return 0
    kelly_pct = edge_cents / price_cents
    count = int((kelly_fraction * kelly_pct * max_position_usd * 100) / price_cents)
    return max(0, min(count, max_contracts))
