"""Order pricing / maker discipline helpers."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol

log = logging.getLogger(__name__)


def limit_price_cents(
    *,
    fair_prob_pct: float,
    ask_cents: int,
    side: str = "yes",
    buffer_cents: int = 1,
    max_cents: int = 99,
) -> int:
    """Choose a limit buy price at or below fair, never above ask.

    Posts below model-fair by buffer_cents when possible (maker-friendly).
    For YES: fair is model yes%; for NO: fair is 100 - model yes%.
    """
    if side == "no":
        fair = int(round(100.0 - float(fair_prob_pct)))
    else:
        fair = int(round(float(fair_prob_pct)))
    target = max(1, fair - max(0, buffer_cents))
    if ask_cents > 0:
        target = min(target, ask_cents)
    return max(1, min(int(target), max_cents))


def should_cancel_for_first_pitch(game_has_started: bool) -> bool:
    """Cancel resting orders once the game is underway."""
    return bool(game_has_started)


def classify_fill_role(limit_cents: int, fill_cents: int, ask_at_entry: int) -> str:
    """Maker if fill better than ask (rested), else taker-ish.

    Heuristic: filled at ask → taker; filled below ask → maker/passive.
    """
    if fill_cents <= 0:
        return "unknown"
    if ask_at_entry > 0 and fill_cents < ask_at_entry:
        return "maker"
    if ask_at_entry > 0 and fill_cents >= ask_at_entry:
        return "taker"
    if fill_cents < limit_cents:
        return "maker"
    return "taker"


def cancel_resting_orders_for_started_games(
    orders: List[dict],
    ticker_game_started: Callable[[str], bool],
    cancel_fn: Callable[[str], bool],
) -> int:
    """Cancel resting GTC orders whose market game has started.

    Args:
        orders: list of order dicts with order_id/ticker/status
        ticker_game_started: returns True if that ticker's game started
        cancel_fn: cancel_order(order_id) -> bool

    Returns:
        Number of successful cancels.
    """
    n = 0
    for o in orders:
        status = (o.get("status") or "").lower()
        if status and status not in ("resting", "open", "pending"):
            continue
        ticker = o.get("ticker") or o.get("market_ticker") or ""
        oid = o.get("order_id") or o.get("id") or ""
        if not oid or not ticker:
            continue
        if not should_cancel_for_first_pitch(ticker_game_started(ticker)):
            continue
        if cancel_fn(oid):
            n += 1
            log.info("Cancelled resting order %s for started game ticker %s", oid, ticker)
    return n
