"""Order pricing / maker discipline helpers."""
from __future__ import annotations

from typing import Optional


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
