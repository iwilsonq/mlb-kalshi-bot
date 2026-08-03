"""In-session game state for latency / SP-scratch invalidation (Phase 3)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from slugger.types import GameInfo

log = logging.getLogger(__name__)


@dataclass
class GameStateTracker:
    """Track probable pitchers between poll cycles.

    If a starting pitcher changes (scratch / late change), the game is marked
    dirty so the bot skips trading until re-hydrated cleanly next cycle.
    """

    # game_id → (away_pitcher_id, home_pitcher_id)
    _pitchers: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    _dirty: Dict[int, str] = field(default_factory=dict)

    def observe(self, game: GameInfo) -> Optional[str]:
        """Update tracker. Returns invalidate reason if SP changed, else None."""
        gid = game.game_id
        cur = (int(game.away_pitcher_id or 0), int(game.home_pitcher_id or 0))
        prev = self._pitchers.get(gid)
        self._pitchers[gid] = cur
        if prev is None:
            self._dirty.pop(gid, None)
            return None
        if prev == cur:
            self._dirty.pop(gid, None)
            return None
        # Pitcher id changed after we had seen the game
        if prev[0] != cur[0] and cur[0] and prev[0]:
            reason = f"away SP scratch/change {prev[0]}→{cur[0]}"
        elif prev[1] != cur[1] and cur[1] and prev[1]:
            reason = f"home SP scratch/change {prev[1]}→{cur[1]}"
        else:
            reason = f"pitcher ids changed {prev}→{cur}"
        self._dirty[gid] = reason
        log.warning("Game %s invalidated: %s", gid, reason)
        return reason

    def is_invalid(self, game_id: int) -> bool:
        return game_id in self._dirty

    def invalidate_reason(self, game_id: int) -> Optional[str]:
        return self._dirty.get(game_id)

    def clear(self, game_id: int) -> None:
        self._dirty.pop(game_id, None)
