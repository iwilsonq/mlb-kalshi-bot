"""Historical point-in-time backtest harness.

Reconstruct pitcher/batter profiles from game logs *before* a target date,
persist GameContext snapshots, and replay strategies without future leakage.
"""
from slugger.backtest.pit import (
    pitcher_profile_from_logs,
    batter_profile_from_logs,
    filter_logs_before,
)
from slugger.backtest.snapshot import (
    save_snapshot,
    load_snapshot,
    game_context_to_dict,
    game_context_from_dict,
)
from slugger.backtest.replay import replay_strategies, ReplayResult

__all__ = [
    "pitcher_profile_from_logs",
    "batter_profile_from_logs",
    "filter_logs_before",
    "save_snapshot",
    "load_snapshot",
    "game_context_to_dict",
    "game_context_from_dict",
    "replay_strategies",
    "ReplayResult",
]
