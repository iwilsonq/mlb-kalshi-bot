"""Replay strategies against point-in-time GameContext snapshots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from slugger.calibration import CalibrationLayer
from slugger.config import Config
from slugger.kalshi_client import FixtureMarketClient
from slugger.strategies import STRATEGY_PIPELINE
from slugger.types import GameContext, TradeSignal


@dataclass
class ReplayResult:
    signals: List[TradeSignal] = field(default_factory=list)
    by_strategy: Dict[str, int] = field(default_factory=dict)


def replay_strategies(
    ctx: GameContext,
    markets_by_event: Dict[str, List[dict]],
    config: Optional[Config] = None,
    *,
    enabled: Optional[List[str]] = None,
    calibration: Optional[CalibrationLayer] = None,
) -> ReplayResult:
    """Run STRATEGY_PIPELINE on a historical context with fixture markets.

    Does not place orders — only collects TradeSignals.
    """
    config = config or Config(dry_run=True, enabled_strategies=tuple(enabled or ("pitcher_ks",)))
    if enabled is not None:
        # frozen dataclass — rebuild
        config = Config(
            dry_run=True,
            enabled_strategies=tuple(enabled),
            min_edge_cents=config.min_edge_cents,
            edge_cost_buffer_cents=config.edge_cost_buffer_cents,
            kelly_fraction=config.kelly_fraction,
            max_position_usd=config.max_position_usd,
            max_contracts_per_trade=config.max_contracts_per_trade,
            log_dir=config.log_dir,
        )
    client = FixtureMarketClient(markets=markets_by_event)
    cal = calibration or CalibrationLayer()
    enabled_set = set(config.enabled_strategies)
    prior: List[TradeSignal] = []
    result = ReplayResult()

    for name, fn in STRATEGY_PIPELINE:
        if name not in enabled_set:
            continue
        sigs = fn(ctx, client, config, prior, calibration=cal)
        prior.extend(sigs)
        result.signals.extend(sigs)
        result.by_strategy[name] = result.by_strategy.get(name, 0) + len(sigs)

    return result
