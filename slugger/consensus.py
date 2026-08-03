"""Optional external consensus prior for market gating (Phase 3).

If CONSENSUS_PRICES_PATH JSON is present:
  { "TICKER": {"fair_cents": 42}, ... }

A trade is allowed only when Kalshi ask is cheaper than consensus fair
by min_edge_cents (or file missing → no consensus gate).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)


def load_consensus_prices(path: Optional[str] = None) -> Dict[str, float]:
    path = path or os.getenv("CONSENSUS_PRICES_PATH", "")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        log.debug("Consensus file missing: %s", path)
        return {}
    try:
        data = json.loads(p.read_text())
        out: Dict[str, float] = {}
        for k, v in data.items():
            if isinstance(v, dict) and "fair_cents" in v:
                out[k] = float(v["fair_cents"])
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        return out
    except Exception as exc:
        log.warning("Failed to load consensus prices: %s", exc)
        return {}


def consensus_allows_trade(
    ticker: str,
    ask_cents: int,
    consensus: Dict[str, float],
    min_edge_cents: float = 3.0,
) -> bool:
    """True if no consensus, or ask is at least min_edge below consensus fair."""
    if not consensus:
        return True
    fair = consensus.get(ticker)
    if fair is None:
        return True  # no line for this market — do not block
    return float(ask_cents) <= float(fair) - float(min_edge_cents)
