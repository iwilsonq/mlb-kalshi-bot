"""Phase 0 live recorder (mlb-kalshi-bot-z36).

Synchronized capture of:
  - Kalshi websocket streams (ticker / trade / orderbook_delta) for
    in-play MLB game-winner and totals markets
  - MLB Stats API GUMBO live play-by-play feeds

Zero trading. Every record carries a local receive timestamp (`recv_ts`)
so downstream analysis can measure our feed latency vs price moves
(adverse-selection check) and overshoot vs a win-probability anchor.
"""
from slugger.recorder.recorder import run_recorder  # noqa: F401
