"""Phase 0 recorder orchestrator (mlb-kalshi-bot-z36).

Discovers today's MLB games, maps them to Kalshi game-winner and totals
markets, then records two synchronized streams to logs/recorder/<date>/:

  kalshi.jsonl  - every websocket frame (ticker / trade / orderbook_delta /
                  orderbook_snapshot), stamped with local receive time
  gumbo.jsonl   - compact game-state snapshots + completed plays per game

Zero trading. Run it before the first pitch of the day:

    python3 main.py record
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date as date_cls
from pathlib import Path
from typing import Dict, List, Optional

from slugger.config import Config
from slugger.mlb_data import get_todays_games
from slugger.recorder.gumbo import poll_game
from slugger.recorder.kalshi_ws import KalshiWSRecorder
from slugger.recorder.writer import JsonlWriter
from slugger.tickers import game_event_ticker, total_event_ticker

log = logging.getLogger(__name__)


def _discover_markets(config: Config, games) -> Dict[str, List[str]]:
    """Map event tickers -> market tickers for game-winner and totals.

    Returns dict of event_ticker -> [market_ticker, ...].
    """
    client = config.create_kalshi_client()
    found: Dict[str, List[str]] = {}
    for game in games:
        for build in (game_event_ticker, total_event_ticker):
            event_ticker = build(game)
            if not event_ticker:
                continue
            try:
                markets = client.get_event_markets(event_ticker)
            except Exception as e:
                log.debug("Market lookup failed for %s: %s", event_ticker, e)
                continue
            tickers = [m["ticker"] for m in markets if m.get("ticker")]
            if tickers:
                found[event_ticker] = tickers
                log.info("  %s: %d markets", event_ticker, len(tickers))
    return found


async def _run(config: Config, target_date: Optional[str]) -> None:
    day = target_date or date_cls.today().strftime("%Y-%m-%d")
    out_dir = Path(config.log_dir) / "recorder" / day
    kalshi_writer = JsonlWriter(out_dir / "kalshi.jsonl")
    gumbo_writer = JsonlWriter(out_dir / "gumbo.jsonl")
    log.info("Recording to %s", out_dir)

    games = get_todays_games(day)
    if not games:
        log.error("No MLB games found for %s — nothing to record", day)
        return
    log.info("Found %d games for %s", len(games), day)

    market_map = _discover_markets(config, games)
    all_market_tickers = [t for ts in market_map.values() for t in ts]
    if not all_market_tickers:
        log.error("No open Kalshi markets found for %s — nothing to record", day)
        return
    log.info(
        "Recording %d Kalshi markets across %d events",
        len(all_market_tickers), len(market_map),
    )

    # Manifest record: lets analysis join game_pk <-> event/market tickers
    kalshi_writer.write({
        "src": "recorder",
        "type": "manifest",
        "date": day,
        "games": [
            {
                "game_pk": g.game_id,
                "away": g.away_abbrev,
                "home": g.home_abbrev,
                "game_datetime": g.game_datetime,
                "game_event_ticker": game_event_ticker(g),
                "total_event_ticker": total_event_ticker(g),
            }
            for g in games
        ],
        "markets": market_map,
    })

    stop_event = asyncio.Event()

    def on_ws_message(msg: dict, recv_ts: float) -> None:
        kalshi_writer.write({
            "recv_ts": recv_ts,
            "src": "kalshi",
            "type": msg.get("type", "unknown"),
            "sid": msg.get("sid"),
            "seq": msg.get("seq"),
            "msg": msg.get("msg", msg),
        })

    ws = KalshiWSRecorder(
        api_key_id=config.kalshi_api_key_id,
        private_key_pem=Path(config.kalshi_private_key_path).expanduser().read_text().strip(),
        market_tickers=all_market_tickers,
        on_message=on_ws_message,
        use_demo=config.use_demo,
    )

    ws_task = asyncio.ensure_future(ws.run())
    gumbo_tasks = [
        asyncio.ensure_future(poll_game(g.game_id, gumbo_writer.write, stop_event))
        for g in games if g.game_id
    ]

    try:
        # Run until every game is Final (gumbo pollers exit on Final)
        await asyncio.gather(*gumbo_tasks)
        log.info("All games final — shutting down recorder")
    finally:
        stop_event.set()
        ws.stop()
        ws_task.cancel()
        try:
            await ws_task
        except (asyncio.CancelledError, Exception):
            pass
        kalshi_writer.close()
        gumbo_writer.close()


def run_recorder(config: Config, target_date: Optional[str] = None) -> None:
    """Blocking entry point for `python3 main.py record`."""
    try:
        asyncio.run(_run(config, target_date))
    except KeyboardInterrupt:
        log.info("Recorder interrupted — files flushed")
