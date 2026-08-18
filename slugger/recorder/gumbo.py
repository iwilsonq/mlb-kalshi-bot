"""GUMBO live play-by-play poller for the Phase 0 recorder.

Polls statsapi.mlb.com's live feed for one game and emits:
  - gumbo_state: compact game-state snapshot whenever the feed timestamp
    changes (inning, half, outs, score, base runners) — the exact inputs a
    win-probability anchor needs
  - gumbo_play:  each newly-completed play (event type, description, score)

Poll cadence adapts to game state: slow while Preview, fast while Live,
exits on Final. Payloads are trimmed server-side via the `fields` param.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

import requests

log = logging.getLogger(__name__)

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

# Server-side field filter: keeps the response small enough to poll fast.
FIELDS = ",".join([
    "metaData", "timeStamp",
    "gameData", "status", "abstractGameState", "detailedState",
    "liveData",
    "plays", "allPlays", "result", "event", "eventType", "description",
    "rbi", "awayScore", "homeScore",
    "about", "atBatIndex", "isComplete", "inning", "halfInning", "endTime",
    "linescore", "currentInning", "inningHalf", "isTopInning",
    "outs", "balls", "strikes",
    "teams", "home", "away", "runs",
    "offense", "first", "second", "third", "id",
])

POLL_LIVE_SEC = 3.0
POLL_PREVIEW_SEC = 30.0
POLL_ERROR_SEC = 10.0


def _fetch_feed(game_pk: int) -> Dict[str, Any]:
    resp = requests.get(
        FEED_URL.format(pk=game_pk),
        params={"fields": FIELDS},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _compact_state(feed: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the win-probability-relevant game state."""
    live = feed.get("liveData", {}) or {}
    ls = live.get("linescore", {}) or {}
    offense = ls.get("offense", {}) or {}
    teams = ls.get("teams", {}) or {}
    return {
        "inning": ls.get("currentInning"),
        "half": ls.get("inningHalf"),
        "is_top": ls.get("isTopInning"),
        "outs": ls.get("outs"),
        "balls": ls.get("balls"),
        "strikes": ls.get("strikes"),
        "home_runs": (teams.get("home", {}) or {}).get("runs"),
        "away_runs": (teams.get("away", {}) or {}).get("runs"),
        "on_first": bool(offense.get("first")),
        "on_second": bool(offense.get("second")),
        "on_third": bool(offense.get("third")),
    }


def _status(feed: Dict[str, Any]) -> str:
    return (
        (feed.get("gameData", {}) or {})
        .get("status", {})
        .get("abstractGameState", "")
    )


async def poll_game(
    game_pk: int,
    write: Callable[[Dict[str, Any]], None],
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """Poll one game's GUMBO feed until Final (or stop_event)."""
    loop = asyncio.get_event_loop()
    last_feed_ts: Optional[str] = None
    last_play_idx = -1
    last_state: Optional[Dict[str, Any]] = None

    log.info("GUMBO poller started for game %s", game_pk)
    while stop_event is None or not stop_event.is_set():
        try:
            feed = await loop.run_in_executor(None, _fetch_feed, game_pk)
            recv_ts = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("GUMBO fetch failed for %s: %s", game_pk, e)
            await asyncio.sleep(POLL_ERROR_SEC)
            continue

        status = _status(feed)
        feed_ts = (feed.get("metaData", {}) or {}).get("timeStamp")

        if feed_ts != last_feed_ts:
            last_feed_ts = feed_ts

            # State snapshot (only when it actually changed)
            state = _compact_state(feed)
            if state != last_state:
                last_state = state
                write({
                    "recv_ts": recv_ts,
                    "src": "gumbo",
                    "type": "gumbo_state",
                    "game_pk": game_pk,
                    "feed_ts": feed_ts,
                    "status": status,
                    "state": state,
                })

            # Newly completed plays
            all_plays = (
                (feed.get("liveData", {}) or {}).get("plays", {}) or {}
            ).get("allPlays", []) or []
            for play in all_plays:
                about = play.get("about", {}) or {}
                idx = about.get("atBatIndex", -1)
                if idx <= last_play_idx or not about.get("isComplete"):
                    continue
                last_play_idx = idx
                result = play.get("result", {}) or {}
                write({
                    "recv_ts": recv_ts,
                    "src": "gumbo",
                    "type": "gumbo_play",
                    "game_pk": game_pk,
                    "feed_ts": feed_ts,
                    "at_bat_index": idx,
                    "inning": about.get("inning"),
                    "half": about.get("halfInning"),
                    "end_time": about.get("endTime"),
                    "event": result.get("event"),
                    "event_type": result.get("eventType"),
                    "description": result.get("description"),
                    "rbi": result.get("rbi"),
                    "away_score": result.get("awayScore"),
                    "home_score": result.get("homeScore"),
                })

        if status == "Final":
            write({
                "recv_ts": time.time(),
                "src": "gumbo",
                "type": "gumbo_final",
                "game_pk": game_pk,
                "feed_ts": feed_ts,
            })
            log.info("GUMBO poller finished for game %s (Final)", game_pk)
            return

        interval = POLL_LIVE_SEC if status == "Live" else POLL_PREVIEW_SEC
        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(interval)

    log.info("GUMBO poller stopped for game %s", game_pk)
