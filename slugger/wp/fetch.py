"""Build a WP training dataset from historical GUMBO feeds.

For each completed regular-season game, extract the game state BEFORE each
plate appearance (inning, half, outs, base state, home-perspective score
diff) plus the final home_win label. Per-game rows are cached to
logs/wp/games/{gamePk}.json so re-runs are incremental.

CLI:
    python3 -m slugger.wp.fetch 2025-04-01 2025-09-28
"""
from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
DEFAULT_CACHE_DIR = "logs/wp/games"
MAX_WORKERS = 8

# Server-side field filter (same pattern as slugger/recorder/gumbo.py):
# only what pre-PA state reconstruction needs.
FIELDS = ",".join([
    "gameData", "status", "abstractGameState",
    "liveData",
    "plays", "allPlays",
    "about", "atBatIndex", "inning", "halfInning", "isComplete",
    "count", "outs",
    "result", "homeScore", "awayScore",
    "matchup", "postOnFirst", "postOnSecond", "postOnThird", "id",
])

INNING_CAP = 9
DIFF_CLAMP = 8


def list_final_games(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """[{game_pk, date}] for completed regular-season games in the range.

    Dates are YYYY-MM-DD (inclusive). Uses statsapi.schedule; keeps only
    game_type == 'R' with a Final status.
    """
    import statsapi

    sched = statsapi.schedule(start_date=start_date, end_date=end_date)
    games: List[Dict[str, Any]] = []
    seen = set()
    for g in sched:
        if g.get("game_type") != "R":
            continue
        if not str(g.get("status", "")).startswith("Final"):
            continue
        pk = g.get("game_id")
        if not pk or pk in seen:
            continue
        seen.add(pk)
        games.append({"game_pk": int(pk), "date": (g.get("game_date") or "")[:10]})
    return games


def _fetch_feed(game_pk: int) -> Dict[str, Any]:
    resp = requests.get(
        FEED_URL.format(pk=game_pk),
        params={"fields": FIELDS},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def extract_pa_states(all_plays: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Pre-PA states + home_win label from a GUMBO allPlays list.

    State before play i is reconstructed from play i-1:
      - outs / runners reset at each half-inning boundary
      - otherwise outs = prev count.outs, runners = prev matchup.postOn*
      - score = prev result.homeScore/awayScore (0-0 before the first play)

    Returns None if the game can't be labelled (no plays / tied score).
    """
    plays = [p for p in all_plays if (p.get("about") or {}).get("isComplete")]
    if not plays:
        return None

    last_result = plays[-1].get("result") or {}
    final_home = last_result.get("homeScore")
    final_away = last_result.get("awayScore")
    if final_home is None or final_away is None or final_home == final_away:
        return None
    home_win = 1 if final_home > final_away else 0

    rows: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for play in plays:
        about = play.get("about") or {}
        inning = about.get("inning")
        half = about.get("halfInning")
        if inning is None or half not in ("top", "bottom"):
            prev = play
            continue
        is_top = half == "top"

        if prev is None:
            outs, on1, on2, on3 = 0, False, False, False
            home_score = away_score = 0
        else:
            p_about = prev.get("about") or {}
            p_result = prev.get("result") or {}
            home_score = int(p_result.get("homeScore") or 0)
            away_score = int(p_result.get("awayScore") or 0)
            same_half = (
                p_about.get("inning") == inning
                and p_about.get("halfInning") == half
            )
            if same_half:
                p_count = prev.get("count") or {}
                outs = int(p_count.get("outs") or 0)
                p_matchup = prev.get("matchup") or {}
                on1 = bool(p_matchup.get("postOnFirst"))
                on2 = bool(p_matchup.get("postOnSecond"))
                on3 = bool(p_matchup.get("postOnThird"))
            else:
                outs, on1, on2, on3 = 0, False, False, False

        if outs >= 3:
            # Defensive: a completed half-inning leaking into the next play
            prev = play
            continue

        diff = home_score - away_score
        diff = max(-DIFF_CLAMP, min(DIFF_CLAMP, diff))
        rows.append({
            "inning": min(int(inning), INNING_CAP),
            "is_top": is_top,
            "outs": outs,
            "on1": on1,
            "on2": on2,
            "on3": on3,
            "score_diff": diff,
            "home_win": home_win,
        })
        prev = play

    return rows or None


def fetch_game_rows(
    game_pk: int,
    date: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Dict[str, Any]]:
    """Extracted rows for one game, cached to {cache_dir}/{gamePk}.json."""
    cache = Path(cache_dir) / f"{game_pk}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            return d.get("rows", [])
        except Exception:
            pass  # corrupt cache — refetch

    feed = _fetch_feed(game_pk)
    status = (
        (feed.get("gameData", {}) or {}).get("status", {}) or {}
    ).get("abstractGameState", "")
    all_plays = (
        (feed.get("liveData", {}) or {}).get("plays", {}) or {}
    ).get("allPlays", []) or []

    rows: List[Dict[str, Any]] = []
    if status == "Final":
        extracted = extract_pa_states(all_plays)
        if extracted:
            for r in extracted:
                r["game_pk"] = game_pk
                r["date"] = date
            rows = extracted

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"game_pk": game_pk, "date": date, "rows": rows}))
    return rows


def build_dataset(
    start_date: str,
    end_date: str,
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_workers: int = MAX_WORKERS,
    deadline_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """All pre-PA rows for completed regular-season games in the date range.

    Cached games cost no network. deadline_sec (wall-clock budget) lets a
    long fetch stop early and keep what it has.
    """
    games = list_final_games(start_date, end_date)
    log.info("WP dataset: %d final regular-season games %s..%s",
             len(games), start_date, end_date)

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_game_rows, g["game_pk"], g["date"], cache_dir): g
            for g in games
        }
        for fut in as_completed(futures):
            if deadline_sec is not None and time.time() - t0 > deadline_sec:
                log.warning("WP dataset: deadline hit; stopping with %d games", n_done)
                for f in futures:
                    f.cancel()
                break
            g = futures[fut]
            try:
                rows.extend(fut.result())
                n_done += 1
            except Exception as exc:
                n_err += 1
                log.debug("WP fetch failed for %s: %s", g["game_pk"], exc)
            if n_done and n_done % 200 == 0:
                log.info("WP dataset: %d/%d games (%.0fs)", n_done, len(games), time.time() - t0)
    if n_err:
        log.warning("WP dataset: %d games failed to fetch", n_err)
    return rows


def load_cached_rows(cache_dir: str = DEFAULT_CACHE_DIR) -> List[Dict[str, Any]]:
    """All rows already on disk (no network)."""
    rows: List[Dict[str, Any]] = []
    d = Path(cache_dir)
    if not d.exists():
        return rows
    for p in sorted(d.glob("*.json")):
        try:
            rows.extend(json.loads(p.read_text()).get("rows", []))
        except Exception:
            continue
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-06-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-06-07"
    deadline = float(sys.argv[3]) if len(sys.argv) > 3 else None
    out = build_dataset(start, end, deadline_sec=deadline)
    games = len({r["game_pk"] for r in out})
    print(f"rows={len(out)} games={games} range={start}..{end}")
