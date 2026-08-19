#!/usr/bin/env python3
"""Earliest scheduled first pitch for a date (mlb-kalshi-bot-gdq).

The recorder has to be running before the first pitch of the day, and that
time moves around a lot — 2026-08-19's first game started 09:35 PDT. Rather
than hard-code a launch time, ask the schedule.

Deliberately lightweight: `statsapi.schedule` is one HTTP call, unlike
`slugger.mlb_data.get_todays_games`, which pulls every game's live feed.
scripts/daily_recorder.sh calls this on a timer, so it must be cheap and
must not need Kalshi credentials.

Usage:
    python3 scripts/first_pitch.py                 # human summary
    python3 scripts/first_pitch.py --epoch         # unix seconds, for sleep math
    python3 scripts/first_pitch.py 2026-08-19      # a specific date

Exit codes:
    0  a first pitch was found
    2  the schedule loaded but there are no games (an off day, not an error)
    1  the schedule could not be fetched or parsed
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys


def _parse_iso_utc(value: str) -> dt.datetime:
    """Parse statsapi's '2026-08-19T16:35:00Z' into an aware UTC datetime."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def first_pitch(day: str) -> dt.datetime:
    """Earliest scheduled start on `day`, as an aware UTC datetime.

    Raises LookupError if the slate is empty, RuntimeError if the schedule
    cannot be fetched.
    """
    try:
        import statsapi
    except ImportError as e:  # pragma: no cover - environment problem
        raise RuntimeError(f"mlb-statsapi not installed: {e}") from e

    try:
        games = statsapi.schedule(date=day)
    except Exception as e:
        raise RuntimeError(f"schedule fetch failed for {day}: {e}") from e

    # No game_type filter on purpose: the recorder records whatever
    # statsapi.schedule returns (slugger/recorder/recorder.py ->
    # get_todays_games), so this must agree with it or we would start late
    # for a slate the recorder considers real.
    starts = []
    for g in games:
        raw = g.get("game_datetime")
        if not raw:
            continue
        try:
            starts.append(_parse_iso_utc(raw))
        except ValueError:
            continue

    if not starts:
        raise LookupError(f"no scheduled games for {day}")
    return min(starts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "date", nargs="?", default=dt.date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD", help="Date to query (default: today, local)",
    )
    ap.add_argument(
        "--epoch", action="store_true",
        help="Print only unix seconds of the first pitch (for shell arithmetic)",
    )
    args = ap.parse_args()

    try:
        fp = first_pitch(args.date)
    except LookupError as e:
        print(e, file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    if args.epoch:
        print(int(fp.timestamp()))
        return 0

    local = fp.astimezone()
    print(f"{args.date}: first pitch {local:%H:%M %Z} ({fp:%Y-%m-%dT%H:%M:%SZ})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
