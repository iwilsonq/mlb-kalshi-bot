#!/usr/bin/env bash
#
# Daily recorder job (mlb-kalshi-bot-gdq). This is what launchd runs.
#
# Fires once in the early morning, asks the schedule when the first pitch is,
# sleeps until LEAD minutes before it, and starts the recorder. First pitch
# moves by hours across the season (2026-08-19 opened at 09:35 PDT), so a
# fixed launch time either misses day games or burns hours of disk on an
# empty book.
#
#   scripts/daily_recorder.sh              # today
#   scripts/daily_recorder.sh 2026-08-19   # a specific date
#
# Env:
#   RECORDER_LEAD_MIN  minutes before first pitch to start (default 15)
#   PYTHON             interpreter (default python3)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DAY="${1:-$(date +%Y-%m-%d)}"
LEAD_MIN="${RECORDER_LEAD_MIN:-15}"
PY="${PYTHON:-python3}"
LOG="logs/recorder/daily.log"

mkdir -p logs/recorder
say() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*" | tee -a "$LOG"; }

# Buffer stderr rather than tee it: statsapi's dependencies emit a urllib3
# LibreSSL warning on every single call, and a daily log that is mostly
# known-harmless warnings is a daily log nobody reads.
errfile="$(mktemp -t recorder_fp)"
trap 'rm -f "$errfile"' EXIT

set +e
first_pitch=$("$PY" scripts/first_pitch.py --epoch "$DAY" 2>"$errfile")
status=$?
set -e

if [ "$status" -eq 2 ]; then
    say "$DAY: no games scheduled — nothing to record"
    exit 0
elif [ "$status" -ne 0 ]; then
    say "$DAY: could not read the schedule (exit $status) — not starting"
    sed 's/^/    /' "$errfile" | tee -a "$LOG"
    exit 1
fi

now=$(date +%s)
target=$((first_pitch - LEAD_MIN * 60))

if [ "$target" -gt "$now" ]; then
    wait_s=$((target - now))
    say "$DAY: first pitch $(date -r "$first_pitch" '+%H:%M %Z'), starting in $((wait_s / 60))m"
    # caffeinate the wait too: an idle-slept machine wakes on the timer, but
    # only after the sleep has already overshot.
    caffeinate -is sleep "$wait_s"
else
    # Started late (machine was asleep, job added mid-day, manual re-run).
    # A partial slate is worth more than no slate.
    say "$DAY: first pitch already passed — starting now, slate will be partial"
fi

say "$DAY: launching recorder"

# --foreground, not detached: launchd tears down the job's whole process
# group the moment this script exits, so a backgrounded recorder dies
# seconds after it starts and leaves an empty slate behind. Staying in the
# foreground makes launchd the owner of the recorder for the whole day,
# which is what it is good at.
rc=0
scripts/start_recorder.sh --foreground "$DAY" || rc=$?
case "$rc" in
    0) say "$DAY: recorder finished (all games final)" ;;
    3) say "$DAY: a recorder was already running — leaving it alone" ; rc=0 ;;
    *) say "$DAY: recorder exited $rc — see logs/recorder/$DAY/recorder.out" ;;
esac
exit "$rc"
