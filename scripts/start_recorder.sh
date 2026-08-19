#!/usr/bin/env bash
#
# Start the Phase 0 recorder for a slate (mlb-kalshi-bot-gdq).
#
#   scripts/start_recorder.sh                     # today, detached
#   scripts/start_recorder.sh 2026-08-19          # a specific date
#   scripts/start_recorder.sh --foreground [DATE] # block until the slate ends
#
# Two modes, and the difference matters:
#
#   detached (default)  nohup + background, returns immediately. For a
#                       terminal, where you want your prompt back.
#   --foreground        blocks for the whole slate. This is what launchd
#                       must use: launchd signals the job's entire process
#                       group when the main process exits, so a detached
#                       recorder is killed seconds after launch — verified,
#                       and it fails silently (a directory with a truncated
#                       recorder.out and no data).
#
# caffeinate -is keeps the machine awake while the recorder runs. It does
# NOT survive closing the lid on battery — a recording machine has to be
# plugged in, or have its lid open, or the streams stop mid-slate.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

FOREGROUND=0
if [ "${1:-}" = "--foreground" ]; then
    FOREGROUND=1
    shift
fi

DAY="${1:-$(date +%Y-%m-%d)}"
OUT="logs/recorder/$DAY"
PY="${PYTHON:-python3}"

# Two recorders would interleave lines into the same jsonl files and double
# the Kalshi subscriptions. Refuse rather than corrupt a slate.
# Exit 3 (not 1) so callers can tell "already recording, all is well" from
# "the recorder failed to start".
if running=$(pgrep -f 'main[.]py record' | tr '\n' ' '); then
    echo "refusing to start: 'main.py record' already running (pid ${running% })" >&2
    echo "  kill it first, or check: python3 scripts/recorder_status.py" >&2
    exit 3
fi

mkdir -p "$OUT"

# nohup in both modes: a closing terminal (detached) or a wedged launchd
# session should not SIGHUP a recording mid-slate.
nohup caffeinate -is "$PY" main.py record --date "$DAY" >>"$OUT/recorder.out" 2>&1 &
pid=$!
echo "$pid" > "$OUT/recorder.pid"

if [ "$FOREGROUND" -eq 1 ]; then
    # Pass shutdown on to the recorder. SIGINT is the one it handles
    # explicitly (flush + close); every JSONL write already fsyncs, so even
    # a SIGKILL costs no recorded data.
    trap 'kill -INT "$pid" 2>/dev/null || true; pkill -INT -P "$pid" 2>/dev/null || true' INT TERM
    echo "recorder running in foreground for $DAY (pid $pid) -> $OUT/"
    rc=0
    wait "$pid" || rc=$?
    rm -f "$OUT/recorder.pid"
    exit "$rc"
fi

disown "$pid" 2>/dev/null || true

# The recorder dies early on predictable problems (no games, no open Kalshi
# markets, bad credentials). Fail loudly here instead of leaving a stale
# pidfile and an empty directory for someone to find hours later.
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$OUT/recorder.pid"
    echo "recorder exited within 5s — last lines of $OUT/recorder.out:" >&2
    tail -n 15 "$OUT/recorder.out" >&2 || true
    exit 1
fi

echo "recorder started for $DAY (pid $pid) -> $OUT/"
