#!/usr/bin/env bash
#
# Install (or refresh) the launchd job that starts the recorder every day
# (mlb-kalshi-bot-gdq).
#
#   scripts/install_recorder_job.sh            # install, fires 05:00 local
#   RECORDER_HOUR=4 scripts/install_recorder_job.sh
#   scripts/install_recorder_job.sh --uninstall
#
# The job only *wakes up* at RECORDER_HOUR; daily_recorder.sh then asks the
# schedule for the real first pitch and sleeps until 15 minutes before it.
# So RECORDER_HOUR just needs to be earlier than any plausible first pitch,
# not accurate.
#
# Paths and the interpreter are baked in at install time: launchd runs jobs
# with a near-empty environment, so nothing here can rely on a login shell.
set -euo pipefail

LABEL="com.slugger.recorder"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOUR="${RECORDER_HOUR:-5}"
MINUTE="${RECORDER_MINUTE:-0}"

uninstall() {
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled $LABEL"
}

if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    exit 0
fi

PY="$(command -v "${PYTHON:-python3}")"
[ -n "$PY" ] || { echo "no python3 on PATH" >&2; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs/recorder"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/scripts/daily_recorder.sh</string>
  </array>

  <key>WorkingDirectory</key>
  <string>$REPO</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin</string>
    <key>PYTHON</key>
    <string>$PY</string>
  </dict>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>$HOUR</integer>
    <key>Minute</key><integer>$MINUTE</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>$REPO/logs/recorder/launchd.out</string>
  <key>StandardErrorPath</key>
  <string>$REPO/logs/recorder/launchd.err</string>

  <!-- The job spends most of its life in 'sleep' waiting for first pitch;
       launchd must not treat that as a hung task worth throttling. -->
  <key>ExitTimeOut</key>
  <integer>0</integer>
</dict>
</plist>
PLIST_EOF

# bootout first so re-running picks up an edited plist.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

printf 'installed %s\n' "$PLIST"
printf '  fires daily at %02d:%02d local, then waits for first pitch\n' "$HOUR" "$MINUTE"
printf '  python:   %s\n' "$PY"
printf '  logs:     %s/logs/recorder/{daily.log,launchd.err}\n' "$REPO"
printf '  run now:  launchctl kickstart gui/%s/%s\n' "$(id -u)" "$LABEL"
printf '  remove:   scripts/install_recorder_job.sh --uninstall\n'
