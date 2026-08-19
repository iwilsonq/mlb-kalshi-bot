#!/usr/bin/env python3
"""Quick health/status view of today's Phase 0 recording.

Usage:
    python3 scripts/recorder_status.py [YYYY-MM-DD]
    watch -n 30 python3 scripts/recorder_status.py   # live view
"""
import collections
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

day = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().strftime("%Y-%m-%d")
rec_dir = Path("logs/recorder") / day

if not rec_dir.exists():
    sys.exit(f"No recording directory: {rec_dir}")

# ── Process alive? ────────────────────────────────────────────────────────
alive = subprocess.run(
    ["pgrep", "-f", "main.py record"], capture_output=True, text=True
).stdout.strip()
print(f"=== Recorder status {day} ===")
print(f"process:   {'RUNNING (pid ' + alive.split()[0] + ')' if alive else 'NOT RUNNING'}")

# ── Kalshi stream ─────────────────────────────────────────────────────────
kalshi_path = rec_dir / "kalshi.jsonl"
types = collections.Counter()
last_ts = 0.0
manifest = None
if kalshi_path.exists():
    for line in kalshi_path.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        types[r.get("type", "?")] += 1
        last_ts = max(last_ts, r.get("recv_ts", 0))
        if r.get("type") == "manifest":
            manifest = r
    age = time.time() - last_ts if last_ts else float("inf")
    size_mb = kalshi_path.stat().st_size / 1e6
    print(f"kalshi:    {size_mb:.0f} MB | last record {age:.0f}s ago"
          + ("  ⚠ STALE" if age > 120 else ""))
    print(f"           deltas={types['orderbook_delta']:,} tickers={types['ticker']:,}"
          f" trades={types['trade']:,} disconnects={types['recorder_disconnect']}")

# ── GUMBO stream / per-game state ────────────────────────────────────────
gumbo_path = rec_dir / "gumbo.jsonl"
game_state = {}   # pk -> latest state record
plays = collections.Counter()
finals = set()
if gumbo_path.exists():
    for line in gumbo_path.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        pk = r.get("game_pk")
        if r["type"] == "gumbo_state":
            game_state[pk] = r
        elif r["type"] == "gumbo_play":
            plays[pk] += 1
        elif r["type"] == "gumbo_final":
            finals.add(pk)

names = {}
if manifest:
    for g in manifest.get("msg", manifest).get("games", []):
        names[g["game_pk"]] = f"{g['away']}@{g['home']}"

n_live = sum(1 for s in game_state.values() if s.get("status") == "Live")
print(f"gumbo:     {len(game_state)} games tracked | {n_live} live | "
      f"{len(finals)} final | {sum(plays.values())} plays captured")
for pk, s in sorted(game_state.items(), key=lambda kv: kv[1].get("status", "")):
    st = s.get("state", {})
    status = "FINAL" if pk in finals else s.get("status", "?")
    line = f"  {names.get(pk, pk):<12} {status:<8}"
    if status == "Live":
        line += (f" {st.get('half','')} {st.get('inning','')}"
                 f" | {st.get('away_runs')}-{st.get('home_runs')}"
                 f" | {st.get('outs')} out | {plays[pk]} plays")
    print(line)
