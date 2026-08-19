"""Single-instance guard for the trading loop (mlb-kalshi-bot-19v).

Two concurrent `main.py run` processes would double-trade every signal.
Neither would see the other: `logs/placed_<date>.json` is read once at
startup and appended per placement, and the per-game and daily risk budgets
live in process memory, so both caps would silently double.

Implemented with `flock`, not a PID file. The kernel releases an flock when
the holding process dies for any reason — SIGKILL, panic, power cut — so a
stale lock cannot exist, and none of the "is that PID still alive, or has it
been recycled onto someone else's process?" logic has to be written or
trusted. The file's *contents* do persist after a crash, which is the useful
half: they say what the last instance was running.

The recorded config summary is arguably worth more than the lock. A running
bot holds its .env and its code in memory, so editing DRY_RUN or toggling a
gate mid-session has no effect on it — while `logs/calibration.json` and
`logs/ks_model.json` *are* re-read from disk, so a live bot can pick up a
half-finished model change. Recording what an instance actually started with
turns that from something you infer into something you read. The git commit
in particular is what lets a fill be tied to the code that produced it.

Usage:

    with InstanceLock.acquire(config) as lock:   # raises AlreadyRunning
        run_bot(config)
"""
from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

LOCK_FILENAME = "bot.lock"


class AlreadyRunning(RuntimeError):
    """Raised when another instance holds the lock.

    Carries whatever that instance recorded about itself, so the error can
    name the process instead of just refusing.
    """

    def __init__(self, holder: Optional[Dict[str, Any]], path: Path):
        self.holder = holder or {}
        self.path = path
        pid = self.holder.get("pid", "unknown")
        started = self.holder.get("started_at", "unknown time")
        super().__init__(
            f"another slugger instance is already running (pid {pid}, "
            f"started {started}); lock held on {path}"
        )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


@dataclass
class InstanceInfo:
    """What a running instance committed to at startup."""
    pid: int
    started_at: str
    host: str
    command: List[str]
    git_commit: str = ""
    git_dirty: bool = False
    dry_run: bool = True
    use_demo: bool = True
    enabled_strategies: List[str] = field(default_factory=list)
    log_dir: str = "logs"

    @classmethod
    def describe(cls, config) -> "InstanceInfo":
        status = _git("status", "--porcelain")
        return cls(
            pid=os.getpid(),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            host=socket.gethostname(),
            command=list(sys.argv),
            git_commit=_git("rev-parse", "--short", "HEAD"),
            # A dirty tree means the commit does not identify the code that
            # placed the trade. Worth recording loudly rather than pretending.
            git_dirty=bool(status),
            dry_run=bool(getattr(config, "dry_run", True)),
            use_demo=bool(getattr(config, "use_demo", True)),
            enabled_strategies=list(getattr(config, "enabled_strategies", ()) or ()),
            log_dir=str(getattr(config, "log_dir", "logs")),
        )

    def summary(self) -> str:
        mode = "DRY-RUN" if self.dry_run else "*** LIVE MONEY ***"
        env = "demo" if self.use_demo else "prod"
        commit = self.git_commit or "unknown"
        if self.git_dirty:
            commit += "+dirty"
        strategies = ",".join(self.enabled_strategies) or "none"
        return (f"pid {self.pid} on {self.host} · started {self.started_at} · "
                f"{mode} · {env} · {commit} · [{strategies}]")


def lock_path(config) -> Path:
    return Path(getattr(config, "log_dir", "logs")) / LOCK_FILENAME


class InstanceLock:
    """Exclusive, self-releasing lock on the trading loop."""

    def __init__(self, path: Path, info: InstanceInfo, fd: int):
        self.path = path
        self.info = info
        self._fd: Optional[int] = fd

    # ── Acquire / release ────────────────────────────────────────────────

    @classmethod
    def acquire(cls, config) -> "InstanceLock":
        path = lock_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        info = InstanceInfo.describe(config)

        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                os.close(fd)
                raise
            holder = _read_json(fd)
            os.close(fd)
            raise AlreadyRunning(holder, path) from None

        # Only now is it ours to overwrite. Truncate first so a shorter
        # record cannot leave a tail of the previous instance's JSON behind.
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(asdict(info), indent=2) + "\n").encode())
        os.fsync(fd)
        log.info("Instance lock acquired: %s", info.summary())
        return cls(path, info, fd)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        # Remove on a clean exit so the file's presence means "an instance
        # ran and did not shut down cleanly", which is worth noticing.
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.debug("Could not remove %s: %s", self.path, exc)

    def __enter__(self) -> "InstanceLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ─── Inspection ──────────────────────────────────────────────────────────────

def _read_json(fd: int) -> Optional[Dict[str, Any]]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 65536).decode("utf-8", "replace")
        return json.loads(raw) if raw.strip() else None
    except Exception:
        return None


def read_status(config) -> Dict[str, Any]:
    """Report whether an instance is running, and what it started with.

    Liveness is tested by trying to take the lock: if flock succeeds, nobody
    holds it. That is the same primitive the running process uses, so the
    answer cannot drift from reality the way a PID check can.

    Returns {"state": "running"|"crashed"|"clean", "info": {...}|None}.
      running  an instance holds the lock right now
      crashed  a record exists but no one holds it — the last instance died
               without releasing (and its risk budgets died with it)
      clean    no lock file; nothing has run, or everything exited cleanly
    """
    path = lock_path(config)
    if not path.exists():
        return {"state": "clean", "info": None, "path": str(path)}

    fd = os.open(path, os.O_RDWR)
    try:
        info = _read_json(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"state": "running", "info": info, "path": str(path)}
        # We got it, so nobody was holding it. Give it straight back.
        fcntl.flock(fd, fcntl.LOCK_UN)
        return {"state": "crashed", "info": info, "path": str(path)}
    finally:
        os.close(fd)


def format_status(config) -> str:
    """One-line human summary for `main.py status`."""
    st = read_status(config)
    info = st.get("info") or {}
    if st["state"] == "clean":
        return "bot instance: not running"
    try:
        summary = InstanceInfo(**info).summary()
    except Exception:
        summary = json.dumps(info)
    if st["state"] == "running":
        return f"bot instance: RUNNING — {summary}"
    return (f"bot instance: not running; last instance exited without "
            f"releasing the lock — {summary}")
