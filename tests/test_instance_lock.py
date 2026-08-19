"""Tests for the single-instance guard (slugger/instance_lock.py, bead 19v).

The property that matters is the one a unit test usually cannot reach: a
second *process* must be refused. Several tests below fork a real child so
the flock is genuinely contended, because an in-process test would pass even
if the lock were a no-op (flock is per-open-file-description, and a same-fd
re-lock succeeds).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from slugger.instance_lock import (
    AlreadyRunning,
    InstanceInfo,
    InstanceLock,
    format_status,
    lock_path,
    read_status,
)

REPO = Path(__file__).resolve().parent.parent


@dataclass
class FakeConfig:
    log_dir: str
    dry_run: bool = True
    use_demo: bool = True
    enabled_strategies: tuple = ("pitcher_ks", "player_hits")


def cfg(tmp_path) -> FakeConfig:
    return FakeConfig(log_dir=str(tmp_path))


# ─── Acquire / release ───────────────────────────────────────────────────────

def test_acquire_writes_the_config_it_started_with(tmp_path):
    """The recorded config is the point: a running bot holds .env in memory,
    so the file on disk does not tell you what it is actually doing."""
    with InstanceLock.acquire(cfg(tmp_path)) as lock:
        data = json.loads(lock_path(cfg(tmp_path)).read_text())
        assert data["pid"] == os.getpid()
        assert data["dry_run"] is True
        assert data["use_demo"] is True
        assert data["enabled_strategies"] == ["pitcher_ks", "player_hits"]
        assert data["started_at"]
        assert "command" in data
        assert "git_commit" in data


def test_clean_release_removes_the_file(tmp_path):
    c = cfg(tmp_path)
    with InstanceLock.acquire(c):
        assert lock_path(c).exists()
    assert not lock_path(c).exists()
    assert read_status(c)["state"] == "clean"


def test_release_is_idempotent(tmp_path):
    lock = InstanceLock.acquire(cfg(tmp_path))
    lock.release()
    lock.release()   # must not raise


def test_lock_released_even_when_the_run_raises(tmp_path):
    c = cfg(tmp_path)
    with pytest.raises(ValueError):
        with InstanceLock.acquire(c):
            raise ValueError("strategy blew up")
    assert not lock_path(c).exists()
    InstanceLock.acquire(c).release()   # reacquirable


def test_a_stale_record_does_not_block_a_new_instance(tmp_path):
    """A crashed instance leaves the file behind; the kernel drops its flock.

    This is the whole reason for using flock over a PID file — no liveness
    check to get wrong, and no recycled-PID false positive.
    """
    c = cfg(tmp_path)
    lock_path(c).parent.mkdir(parents=True, exist_ok=True)
    lock_path(c).write_text(json.dumps({"pid": 999999, "started_at": "then"}))
    with InstanceLock.acquire(c):
        assert json.loads(lock_path(c).read_text())["pid"] == os.getpid()


def test_stale_record_is_fully_overwritten(tmp_path):
    """A shorter record must not leave a tail of the old JSON behind."""
    c = cfg(tmp_path)
    lock_path(c).parent.mkdir(parents=True, exist_ok=True)
    lock_path(c).write_text("x" * 20000)
    with InstanceLock.acquire(c):
        json.loads(lock_path(c).read_text())   # parses => no trailing garbage


# ─── Real contention, across processes ───────────────────────────────────────

_HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {repo!r})
    from dataclasses import dataclass
    from slugger.instance_lock import InstanceLock

    @dataclass
    class C:
        log_dir: str
        dry_run: bool = False
        use_demo: bool = False
        enabled_strategies: tuple = ("pitcher_ks",)

    lock = InstanceLock.acquire(C(log_dir={log_dir!r}))
    print("HELD", flush=True)
    time.sleep(30)
""")


@pytest.fixture
def holder(tmp_path):
    """A separate process holding the lock for the duration of a test."""
    proc = subprocess.Popen(
        [sys.executable, "-c",
         _HOLDER.format(repo=str(REPO), log_dir=str(tmp_path))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "HELD", proc.stderr.read()
    yield proc
    proc.kill()
    proc.wait(timeout=10)


def test_second_process_is_refused(tmp_path, holder):
    with pytest.raises(AlreadyRunning) as exc:
        InstanceLock.acquire(cfg(tmp_path))
    # The refusal names the holder rather than just failing.
    assert exc.value.holder["pid"] == holder.pid
    assert exc.value.holder["dry_run"] is False
    assert str(holder.pid) in str(exc.value)


def test_status_reports_running_while_held(tmp_path, holder):
    st = read_status(cfg(tmp_path))
    assert st["state"] == "running"
    assert st["info"]["pid"] == holder.pid
    line = format_status(cfg(tmp_path))
    assert "RUNNING" in line
    # A live-money instance must be unmissable in the status line.
    assert "LIVE MONEY" in line


def test_status_does_not_steal_the_lock(tmp_path, holder):
    """read_status probes with flock; it must hand the lock straight back."""
    for _ in range(3):
        assert read_status(cfg(tmp_path))["state"] == "running"
    with pytest.raises(AlreadyRunning):
        InstanceLock.acquire(cfg(tmp_path))


def test_killed_holder_frees_the_lock(tmp_path, holder):
    holder.kill()
    holder.wait(timeout=10)
    # File still there, but nobody holds it: that is "crashed", not "running".
    st = read_status(cfg(tmp_path))
    assert st["state"] == "crashed"
    assert "not running" in format_status(cfg(tmp_path))
    InstanceLock.acquire(cfg(tmp_path)).release()


# ─── Recorded metadata ───────────────────────────────────────────────────────

def test_summary_flags_live_money_and_dirty_trees(tmp_path):
    info = InstanceInfo(
        pid=1, started_at="2026-08-19T00:00:00+00:00", host="h", command=["x"],
        git_commit="abc1234", git_dirty=True, dry_run=False, use_demo=False,
        enabled_strategies=["pitcher_ks"],
    )
    s = info.summary()
    assert "LIVE MONEY" in s
    # A dirty tree means the commit does not identify the code that traded.
    assert "abc1234+dirty" in s
    assert "prod" in s


def test_summary_is_calm_about_dry_runs(tmp_path):
    info = InstanceInfo(
        pid=1, started_at="t", host="h", command=["x"],
        git_commit="abc1234", dry_run=True, use_demo=True,
    )
    s = info.summary()
    assert "DRY-RUN" in s and "LIVE MONEY" not in s
    assert "demo" in s


def test_status_survives_a_corrupt_lock_file(tmp_path):
    c = cfg(tmp_path)
    lock_path(c).parent.mkdir(parents=True, exist_ok=True)
    lock_path(c).write_text("{not json")
    assert read_status(c)["state"] == "crashed"
    assert isinstance(format_status(c), str)
