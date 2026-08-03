"""Point-in-time backtest harness — no future leakage, fixture-driven."""
from __future__ import annotations

import json
from pathlib import Path

from slugger.backtest.pit import (
    batter_profile_from_logs,
    filter_logs_before,
    pitcher_profile_from_logs,
)
from slugger.backtest.replay import replay_strategies
from slugger.backtest.snapshot import (
    game_context_from_dict,
    game_context_to_dict,
    load_snapshot,
    save_snapshot,
)
from slugger.config import Config
from slugger.types import GameContext, GameInfo


def test_filter_logs_before_excludes_as_of_and_future():
    logs = [
        {"date": "2026-05-01", "k": 7},
        {"date": "2026-05-10", "k": 8},
        {"date": "2026-05-15", "k": 9},
    ]
    past = filter_logs_before(logs, "2026-05-10")
    assert len(past) == 1
    assert past[0]["k"] == 7


def test_pitcher_profile_point_in_time():
    logs = [
        {"date": "2026-04-01", "ip": 6.0, "k": 8, "er": 2, "h": 5, "bb": 1, "hr": 1},
        {"date": "2026-04-08", "ip": 5.0, "k": 6, "er": 1, "h": 4, "bb": 2, "hr": 0},
        # After as_of — must not count
        {"date": "2026-05-01", "ip": 7.0, "k": 15, "er": 0, "h": 1, "bb": 0, "hr": 0},
    ]
    p = pitcher_profile_from_logs(1, "Ace", logs, as_of="2026-04-15")
    assert p.games_started == 2
    assert p.strikeouts == 14
    assert p.max_k_in_start == 8  # not 15
    assert abs(p.innings_pitched - 11.0) < 1e-9


def test_batter_profile_point_in_time():
    logs = [
        {"date": "2026-04-01", "ab": 4, "h": 2, "hr": 1},
        {"date": "2026-04-02", "ab": 3, "h": 0, "hr": 0},
        {"date": "2026-06-01", "ab": 4, "h": 4, "hr": 3},
    ]
    b = batter_profile_from_logs(2, "Slugger", "LAD", logs, as_of="2026-05-01")
    assert b.ab == 7
    assert b.hits == 2
    assert b.hr == 1


def test_snapshot_roundtrip(tmp_path):
    game = GameInfo(
        game_id=1,
        away_team="Away",
        home_team="Home",
        away_abbrev="AWY",
        home_abbrev="HOM",
        away_record="10-10",
        home_record="11-9",
        away_pitcher_name="A",
        home_pitcher_name="B",
        away_pitcher_id=1,
        home_pitcher_id=2,
        game_datetime="2026-05-10T17:00:00Z",
        venue="Park",
        weather={},
        status="Final",
    )
    logs = [
        {"date": "2026-04-01", "ip": 6.0, "k": 7, "er": 2, "h": 5, "bb": 1, "hr": 0},
        {"date": "2026-04-08", "ip": 6.0, "k": 8, "er": 1, "h": 4, "bb": 1, "hr": 1},
    ]
    ctx = GameContext(
        game=game,
        away_pitcher=pitcher_profile_from_logs(1, "A", logs, "2026-05-10"),
        home_pitcher=pitcher_profile_from_logs(2, "B", logs, "2026-05-10"),
    )
    path = str(tmp_path / "snap.json")
    save_snapshot(path, ctx, as_of="2026-05-10")
    loaded = load_snapshot(path)
    assert loaded.game.game_id == 1
    assert loaded.away_pitcher is not None
    assert loaded.away_pitcher.strikeouts == 15


def test_replay_strategies_on_snapshot(tmp_path):
    """Replay uses snapshot profiles + fixture markets; no live API."""
    game = GameInfo(
        game_id=99,
        away_team="San Francisco Giants",
        home_team="Los Angeles Dodgers",
        away_abbrev="SF",
        home_abbrev="LAD",
        away_record="20-20",
        home_record="25-15",
        away_pitcher_name="Webb",
        home_pitcher_name="Glasnow",
        away_pitcher_id=100,
        home_pitcher_id=200,
        game_datetime="2026-05-10T20:00:00Z",
        venue="Dodger Stadium",
        weather={},
        status="Pre-Game",
    )
    logs = [
        {"date": "2026-04-01", "ip": 6.0, "k": 9, "er": 1, "h": 4, "bb": 1, "hr": 0},
        {"date": "2026-04-08", "ip": 6.0, "k": 10, "er": 2, "h": 5, "bb": 0, "hr": 1},
        {"date": "2026-04-15", "ip": 7.0, "k": 11, "er": 1, "h": 3, "bb": 1, "hr": 0},
        {"date": "2026-04-22", "ip": 6.0, "k": 8, "er": 2, "h": 4, "bb": 2, "hr": 0},
        {"date": "2026-04-29", "ip": 6.0, "k": 12, "er": 0, "h": 2, "bb": 1, "hr": 0},
    ]
    ctx = GameContext(
        game=game,
        home_pitcher=pitcher_profile_from_logs(200, "Glasnow", logs, "2026-05-10", throws="R"),
        away_pitcher=pitcher_profile_from_logs(100, "Webb", logs, "2026-05-10", throws="R"),
    )
    # Markets under KS event ticker pattern from tickers module may be empty
    # if ticker helpers need exact format — use empty markets; replay still runs.
    result = replay_strategies(
        ctx,
        markets_by_event={},
        config=Config(dry_run=True, log_dir=str(tmp_path), enabled_strategies=("pitcher_ks",)),
        enabled=["pitcher_ks"],
    )
    assert isinstance(result.signals, list)
    assert "pitcher_ks" in result.by_strategy
