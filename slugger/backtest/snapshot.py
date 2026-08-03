"""Serialize / deserialize GameContext snapshots for offline replay."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from slugger.types import (
    BatterProfile,
    GameContext,
    GameInfo,
    PitcherProfile,
    TeamProfile,
)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def game_context_to_dict(ctx: GameContext) -> dict:
    return {
        "game": _to_plain(ctx.game),
        "away_pitcher": _to_plain(ctx.away_pitcher) if ctx.away_pitcher else None,
        "home_pitcher": _to_plain(ctx.home_pitcher) if ctx.home_pitcher else None,
        "away_batters": [_to_plain(b) for b in ctx.away_batters],
        "home_batters": [_to_plain(b) for b in ctx.home_batters],
        "away_team": _to_plain(ctx.away_team) if ctx.away_team else None,
        "home_team": _to_plain(ctx.home_team) if ctx.home_team else None,
    }


def _pitcher(d: Optional[dict]) -> Optional[PitcherProfile]:
    if not d:
        return None
    return PitcherProfile(**{k: v for k, v in d.items() if k in PitcherProfile.__dataclass_fields__})


def _batter(d: dict) -> BatterProfile:
    return BatterProfile(**{k: v for k, v in d.items() if k in BatterProfile.__dataclass_fields__})


def _team(d: Optional[dict]) -> Optional[TeamProfile]:
    if not d:
        return None
    return TeamProfile(**{k: v for k, v in d.items() if k in TeamProfile.__dataclass_fields__})


def game_context_from_dict(data: dict) -> GameContext:
    g = data["game"]
    game = GameInfo(**{k: v for k, v in g.items() if k in GameInfo.__dataclass_fields__})
    return GameContext(
        game=game,
        away_pitcher=_pitcher(data.get("away_pitcher")),
        home_pitcher=_pitcher(data.get("home_pitcher")),
        away_batters=[_batter(b) for b in data.get("away_batters") or []],
        home_batters=[_batter(b) for b in data.get("home_batters") or []],
        away_team=_team(data.get("away_team")),
        home_team=_team(data.get("home_team")),
    )


def save_snapshot(path: str, ctx: GameContext, *, as_of: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": as_of,
        "context": game_context_to_dict(ctx),
    }
    p.write_text(json.dumps(payload, indent=2))


def load_snapshot(path: str) -> GameContext:
    data = json.loads(Path(path).read_text())
    if "context" in data:
        return game_context_from_dict(data["context"])
    return game_context_from_dict(data)
