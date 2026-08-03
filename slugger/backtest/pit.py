"""Point-in-time profile reconstruction from game logs.

Season stats on the live MLB API are cumulative *today*; backtests must
recompute from starts/games with date < as_of only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from slugger.types import BatterProfile, PitcherProfile


def filter_logs_before(logs: List[dict], as_of: str) -> List[dict]:
    """Keep log rows with date strictly before as_of (YYYY-MM-DD)."""
    out = []
    for row in logs:
        d = row.get("date") or row.get("game_date") or ""
        if isinstance(d, str) and len(d) >= 10 and d[:10] < as_of:
            out.append(row)
    return out


def pitcher_profile_from_logs(
    player_id: int,
    name: str,
    logs: List[dict],
    as_of: str,
    *,
    throws: str = "R",
) -> PitcherProfile:
    """Build PitcherProfile from starts strictly before as_of.

    Expected log keys (flexible): date, innings_pitched|ip, strikeouts|k,
    earned_runs|er, hits|h, walks|bb, home_runs|hr.
    """
    past = filter_logs_before(logs, as_of)
    ip = 0.0
    k = 0
    er = 0.0
    h = 0
    bb = 0
    hr = 0
    gs = 0
    k_list: List[int] = []
    for row in past:
        gs += 1
        ip_i = float(row.get("innings_pitched", row.get("ip", 0)) or 0)
        k_i = int(row.get("strikeouts", row.get("k", 0)) or 0)
        er_i = float(row.get("earned_runs", row.get("er", 0)) or 0)
        h_i = int(row.get("hits", row.get("h", 0)) or 0)
        bb_i = int(row.get("walks", row.get("bb", 0)) or 0)
        hr_i = int(row.get("home_runs", row.get("hr", 0)) or 0)
        ip += ip_i
        k += k_i
        er += er_i
        h += h_i
        bb += bb_i
        hr += hr_i
        k_list.append(k_i)

    recent = past[-5:] if past else []
    recent_k = [int(r.get("strikeouts", r.get("k", 0)) or 0) for r in recent]
    recent_ip = [float(r.get("innings_pitched", r.get("ip", 0)) or 0) for r in recent]
    recent_er = [float(r.get("earned_runs", r.get("er", 0)) or 0) for r in recent]

    def _era(er_sum: float, ip_sum: float) -> float:
        return (er_sum / ip_sum * 9.0) if ip_sum > 0 else 0.0

    return PitcherProfile(
        player_id=player_id,
        name=name,
        era=_era(er, ip),
        whip=((h + bb) / ip) if ip > 0 else 0.0,
        k_per_9=(k / ip * 9.0) if ip > 0 else 0.0,
        bb_per_9=(bb / ip * 9.0) if ip > 0 else 0.0,
        hr_per_9=(hr / ip * 9.0) if ip > 0 else 0.0,
        innings_pitched=ip,
        strikeouts=k,
        games_started=gs,
        recent_era=_era(sum(recent_er), sum(recent_ip)),
        recent_k_per_start=(sum(recent_k) / len(recent_k)) if recent_k else 0.0,
        recent_ip_per_start=(sum(recent_ip) / len(recent_ip)) if recent_ip else 0.0,
        max_k_in_start=max(k_list) if k_list else 0,
        k_per_start_list=k_list,
        throws=throws,
    )


def batter_profile_from_logs(
    player_id: int,
    name: str,
    team: str,
    logs: List[dict],
    as_of: str,
    *,
    batting_order: int = 0,
) -> BatterProfile:
    """Build BatterProfile from games strictly before as_of.

    Expected keys: date, ab, hits|h, hr, avg optional.
    """
    past = filter_logs_before(logs, as_of)
    ab = 0
    hits = 0
    hr = 0
    for row in past:
        ab += int(row.get("ab", 0) or 0)
        hits += int(row.get("hits", row.get("h", 0)) or 0)
        hr += int(row.get("hr", row.get("home_runs", 0)) or 0)

    recent = past[-7:] if past else []
    recent_hr = sum(int(r.get("hr", r.get("home_runs", 0)) or 0) for r in recent)
    recent_ab = sum(int(r.get("ab", 0) or 0) for r in recent)
    recent_h = sum(int(r.get("hits", r.get("h", 0)) or 0) for r in recent)

    avg = (hits / ab) if ab > 0 else 0.0
    return BatterProfile(
        player_id=player_id,
        name=name,
        team=team,
        ab=ab,
        hits=hits,
        hr=hr,
        avg=avg,
        recent_hr=recent_hr,
        recent_avg=(recent_h / recent_ab) if recent_ab > 0 else 0.0,
        batting_order=batting_order,
    )
