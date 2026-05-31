"""MLB data sourcing via Stats API + Baseball Savant (Statcast).

Data hierarchy:
  1. MLB Stats API (statsapi.mlb.com) — schedules, rosters, box scores, player stats
  2. Baseball Savant / pybaseball — Statcast metrics (exit velo, xBA, pitch mix, etc.)

All functions return plain dicts/lists for easy testing and serialization.
"""
from __future__ import annotations
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import statsapi
from cachetools import TTLCache

# ─── Shared types (canonical definitions in slugger.types) ────────────────────
# Re-exported here so existing callers that import from slugger.mlb_data
# continue to work without changes.
from slugger.types import (  # noqa: F401
    BatterProfile,
    GameContext,
    GameInfo,
    Lineup,
    MLBDataProvider,
    PitcherProfile,
    TeamProfile,
)

log = logging.getLogger(__name__)

# Suppress pybaseball FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

MLB_API = "https://statsapi.mlb.com/api/v1"

# ─── Shared thread pool for parallel I/O ─────────────────────────────────────
# Max workers kept moderate to avoid overwhelming the MLB API / Statcast.
_POOL = ThreadPoolExecutor(max_workers=8)

# ─── TTL caches — survive across poll cycles, auto-expire ────────────────────
# Pitcher & batter profiles change slowly; 5-minute TTL avoids redundant calls
# while staying reasonably fresh.  Team profiles change even less often.
_pitcher_cache: TTLCache = TTLCache(maxsize=256, ttl=300)   # 5 min
_batter_cache:  TTLCache = TTLCache(maxsize=512, ttl=300)   # 5 min
_team_cache_ttl: TTLCache = TTLCache(maxsize=64,  ttl=600)  # 10 min


# ─── Team Lookup ─────────────────────────────────────────────────────────────

# MLB team ID mapping (from statsapi)
_TEAM_CACHE: Dict[str, int] = {}

def _build_team_cache():
    """Build abbrev → team_id mapping."""
    global _TEAM_CACHE
    if _TEAM_CACHE:
        return
    try:
        teams = statsapi.get("teams", {"sportId": 1})
        for t in teams.get("teams", []):
            _TEAM_CACHE[t["abbreviation"]] = t["id"]
            _TEAM_CACHE[t["teamName"].lower()] = t["id"]
            _TEAM_CACHE[t["name"].lower()] = t["id"]
    except Exception as e:
        log.warning("Failed to build team cache: %s", e)

def get_team_id(name_or_abbrev: str) -> Optional[int]:
    """Resolve team name/abbreviation to MLB team ID."""
    _build_team_cache()
    return _TEAM_CACHE.get(name_or_abbrev) or _TEAM_CACHE.get(name_or_abbrev.lower())


# ─── Schedule & Game Info ────────────────────────────────────────────────────

def get_todays_games(target_date: Optional[str] = None) -> List[GameInfo]:
    """Get all MLB games for a date (default: today).
    
    Returns GameInfo objects with probable pitchers, venue, weather.
    """
    dt = target_date or date.today().strftime("%Y-%m-%d")
    games = []
    try:
        sched = statsapi.schedule(date=dt)
    except Exception as e:
        log.error("Failed to fetch schedule: %s", e)
        return []
    
    # ── Fetch all live feeds in parallel ──────────────────────────────────
    def _fetch_feed(game_pk: int) -> dict:
        """Fetch a single game's live feed; return parsed JSON or {}."""
        try:
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                timeout=10,
            )
            return resp.json()
        except Exception:
            return {}

    game_pks = [g.get("game_id") for g in sched if g.get("game_id")]
    feed_map: Dict[int, dict] = {}
    if game_pks:
        futures = {_POOL.submit(_fetch_feed, pk): pk for pk in game_pks}
        for fut in as_completed(futures):
            pk = futures[fut]
            try:
                feed_map[pk] = fut.result()
            except Exception:
                feed_map[pk] = {}
    log.debug("Fetched %d live feeds in parallel", len(feed_map))

    for g in sched:
        game_pk = g.get("game_id")
        weather = {}
        venue_name = ""
        away_pitcher_id = 0
        home_pitcher_id = 0
        away_abbrev = ""
        home_abbrev = ""

        feed = feed_map.get(game_pk, {})
        if feed:
            gd = feed.get("gameData", {})
            weather = gd.get("weather", {})
            venue_name = gd.get("venue", {}).get("name", "")
            probs = gd.get("probablePitchers", {})
            away_pitcher_id = probs.get("away", {}).get("id", 0)
            home_pitcher_id = probs.get("home", {}).get("id", 0)
            teams_gd = gd.get("teams", {})
            away_abbrev = teams_gd.get("away", {}).get("abbreviation", "")
            home_abbrev = teams_gd.get("home", {}).get("abbreviation", "")

        # Fall back to name slice only if the feed didn't supply abbreviations
        if not away_abbrev:
            away_abbrev = g.get("away_name", "")[:3].upper()
        if not home_abbrev:
            home_abbrev = g.get("home_name", "")[:3].upper()

        games.append(GameInfo(
            game_id=game_pk or 0,
            away_team=g.get("away_name", ""),
            home_team=g.get("home_name", ""),
            away_abbrev=away_abbrev,
            home_abbrev=home_abbrev,
            away_record="",
            home_record="",
            away_pitcher_name=g.get("away_probable_pitcher", "TBD"),
            home_pitcher_name=g.get("home_probable_pitcher", "TBD"),
            away_pitcher_id=away_pitcher_id,
            home_pitcher_id=home_pitcher_id,
            game_datetime=g.get("game_datetime", ""),
            venue=venue_name,
            weather=weather,
            status=g.get("status", ""),
        ))
    return games


@dataclass
class PitcherGameStatus:
    """A pitcher's current status within an in-progress game."""
    pitcher_id: int
    name: str = ""
    is_active: bool = False      # currently on the mound
    has_pitched: bool = False    # appeared in the game at all
    current_ks: int = 0          # strikeouts recorded so far
    ip_today: float = 0.0        # innings pitched today
    current_inning: int = 0      # current game inning (for remaining-IP estimate)


def get_pitcher_game_status(game_id: int, pitcher_id: int) -> PitcherGameStatus:
    """Return a pitcher's live in-game status.

    Uses the MLB live feed to determine whether the pitcher is still on the
    mound, how many strikeouts they have, and how many innings they've thrown.
    This prevents the bot from buying K markets for a pitcher who has already
    been pulled.
    """
    status = PitcherGameStatus(pitcher_id=pitcher_id)
    try:
        feed = get_live_game_feed(game_id)
        if not feed:
            return status

        ld   = feed.get("liveData", {})
        ls   = ld.get("linescore", {})
        bs   = ld.get("boxscore", {}).get("teams", {})

        status.current_inning = ls.get("currentInning", 0) or 0

        # Is this pitcher currently on the mound?
        current_pitcher_id = (
            ls.get("defense", {}).get("pitcher", {}).get("id")
        )
        status.is_active = current_pitcher_id == pitcher_id

        # Find pitcher stats in the boxscore (search both sides)
        for side in ("away", "home"):
            side_data = bs.get(side, {})
            pitchers_used = side_data.get("pitchers", [])
            if pitcher_id not in pitchers_used:
                continue

            status.has_pitched = True
            pdata  = side_data.get("players", {}).get(f"ID{pitcher_id}", {})
            status.name = pdata.get("person", {}).get("fullName", "")
            pstats = pdata.get("stats", {}).get("pitching", {})
            status.current_ks = int(pstats.get("strikeOuts", 0))
            status.ip_today   = _safe_float(pstats.get("inningsPitched", "0"))
            break

    except Exception as exc:
        log.debug("Could not fetch game status for pitcher %d: %s", pitcher_id, exc)

    return status


def get_live_game_feed(game_id: int) -> Dict[str, Any]:
    """Get live game feed with play-by-play data."""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error("Failed to get live feed for game %d: %s", game_id, e)
        return {}


# ─── Pitcher Data ────────────────────────────────────────────────────────────

def get_pitcher_profile(player_id: int, season: int = None) -> PitcherProfile:
    """Build a comprehensive pitcher profile from Stats API + Statcast.

    Results are cached for 5 minutes to avoid redundant API calls across
    poll cycles.  Internal sub-fetches (season stats, handedness, game log,
    Statcast) run in parallel where possible.
    """
    if season is None:
        season = date.today().year

    cache_key = (player_id, season)
    if cache_key in _pitcher_cache:
        log.debug("Pitcher cache hit: %d", player_id)
        return _pitcher_cache[cache_key]

    profile = PitcherProfile(player_id=player_id, name="")

    # ── Define independent sub-fetches ─────────────────────────────────────

    def _fetch_season_stats() -> dict:
        """Season stats from Stats API."""
        try:
            return statsapi.player_stat_data(player_id, group="pitching", type="season")
        except Exception as e:
            log.warning("Stats API pitcher data failed for %d: %s", player_id, e)
            return {}

    def _fetch_handedness() -> str:
        """Pitcher throwing hand."""
        try:
            people = statsapi.get("people", {"personIds": player_id})
            return people.get("people", [{}])[0].get("pitchHand", {}).get("code", "")
        except Exception as e:
            log.debug("Could not fetch pitcher handedness for %d: %s", player_id, e)
            return ""

    def _fetch_game_log() -> list:
        """Recent game log splits."""
        try:
            url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}"
            resp = requests.get(url, timeout=10)
            return resp.json().get("stats", [{}])[0].get("splits", [])
        except Exception as e:
            log.debug("Game log fetch failed for pitcher %d: %s", player_id, e)
            return []

    def _fetch_statcast() -> dict:
        """Statcast metrics (returns raw data dict for later enrichment)."""
        try:
            from pybaseball import statcast_pitcher
        except ImportError:
            return {}
        end = date.today()
        start = end - timedelta(days=30)
        log.debug("Statcast pitcher fetch: %d  %s -> %s", player_id, start, end)
        try:
            data = statcast_pitcher(str(start), str(end), player_id=player_id)
            if data is not None and len(data) > 0:
                return {"data": data}
        except Exception as exc:
            log.debug("Statcast pitcher fetch failed for %d: %s", player_id, exc)
        return {}

    # ── Fire all sub-fetches in parallel (dedicated pool to avoid deadlock
    #    with the shared _POOL used by hydrate_game) ────────────────────────
    with ThreadPoolExecutor(max_workers=4) as local_pool:
        fut_season   = local_pool.submit(_fetch_season_stats)
        fut_hand     = local_pool.submit(_fetch_handedness)
        fut_gamelog  = local_pool.submit(_fetch_game_log)
        fut_statcast = local_pool.submit(_fetch_statcast)

        # ── Collect results ────────────────────────────────────────────────
        season_data = fut_season.result()
        if season_data:
            name = season_data.get("first_name", "") + " " + season_data.get("last_name", "")
            profile.name = name.strip()
            if season_data.get("stats"):
                s = season_data["stats"][0].get("stats", {})
                profile.era = _safe_float(s.get("era"))
                profile.whip = _safe_float(s.get("whip"))
                profile.k_per_9 = _safe_float(s.get("strikeoutsPer9Inn"))
                profile.bb_per_9 = _safe_float(s.get("walksPer9Inn"))
                profile.hr_per_9 = _safe_float(s.get("homeRunsPer9"))
                profile.innings_pitched = _safe_float(s.get("inningsPitched"))
                profile.strikeouts = int(s.get("strikeOuts", 0))
                profile.games_started = int(s.get("gamesStarted", 0))

        profile.throws = fut_hand.result()

        splits = fut_gamelog.result()
        if splits:
            all_ks = [int(g["stat"].get("strikeOuts", 0)) for g in splits]
            profile.k_per_start_list = all_ks
            profile.max_k_in_start = max(all_ks) if all_ks else 0
            recent = splits[-5:] if len(splits) >= 5 else splits
            if recent:
                total_er = sum(_safe_float(g["stat"].get("earnedRuns")) for g in recent)
                total_ip = sum(_safe_float(g["stat"].get("inningsPitched")) for g in recent)
                total_k = sum(int(g["stat"].get("strikeOuts", 0)) for g in recent)
                if total_ip > 0:
                    profile.recent_era = (total_er / total_ip) * 9.0
                profile.recent_k_per_start = total_k / len(recent) if recent else 0
                profile.recent_ip_per_start = total_ip / len(recent) if recent else 0

        sc_result = fut_statcast.result()
        if sc_result.get("data") is not None:
            _apply_pitcher_statcast(profile, sc_result["data"])

    _pitcher_cache[cache_key] = profile
    return profile


def _apply_pitcher_statcast(profile: PitcherProfile, data):
    """Apply pre-fetched Statcast DataFrame to a pitcher profile."""
    if data is None or len(data) == 0:
        return

    n_pitches = len(data)

    # Pitch mix
    if "pitch_type" in data.columns:
        mix = data["pitch_type"].value_counts(normalize=True)
        profile.pitch_mix = {str(k): round(float(v), 3) for k, v in mix.items()}

    # Average fastball velocity
    fastballs = data[data["pitch_type"].isin(["FF", "SI"])]
    if len(fastballs) > 0 and "release_speed" in fastballs.columns:
        profile.avg_fastball_velo = float(fastballs["release_speed"].mean())

    # Whiff rate (swinging strikes / swings)
    if "description" in data.columns:
        swings = data[data["description"].str.contains(
            "swinging_strike|foul|hit_into_play|missed_bunt", na=False
        )]
        whiffs = data[data["description"].str.contains(
            "swinging_strike|swinging_strike_blocked", na=False
        )]
        if len(swings) > 0:
            profile.whiff_rate = len(whiffs) / len(swings)

    log.debug(
        "Statcast pitcher %s: %d pitches  velo=%.1f  whiff=%.3f  mix=%s",
        profile.name, n_pitches,
        profile.avg_fastball_velo, profile.whiff_rate,
        {k: f"{v:.0%}" for k, v in list(profile.pitch_mix.items())[:3]},
    )


# ─── Batter Data ─────────────────────────────────────────────────────────────

def get_batter_profile(player_id: int, season: int = None) -> BatterProfile:
    """Build a comprehensive batter profile from Stats API + Statcast.

    Results are cached for 5 minutes.  Internal sub-fetches (season stats,
    game log, Statcast, platoon splits) run in parallel.
    """
    if season is None:
        season = date.today().year

    cache_key = (player_id, season)
    if cache_key in _batter_cache:
        log.debug("Batter cache hit: %d", player_id)
        return _batter_cache[cache_key]

    profile = BatterProfile(player_id=player_id, name="", team="")

    # ── Define independent sub-fetches ─────────────────────────────────────

    def _fetch_season_stats() -> dict:
        try:
            return statsapi.player_stat_data(player_id, group="hitting", type="season")
        except Exception as e:
            log.warning("Stats API batter data failed for %d: %s", player_id, e)
            return {}

    def _fetch_game_log() -> list:
        try:
            url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&group=hitting&season={season}"
            resp = requests.get(url, timeout=10)
            return resp.json().get("stats", [{}])[0].get("splits", [])
        except Exception as e:
            log.debug("Game log fetch failed for batter %d: %s", player_id, e)
            return []

    def _fetch_statcast() -> dict:
        try:
            from pybaseball import statcast_batter
        except ImportError:
            return {}
        end = date.today()
        start = end - timedelta(days=30)
        log.debug("Statcast batter fetch: %d  %s -> %s", player_id, start, end)
        try:
            data = statcast_batter(str(start), str(end), player_id=player_id)
            if data is not None and len(data) > 0:
                return {"data": data}
        except Exception as exc:
            log.debug("Statcast batter fetch failed for %d: %s", player_id, exc)
        return {}

    def _fetch_splits() -> dict:
        try:
            url = (
                f"{MLB_API}/people/{player_id}/stats"
                f"?stats=statSplits&group=hitting&season={season}&sitCodes=vl,vr"
            )
            resp = requests.get(url, timeout=10)
            return resp.json()
        except Exception as e:
            log.debug("Splits fetch failed for batter %d: %s", player_id, e)
            return {}

    # ── Fire all sub-fetches in parallel (dedicated pool to avoid deadlock
    #    with the shared _POOL used by hydrate_game) ────────────────────────
    with ThreadPoolExecutor(max_workers=4) as local_pool:
        fut_season   = local_pool.submit(_fetch_season_stats)
        fut_gamelog  = local_pool.submit(_fetch_game_log)
        fut_statcast = local_pool.submit(_fetch_statcast)
        fut_splits   = local_pool.submit(_fetch_splits)

        # ── Collect results ────────────────────────────────────────────────
        season_data = fut_season.result()
        if season_data:
            name = season_data.get("first_name", "") + " " + season_data.get("last_name", "")
            profile.name = name.strip()
            profile.team = season_data.get("current_team", "")
            if season_data.get("stats"):
                s = season_data["stats"][0].get("stats", {})
                profile.avg = _safe_float(s.get("avg"))
                profile.obp = _safe_float(s.get("obp"))
                profile.slg = _safe_float(s.get("slg"))
                profile.ops = _safe_float(s.get("ops"))
                profile.hr = int(s.get("homeRuns", 0))
                profile.ab = int(s.get("atBats", 0))
                profile.hits = int(s.get("hits", 0))
                if profile.ab > 0:
                    profile.hr_per_ab = profile.hr / profile.ab
                    ks = int(s.get("strikeOuts", 0))
                    bbs = int(s.get("baseOnBalls", 0))
                    pa = int(s.get("plateAppearances", profile.ab))
                    if pa > 0:
                        profile.k_rate = ks / pa
                        profile.bb_rate = bbs / pa

        splits = fut_gamelog.result()
        if splits:
            recent = splits[-7:] if len(splits) >= 7 else splits
            if recent:
                total_h = sum(int(g["stat"].get("hits", 0)) for g in recent)
                total_ab = sum(int(g["stat"].get("atBats", 0)) for g in recent)
                total_hr = sum(int(g["stat"].get("homeRuns", 0)) for g in recent)
                if total_ab > 0:
                    profile.recent_avg = total_h / total_ab
                    profile.recent_ops = (total_h + sum(int(g["stat"].get("baseOnBalls", 0)) for g in recent)) / (total_ab + sum(int(g["stat"].get("baseOnBalls", 0)) for g in recent))
                profile.recent_hr = total_hr

        sc_result = fut_statcast.result()
        if sc_result.get("data") is not None:
            _apply_batter_statcast(profile, sc_result["data"])

        splits_data = fut_splits.result()
        if splits_data:
            _apply_batter_splits(profile, splits_data)

    _batter_cache[cache_key] = profile
    return profile


def _apply_batter_statcast(profile: BatterProfile, data):
    """Apply pre-fetched Statcast DataFrame to a batter profile."""
    if data is None or len(data) == 0:
        return

    n_pitches = len(data)

    # Exit velocity (batted balls only — launch_speed is null for non-contact)
    if "launch_speed" in data.columns:
        ev = data["launch_speed"].dropna()
        if len(ev) > 0:
            profile.avg_exit_velo = float(ev.mean())

    # Barrel rate: use Statcast's own barrel column if available, otherwise
    # approximate with EV >= 98 mph + launch angle 26-30 deg (simplified).
    if "barrel" in data.columns:
        batted = data[data["launch_speed"].notna()]
        if len(batted) > 0:
            profile.barrel_rate = float(data["barrel"].fillna(0).sum() / len(batted))
    elif "launch_speed" in data.columns and "launch_angle" in data.columns:
        batted = data.dropna(subset=["launch_speed", "launch_angle"])
        if len(batted) > 0:
            barrels = batted[
                (batted["launch_speed"] >= 98) &
                (batted["launch_angle"].between(26, 30))
            ]
            profile.barrel_rate = len(barrels) / len(batted)

    # Hard hit rate (EV >= 95 mph on batted balls)
    if "launch_speed" in data.columns:
        ev_batted = data["launch_speed"].dropna()
        if len(ev_batted) > 0:
            profile.hard_hit_rate = float((ev_batted >= 95).mean())

    # xBA and xSLG
    if "estimated_ba_using_speedangle" in data.columns:
        xba = data["estimated_ba_using_speedangle"].dropna()
        if len(xba) > 0:
            profile.xba = float(xba.mean())
    if "estimated_slg_using_speedangle" in data.columns:
        xslg = data["estimated_slg_using_speedangle"].dropna()
        if len(xslg) > 0:
            profile.xslg = float(xslg.mean())

    log.debug(
        "Statcast batter %s: %d pitches  ev=%.1f  barrel=%.3f  hard_hit=%.3f  xba=%.3f",
        profile.name, n_pitches,
        profile.avg_exit_velo, profile.barrel_rate,
        profile.hard_hit_rate, profile.xba,
    )


def _apply_batter_splits(profile: BatterProfile, splits_json: dict):
    """Apply pre-fetched platoon splits JSON to a batter profile."""
    try:
        stats = splits_json.get("stats", [{}])
        for stat_group in stats:
            for split in stat_group.get("splits", []):
                s = split.get("stat", {})
                code = split.get("split", {}).get("code", "")
                avg = _safe_float(s.get("avg"))
                hr  = int(s.get("homeRuns", 0))
                ab  = int(s.get("atBats", 0))
                if code == "vl":
                    profile.vs_lhp_avg = avg
                    profile.vs_lhp_hr  = hr
                    profile.vs_lhp_ab  = ab
                elif code == "vr":
                    profile.vs_rhp_avg = avg
                    profile.vs_rhp_hr  = hr
                    profile.vs_rhp_ab  = ab
    except Exception as e:
        log.debug("Splits apply failed: %s", e)


# ─── Team Data ───────────────────────────────────────────────────────────────

def get_team_profile(team_name_or_id, season: int = None) -> TeamProfile:
    """Get team-level stats.

    Results are cached for 10 minutes since team-level stats change
    very slowly (once per completed game).
    """
    if season is None:
        season = date.today().year

    # Resolve team ID
    if isinstance(team_name_or_id, int):
        team_id = team_name_or_id
    else:
        team_id = get_team_id(team_name_or_id)
        if not team_id:
            return TeamProfile(name=str(team_name_or_id), abbrev="???", team_id=0)

    cache_key = (team_id, season)
    if cache_key in _team_cache_ttl:
        log.debug("Team cache hit: %s (%d)", team_name_or_id, team_id)
        return _team_cache_ttl[cache_key]

    profile = TeamProfile(name="", abbrev="", team_id=team_id)

    # ── Define independent sub-fetches ─────────────────────────────────────

    def _fetch_team_info() -> dict:
        try:
            teams = statsapi.get("teams", {"teamId": team_id})
            if teams.get("teams"):
                return teams["teams"][0]
        except Exception:
            pass
        return {}

    def _fetch_batting() -> dict:
        try:
            url = f"{MLB_API}/teams/{team_id}/stats?stats=season&group=hitting&season={season}"
            resp = requests.get(url, timeout=10)
            return resp.json()
        except Exception as e:
            log.debug("Team batting stats failed: %s", e)
            return {}

    def _fetch_pitching() -> dict:
        try:
            url = f"{MLB_API}/teams/{team_id}/stats?stats=season&group=pitching&season={season}"
            resp = requests.get(url, timeout=10)
            return resp.json()
        except Exception as e:
            log.debug("Team pitching stats failed: %s", e)
            return {}

    def _fetch_standings() -> dict:
        try:
            return statsapi.standings_data(season=season)
        except Exception:
            return {}

    # ── Fire all sub-fetches in parallel (dedicated pool to avoid deadlock
    #    with the shared _POOL used by hydrate_game) ────────────────────────
    with ThreadPoolExecutor(max_workers=4) as local_pool:
        fut_info      = local_pool.submit(_fetch_team_info)
        fut_batting   = local_pool.submit(_fetch_batting)
        fut_pitching  = local_pool.submit(_fetch_pitching)
        fut_standings = local_pool.submit(_fetch_standings)

        # ── Collect results ────────────────────────────────────────────────
        team_info = fut_info.result()
        if team_info:
            profile.name = team_info.get("name", "")
            profile.abbrev = team_info.get("abbreviation", "")

        batting_data = fut_batting.result()
        stats = batting_data.get("stats", [{}])
        if stats and stats[0].get("splits"):
            s = stats[0]["splits"][0].get("stat", {})
            profile.team_avg = _safe_float(s.get("avg"))
            profile.team_ops = _safe_float(s.get("ops"))
            profile.team_hr = int(s.get("homeRuns", 0))
            runs = int(s.get("runs", 0))
            games = int(s.get("gamesPlayed", 1))
            profile.runs_per_game = runs / games if games > 0 else 0
            pa = int(s.get("plateAppearances", 1))
            ks = int(s.get("strikeOuts", 0))
            profile.k_rate = ks / pa if pa > 0 else 0

        pitching_data = fut_pitching.result()
        stats = pitching_data.get("stats", [{}])
        if stats and stats[0].get("splits"):
            s = stats[0]["splits"][0].get("stat", {})
            profile.team_era = _safe_float(s.get("era"))
            profile.team_whip = _safe_float(s.get("whip"))

        standings = fut_standings.result()
        for div_id, div_data in standings.items():
            for t in div_data.get("teams", []):
                if t.get("team_id") == team_id:
                    profile.wins = int(t.get("w", 0))
                    profile.losses = int(t.get("l", 0))
                    profile.run_diff = int(t.get("run_diff", 0))

    _team_cache_ttl[cache_key] = profile
    return profile


# ─── Lineup Detection ────────────────────────────────────────────────────────

def get_lineup(game_id: int, team: str = "away") -> Lineup:
    """Try to get confirmed lineup from live game feed.
    
    Lineups are typically posted 1-2 hours before game time.
    """
    feed = get_live_game_feed(game_id)
    if not feed:
        return Lineup(team=team, batters=[], confirmed=False)
    
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    teams_box = boxscore.get("teams", {})
    team_box = teams_box.get(team, {})
    
    batting_order = team_box.get("battingOrder", [])
    players = team_box.get("players", {})
    
    if not batting_order:
        return Lineup(team=team, batters=[], confirmed=False)
    
    batters = []
    for i, pid in enumerate(batting_order):
        player_data = players.get(f"ID{pid}", {})
        person = player_data.get("person", {})
        pos = player_data.get("position", {})
        batters.append({
            "player_id": pid,
            "name": person.get("fullName", ""),
            "position": pos.get("abbreviation", ""),
            "order": i + 1,
        })
    
    return Lineup(
        team=team,
        batters=batters,
        confirmed=len(batters) > 0,
    )


# ─── Player Lookup ───────────────────────────────────────────────────────────

def lookup_player(name: str) -> Optional[int]:
    """Find MLB player ID by name. Returns player_id or None."""
    try:
        results = statsapi.lookup_player(name)
        if results:
            return results[0]["id"]
    except Exception:
        pass
    return None


def lookup_player_by_kalshi_ticker(ticker: str) -> Optional[int]:
    """Extract player name from Kalshi ticker and look up MLB ID.
    
    Ticker format: KXMLBHR-26MAY112210SFLAD-LADSOHTANI17-2
    Player part: LADSOHTANI17 → S. Ohtani
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    
    player_part = parts[2]  # e.g., "LADSOHTANI17"
    # Remove team prefix (3 chars) and jersey number (trailing digits)
    if len(player_part) < 4:
        return None
    
    # Strip team code (first 3 chars)
    name_and_num = player_part[3:]
    # Strip trailing jersey number
    name = ""
    for i, c in enumerate(name_and_num):
        if c.isdigit() and all(x.isdigit() for x in name_and_num[i:]):
            name = name_and_num[:i]
            break
    
    if not name or len(name) < 2:
        return None
    
    # First char is initial, rest is last name: SOHTANI → S. Ohtani → "Ohtani"
    first_initial = name[0]
    last_name = name[1:]
    
    # Search by last name, filter by first initial
    try:
        results = statsapi.lookup_player(last_name)
        for r in results:
            if r.get("firstName", "")[0:1].upper() == first_initial.upper():
                return r["id"]
        # Fallback: return first match
        if results:
            return results[0]["id"]
    except Exception:
        pass
    
    return None


# ─── Batter vs Pitcher ──────────────────────────────────────────────────────

def get_bvp_stats(batter_id: int, pitcher_id: int) -> Dict[str, Any]:
    """Get batter vs pitcher career matchup data.
    
    Returns dict with ab, hits, hr, k, bb, avg (or empty if no data).
    """
    try:
        url = (
            f"{MLB_API}/people/{batter_id}/stats"
            f"?stats=vsPlayer,vsPlayerTotal&group=hitting"
            f"&opposingPlayerId={pitcher_id}"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                return {
                    "ab": int(s.get("atBats", 0)),
                    "hits": int(s.get("hits", 0)),
                    "hr": int(s.get("homeRuns", 0)),
                    "k": int(s.get("strikeOuts", 0)),
                    "bb": int(s.get("baseOnBalls", 0)),
                    "avg": _safe_float(s.get("avg")),
                }
    except Exception as e:
        log.debug("BvP lookup failed for %d vs %d: %s", batter_id, pitcher_id, e)
    
    return {}


# ─── Utilities ───────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0) -> float:
    """Safely parse a stat value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ─── Live Data Provider ──────────────────────────────────────────────────────

class LiveMLBDataProvider:
    """Production adapter — fetches data from MLB Stats API + Statcast.

    Wraps the existing module-level functions and returns fully-hydrated
    GameContext objects so callers never need to call individual fetch
    functions directly.
    """

    def get_game_contexts(
        self,
        target_date: Optional[str] = None,
    ) -> List[GameContext]:
        """Fetch today's games and hydrate each into a GameContext.

        For each game, fetches (in parallel where possible):
          - Both pitcher profiles
          - Both team profiles
          - Both lineups + batter profiles

        Returns a list of GameContext objects ready for strategy evaluation.
        """
        games = get_todays_games(target_date)
        if not games:
            return []

        contexts: List[GameContext] = []
        for game in games:
            ctx = self.hydrate_game(game)
            contexts.append(ctx)
        return contexts

    def hydrate_game(self, game: GameInfo) -> GameContext:
        """Build a complete GameContext for a single game.

        Uses the shared thread pool (_POOL) for parallel I/O.
        """
        ctx = GameContext(game=game)

        # ── Pitchers (parallel) ────────────────────────────────────────────
        pitcher_futs = {}
        if game.away_pitcher_id:
            pitcher_futs["away"] = _POOL.submit(get_pitcher_profile, game.away_pitcher_id)
        if game.home_pitcher_id:
            pitcher_futs["home"] = _POOL.submit(get_pitcher_profile, game.home_pitcher_id)

        if "away" in pitcher_futs:
            try:
                ctx.away_pitcher = pitcher_futs["away"].result()
            except Exception as exc:
                log.warning("Away pitcher profile failed for game %s: %s", game.game_id, exc)
        if "home" in pitcher_futs:
            try:
                ctx.home_pitcher = pitcher_futs["home"].result()
            except Exception as exc:
                log.warning("Home pitcher profile failed for game %s: %s", game.game_id, exc)

        # ── Teams (parallel) ──────────────────────────────────────────────
        team_futs = {}
        if game.away_abbrev:
            team_futs["away"] = _POOL.submit(get_team_profile, game.away_abbrev)
        if game.home_abbrev:
            team_futs["home"] = _POOL.submit(get_team_profile, game.home_abbrev)

        if "away" in team_futs:
            try:
                ctx.away_team = team_futs["away"].result()
            except Exception as exc:
                log.debug("Away team profile failed: %s", exc)
        if "home" in team_futs:
            try:
                ctx.home_team = team_futs["home"].result()
            except Exception as exc:
                log.debug("Home team profile failed: %s", exc)

        # ── Lineups + batter profiles ─────────────────────────────────────
        if game.game_id:
            ctx.away_batters, ctx.home_batters = self._fetch_batters(game)

        return ctx

    def _fetch_batters(self, game: GameInfo) -> Tuple[List[BatterProfile], List[BatterProfile]]:
        """Fetch confirmed lineups and resolve batter profiles in parallel."""
        away_batters: List[BatterProfile] = []
        home_batters: List[BatterProfile] = []

        try:
            away_lineup = get_lineup(game.game_id, team="away")
            home_lineup = get_lineup(game.game_id, team="home")
        except Exception as exc:
            log.warning("Lineup fetch failed for game %s: %s", game.game_id, exc)
            return away_batters, home_batters

        if not away_lineup.confirmed and not home_lineup.confirmed:
            return away_batters, home_batters

        # Collect all player IDs
        away_pids = [b.get("player_id") for b in (away_lineup.batters or []) if b.get("player_id")]
        home_pids = [b.get("player_id") for b in (home_lineup.batters or []) if b.get("player_id")]
        all_pids = list(set(away_pids + home_pids))

        # Fetch all profiles in parallel
        profile_cache: Dict[int, BatterProfile] = {}
        if all_pids:
            futs = {_POOL.submit(get_batter_profile, pid): pid for pid in all_pids}
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    profile_cache[pid] = fut.result()
                except Exception as exc:
                    log.debug("Could not fetch batter profile %d: %s", pid, exc)

        # Reconstruct ordered lists, stamping batting order position
        if away_lineup.confirmed:
            for i, pid in enumerate(away_pids):
                if pid in profile_cache:
                    bp = profile_cache[pid]
                    bp.batting_order = i + 1
                    away_batters.append(bp)
        if home_lineup.confirmed:
            for i, pid in enumerate(home_pids):
                if pid in profile_cache:
                    bp = profile_cache[pid]
                    bp.batting_order = i + 1
                    home_batters.append(bp)

        return away_batters, home_batters


class FixtureMLBDataProvider:
    """Test adapter — returns pre-built GameContext objects from fixture data.

    Pass a list of GameContext objects at construction time; they're returned
    verbatim by get_game_contexts() regardless of the requested date.
    """

    def __init__(self, contexts: List[GameContext]):
        self._contexts = contexts
        self._by_game_id = {ctx.game.game_id: ctx for ctx in contexts}

    def get_game_contexts(
        self,
        target_date: Optional[str] = None,
    ) -> List[GameContext]:
        return list(self._contexts)

    def hydrate_game(self, game: GameInfo) -> GameContext:
        """Return the fixture context matching this game, or a bare context."""
        if game.game_id in self._by_game_id:
            return self._by_game_id[game.game_id]
        return GameContext(game=game)
