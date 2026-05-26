"""Kalshi ticker construction and team abbreviation mapping.

Single source of truth for:
  - MLB → Kalshi team abbreviation mapping
  - Event ticker construction for all MLB prop types
  - Ticker parsing (datetime extraction, team extraction)

Pure functions, no I/O.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from slugger.types import GameInfo


# ─── Team abbreviation mapping ───────────────────────────────────────────────

# Map MLB Stats API abbreviations → Kalshi codes where they differ.
# Most MLB abbreviations already match Kalshi exactly (LAD, SF, NYY, etc.).
# This table only covers the handful that don't.
API_TO_KALSHI = {
    "SFG": "SF",   # Giants: MLB uses "SFG" in some contexts, Kalshi uses "SF"
    "KCR": "KC",   # Royals
    "SDP": "SD",   # Padres
    "TBR": "TB",   # Rays
    "WSN": "WSH",  # Nationals
    # Legacy fallbacks from old name-slice derivation (kept for safety)
    "SAN": "SF",   "SFN": "SF",
    "LAN": "LAD",  "SDN": "SD",
    "SLN": "STL",  "ANA": "LAA",
    "TAM": "TB",   "NEW": "NYY",
    "LOS": "LAA",
}


def kalshi_team(abbrev: str) -> str:
    """Map an MLB abbreviation to its Kalshi equivalent."""
    return API_TO_KALSHI.get(abbrev.upper(), abbrev.upper())


# ─── Date formatting ─────────────────────────────────────────────────────────

def kalshi_date(game: GameInfo) -> Optional[str]:
    """Format game datetime as Kalshi date+time string (YYMONDDHHMI) in ET.

    Example: "26MAY111810" for a game on May 11, 2026 at 6:10 PM ET.
    Returns None if game has no datetime.
    """
    if not game.game_datetime:
        return None
    try:
        dt = datetime.fromisoformat(game.game_datetime.replace("Z", "+00:00"))
        et = timezone(timedelta(hours=-4))
        dt_et = dt.astimezone(et)
        return dt_et.strftime("%y%b%d").upper() + dt_et.strftime("%H%M")
    except (ValueError, TypeError):
        return None


def _game_teams(game: GameInfo) -> str:
    """Build the away+home team string for a ticker."""
    return f"{kalshi_team(game.away_abbrev)}{kalshi_team(game.home_abbrev)}"


# ─── Event ticker construction ───────────────────────────────────────────────

def game_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi game event ticker.

    Format: KXMLBGAME-{YYMONDDHHMM}{AWAY}{HOME}
    Example: KXMLBGAME-26MAY111810LAACLE
    """
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBGAME-{d}{_game_teams(game)}"


def ks_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi strikeout event ticker."""
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBKS-{d}{_game_teams(game)}"


def hr_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi home run event ticker."""
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBHR-{d}{_game_teams(game)}"


def total_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi total runs event ticker."""
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBTOTAL-{d}{_game_teams(game)}"


def hit_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi player hits event ticker."""
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBHIT-{d}{_game_teams(game)}"


def hrr_event_ticker(game: GameInfo) -> Optional[str]:
    """Construct a Kalshi hits+runs+RBIs event ticker."""
    d = kalshi_date(game)
    if not d:
        return None
    return f"KXMLBHRR-{d}{_game_teams(game)}"


# ─── Ticker parsing ──────────────────────────────────────────────────────────

def parse_game_time_utc(ticker: str) -> Optional[datetime]:
    """Extract game start time (UTC) from the date+time embedded in a ticker.

    Parses the YYMONDDHHMI portion of tickers like:
      KXMLBGAME-26MAY121940KCCWS
    Returns a timezone-aware UTC datetime, or None if parsing fails.
    """
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})(\d{4})", ticker)
    if not m:
        return None
    year = int(m.group(1)) + 2000
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(m.group(2))
    if not month:
        return None
    day = int(m.group(3))
    hh, mm = int(m.group(4)[:2]), int(m.group(4)[2:])
    et = timezone(timedelta(hours=-4))
    return datetime(year, month, day, hh, mm, tzinfo=et).astimezone(timezone.utc)


def extract_teams(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (away, home) team abbreviations from a ticker.

    Parses the teams portion after the YYMONDDHHMI date segment.
    e.g. KXMLBGAME-26MAY121940KCCWS -> ("KC", "CWS")

    Returns (None, None) if parsing fails.
    """
    m = re.search(r"\d{4}([A-Z]+)$", ticker.rsplit("-", 1)[0])
    if not m:
        return None, None
    teams_str = m.group(1)
    # Try 3-char then 2-char suffix match for the home team
    for n in (3, 2):
        candidate_home = teams_str[-n:]
        candidate_away = teams_str[:-n]
        if candidate_away:  # must have something left for away
            return candidate_away, candidate_home
    return None, None
