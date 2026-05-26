"""Shared domain types for Slugger MLB trading bot.

All dataclasses and protocols that cross module boundaries live here.
This is a leaf module — it imports nothing from the slugger package,
so any slugger module can import from it without circular dependencies.

Types extracted from:
  - slugger.mlb_data   (GameInfo, PitcherProfile, BatterProfile, TeamProfile,
                         Lineup, GameContext, MLBDataProvider)
  - slugger.strategies  (TradeSignal)
  - slugger.signal_pipeline (ModelResult, MarketSpec, ModelFn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MLB DOMAIN TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class GameInfo:
    """Scheduled MLB game with key metadata."""
    game_id: int
    away_team: str
    home_team: str
    away_abbrev: str
    home_abbrev: str
    away_record: str          # "21-20"
    home_record: str
    away_pitcher_name: str    # Probable starter
    home_pitcher_name: str
    away_pitcher_id: int
    home_pitcher_id: int
    game_datetime: str        # ISO UTC
    venue: str
    weather: Dict[str, str]   # condition, temp, wind
    status: str               # "Pre-Game", "In Progress", "Final"


@dataclass
class PitcherProfile:
    """Pitcher stats for prediction models."""
    player_id: int
    name: str
    # Season stats
    era: float = 0.0
    whip: float = 0.0
    k_per_9: float = 0.0
    bb_per_9: float = 0.0
    hr_per_9: float = 0.0
    innings_pitched: float = 0.0
    strikeouts: int = 0
    games_started: int = 0
    # Recent form (last 5 starts)
    recent_era: float = 0.0
    recent_k_per_start: float = 0.0
    recent_ip_per_start: float = 0.0
    max_k_in_start: int = 0           # ceiling: most Ks in any start this season
    k_per_start_list: List[int] = field(default_factory=list)  # per-start K counts
    # Handedness
    throws: str = ""              # "L", "R", or "S" (switch — rare for pitchers)
    # Statcast (if available)
    avg_fastball_velo: float = 0.0
    whiff_rate: float = 0.0
    chase_rate: float = 0.0
    barrel_rate_against: float = 0.0
    xera: float = 0.0
    pitch_mix: Dict[str, float] = field(default_factory=dict)  # {pitch_type: pct}


@dataclass
class BatterProfile:
    """Batter stats for prediction models."""
    player_id: int
    name: str
    team: str
    # Season stats
    avg: float = 0.0
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0
    hr: int = 0
    ab: int = 0
    hits: int = 0
    k_rate: float = 0.0          # K%
    bb_rate: float = 0.0         # BB%
    hr_per_ab: float = 0.0
    # Recent form (last 10 games)
    recent_avg: float = 0.0
    recent_ops: float = 0.0
    recent_hr: int = 0
    # Statcast
    avg_exit_velo: float = 0.0
    barrel_rate: float = 0.0
    hard_hit_rate: float = 0.0
    xba: float = 0.0
    xslg: float = 0.0
    # Platoon splits (vs LHP / vs RHP)
    vs_lhp_avg: float = 0.0
    vs_lhp_hr: int = 0
    vs_lhp_ab: int = 0
    vs_rhp_avg: float = 0.0
    vs_rhp_hr: int = 0
    vs_rhp_ab: int = 0
    # Lineup position (set by _fetch_batters when lineup is confirmed)
    batting_order: int = 0        # 1-9 lineup slot, 0 = unknown


@dataclass
class TeamProfile:
    """Team-level batting/pitching stats."""
    name: str
    abbrev: str
    team_id: int
    # Batting
    team_avg: float = 0.0
    team_ops: float = 0.0
    team_hr: int = 0
    runs_per_game: float = 0.0
    k_rate: float = 0.0
    # Pitching
    team_era: float = 0.0
    team_whip: float = 0.0
    bullpen_era: float = 0.0
    # Record
    wins: int = 0
    losses: int = 0
    run_diff: int = 0


@dataclass
class Lineup:
    """Game-day batting lineup."""
    team: str
    batters: List[Dict[str, Any]]  # [{player_id, name, position, order}]
    confirmed: bool = False


@dataclass
class GameContext:
    """Everything needed to evaluate a single game — no further API calls required.

    Built once per game by an MLBDataProvider, then passed through the
    processing pipeline.  Strategies receive this instead of calling
    mlb_data functions directly.
    """
    game: GameInfo
    away_pitcher: Optional[PitcherProfile] = None
    home_pitcher: Optional[PitcherProfile] = None
    away_batters: List[BatterProfile] = field(default_factory=list)
    home_batters: List[BatterProfile] = field(default_factory=list)
    away_team: Optional[TeamProfile] = None
    home_team: Optional[TeamProfile] = None


@runtime_checkable
class MLBDataProvider(Protocol):
    """Seam for MLB data sourcing.

    Production adapter calls live APIs.  Test adapter returns fixtures.
    """

    def get_game_contexts(
        self,
        target_date: Optional[str] = None,
    ) -> List[GameContext]:
        """Fetch fully-hydrated game contexts for a date (default: today)."""
        ...

    def hydrate_game(self, game: GameInfo) -> GameContext:
        """Hydrate a single GameInfo into a full GameContext."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRADING TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class TradeSignal:
    """A suggested trade from a strategy."""
    ticker: str
    action: str           # "buy"
    side: str             # "yes" or "no"
    count: int            # number of contracts
    price: int            # limit price in cents (1-99)
    strategy: str         # e.g. "game_winner"
    confidence: float     # 0.0-1.0
    edge_cents: float     # expected edge in cents
    reason: str = ""      # human-readable rationale
    model_prob_pct: float = 0.0  # calibrated model probability (0-100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIGNAL PIPELINE TYPES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ModelResult:
    """Output from a strategy's probability model for a single market.

    Attributes:
        prob_pct:  Model probability as integer percentage (0-100).
        reason:    Human-readable explanation for logging / journal.
    """
    prob_pct: int
    reason: str


@dataclass
class MarketSpec:
    """Describes how to find and evaluate markets for a strategy.

    Attributes:
        event_ticker:       Kalshi event ticker to query.
        strategy_name:      Name for TradeSignal.strategy and journal recording.
        title_keywords:     Market title must contain at least one of these
                            (case-insensitive).  Empty list = no keyword filter.
        player_name:        If set, market title must contain this player's last
                            name (case-insensitive).
        ticker_suffix:      If set, market ticker must end with this suffix
                            (case-insensitive, e.g. "-LAD" for home team).
        threshold_pattern:  Regex pattern to extract a numeric threshold from the
                            title.  Must have one capture group yielding a number.
                            If None, threshold is not parsed (passed as None to model).
        threshold_ceil:     If True, ceil the parsed threshold (for "over 6.5" -> 7).
        min_threshold:      Skip markets with threshold below this value.
        min_model_prob:     Minimum model probability (%) to consider trading YES.
        min_edge_cents:     Minimum edge in cents to trade (overrides config if higher).
        max_signals:        Maximum number of YES signals to return (sorted by edge).
                            0 = unlimited.
        confidence_fn:      Compute TradeSignal.confidence from edge_cents.
                            Default: min(0.5 + edge/100, 0.85).
        no_side:            If True, also evaluate NO-side trades.
        no_max_model_prob:  For NO-side: only buy NO when model YES prob <= this (%).
        no_min_edge_cents:  For NO-side: minimum edge to buy NO.
    """
    event_ticker: str
    strategy_name: str
    title_keywords: List[str] = field(default_factory=list)
    player_name: str = ""
    ticker_suffix: str = ""
    threshold_pattern: Optional[str] = None
    threshold_ceil: bool = False
    min_threshold: int = 0
    min_model_prob: int = 0
    min_edge_cents: int = 0
    max_signals: int = 0
    confidence_fn: Optional[Callable[[float], float]] = None
    no_side: bool = False
    no_max_model_prob: int = 10
    no_min_edge_cents: int = 5


# Probability model callable type:
#   (market_title, threshold_or_none, market_price_cents) -> ModelResult or None
#   Return None to skip this market entirely.
ModelFn = Callable[[str, Optional[int], int], Optional[ModelResult]]
