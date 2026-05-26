"""Configuration for Slugger MLB trading bot."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

def _load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val

_load_env()

@dataclass(frozen=True)
class Config:
    """Immutable bot configuration from environment."""
    # Kalshi auth
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    use_demo: bool = False
    
    # Trading
    dry_run: bool = True
    max_position_usd: float = 5.0
    max_contracts_per_trade: int = 10
    min_edge_cents: int = 3       # Minimum edge in cents to trade
    kelly_fraction: float = 0.25  # Quarter-Kelly
    
    # Circuit breaker
    cb_max_loss_usd: float = 10.0
    cb_max_consecutive_losses: int = 3
    
    # Market selection
    enabled_strategies: tuple = ("game_winner", "pitcher_ks", "player_hr", "player_hits")
    min_liquidity_dollars: float = 5.0
    min_volume: int = 0

    # Risk management
    max_signals_per_game: int = 5     # Cap correlated same-game exposure
    max_exposure_per_game_usd: float = 0.0  # 0 = no dollar cap (use signal count only)

    # Combo / parlay
    combo_max_legs: int = 3           # Max legs per combo (2-3)
    
    # Timing
    pregame_hours: float = 2.0   # Start analyzing N hours before game
    poll_interval_sec: int = 60  # How often to rescan markets
    
    # Data
    statcast_lookback_days: int = 30
    min_sample_size: int = 20    # Min PA/BF for stats to be meaningful
    
    # Telegram (optional)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    
    # Logging
    log_dir: str = "logs"
    
    @property
    def api_base(self) -> str:
        if self.use_demo:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://external-api.kalshi.com/trade-api/v2"
    
    @classmethod
    def from_env(cls) -> Config:
        """Build Config from environment variables."""
        key_id = os.getenv("KALSHI_API_KEY_ID", os.getenv("KALSHI_KEY_ID", ""))
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        if key_path and key_path.startswith("~"):
            key_path = str(Path(key_path).expanduser())

        return cls(
            kalshi_api_key_id=key_id,
            kalshi_private_key_path=key_path,
            use_demo=os.getenv("USE_DEMO", "false").lower() == "true",
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
            max_position_usd=float(os.getenv("MAX_POSITION_USD", "5")),
            max_contracts_per_trade=int(os.getenv("MAX_CONTRACTS_PER_TRADE", "10")),
            min_edge_cents=int(os.getenv("MIN_EDGE_CENTS", "3")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            cb_max_loss_usd=float(os.getenv("CIRCUIT_BREAKER_MAX_LOSS_USD", "10")),
            cb_max_consecutive_losses=int(os.getenv("CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES", "3")),
            enabled_strategies=tuple(os.getenv("ENABLED_STRATEGIES", "game_winner,pitcher_ks,player_hr,player_hits").split(",")),
            min_liquidity_dollars=float(os.getenv("MIN_LIQUIDITY_DOLLARS", "5")),
            min_volume=int(os.getenv("MIN_VOLUME", "0")),
            max_signals_per_game=int(os.getenv("MAX_SIGNALS_PER_GAME", "5")),
            max_exposure_per_game_usd=float(os.getenv("MAX_EXPOSURE_PER_GAME_USD", "0")),
            combo_max_legs=int(os.getenv("COMBO_MAX_LEGS", "3")),
            pregame_hours=float(os.getenv("PREGAME_HOURS", "2")),
            poll_interval_sec=int(os.getenv("POLL_INTERVAL_SEC", "60")),
            statcast_lookback_days=int(os.getenv("STATCAST_LOOKBACK_DAYS", "30")),
            min_sample_size=int(os.getenv("MIN_SAMPLE_SIZE", "20")),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )

    def create_kalshi_client(self):
        """Instantiate a KalshiClient from config."""
        from .kalshi_client import KalshiClient

        if not self.kalshi_api_key_id or not self.kalshi_private_key_path:
            raise ValueError(
                "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set. "
                "Copy your API key ID from https://kalshi.com/account/api and "
                "save your private key PEM to a file, then set the paths in .env."
            )
        return KalshiClient(
            api_key_id=self.kalshi_api_key_id,
            private_key_path=self.kalshi_private_key_path,
            use_demo=self.use_demo,
        )
