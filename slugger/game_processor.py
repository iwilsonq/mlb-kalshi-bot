"""Game processing engine — scan, signal, trade.

Encapsulates the core bot loop: fetch games, hydrate contexts, run
strategies, execute trades, and manage the daily ledger + circuit breaker.

Extracted from main.py so the game-processing logic is testable and
reusable without the CLI layer.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set

import requests

from slugger.config import Config
from slugger.kalshi_client import KalshiClient
from slugger.mlb_data import (
    GameContext, GameInfo, LiveMLBDataProvider, get_todays_games,
)
from slugger.strategies import BATTER_STRATEGIES, STRATEGIES, TradeSignal, strategy_combo
from slugger.tickers import game_event_ticker
import slugger.journal as journal

log = logging.getLogger("slugger")


def game_markets(client: KalshiClient, game: GameInfo, config: Config) -> List[dict]:
    """Fetch markets for a single game via its Kalshi event ticker."""
    event_ticker = game_event_ticker(game)
    if not event_ticker:
        return []

    try:
        return client.get_event_markets(event_ticker, min_liquidity=config.min_liquidity_dollars)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            log.debug("No Kalshi markets found for event %s", event_ticker)
            return []
        log.warning("Failed to fetch markets for %s: %s", event_ticker, e)
        return []


# ─── Daily trade ledger ───────────────────────────────────────────────────────

def ledger_path(log_dir: str) -> Path:
    """Return path to today's trade ledger file."""
    today = date.today().isoformat()
    return Path(log_dir) / f"placed_{today}.json"


def load_ledger(log_dir: str) -> Set[str]:
    """Load today's set of placed tickers from disk."""
    path = ledger_path(log_dir)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            tickers = set(data) if isinstance(data, list) else set()
            if tickers:
                log.info("Loaded %d ticker(s) from today's ledger", len(tickers))
            return tickers
        except Exception as exc:
            log.warning("Could not read ledger %s: %s", path, exc)
    return set()


def save_ledger(tickers: Set[str], log_dir: str) -> None:
    """Persist today's set of placed tickers to disk."""
    path = ledger_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(tickers), indent=2))


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """Monitors losses and trips the bot if thresholds are exceeded."""

    def __init__(self, config: Config):
        self.max_loss = config.cb_max_loss_usd
        self.max_consec = config.cb_max_consecutive_losses
        self.total_loss = 0.0
        self.consec_losses = 0
        self.tripped = False

    def record_trade(self, loss_usd: float):
        if loss_usd < 0:
            self.total_loss += abs(loss_usd)
            self.consec_losses += 1
            if self.total_loss > self.max_loss or self.consec_losses >= self.max_consec:
                self.tripped = True
                log.warning(
                    "⚡ Circuit breaker TRIPPED: $%.2f lost, %d consecutive losses",
                    self.total_loss, self.consec_losses,
                )
        else:
            self.consec_losses = 0

    def is_tripped(self) -> bool:
        return self.tripped


# ─── Helpers ──────────────────────────────────────────────────────────────────

def game_matches(game: GameInfo, pattern: str) -> bool:
    """Return True if the game involves a team matching the pattern.

    Matches against away_abbrev, home_abbrev, or the combined
    '{away}{home}' string, all case-insensitive.  Examples:
        "LAD"   matches any game with LAD
        "SFLAD" matches SF @ LAD specifically
        "sf"    matches any Giants game
    """
    p = pattern.upper()
    combined = f"{game.away_abbrev}{game.home_abbrev}".upper()
    return (
        p in game.away_abbrev.upper()
        or p in game.home_abbrev.upper()
        or p in combined
    )


def game_has_started(game: GameInfo, buffer_minutes: int = 5) -> bool:
    """Return True if the game's scheduled start time has passed (with buffer).

    Uses the game datetime embedded in the schedule, NOT the status field,
    which can lag behind reality.  A 5-minute buffer allows for delayed
    first pitches.
    """
    if not game.game_datetime:
        return True  # No datetime → assume started (safe default)
    try:
        dt = datetime.fromisoformat(game.game_datetime.replace("Z", "+00:00"))
        cutoff = dt + timedelta(minutes=buffer_minutes)
        return datetime.now(timezone.utc) > cutoff
    except (ValueError, TypeError):
        return True


# ─── Signal Execution ─────────────────────────────────────────────────────────

def execute_signals(
    signals: List[TradeSignal],
    client: KalshiClient,
    config: Config,
    circuit: CircuitBreaker,
    effective_bankroll: float,
    held_tickers: Set[str],
    placed_tickers: Set[str],
) -> bool:
    """Place orders for a list of signals. Returns True if any signal was acted on."""
    any_acted = False
    for signal in signals:
        if circuit.is_tripped():
            return any_acted

        any_acted = True

        # ── Dedup check ────────────────────────────────────────────────────
        if signal.ticker in held_tickers:
            log.info("  ⏭ %s | %s — already held, skipping", signal.strategy, signal.ticker)
            continue

        # ── Rescale count to live bankroll ─────────────────────────────────
        if effective_bankroll < config.max_position_usd and signal.count > 0:
            scale = effective_bankroll / config.max_position_usd
            signal.count = max(1, int(signal.count * scale))

        log.info(
            "  📊 %s | %s | %d contracts @ %d¢ | Edge: %.1f¢",
            signal.strategy, signal.reason, signal.count,
            signal.price, signal.edge_cents,
        )

        if config.dry_run:
            log.info(
                "     [DRY RUN] Would BUY %s %s %d × %d¢ = $%.2f",
                signal.side.upper(), signal.ticker, signal.count, signal.price,
                signal.count * signal.price / 100,
            )
            held_tickers.add(signal.ticker)
        else:
            if signal.side == "no":
                result = client.create_no_order(
                    ticker=signal.ticker,
                    count=signal.count,
                    no_price=signal.price,
                )
            else:
                result = client.create_yes_order(
                    ticker=signal.ticker,
                    count=signal.count,
                    yes_price=signal.price,
                )
            if result.status in ("accepted", "executed"):
                cost_usd = signal.count * signal.price / 100
                log.info(
                    "     ✅ Order placed: %s (status: %s) cost=$%.2f",
                    result.order_id, result.status, cost_usd,
                )
                held_tickers.add(signal.ticker)
                placed_tickers.add(signal.ticker)
                circuit.record_trade(cost_usd)
                journal.record_trade(
                    log_dir=config.log_dir,
                    ticker=signal.ticker,
                    strategy=signal.strategy,
                    side=signal.side,
                    count=signal.count,
                    price_cents=signal.price,
                    cost_usd=cost_usd,
                    edge_cents=signal.edge_cents,
                    reason=signal.reason,
                    order_id=result.order_id,
                )
            else:
                log.warning(
                    "     ❌ Order failed: %s (status: %s)",
                    result.error or "unknown", result.status,
                )
    return any_acted


# ─── Game Processing ──────────────────────────────────────────────────────────

def process_game(
    ctx: GameContext,
    client: KalshiClient,
    config: Config,
    circuit: CircuitBreaker,
    bankroll_usd: float,
    held_tickers: Set[str],
    placed_tickers: Set[str],
):
    """Run all strategies for a single game and execute trades.

    Accepts a fully-hydrated GameContext — pitcher profiles, batter profiles,
    and team stats are already fetched.  No additional MLB API calls needed.
    """
    game = ctx.game
    log.info("\n🔍 %s @ %s [%s]", game.away_team, game.home_team, game.status)

    # ── Hard gate: refuse to trade if game has already started ──────────
    if game_has_started(game):
        log.info("  ⛔ Game has started (past scheduled time + buffer) — skipping entirely")
        return

    away_pitch = ctx.away_pitcher
    home_pitch = ctx.home_pitcher
    log.info(
        "  Pitchers: %s vs %s",
        away_pitch.name if away_pitch else "TBD",
        home_pitch.name if home_pitch else "TBD",
    )

    # Effective bankroll cap
    effective_bankroll = min(bankroll_usd, config.max_position_usd)
    if effective_bankroll < config.max_position_usd:
        log.info(
            "  ⚠️  Live balance $%.2f < MAX_POSITION_USD $%.2f — "
            "scaling Kelly to $%.2f",
            bankroll_usd, config.max_position_usd, effective_bankroll,
        )

    pitcher_strats = [s for s in config.enabled_strategies if s not in BATTER_STRATEGIES]
    batter_strats  = [s for s in config.enabled_strategies if s in BATTER_STRATEGIES]

    any_signals = False

    # Collect all single-leg signals for potential combo use later
    all_single_leg_signals: List[TradeSignal] = []

    # ── Pitcher / game-level strategies (run once per pitcher) ────────────
    pitchers = [p for p in [away_pitch, home_pitch] if p]

    for strat_name in pitcher_strats:
        if circuit.is_tripped():
            log.warning("⚡ Circuit breaker tripped — stopping")
            return

        strategy = STRATEGIES.get(strat_name)
        if not strategy:
            log.warning("Unknown strategy: %s", strat_name)
            continue

        for pitcher in pitchers:
            signals = strategy(game, pitcher, None, client, config)
            all_single_leg_signals.extend(signals)
            if execute_signals(
                signals, client, config, circuit,
                effective_bankroll, held_tickers, placed_tickers,
            ):
                any_signals = True

    # ── Batter / player-prop strategies (run once per batter) ─────────────
    away_batters = ctx.away_batters
    home_batters = ctx.home_batters
    batter_pitcher_pairs: List[tuple] = []

    if batter_strats:
        # away batters face the home pitcher; home batters face the away pitcher
        batter_pitcher_pairs = (
            [(b, home_pitch) for b in away_batters]
            + [(b, away_pitch) for b in home_batters]
        )

        if not batter_pitcher_pairs:
            log.info("  Skipping batter strategies — lineups not yet posted")
        else:
            if away_batters or home_batters:
                log.info(
                    "  Lineups: %d away batters, %d home batters confirmed",
                    len(away_batters), len(home_batters),
                )
            for batter, opp_pitcher in batter_pitcher_pairs:
                if circuit.is_tripped():
                    log.warning("⚡ Circuit breaker tripped — stopping")
                    return
                for strat_name in batter_strats:
                    strategy = STRATEGIES.get(strat_name)
                    if not strategy:
                        continue
                    signals = strategy(game, opp_pitcher, batter, client, config)
                    all_single_leg_signals.extend(signals)
                    if execute_signals(
                        signals, client, config, circuit,
                        effective_bankroll, held_tickers, placed_tickers,
                    ):
                        any_signals = True

    # ── Combo / parlay strategy ───────────────────────────────────────────
    if "combo" in config.enabled_strategies and not circuit.is_tripped():
        # Use batter data from context (already fetched by provider)
        combo_bp_pairs = batter_pitcher_pairs
        if not combo_bp_pairs:
            combo_bp_pairs = (
                [(b, home_pitch) for b in away_batters]
                + [(b, away_pitch) for b in home_batters]
            )

        combo_signals = strategy_combo(
            game, client, config,
            away_pitcher=away_pitch,
            home_pitcher=home_pitch,
            batter_pitcher_pairs=combo_bp_pairs,
            single_leg_signals=all_single_leg_signals,
        )
        if execute_signals(
            combo_signals, client, config, circuit,
            effective_bankroll, held_tickers, placed_tickers,
        ):
            any_signals = True

    if not any_signals:
        log.info("  No signals found for any strategy.")


# ─── Settlement ───────────────────────────────────────────────────────────────

def settle_pending(client: KalshiClient, config: Config) -> int:
    """Check Kalshi for outcomes on any unsettled journal trades.

    Returns the number of newly recorded settlements.
    Safe to call repeatedly — skips tickers already in the journal.
    """
    records = journal.load_journal(config.log_dir)
    if not records:
        return 0

    trade_tickers    = {r["ticker"] for r in records if r.get("type") == "trade"}
    settled_tickers  = {r["ticker"] for r in records if r.get("type") == "settlement"}
    pending          = trade_tickers - settled_tickers

    if not pending:
        return 0

    found = 0
    for ticker in sorted(pending):
        try:
            settlements = client.get_settlements(ticker=ticker, limit=10)
        except Exception as exc:
            log.warning("Could not fetch settlements for %s: %s", ticker, exc)
            continue

        if not settlements:
            log.debug("%s — not yet settled", ticker)
            continue

        s = settlements[0]
        result      = s.get("market_result", "")
        revenue_usd = s.get("revenue", 0) / 100.0
        yes_cost    = float(s.get("yes_total_cost_dollars", 0))
        fee         = float(s.get("fee_cost", 0))
        settled_at  = s.get("settled_time", "")

        journal.record_settlement(
            log_dir=config.log_dir,
            ticker=ticker,
            market_result=result,
            revenue_usd=revenue_usd,
            yes_cost_usd=yes_cost,
            fee_usd=fee,
            settled_at=settled_at,
        )
        pnl = revenue_usd - yes_cost - fee
        log.info("  📋 Settled %-45s  result=%-4s  P&L $%+.2f", ticker, result or "?", pnl)
        found += 1

    return found


# ─── Main bot loop ────────────────────────────────────────────────────────────

def run(config: Config, game_filter: Optional[str] = None):
    """Main bot loop — scan, signal, trade.

    Args:
        config: Bot configuration.
        game_filter: If set, only process games matching this team pattern
                     and exit after one pass (useful for testing / dry-runs).
    """
    single_pass = game_filter is not None

    log.info("🚀 Starting Slugger bot (v%s)", __import__("slugger").__version__)
    log.info(
        "Config: dry_run=%s  kelly=%.2f  min_edge=%d¢  poll=%ds%s",
        config.dry_run, config.kelly_fraction,
        config.min_edge_cents, config.poll_interval_sec,
        f"  game_filter={game_filter!r}" if game_filter else "",
    )

    client = config.create_kalshi_client()
    circuit = CircuitBreaker(config)
    provider = LiveMLBDataProvider()

    # Load today's ledger — persists placed tickers across invocations
    placed_tickers: Set[str] = load_ledger(config.log_dir)

    while True:
        if circuit.is_tripped():
            log.error("⚡ Circuit breaker tripped — halting bot.")
            break

        # ── Fetch live balance ──────────────────────────────────────────────
        try:
            balance = client.get_balance()
        except Exception as exc:
            log.error("Could not fetch balance: %s — sleeping %ds", exc, config.poll_interval_sec)
            if single_pass:
                break
            time.sleep(config.poll_interval_sec)
            continue

        log.info("💰 Balance: $%.2f", balance)

        if balance < 0.50:
            log.error("Balance too low ($%.2f) — halting bot.", balance)
            break

        # ── Build dedup set from live positions + ledger ────────────────────
        try:
            positions = client.get_positions()
            api_held = {p.get("ticker", "") for p in positions}
        except Exception as exc:
            log.warning("Could not fetch positions (using ledger only): %s", exc)
            api_held = set()

        held_tickers: Set[str] = api_held | placed_tickers
        if held_tickers:
            log.info("Already holding %d ticker(s) — will skip duplicates", len(held_tickers))

        # ── Fetch and filter games ─────────────────────────────────────────
        try:
            games = get_todays_games()
        except Exception as exc:
            log.error("Could not fetch today's games: %s", exc)
            if single_pass:
                break
            time.sleep(config.poll_interval_sec)
            continue

        active_games = [g for g in games if g.status in ("Pre-Game", "Warmup", "Scheduled")]

        if game_filter:
            filtered = [g for g in active_games if game_matches(g, game_filter)]
            if not filtered:
                # Also search all games (not just active) so pre-game filter works
                filtered = [g for g in games if game_matches(g, game_filter)]
                if filtered:
                    log.info(
                        "Game %s found but status is %r — processing anyway",
                        game_filter, filtered[0].status,
                    )
                else:
                    log.error("No game matching %r found in today's schedule.", game_filter)
                    log.info("Today's games: %s", ", ".join(
                        f"{g.away_abbrev}@{g.home_abbrev}" for g in games
                    ))
                    break
            active_games = filtered

        if not active_games:
            log.info("No active games right now. Sleeping %ds...", config.poll_interval_sec)
            if single_pass:
                break
            time.sleep(config.poll_interval_sec)
            continue

        log.info("Processing %d game(s)...", len(active_games))
        for game in active_games:
            if circuit.is_tripped():
                log.warning("⚡ Circuit breaker tripped mid-scan — stopping.")
                break
            try:
                ctx = provider.hydrate_game(game)
                process_game(
                    ctx, client, config, circuit,
                    bankroll_usd=balance,
                    held_tickers=held_tickers,
                    placed_tickers=placed_tickers,
                )
            except Exception as exc:
                log.error("Error processing %s: %s", game.game_id, exc)

        # Persist ledger after each scan pass
        save_ledger(placed_tickers, config.log_dir)

        # Auto-settle: check for outcomes on any open journal trades
        if not single_pass:
            try:
                n = settle_pending(client, config)
                if n:
                    overall, _ = journal.get_stats(journal.load_journal(config.log_dir))
                    log.info(
                        "📊 Running P&L: $%+.2f  (%dW/%dL, %.0f%% win rate)",
                        overall.total_pnl_usd,
                        overall.wins,
                        overall.losses,
                        (overall.win_rate or 0) * 100,
                    )
            except Exception as exc:
                log.debug("Auto-settle failed: %s", exc)

        if single_pass:
            log.info("— Single-pass complete —")
            break

        log.info("— Scan complete — sleeping %ds", config.poll_interval_sec)
        time.sleep(config.poll_interval_sec)
