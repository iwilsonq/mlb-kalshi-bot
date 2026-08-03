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
from typing import List, Optional, Set, Tuple

import requests

from slugger.calibration import CalibrationLayer
from slugger.config import Config
from slugger.kalshi_client import market_quotes
from slugger.mlb_data import LiveMLBDataProvider, get_todays_games
from slugger.signal_pipeline import load_calibration
from slugger.consensus import consensus_allows_trade, load_consensus_prices
from slugger.execution import (
    cancel_resting_orders_for_started_games,
    classify_fill_role,
    limit_price_cents,
)
from slugger.game_state import GameStateTracker
from slugger.risk import GameFactorBudget, StrategyHealthMonitor
from slugger.sizing import DailyRiskBudget, daily_spent_from_journal, kelly_count
from slugger.strategies import STRATEGY_PIPELINE
from slugger.tickers import game_event_ticker, parse_game_time_utc
from slugger.types import GameContext, GameInfo, MarketClient, TradeSignal
import slugger.journal as journal

log = logging.getLogger("slugger")


def game_markets(client: MarketClient, game: GameInfo, config: Config) -> List[dict]:
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
    """Monitors losses and trips the bot if thresholds are exceeded.

    Fed by settlement P&L (not placement cost).  A negative pnl_usd
    means a loss; a positive pnl_usd means a win.
    """

    def __init__(self, config: Config):
        self.max_loss = config.cb_max_loss_usd
        self.max_consec = config.cb_max_consecutive_losses
        self.total_loss = 0.0
        self.consec_losses = 0
        self.tripped = False

    def record_settlement(self, pnl_usd: float):
        """Record a settlement outcome.

        Args:
            pnl_usd: Profit/loss in dollars.  Negative = loss, positive = win.
        """
        if pnl_usd < 0:
            self.total_loss += abs(pnl_usd)
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


# ─── Per-Game Risk Budget ─────────────────────────────────────────────────────

class GameBudget:
    """Tracks per-game signal count and dollar exposure limits.

    Passed through all execute_signals calls within a single game to
    prevent over-concentration in correlated same-game positions.
    """

    def __init__(self, max_signals: int, max_exposure_usd: float):
        self.max_signals = max_signals
        self.max_exposure_usd = max_exposure_usd
        self.signals_placed = 0
        self.exposure_usd = 0.0

    def can_place(self, cost_usd: float = 0.0) -> bool:
        """Return True if the budget allows another signal."""
        if self.signals_placed >= self.max_signals:
            return False
        if self.max_exposure_usd > 0 and self.exposure_usd + cost_usd > self.max_exposure_usd:
            return False
        return True

    def record(self, cost_usd: float) -> None:
        """Record a placed signal against the budget."""
        self.signals_placed += 1
        self.exposure_usd += cost_usd

    @property
    def remaining(self) -> int:
        return max(0, self.max_signals - self.signals_placed)


# ─── Signal Execution ─────────────────────────────────────────────────────────

def execute_signals(
    signals: List[TradeSignal],
    client: MarketClient,
    config: Config,
    circuit: CircuitBreaker,
    effective_bankroll: float,
    held_tickers: Set[str],
    placed_tickers: Set[str],
    budget: Optional["GameBudget"] = None,
    daily: Optional[DailyRiskBudget] = None,
    game_factor: Optional[GameFactorBudget] = None,
) -> bool:
    """Place orders for a list of signals. Returns True if any signal was acted on."""
    any_acted = False
    for signal in signals:
        if circuit.is_tripped():
            return any_acted

        # ── Per-game budget check ──────────────────────────────────────────
        if budget and not budget.can_place():
            log.info(
                "  🛑 Game budget exhausted (%d/%d signals) — skipping remaining",
                budget.signals_placed, budget.max_signals,
            )
            return any_acted

        if daily is not None and daily.max_fraction > 0 and daily.remaining_usd <= 0:
            log.info(
                "  🛑 Daily bankroll fraction cap reached ($%.2f/$%.2f) — skipping remaining",
                daily.spent_usd, daily.cap_usd,
            )
            return any_acted

        any_acted = True

        # ── Dedup check ────────────────────────────────────────────────────
        if signal.ticker in held_tickers:
            log.info("  ⏭ %s | %s — already held, skipping", signal.strategy, signal.ticker)
            continue

        # ── Maker/limit discipline: never bid above fair − buffer ──────────
        if signal.model_prob_pct > 0:
            signal.price = limit_price_cents(
                fair_prob_pct=signal.model_prob_pct,
                ask_cents=signal.ask_cents or signal.price,
                side=signal.side,
                buffer_cents=1,
            )

        # ── Optional consensus prior gate ──────────────────────────────────
        consensus = load_consensus_prices()
        if not consensus_allows_trade(
            signal.ticker, signal.price, consensus,
            min_edge_cents=float(config.min_edge_cents),
        ):
            log.info(
                "  ⏭ %s | %s — blocked by consensus prior (ask %d¢)",
                signal.strategy, signal.ticker, signal.price,
            )
            continue

        # ── Binary Kelly size with bankroll + daily remaining ──────────────
        remaining_daily = daily.remaining_usd if daily is not None and daily.max_fraction > 0 else None
        signal.count = kelly_count(
            signal.edge_cents,
            signal.price,
            config.kelly_fraction,
            config.max_position_usd,
            config.max_contracts_per_trade,
            model_prob_pct=signal.model_prob_pct or None,
            bankroll_usd=effective_bankroll,
            remaining_daily_usd=remaining_daily,
        )
        if signal.count <= 0:
            log.info(
                "  ⏭ %s | %s — sized to 0 contracts (edge/daily/bankroll)",
                signal.strategy, signal.ticker,
            )
            continue

        log.info(
            "  📊 %s | %s | %d contracts @ %d¢ | Edge: %.1f¢ | model=%.1f%%",
            signal.strategy, signal.reason, signal.count,
            signal.price, signal.edge_cents, signal.model_prob_pct,
        )

        cost_est = signal.count * signal.price / 100

        # ── Dollar exposure check ──────────────────────────────────────────
        if budget and not budget.can_place(cost_est):
            log.info(
                "  🛑 Game exposure limit ($%.2f/$%.2f) — skipping %s",
                budget.exposure_usd, budget.max_exposure_usd, signal.ticker,
            )
            continue

        # ── Same-game factor (correlated markets across prop types) ───────
        if game_factor is not None and not game_factor.can_place(signal.ticker, cost_est):
            log.info(
                "  🛑 Game-factor budget for %s exhausted — skipping %s",
                signal.ticker.split("-")[1] if "-" in signal.ticker else signal.ticker,
                signal.ticker,
            )
            continue

        if daily is not None and daily.max_fraction > 0 and not daily.can_spend(cost_est):
            log.info(
                "  🛑 Daily risk remaining $%.2f < cost $%.2f — skipping %s",
                daily.remaining_usd, cost_est, signal.ticker,
            )
            continue

        if config.dry_run:
            log.info(
                "     [DRY RUN] Would BUY %s %s %d × %d¢ = $%.2f",
                signal.side.upper(), signal.ticker, signal.count, signal.price,
                cost_est,
            )
            held_tickers.add(signal.ticker)
            if budget:
                budget.record(cost_est)
            if game_factor is not None:
                game_factor.record(signal.ticker, cost_est)
            if daily is not None:
                daily.record(cost_est)
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
            if result.status in ("accepted", "executed", "resting", "partially_filled"):
                fill_px = getattr(result, "fill_price_cents", 0) or 0
                fill_n = getattr(result, "fill_count", 0) or 0
                # Cost uses fill price when known, else limit
                px_for_cost = fill_px if fill_px > 0 else signal.price
                cost_usd = signal.count * px_for_cost / 100
                fill_role = classify_fill_role(
                    signal.price, fill_px or signal.price, signal.ask_cents or signal.price,
                )
                status_icon = "✅" if result.status == "executed" else "🕐"
                log.info(
                    "     %s Order placed: %s (status: %s) cost=$%.2f"
                    " fill=%s¢ n=%s role=%s",
                    status_icon, result.order_id, result.status, cost_usd,
                    fill_px or "—", fill_n or "—", fill_role,
                )
                held_tickers.add(signal.ticker)
                placed_tickers.add(signal.ticker)
                if budget:
                    budget.record(cost_usd)
                if game_factor is not None:
                    game_factor.record(signal.ticker, cost_usd)
                if daily is not None:
                    daily.record(cost_usd)
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
                    raw_model_prob_pct=signal.raw_model_prob_pct,
                    model_prob_pct=signal.model_prob_pct,
                    gross_edge_cents=signal.gross_edge_cents,
                    cost_buffer_cents=signal.cost_buffer_cents,
                    bid_cents=signal.bid_cents,
                    ask_cents=signal.ask_cents,
                    mid_cents=signal.mid_cents,
                    spread_cents=signal.spread_cents,
                    fill_price_cents=fill_px,
                    fill_count=fill_n,
                    fill_status=result.status,
                    filled_at=result.created_at or "",
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
    client: MarketClient,
    config: Config,
    circuit: CircuitBreaker,
    bankroll_usd: float,
    held_tickers: Set[str],
    placed_tickers: Set[str],
    calibration: Optional[CalibrationLayer] = None,
    daily: Optional[DailyRiskBudget] = None,
    health: Optional[StrategyHealthMonitor] = None,
    game_factor: Optional[GameFactorBudget] = None,
    game_state: Optional[GameStateTracker] = None,
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

    # ── SP scratch / pitcher change within poll cycle ────────────────────
    if game_state is not None:
        reason = game_state.observe(game)
        if reason or game_state.is_invalid(game.game_id):
            log.info(
                "  ⛔ Skipping trade: %s",
                reason or game_state.invalidate_reason(game.game_id),
            )
            return

    log.info(
        "  Pitchers: %s vs %s",
        ctx.away_pitcher.name if ctx.away_pitcher else "TBD",
        ctx.home_pitcher.name if ctx.home_pitcher else "TBD",
    )
    if ctx.away_batters or ctx.home_batters:
        log.info(
            "  Lineups: %d away batters, %d home batters confirmed",
            len(ctx.away_batters), len(ctx.home_batters),
        )

    # Kelly uses full live balance as bankroll; max_position_usd caps per trade.
    effective_bankroll = bankroll_usd if bankroll_usd > 0 else config.max_position_usd
    if bankroll_usd < config.max_position_usd:
        log.info(
            "  ⚠️  Live balance $%.2f < MAX_POSITION_USD $%.2f — "
            "per-trade cap is balance",
            bankroll_usd, config.max_position_usd,
        )

    any_signals = False

    # ── Per-game risk budget ───────────────────────────────────────────────
    budget = GameBudget(
        max_signals=config.max_signals_per_game,
        max_exposure_usd=config.max_exposure_per_game_usd,
    )
    if game_factor is None:
        game_factor = GameFactorBudget(
            max_signals_per_game=config.max_signals_per_game,
            max_exposure_usd=config.max_exposure_per_game_usd,
        )

    # Accumulated signals from all prior strategies — fed to each subsequent
    # strategy so combo (last in the pipeline) can see all single-leg signals.
    all_prior_signals: List[TradeSignal] = []

    # ── Run strategy pipeline in order ─────────────────────────────────────
    base_enabled = set(config.enabled_strategies)
    for strat_name, strat_fn in STRATEGY_PIPELINE:
        if strat_name not in base_enabled:
            continue
        if health is not None and not health.is_enabled(strat_name, base_enabled):
            log.info("  ⛔ %s auto-disabled (rolling health) — skipping", strat_name)
            continue
        if circuit.is_tripped():
            log.warning("⚡ Circuit breaker tripped — stopping")
            return

        signals = strat_fn(ctx, client, config, all_prior_signals, calibration=calibration)
        all_prior_signals.extend(signals)
        if execute_signals(
            signals, client, config, circuit,
            effective_bankroll, held_tickers, placed_tickers,
            budget=budget,
            daily=daily,
            game_factor=game_factor,
        ):
            any_signals = True

    if budget.signals_placed > 0:
        log.info(
            "  📋 Game budget: %d/%d signals placed, $%.2f exposed",
            budget.signals_placed, budget.max_signals, budget.exposure_usd,
        )
    elif not any_signals:
        log.info("  No signals found for any strategy.")


# ─── Settlement ───────────────────────────────────────────────────────────────

def settle_pending(
    client: MarketClient,
    config: Config,
    circuit: Optional[CircuitBreaker] = None,
    health: Optional[StrategyHealthMonitor] = None,
) -> int:
    """Check Kalshi for outcomes on any unsettled journal trades.

    Returns the number of newly recorded settlements.
    Safe to call repeatedly — skips tickers already in the journal.

    If a CircuitBreaker is provided, each settlement's P&L is fed into
    it so the breaker can trip on consecutive or total losses.
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

        settlement_price = None
        if result == "yes":
            settlement_price = 100
        elif result == "no":
            settlement_price = 0

        journal.record_settlement(
            log_dir=config.log_dir,
            ticker=ticker,
            market_result=result,
            revenue_usd=revenue_usd,
            yes_cost_usd=yes_cost,
            fee_usd=fee,
            settled_at=settled_at,
            settlement_price_cents=settlement_price,
        )
        pnl = revenue_usd - yes_cost - fee
        if circuit is not None:
            circuit.record_settlement(pnl)
        if health is not None:
            strat, cost = _trade_cost_for_ticker(records, ticker)
            health.observe(strat, pnl, cost if cost > 0 else yes_cost)
        log.info("  📋 Settled %-45s  result=%-4s  P&L $%+.2f", ticker, result or "?", pnl)
        found += 1

    return found


def _trade_cost_for_ticker(records: List[dict], ticker: str) -> Tuple[str, float]:
    """Return (strategy, cost_usd) for the latest trade on ticker."""
    strategy, cost = "unknown", 0.0
    for r in records:
        if r.get("type") == "trade" and r.get("ticker") == ticker:
            strategy = r.get("strategy", "unknown")
            cost = float(r.get("cost_usd") or 0.0)
    return strategy, cost


# ─── CLV snapshots ────────────────────────────────────────────────────────────

CLV_MIN_HOURS = 1.0
CLV_MAX_HOURS = 48.0  # stop trying after 2 days


def snapshot_pending_clv(
    client: MarketClient,
    config: Config,
    min_hours: float = CLV_MIN_HOURS,
    max_hours: float = CLV_MAX_HOURS,
) -> int:
    """Fetch mid/bid/ask for trades that are ≥ min_hours old and lack a CLV row.

    Returns number of new CLV records written.  Safe to call every bot loop.
    Requires client.get_market(ticker) when available.
    """
    get_market = getattr(client, "get_market", None)
    if not callable(get_market):
        return 0

    records = journal.load_journal(config.log_dir)
    if not records:
        return 0

    trades = [r for r in records if r.get("type") == "trade"]
    clv_done = {r["ticker"] for r in records if r.get("type") == "clv"}
    settled = {r["ticker"] for r in records if r.get("type") == "settlement"}

    now = datetime.now(timezone.utc)
    written = 0

    for trade in trades:
        ticker = trade.get("ticker", "")
        if not ticker or ticker in clv_done:
            continue
        # Still useful after settlement for research, but prefer pre-settle CLV
        placed_raw = trade.get("placed_at") or ""
        if not placed_raw:
            continue
        try:
            placed = datetime.fromisoformat(placed_raw.replace("Z", "+00:00"))
            if placed.tzinfo is None:
                placed = placed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        hours = (now - placed).total_seconds() / 3600.0
        if hours < min_hours:
            continue
        if hours > max_hours and ticker in settled:
            continue  # give up on ancient settled trades without CLV

        try:
            market = get_market(ticker)
        except Exception as exc:
            log.debug("CLV: get_market(%s) failed: %s", ticker, exc)
            continue
        entry_mid = float(trade.get("mid_cents") or 0)
        if entry_mid <= 0:
            entry_ask = float(trade.get("ask_cents") or trade.get("price_cents") or 0)
            entry_bid = float(trade.get("bid_cents") or 0)
            entry_mid = (entry_bid + entry_ask) / 2.0 if entry_bid and entry_ask else entry_ask

        if not market:
            # Market gone (likely settled). After max_hours, still emit a CLV
            # stub so we do not poll forever; mid=0 marks unavailable.
            if hours <= max_hours:
                continue
            journal.record_clv(
                log_dir=config.log_dir,
                ticker=ticker,
                strategy=trade.get("strategy", ""),
                placed_at=placed_raw,
                bid_cents=0,
                ask_cents=0,
                mid_cents=0.0,
                spread_cents=0,
                entry_mid_cents=entry_mid,
                hours_after_entry=hours,
            )
            written += 1
            clv_done.add(ticker)
            continue

        q = market_quotes(market)
        if q["mid_cents"] <= 0 and q["ask_cents"] <= 0:
            continue

        journal.record_clv(
            log_dir=config.log_dir,
            ticker=ticker,
            strategy=trade.get("strategy", ""),
            placed_at=placed_raw,
            bid_cents=int(q["bid_cents"]),
            ask_cents=int(q["ask_cents"]),
            mid_cents=float(q["mid_cents"] or q["ask_cents"]),
            spread_cents=int(q["spread_cents"]),
            entry_mid_cents=entry_mid,
            hours_after_entry=hours,
        )
        written += 1
        clv_done.add(ticker)

    if written:
        log.info("📈 Recorded %d CLV snapshot(s)", written)
    return written


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
        "Config: dry_run=%s  kelly=%.2f  daily_frac=%.0f%%  min_edge=%d¢  "
        "cost_buffer=%d¢  max_game_exposure=$%.2f  strategies=%s  poll=%ds%s",
        config.dry_run, config.kelly_fraction,
        config.max_bankroll_fraction_per_day * 100,
        config.min_edge_cents, config.edge_cost_buffer_cents,
        config.max_exposure_per_game_usd,
        ",".join(config.enabled_strategies),
        config.poll_interval_sec,
        f"  game_filter={game_filter!r}" if game_filter else "",
    )

    client = config.create_kalshi_client()
    circuit = CircuitBreaker(config)
    provider = LiveMLBDataProvider()

    # Load calibration curves (if available) for probability adjustment
    cal_path = str(Path(config.log_dir) / "calibration.json")
    calibration = load_calibration(cal_path)

    # Rolling strategy health from journal history
    health = StrategyHealthMonitor(
        window_n=config.strategy_health_window,
        min_trades=config.strategy_health_min_trades,
        min_roi_pct=config.strategy_health_min_roi_pct,
        max_brier_deficit=config.strategy_health_max_brier_deficit,
    )
    health.load_from_journal(journal.load_journal(config.log_dir))
    if health.disabled:
        log.warning("Strategies auto-disabled at start: %s", ", ".join(sorted(health.disabled)))

    game_state = GameStateTracker()

    # Load today's ledger — persists placed tickers across invocations
    placed_tickers: Set[str] = load_ledger(config.log_dir)

    while True:
        if circuit.is_tripped():
            log.error("⚡ Circuit breaker tripped — halting bot.")
            break

        # ── CLV snapshots for open trades ≥1h old ─────────────────────────
        try:
            snapshot_pending_clv(client, config)
        except Exception as exc:
            log.debug("CLV snapshot pass failed: %s", exc)

        # ── Cancel resting orders for games that have started ─────────────
        cancel_fn = getattr(client, "cancel_order", None)
        get_orders = getattr(client, "get_orders", None)
        if callable(cancel_fn) and callable(get_orders):
            try:
                def _ticker_started(ticker: str) -> bool:
                    dt = parse_game_time_utc(ticker)
                    if dt is None:
                        return False
                    # 5-minute buffer after scheduled start
                    return datetime.now(timezone.utc) >= dt + timedelta(minutes=5)

                orders = get_orders(status="resting") or []
                n_cancel = cancel_resting_orders_for_started_games(
                    orders, _ticker_started, cancel_fn,
                )
                if n_cancel:
                    log.info("Cancelled %d resting order(s) for started games", n_cancel)
            except Exception as exc:
                log.debug("Resting-order cancel pass failed: %s", exc)

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

        # ── Daily bankroll fraction budget ────────────────────────────────
        today = date.today().isoformat()
        already_spent = daily_spent_from_journal(
            journal.load_journal(config.log_dir), today,
        )
        daily = DailyRiskBudget(
            bankroll_usd=balance,
            max_fraction=config.max_bankroll_fraction_per_day,
            spent_usd=already_spent,
        )
        if daily.max_fraction > 0:
            log.info(
                "📅 Daily risk: spent $%.2f / cap $%.2f (%.0f%% of bankroll)",
                daily.spent_usd, daily.cap_usd, daily.max_fraction * 100,
            )

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
                    calibration=calibration,
                    daily=daily,
                    health=health,
                    game_state=game_state,
                )
            except Exception as exc:
                log.error("Error processing %s: %s", game.game_id, exc)

        # Persist ledger after each scan pass
        save_ledger(placed_tickers, config.log_dir)

        # Auto-settle: check for outcomes on any open journal trades
        if not single_pass:
            try:
                n = settle_pending(client, config, circuit=circuit, health=health)
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
