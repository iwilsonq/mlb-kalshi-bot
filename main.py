"""Slugger — MLB Kalshi trading bot.

Usage:
    python main.py run                      Start the bot loop (all games)
    python main.py run --game LAD           Single pass, only LAD games
    python main.py run --game SFLAD         Single pass, SF @ LAD specifically
    python main.py status                   Show today's games and market status
    python main.py check                    Test Kalshi API connection
    python main.py settle                   Fetch outcomes for unsettled journal trades
    python main.py stats                    Print win rate / ROI per strategy
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Set

from slugger.config import Config
from slugger.kalshi_client import KalshiClient
from slugger.mlb_data import (
    BatterProfile, get_batter_profile, get_lineup,
    get_pitcher_profile, get_todays_games,
)
from slugger.strategies import BATTER_STRATEGIES, STRATEGIES, TradeSignal, strategy_combo
import slugger.journal as journal

log = logging.getLogger("slugger")


def _setup_logging(verbose: bool = False):
    """Configure logging with console handler."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=fmt, level=level, force=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_price(market: dict) -> str:
    """Format bid/ask from a Kalshi market dict."""
    if "yes_ask_dollars" in market:
        bid = market.get("yes_bid_dollars", "n/a")
        ask = market.get("yes_ask_dollars", "n/a")
        return f"${bid} – ${ask}"
    return "n/a"


def _game_event_ticker(game: "GameInfo") -> Optional[str]:
    """Construct a Kalshi game event ticker from GameInfo.

    Kalshi format: KXMLBGAME-{YYMONDDHHMM}{AWAY}{HOME}
    Example:   KXMLBGAME-26MAY111810LAACLE

    MLB API times are UTC; Kalshi tickers use ET times.
    """
    if not game.game_datetime:
        return None

    try:
        dt = datetime.fromisoformat(game.game_datetime.replace("Z", "+00:00"))
        et = timezone(timedelta(hours=-4))
        dt_et = dt.astimezone(et)
        date_str = dt_et.strftime("%y%b%d").upper()  # e.g. 26MAY11
        time_str = dt_et.strftime("%H%M")             # e.g. 1810
    except (ValueError, TypeError):
        return None

    # Map MLB Stats API abbreviations → Kalshi codes where they differ.
    API_TO_KALSHI = {
        "SFG": "SF",   "KCR": "KC",  "SDP": "SD",
        "TBR": "TB",   "WSN": "WSH",
        # Legacy fallbacks
        "SAN": "SF",   "SFN": "SF",  "LAN": "LAD",
        "SDN": "SD",   "SLN": "STL", "ANA": "LAA",
        "TAM": "TB",   "NEW": "NYY", "LOS": "LAA",
    }
    away = API_TO_KALSHI.get(game.away_abbrev.upper(), game.away_abbrev.upper())
    home = API_TO_KALSHI.get(game.home_abbrev.upper(), game.home_abbrev.upper())

    return f"KXMLBGAME-{date_str}{time_str}{away}{home}"


def _game_markets(client: KalshiClient, game: "GameInfo", config: Config) -> List[dict]:
    """Fetch markets for a single game via its Kalshi event ticker."""
    event_ticker = _game_event_ticker(game)
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

def _ledger_path(log_dir: str) -> Path:
    """Return path to today's trade ledger file."""
    today = date.today().isoformat()
    return Path(log_dir) / f"placed_{today}.json"


def _load_ledger(log_dir: str) -> Set[str]:
    """Load today's set of placed tickers from disk."""
    path = _ledger_path(log_dir)
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


def _save_ledger(tickers: Set[str], log_dir: str) -> None:
    """Persist today's set of placed tickers to disk."""
    path = _ledger_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(tickers), indent=2))


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_check(config: Config):
    """Test Kalshi API connectivity."""
    client = config.create_kalshi_client()
    try:
        balance = client.get_balance()
        log.info("✅ Connected to Kalshi — Balance: $%.2f", balance)

        positions = client.get_positions()
        log.info("Open positions: %d", len(positions))
        for p in positions:
            log.info("  %s", p.get("ticker", "?"))
    except Exception as e:
        log.error("❌ Connection failed: %s", e)
        sys.exit(1)


def cmd_status(config: Config):
    """Show today's MLB games and relevant market info."""
    games = get_todays_games()
    if not games:
        log.info("No games scheduled today.")
        return

    log.info("Today's MLB games (%d):", len(games))
    for g in games:
        log.info(
            "  %s @ %s | %s | Pitchers: %s vs %s",
            g.away_team, g.home_team,
            g.game_datetime[:16] if g.game_datetime else "TBD",
            g.away_pitcher_name, g.home_pitcher_name,
        )

    client = config.create_kalshi_client()
    for g in games:
        game_markets = _game_markets(client, g, config)
        if not game_markets:
            continue
        log.info("\n🎮 %s @ %s (%d markets):", g.away_team, g.home_team, len(game_markets))
        for m in game_markets[:5]:
            log.info(
                "  %s | %s",
                m.get("title", "")[:60],
                _fmt_price(m),
            )


def _fetch_lineup_profiles(
    game: "GameInfo",
    config: Config,
) -> tuple:
    """Fetch BatterProfiles for both confirmed lineups.

    Returns:
        (away_batters, home_batters) — each a list of BatterProfile.
        Either list may be empty if the lineup is not yet confirmed.

    Batter profiles are fetched in parallel using a thread pool for
    significant speedup (18 batters x 4 API calls each).
    """
    away_batters: List[BatterProfile] = []
    home_batters: List[BatterProfile] = []

    if not game.game_id:
        return away_batters, home_batters

    # Fetch both lineups (these are cheap — single API call each)
    try:
        away_lineup = get_lineup(game.game_id, team="away")
        home_lineup = get_lineup(game.game_id, team="home")
    except Exception as exc:
        log.warning("Lineup fetch failed for game %s: %s", game.game_id, exc)
        return away_batters, home_batters

    if not away_lineup.confirmed and not home_lineup.confirmed:
        log.info("  Lineups not yet posted — batter strategies will be skipped")
        return away_batters, home_batters

    # Collect all unique player IDs across both lineups
    away_pids = [b.get("player_id") for b in (away_lineup.batters or []) if b.get("player_id")]
    home_pids = [b.get("player_id") for b in (home_lineup.batters or []) if b.get("player_id")]
    all_pids = list(set(away_pids + home_pids))

    # ── Fetch all batter profiles in parallel ──────────────────────────────
    profile_cache: dict = {}
    if all_pids:
        with ThreadPoolExecutor(max_workers=min(len(all_pids), 10)) as pool:
            futures = {pool.submit(get_batter_profile, pid): pid for pid in all_pids}
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    profile_cache[pid] = fut.result()
                except Exception as exc:
                    log.debug("Could not fetch batter profile %d: %s", pid, exc)

    # Reconstruct ordered lists from the cache
    if away_lineup.confirmed:
        for pid in away_pids:
            if pid in profile_cache:
                away_batters.append(profile_cache[pid])
    if home_lineup.confirmed:
        for pid in home_pids:
            if pid in profile_cache:
                home_batters.append(profile_cache[pid])

    if away_batters or home_batters:
        log.info(
            "  Lineups: %d away batters, %d home batters confirmed",
            len(away_batters), len(home_batters),
        )
    else:
        log.info("  Lineups not yet posted — batter strategies will be skipped")

    return away_batters, home_batters


def _game_matches(game: "GameInfo", pattern: str) -> bool:
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
        p in game.away_abbrev.upper() or
        p in game.home_abbrev.upper() or
        p in combined
    )


def cmd_run(config: Config, game_filter: Optional[str] = None):
    """Main bot loop — scan, signal, trade.

    Args:
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

    # Load today's ledger — persists placed tickers across invocations
    placed_tickers: Set[str] = _load_ledger(config.log_dir)

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
            filtered = [g for g in active_games if _game_matches(g, game_filter)]
            if not filtered:
                # Also search all games (not just active) so pre-game filter works
                filtered = [g for g in games if _game_matches(g, game_filter)]
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
                process_game(
                    game, client, config, circuit,
                    bankroll_usd=balance,
                    held_tickers=held_tickers,
                    placed_tickers=placed_tickers,
                )
            except Exception as exc:
                log.error("Error processing %s: %s", game.game_id, exc)

        # Persist ledger after each scan pass
        _save_ledger(placed_tickers, config.log_dir)

        # Auto-settle: check for outcomes on any open journal trades
        if not single_pass:
            try:
                n = _settle_pending(client, config)
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


def _execute_signals(
    signals: List[TradeSignal],
    client: KalshiClient,
    config: Config,
    circuit: "CircuitBreaker",
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


def _game_has_started(game: "GameInfo", buffer_minutes: int = 5) -> bool:
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


def process_game(
    game: "GameInfo",
    client: KalshiClient,
    config: Config,
    circuit: "CircuitBreaker",
    bankroll_usd: float,
    held_tickers: Set[str],
    placed_tickers: Set[str],
):
    """Run all strategies for a single game and execute trades."""
    log.info("\n🔍 %s @ %s [%s]", game.away_team, game.home_team, game.status)

    # ── Hard gate: refuse to trade if game has already started ──────────
    if _game_has_started(game):
        log.info("  ⛔ Game has started (past scheduled time + buffer) — skipping entirely")
        return

    # ── Pitcher profiles (fetched in parallel) ────────────────────────────
    away_pitch = None
    home_pitch = None
    with ThreadPoolExecutor(max_workers=2) as pitch_pool:
        futures = {}
        if game.away_pitcher_id:
            futures["away"] = pitch_pool.submit(get_pitcher_profile, game.away_pitcher_id)
        if game.home_pitcher_id:
            futures["home"] = pitch_pool.submit(get_pitcher_profile, game.home_pitcher_id)
        if "away" in futures:
            try:
                away_pitch = futures["away"].result()
            except Exception as exc:
                log.warning("Away pitcher profile failed: %s", exc)
        if "home" in futures:
            try:
                home_pitch = futures["home"].result()
            except Exception as exc:
                log.warning("Home pitcher profile failed: %s", exc)
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

    # ── Pitcher / game-level strategies (run once per pitcher) ────────────────
    # Run each strategy for both the away and home starter so markets for each
    # pitcher are evaluated with that pitcher's own λ, not a shared one.
    #
    # When the game is in progress, check the live feed to skip any pitcher
    # who has already been pulled — their K total is final and buying into
    # an unmet threshold is a guaranteed loss.
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
            if _execute_signals(
                signals, client, config, circuit,
                effective_bankroll, held_tickers, placed_tickers,
            ):
                any_signals = True

    # ── Batter / player-prop strategies (run once per batter) ─────────────
    if batter_strats:
        away_batters, home_batters = _fetch_lineup_profiles(game, config)

        # away batters face the home pitcher; home batters face the away pitcher
        batter_pitcher_pairs: List[tuple] = (
            [(b, home_pitch) for b in away_batters] +
            [(b, away_pitch) for b in home_batters]
        )

        if not batter_pitcher_pairs:
            log.info("  Skipping batter strategies — lineups not yet posted")
        else:
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
                    if _execute_signals(
                        signals, client, config, circuit,
                        effective_bankroll, held_tickers, placed_tickers,
                    ):
                        any_signals = True

    # ── Combo / parlay strategy ───────────────────────────────────────────
    # Runs after all single-leg strategies so it can see every signal from
    # this game and build the best multi-leg combinations.
    if "combo" in config.enabled_strategies and not circuit.is_tripped():
        combo_signals = strategy_combo(game, all_single_leg_signals, client, config)
        if _execute_signals(
            combo_signals, client, config, circuit,
            effective_bankroll, held_tickers, placed_tickers,
        ):
            any_signals = True

    if not any_signals:
        log.info("  No signals found for any strategy.")


# ─── Settlement & Stats commands ──────────────────────────────────────────────

def _settle_pending(client: KalshiClient, config: Config) -> int:
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


def cmd_settle(config: Config):
    """Manually fetch and record settlements for all unsettled journal trades."""
    records = journal.load_journal(config.log_dir)
    if not records:
        log.info("Journal is empty — nothing to settle.")
        return

    trade_tickers   = {r["ticker"] for r in records if r.get("type") == "trade"}
    settled_tickers = {r["ticker"] for r in records if r.get("type") == "settlement"}
    pending         = trade_tickers - settled_tickers

    if not pending:
        log.info("All %d trade(s) already settled.", len(trade_tickers))
        return

    log.info("Checking %d unsettled ticker(s)...", len(pending))
    client = config.create_kalshi_client()
    found = _settle_pending(client, config)
    log.info("Recorded %d new settlement(s). Run 'stats' to see updated P&L.", found)


def cmd_stats(config: Config):
    """Print win rate and ROI per strategy from the trade journal."""
    records = journal.load_journal(config.log_dir)
    if not records:
        log.info("No journal records found at %s", config.log_dir)
        return

    overall, per_strategy = journal.get_stats(records)
    print(journal.format_stats(overall, per_strategy))


def cmd_calibrate(config: Config):
    """Analyze model calibration: predicted probability vs. actual outcomes.

    Joins signals.jsonl against journal.jsonl settlements to bucket
    predictions into probability bands and show how often they actually hit.
    This reveals whether the model is over- or under-confident at each level.
    """
    signals = journal.load_signals(config.log_dir)
    if not signals:
        print("No signal records found. Run the bot to generate signals.jsonl.")
        return

    # Build settlement lookup from the trade journal
    records = journal.load_journal(config.log_dir)
    settlements = {
        r["ticker"]: r for r in records if r.get("type") == "settlement"
    }

    # Bucket signals by strategy and probability band
    from collections import defaultdict
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
             (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]

    def band_label(lo, hi):
        return f"{lo:>2}-{min(hi, 100):>3}%"

    for strategy in ["overall", "pitcher_ks", "player_hits", "player_hr", "game_winner", "total_runs"]:
        strat_signals = (
            signals if strategy == "overall"
            else [s for s in signals if s["strategy"] == strategy]
        )
        if not strat_signals:
            continue

        print(f"\n{'=' * 78}")
        print(f"  CALIBRATION: {strategy}")
        print(f"{'=' * 78}")
        print(f"  {'Predicted':>10}  {'Signals':>8}  {'Settled':>8}  {'Wins':>6}  "
              f"{'Actual%':>8}  {'Avg Pred':>9}  {'Gap':>6}")
        print(f"  {'-' * 68}")

        for lo, hi in bands:
            bucket = [s for s in strat_signals if lo <= s["model_prob_pct"] < hi]
            if not bucket:
                continue

            settled_count = 0
            wins = 0
            pred_sum = 0
            for s in bucket:
                pred_sum += s["model_prob_pct"]
                sett = settlements.get(s["ticker"])
                if sett:
                    settled_count += 1
                    if sett["market_result"] == "yes":
                        wins += 1

            avg_pred = pred_sum / len(bucket)
            if settled_count > 0:
                actual_pct = wins / settled_count * 100
                gap = actual_pct - avg_pred
                print(
                    f"  {band_label(lo, hi):>10}  {len(bucket):>8}  {settled_count:>8}  "
                    f"{wins:>6}  {actual_pct:>7.1f}%  {avg_pred:>8.1f}%  {gap:>+5.1f}%"
                )
            else:
                print(
                    f"  {band_label(lo, hi):>10}  {len(bucket):>8}  "
                    f"{'—':>8}  {'—':>6}  {'—':>8}  {avg_pred:>8.1f}%  {'—':>6}"
                )

        # Summary row
        total = len(strat_signals)
        total_settled = sum(1 for s in strat_signals if s["ticker"] in settlements)
        total_wins = sum(
            1 for s in strat_signals
            if settlements.get(s["ticker"], {}).get("market_result") == "yes"
        )
        traded_count = sum(1 for s in strat_signals if s.get("traded"))
        skipped_count = total - traded_count
        skipped_wins = sum(
            1 for s in strat_signals
            if not s.get("traded")
            and settlements.get(s["ticker"], {}).get("market_result") == "yes"
        )

        print(f"  {'-' * 68}")
        print(f"  Total signals: {total}  (traded: {traded_count}, skipped: {skipped_count})")
        if total_settled:
            print(f"  Settled: {total_settled}  wins: {total_wins}  "
                  f"({total_wins/total_settled*100:.1f}% actual)")
        if skipped_count and skipped_wins:
            print(f"  Skipped signals that WON: {skipped_wins}  (missed value)")
        print()


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
                log.warning("⚡ Circuit breaker TRIPPED: $%.2f lost, %d consecutive losses",
                            self.total_loss, self.consec_losses)
        else:
            self.consec_losses = 0

    def is_tripped(self) -> bool:
        return self.tripped


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="slugger", description="MLB Kalshi trading bot")
    parser.add_argument(
        "command",
        choices=["run", "status", "check", "settle", "stats", "calibrate"],
        help="Command",
    )
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--game", metavar="PATTERN",
        help="Filter to a specific game by team abbreviation (e.g. LAD, SFLAD). "
             "Implies a single pass — exits after one scan.",
    )

    args = parser.parse_args()

    # Set up logging with a console handler
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=fmt, level=level, force=True)

    config = Config.from_env()
    log.info("Config loaded: %d strategies enabled", len(config.enabled_strategies))
    log.debug("Config: %s", config)

    commands = {
        "check": cmd_check,
        "status": cmd_status,
        "settle": cmd_settle,
        "stats": cmd_stats,
        "calibrate": cmd_calibrate,
    }
    if args.command == "run":
        cmd_run(config, game_filter=args.game)
    else:
        commands[args.command](config)


if __name__ == "__main__":
    main()
