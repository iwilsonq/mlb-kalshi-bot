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
import logging
import sys

from slugger.config import Config
from slugger.mlb_data import get_todays_games
from slugger.game_processor import (
    CircuitBreaker,
    game_markets,
    process_game,
    run as run_bot,
    settle_pending,
)
import slugger.journal as journal

log = logging.getLogger("slugger")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_price(market: dict) -> str:
    """Format bid/ask from a Kalshi market dict."""
    if "yes_ask_dollars" in market:
        bid = market.get("yes_bid_dollars", "n/a")
        ask = market.get("yes_ask_dollars", "n/a")
        return f"${bid} – ${ask}"
    return "n/a"


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
        mkts = game_markets(client, g, config)
        if not mkts:
            continue
        log.info("\n🎮 %s @ %s (%d markets):", g.away_team, g.home_team, len(mkts))
        for m in mkts[:5]:
            log.info(
                "  %s | %s",
                m.get("title", "")[:60],
                _fmt_price(m),
            )


def cmd_run(config: Config, game_filter=None):
    """Delegate to game_processor.run()."""
    run_bot(config, game_filter=game_filter)


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
    found = settle_pending(client, config)
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
