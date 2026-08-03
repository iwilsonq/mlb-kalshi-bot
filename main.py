"""Slugger — MLB Kalshi trading bot.

Usage:
    python main.py run                      Start the bot loop (all games)
    python main.py run --game LAD           Single pass, only LAD games
    python main.py run --game SFLAD         Single pass, SF @ LAD specifically
    python main.py status                   Show today's games and market status
    python main.py check                    Test Kalshi API connection
    python main.py settle                   Fetch outcomes for unsettled journal trades
    python main.py stats                    Print win rate / ROI per strategy
    python main.py report                   Brier/log-loss vs market + ROI heatmaps
"""
from __future__ import annotations
import argparse
import logging
import sys
from typing import Optional

from slugger.config import Config
from slugger.mlb_data import get_todays_games
from slugger.game_processor import (
    game_markets,
    run as run_bot,
    settle_pending,
    snapshot_pending_clv,
)
import slugger.journal as journal
from slugger.reporting import build_report, format_report

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
    """Manually fetch settlements and CLV snapshots for journal trades."""
    records = journal.load_journal(config.log_dir)
    if not records:
        log.info("Journal is empty — nothing to settle.")
        return

    trade_tickers   = {r["ticker"] for r in records if r.get("type") == "trade"}
    settled_tickers = {r["ticker"] for r in records if r.get("type") == "settlement"}
    pending         = trade_tickers - settled_tickers

    client = config.create_kalshi_client()
    try:
        clv_n = snapshot_pending_clv(client, config)
        if clv_n:
            log.info("Recorded %d CLV snapshot(s).", clv_n)
    except Exception as exc:
        log.warning("CLV snapshot pass failed: %s", exc)

    if not pending:
        log.info("All %d trade(s) already settled.", len(trade_tickers))
        return

    log.info("Checking %d unsettled ticker(s)...", len(pending))
    found = settle_pending(client, config)
    log.info("Recorded %d new settlement(s). Run 'stats' to see updated P&L.", found)


def cmd_stats(config: Config, date_filter: Optional[str] = None):
    """Print win rate and ROI per strategy from the trade journal.

    Args:
        date_filter: If set, only include trades placed on this date (YYYY-MM-DD).
                     Use "today" for today's date.  Settlements are included
                     if they match a filtered trade's ticker.
    """
    records = journal.load_journal(config.log_dir)
    if not records:
        log.info("No journal records found at %s", config.log_dir)
        return

    if date_filter:
        if date_filter == "today":
            from datetime import date
            date_filter = date.today().isoformat()

        # Keep trades matching the date + settlements for those tickers
        trade_tickers = {
            r["ticker"] for r in records
            if r.get("type") == "trade" and r.get("date") == date_filter
        }
        records = [
            r for r in records
            if (r.get("type") == "trade" and r.get("date") == date_filter)
            or (r.get("type") == "settlement" and r.get("ticker") in trade_tickers)
        ]
        if not records:
            log.info("No trades found for %s", date_filter)
            return
        print(f"  Filtered to: {date_filter}")

    overall, per_strategy = journal.get_stats(records)
    print(journal.format_stats(overall, per_strategy))


def cmd_report(
    config: Config,
    *,
    market_source: str = "mid",
    min_cell_n: int = 5,
    use_ask: bool = False,
):
    """Print Brier/log-loss vs market and ROI heatmaps from journal + signals."""
    signals = journal.load_signals(config.log_dir)
    records = journal.load_journal(config.log_dir)
    if not signals and not records:
        log.info("No signals or journal at %s — run the bot first.", config.log_dir)
        return

    source = "ask" if use_ask else market_source
    report = build_report(
        signals,
        records,
        market_source=source,
        use_calibrated=True,
        min_cell_n=1,  # keep full heatmap; formatter highlights n≥min_cell_n
    )
    print(format_report(report, min_roi_n=min_cell_n))


def cmd_calibrate(config: Config, fit: bool = False):
    """Analyze model calibration and optionally fit calibration curves.

    Without --fit: shows predicted probability vs. actual outcomes by band.
    With --fit: trains isotonic regression per strategy and saves to
    logs/calibration.json for use by the signal pipeline.
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

    # ── Fit mode: train and save calibration curves ────────────────────────
    if fit:
        from slugger.calibration import CalibrationLayer, backfill_outcomes
        from pathlib import Path

        # Backfill pitcher_ks outcomes from MLB game logs (much more complete
        # than Kalshi settlements — covers every evaluated game, not just
        # markets we traded).
        print("Fetching MLB game logs for pitcher_ks backfill...")
        mlb_outcomes = backfill_outcomes(signals)
        n_mlb = sum(len(v) for v in mlb_outcomes.values())
        print(f"  Backfilled {n_mlb} pitcher_ks outcomes from MLB game logs")

        # Walk-forward: exclude today (and lag window) so live gates never
        # peek at same-day outcomes. Retrain daily via this command.
        from slugger.calibration import DEFAULT_CALIBRATION_LAG_DAYS
        cal = CalibrationLayer.fit_walk_forward(
            signals,
            settlements,
            lag_days=DEFAULT_CALIBRATION_LAG_DAYS,
            mlb_outcomes=mlb_outcomes,
        )
        cal_path = str(Path(config.log_dir) / "calibration.json")
        cal.save(cal_path)
        print(cal.format_report())
        print(f"\nCalibration saved to {cal_path} (as_of={cal.as_of}, lag={cal.lag_days}d)")
        print("The bot will load this automatically on next run.")
        print("Retrain schedule: run `python main.py calibrate --fit` once per day.")

        # Fit trained Ks lambda model from **actual** game-log strikeouts
        # (point-in-time features). Holdout reports Brier vs market when prices known.
        # This is the only thing that puts a trained Ks model in front of the
        # live bot. If it fails, models.expected_ks silently falls back to the
        # hand-tuned blend + KS_LAMBDA_DEFLATOR, so the failure must be loud.
        try:
            from slugger.ks_model import fit_and_save_ks_model, format_ks_fit_report
            from slugger.calibration import (
                _fetch_all_game_logs,
                fetch_team_hitting_game_logs,
            )
            from datetime import date, timedelta

            print("Building Ks training samples from MLB game logs (actual Ks)...")
            game_logs = _fetch_all_game_logs(signals)
            # Point-in-time opponent K%: without per-team game logs every row
            # carries the league constant, the feature has no training variance,
            # and the fit zeroes it — so opponent strength would be ignored.
            team_logs = fetch_team_hitting_game_logs()
            as_of = (date.today() - timedelta(days=DEFAULT_CALIBRATION_LAG_DAYS)).isoformat()
            report = fit_and_save_ks_model(
                signals,
                records,
                game_logs=game_logs,
                team_game_logs=team_logs,
                as_of=as_of,
                model_path=str(Path(config.log_dir) / "ks_model.json"),
                cost_buffer_cents=float(config.edge_cost_buffer_cents),
            )
            print(format_ks_fit_report(report))
        except Exception:
            import traceback
            print("\n❌ Ks model fit FAILED — no ks_model.json written.")
            print("   models.expected_ks will keep using the hand-tuned fallback.")
            traceback.print_exc()
            return 1
        return

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
        choices=["run", "status", "check", "settle", "stats", "calibrate", "report"],
        help="Command",
    )
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument(
        "--game", metavar="PATTERN",
        help="Filter to a specific game by team abbreviation (e.g. LAD, SFLAD). "
             "Implies a single pass — exits after one scan.",
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="(calibrate only) Fit isotonic regression curves and save to logs/calibration.json.",
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help='(stats only) Filter to trades placed on this date. Use "today" for today.',
    )
    parser.add_argument(
        "--ask", action="store_true",
        help="(report only) Use ask as market-implied price instead of mid.",
    )
    parser.add_argument(
        "--min-n", type=int, default=5, metavar="N",
        help="(report only) Highlight ROI cells with at least N samples (default 5).",
    )

    args = parser.parse_args()

    # Set up logging with a console handler
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=fmt, level=level, force=True)

    config = Config.from_env()
    log.info("Config loaded: %d strategies enabled", len(config.enabled_strategies))
    log.debug("Config: %s", config)

    if args.command == "run":
        cmd_run(config, game_filter=args.game)
    elif args.command == "calibrate":
        cmd_calibrate(config, fit=args.fit)
    elif args.command == "stats":
        cmd_stats(config, date_filter=args.date)
    elif args.command == "report":
        cmd_report(config, use_ask=args.ask, min_cell_n=args.min_n)
    elif args.command == "check":
        cmd_check(config)
    elif args.command == "status":
        cmd_status(config)
    elif args.command == "settle":
        cmd_settle(config)


if __name__ == "__main__":
    main()
