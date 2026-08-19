#!/usr/bin/env python3
"""Does Kalshi overreact to in-game events, relative to a WP anchor?

Phase 0 go/no-go for the in-game mean-reversion trader (mlb-kalshi-bot-5vo
-> 4g6). The hypothesis under test: after a salient play the market pushes
past fair value and reverts, by more than it costs to round-trip.

Method — all of it keyed on three clocks, which is the whole point:

    t_true   MLB's own `about.endTime` for the play
    t_seen   when our GUMBO poller actually received it
    quotes   our local receive time for each Kalshi book update

Overshoot is measured as *excess* move, not raw deviation:

    excess(d) = [mid(t_true+d) - mid(t_true-10)] - 100 * (wp_after - wp_before)

Differencing around the event cancels any persistent level disagreement
between model and market, which matters because the WP model has no team,
pitcher or lineup input and is therefore biased in level by construction.
Signing by the direction of the WP move makes positive = market moved
further than fair = overshoot.

Usage:
    python3 scripts/overshoot_analysis.py                 # today
    python3 scripts/overshoot_analysis.py 2026-08-18
    python3 scripts/overshoot_analysis.py 2026-08-18 2026-08-19
    python3 scripts/overshoot_analysis.py --rebuild 2026-08-18

Reading a slate is ~18M lines / 6.4 GB and takes about a minute, so the
extracted series are cached next to the recording.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slugger.recorder.replay import (  # noqa: E402
    Manifest, load_manifest, load_gumbo, iter_quotes,
)

RECORDER_DIR = Path("logs/recorder")
CACHE_VERSION = 1

# An "event" has to move the anchor enough to be worth a trade at all. 3c is
# roughly the round-trip cost bar (1-2c fees + 1-2c spread), so anything
# below it is uninteresting even if the market were wrong about it.
MIN_DWP_CENTS = 3.0

# Offsets around t_true at which the price path is sampled.
OFFSETS = [-60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 45, 60, 90, 120]


# ─── Extraction (cached) ─────────────────────────────────────────────────────

def _cache_path(day_dir: Path) -> Path:
    return day_dir / f"replay_cache_v{CACHE_VERSION}.pkl"


def extract(day_dir: Path, rebuild: bool = False) -> dict:
    """Pull the game-winner book series + scored game states out of a slate."""
    cache = _cache_path(day_dir)
    if cache.exists() and not rebuild:
        with cache.open("rb") as f:
            return pickle.load(f)

    kalshi = day_dir / "kalshi.jsonl"
    gumbo = day_dir / "gumbo.jsonl"
    if not kalshi.exists() or not gumbo.exists():
        raise FileNotFoundError(f"incomplete recording in {day_dir}")

    print(f"  extracting {day_dir.name} "
          f"({kalshi.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    manifest = load_manifest(kalshi)
    states, plays = load_gumbo(gumbo)

    # Only the home team's game-winner market: its YES settles 1 iff the home
    # team wins, which is exactly what the WP model predicts. No sign flips.
    tickers = [g.home_market for g in manifest.games]
    quotes: Dict[str, List[tuple]] = collections.defaultdict(list)
    for tk, q in iter_quotes(kalshi, tickers):
        quotes[tk].append((q.recv_ts, q.exch_ts_ms, q.yes_bid, q.yes_ask))

    data = {
        "manifest": manifest,
        "states": states,
        "plays": plays,
        "quotes": dict(quotes),
    }
    with cache.open("wb") as f:
        pickle.dump(data, f)
    return data


# ─── Event assembly ──────────────────────────────────────────────────────────

def _iso(s: str) -> float:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Book:
    """As-of top-of-book lookup for one market."""

    def __init__(self, rows: Sequence[tuple]):
        self.ts = [r[0] for r in rows]
        self.rows = rows

    def quote(self, t: float) -> Optional[Tuple[int, int]]:
        i = bisect.bisect_right(self.ts, t) - 1
        if i < 0:
            return None
        _, _, bid, ask = self.rows[i]
        if bid is None or ask is None:
            return None
        return bid, ask

    def mid(self, t: float) -> Optional[float]:
        q = self.quote(t)
        return None if q is None else (q[0] + q[1]) / 2.0


class Event:
    __slots__ = ("game", "play", "wp_before", "wp_after", "t_true", "t_seen", "t_next")

    def __init__(self, game, play, wp_before, wp_after, t_true, t_seen, t_next):
        self.game = game
        self.play = play
        self.wp_before = wp_before
        self.wp_after = wp_after
        self.t_true = t_true
        self.t_seen = t_seen
        self.t_next = t_next

    @property
    def dwp(self) -> float:
        """Fair-value move implied by the anchor, in cents."""
        return (self.wp_after - self.wp_before) * 100.0

    @property
    def latency(self) -> float:
        return self.t_seen - self.t_true

    def clear_until(self, t: float) -> bool:
        """True if t lands before the next play, i.e. fair is still valid."""
        return self.t_next is None or t <= self.t_next


def build_events(data: dict) -> Tuple[List[Event], Dict[str, Book]]:
    by_pk = data["manifest"].by_pk()
    books = {tk: Book(rows) for tk, rows in data["quotes"].items()}
    events: List[Event] = []

    for pk, plays in data["plays"].items():
        game = by_pk.get(pk)
        if not game or game.home_market not in books:
            continue
        states = data["states"].get(pk) or []
        state_ts = [s.recv_ts for s in states]

        for j, play in enumerate(plays):
            if not play.end_time:
                continue
            # gumbo.py writes the state snapshot and then the plays from the
            # same poll under one recv_ts, so the state at the play's own
            # timestamp is the post-play state and its predecessor is pre.
            i = bisect.bisect_right(state_ts, play.recv_ts) - 1
            if i < 1:
                continue
            nxt = next((_iso(q.end_time) for q in plays[j + 1:] if q.end_time), None)
            events.append(Event(
                game=game, play=play,
                wp_before=states[i - 1].wp, wp_after=states[i].wp,
                t_true=_iso(play.end_time), t_seen=play.recv_ts, t_next=nxt,
            ))
    return events, books


# ─── Small stats helpers ─────────────────────────────────────────────────────

def pct(sorted_vals: Sequence[float], f: float) -> float:
    return sorted_vals[int(f * (len(sorted_vals) - 1))]


def ols(xs: Sequence[float], ys: Sequence[float]):
    """Slope of y on x with its standard error. Returns (a, b, se, n)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0 or n < 3:
        return 0.0, 0.0, float("inf"), n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, math.sqrt((ssr / (n - 2)) / sxx), n


def mean_se(v: Sequence[float]) -> Tuple[float, float]:
    if len(v) < 2:
        return (v[0] if v else 0.0), float("inf")
    return statistics.mean(v), statistics.pstdev(v) / math.sqrt(len(v))


def h(title: str) -> None:
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


# ─── Report sections ─────────────────────────────────────────────────────────

def report_latency(events: Sequence[Event], data_list: Sequence[dict]) -> float:
    h("1. HOW LATE IS OUR VIEW OF THE GAME?")
    lat = sorted(e.latency for e in events)
    print(f"MLB play end_time -> our GUMBO receipt   (n={len(lat)})")
    for f in (0.05, 0.25, 0.50, 0.75, 0.90, 0.99):
        print(f"    p{int(f * 100):<3} {pct(lat, f):6.1f}s")
    print(f"    mean {statistics.mean(lat):.1f}s")

    skew = sorted(
        r[0] - r[1] / 1000.0
        for d in data_list for rows in d["quotes"].values() for r in rows if r[1]
    )
    print(f"\nSanity check — our clock vs Kalshi's exchange timestamps "
          f"(n={len(skew):,}):")
    print(f"    p25 {pct(skew, .25):+.3f}s   p50 {pct(skew, .50):+.3f}s   "
          f"p99 {pct(skew, .99):+.3f}s")
    print("    Sub-second. The lag above is real information delay, not a "
          "clock artifact.")
    return statistics.median(lat)


def report_reaction_timing(events: Sequence[Event], books) -> None:
    h("2. WHEN DOES THE MARKET ACTUALLY MOVE?")
    print("Fraction of the market's total 180s repricing completed by\n"
          "t_true+offset, over events where the market moved at least 1c.\n")
    frac = collections.defaultdict(list)
    n_ev = 0
    for e in events:
        if abs(e.dwp) < 5 or not e.clear_until(e.t_true + 120):
            continue
        bk = books[e.game.home_market]
        pre, post = bk.mid(e.t_true - 60), bk.mid(e.t_true + 120)
        if pre is None or post is None or abs(post - pre) < 1:
            continue
        n_ev += 1
        for off in OFFSETS:
            mid = bk.mid(e.t_true + off)
            if mid is not None:
                frac[off].append((mid - pre) / (post - pre))
    print(f"    {'offset':>8} {'median':>8} {'mean':>8}      (n={n_ev} events, "
          f"|dWP|>=5c)")
    for off in OFFSETS:
        v = sorted(frac[off])
        if len(v) < 20:
            continue
        bar = "#" * max(0, min(30, int(round(statistics.median(v) * 30))))
        print(f"    {off:>6}s  {pct(v, .5):>7.2f} {statistics.mean(v):>8.2f}  {bar}")


def report_excess(events: Sequence[Event], books) -> None:
    h("3. OVERSHOOT: MARKET MOVE MINUS FAIR MOVE")
    print("Signed excess in cents. 0 = market moved exactly to the anchor's\n"
          "new fair value. Positive = overshoot (the tradable hypothesis).\n"
          "Negative = the market moved less than the anchor says it should.\n")
    rows = collections.defaultdict(list)
    n_ev = 0
    for e in events:
        if abs(e.dwp) < MIN_DWP_CENTS:
            continue
        bk = books[e.game.home_market]
        base = bk.mid(e.t_true - 10)
        if base is None:
            continue
        n_ev += 1
        sgn = 1.0 if e.dwp > 0 else -1.0
        for off in OFFSETS:
            t = e.t_true + off
            if not e.clear_until(t):
                continue
            mid = bk.mid(t)
            if mid is not None:
                rows[off].append(((mid - base) - e.dwp) * sgn)
    print(f"    {'offset':>8} {'n':>5} {'mean':>8} {'median':>8} {'%>0':>6}"
          f"      (n={n_ev} events, |dWP|>={MIN_DWP_CENTS:.0f}c)")
    for off in OFFSETS:
        v = sorted(rows[off])
        if len(v) < 20:
            continue
        print(f"    {off:>6}s {len(v):>5} {statistics.mean(v):>8.2f} "
              f"{pct(v, .5):>8.2f} {100 * sum(1 for x in v if x > 0) / len(v):>5.0f}%")
    print("\n    At -10s the value is -|dWP| by construction (no reaction yet);")
    print("    it rises toward 0 as the market prices the event in.")


def report_response_slope(events: Sequence[Event], books) -> None:
    h("4. HOW BIG IS THE MARKET'S RESPONSE vs THE ANCHOR'S?")
    xs, ys = [], []
    per_event = collections.defaultdict(lambda: ([], []))
    for e in events:
        if not e.clear_until(e.t_true + 120):
            continue
        bk = books[e.game.home_market]
        m0, m1 = bk.mid(e.t_true - 10), bk.mid(e.t_true + 120)
        if m0 is None or m1 is None:
            continue
        xs.append(e.dwp)
        ys.append(m1 - m0)
        key = e.play.event if e.play.event in (
            "Home Run", "Single", "Double", "Strikeout", "Walk",
            "Groundout", "Flyout") else "other"
        per_event[key][0].append(e.dwp)
        per_event[key][1].append(m1 - m0)

    a, b, se, n = ols(xs, ys)
    print(f"    market move = {a:+.2f} + {b:.3f} x anchor move   "
          f"(se {se:.3f}, t={b / se:.1f}, n={n})\n")
    print(f"    {'event':<12} {'n':>5} {'slope':>7} {'mean|dWP|':>10}")
    for k, (bx, by) in sorted(per_event.items(), key=lambda kv: -len(kv[1][0])):
        if len(bx) < 25:
            continue
        _, sb, _, _ = ols(bx, by)
        print(f"    {k:<12} {len(bx):>5} {sb:>7.3f} "
              f"{statistics.mean([abs(x) for x in bx]):>9.2f}c")
    print("\n    A slope below 1 means the anchor moves further than the market.")
    print("    The WP model has no team, pitcher or lineup input, so the")
    print("    parsimonious reading is model over-response, not market")
    print("    under-response — and the tests below do not depend on which.")


def report_capturable(events: Sequence[Event], books, median_latency: float) -> None:
    h("5. WHAT IS LEFT TO CAPTURE AT A GIVEN LATENCY?")
    print("If we learned of the play `latency` seconds after t_true, how much\n"
          "of the market's eventual move would still be ahead of us?\n")
    totals = []
    remaining = collections.defaultdict(list)
    for e in events:
        if abs(e.dwp) < MIN_DWP_CENTS or not e.clear_until(e.t_true + 120):
            continue
        bk = books[e.game.home_market]
        pre, post = bk.mid(e.t_true - 60), bk.mid(e.t_true + 120)
        if pre is None or post is None:
            continue
        total = post - pre
        totals.append(abs(total))
        if abs(total) < 1:
            continue
        for off in (-10, -5, 0, 5, 10, 20, 30, 60):
            mid = bk.mid(e.t_true + off)
            if mid is not None:
                remaining[off].append((post - mid) / total)
    med_total = statistics.median(totals) if totals else 0.0
    print(f"    median |total market move| on these events: {med_total:.2f}c\n")
    print(f"    {'latency':>9} {'frac left':>10} {'cents left':>11}")
    for off in (-10, -5, 0, 5, 10, 20, 30, 60):
        v = remaining[off]
        if len(v) < 20:
            continue
        f = statistics.median(v)
        print(f"    {off:>7}s  {f:>9.2f} {f * med_total:>10.2f}c")
    print(f"\n    Our median latency is {median_latency:.0f}s. Round-trip cost is")
    print("    1-2c of fees (per the bvc fee audit) plus any spread paid.")


def report_tradable(events: Sequence[Event], books) -> None:
    h("6. THE ACTUAL TRADE: FADE THE RESIDUAL AT FEED-RECEIPT TIME")
    print("Enter when our feed delivers the play — the earliest instant we\n"
          "could act — short the market/anchor residual, exit after `hold`.\n"
          "Gross of fees. 'exec' pays the spread to get in; 'mid' is the\n"
          "unattainable best case where we are always filled at mid.\n")

    def sample(min_resid: float, hold: float, use_exec: bool) -> List[float]:
        out = []
        for e in events:
            if abs(e.dwp) < MIN_DWP_CENTS:
                continue
            t_in, t_out = e.t_seen, e.t_seen + hold
            if not e.clear_until(t_out):
                continue
            bk = books[e.game.home_market]
            q = bk.quote(t_in)
            exit_mid = bk.mid(t_out)
            if q is None or exit_mid is None:
                continue
            bid, ask = q
            mid = (bid + ask) / 2.0
            resid = mid - 100.0 * e.wp_after
            if abs(resid) < min_resid:
                continue
            # Market above the anchor -> sell YES, so we hit the bid.
            entry = (bid if resid > 0 else ask) if use_exec else mid
            out.append((-1.0 if resid > 0 else 1.0) * (exit_mid - entry))
        return out

    print(f"    {'fill':>5} {'|resid|':>8} {'hold':>6} {'n':>5} {'mean c':>8} "
          f"{'se':>7} {'t':>6} {'min detectable':>15}")
    for use_exec in (False, True):
        for min_resid in (3, 5, 8):
            for hold in (60, 120):
                v = sample(min_resid, hold, use_exec)
                if len(v) < 30:
                    continue
                mu, se = mean_se(v)
                print(f"    {'exec' if use_exec else 'mid':>5} "
                      f"{min_resid:>7}c {hold:>5}s {len(v):>5} {mu:>+8.3f} "
                      f"{se:>7.3f} {mu / se:>+6.2f} {2 * se:>13.2f}c")
    print("\n    'min detectable' is the smallest true edge this sample could")
    print("    have separated from zero at t=2. Compare it to the 1-2c cost")
    print("    bar: a null result here is informative, not merely underpowered.")


def report_liquidity(data_list: Sequence[dict]) -> None:
    h("7. LIQUIDITY")
    spreads = sorted(
        r[3] - r[2]
        for d in data_list for rows in d["quotes"].values() for r in rows
        if r[2] is not None and r[3] is not None and r[3] > r[2]
    )
    print(f"    game-winner spread, cents (n={len(spreads):,}):  "
          + "   ".join(f"p{int(f * 100)}={pct(spreads, f):.0f}"
                       for f in (.25, .5, .75, .9)))
    print("    Liquidity is not the binding constraint.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dates", nargs="*", metavar="YYYY-MM-DD",
                    help="slates to analyse (default: today)")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the extraction cache and re-read the raw jsonl")
    ap.add_argument("--min-dwp", type=float, default=MIN_DWP_CENTS,
                    help=f"event threshold in cents (default {MIN_DWP_CENTS})")
    args = ap.parse_args()

    globals()["MIN_DWP_CENTS"] = args.min_dwp

    dates = args.dates or [dt.date.today().strftime("%Y-%m-%d")]
    all_events: List[Event] = []
    all_books: Dict[str, Book] = {}
    data_list = []

    print(f"Overshoot analysis — slates: {', '.join(dates)}")
    for day in dates:
        day_dir = RECORDER_DIR / day
        if not day_dir.exists():
            print(f"  !! no recording at {day_dir}", file=sys.stderr)
            continue
        data = extract(day_dir, rebuild=args.rebuild)
        data_list.append(data)
        events, books = build_events(data)
        print(f"  {day}: {len(data['manifest'].games)} games, "
              f"{len(events)} plays, {sum(len(b.ts) for b in books.values()):,} "
              f"book updates")
        all_events.extend(events)
        all_books.update(books)

    if not all_events:
        print("no events — nothing to analyse", file=sys.stderr)
        return 1

    n_big = sum(1 for e in all_events if abs(e.dwp) >= MIN_DWP_CENTS)
    print(f"\n{len(all_events)} plays, {n_big} with |dWP| >= {MIN_DWP_CENTS:.0f}c")

    median_latency = report_latency(all_events, data_list)
    report_reaction_timing(all_events, all_books)
    report_excess(all_events, all_books)
    report_response_slope(all_events, all_books)
    report_capturable(all_events, all_books, median_latency)
    report_tradable(all_events, all_books)
    report_liquidity(data_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
