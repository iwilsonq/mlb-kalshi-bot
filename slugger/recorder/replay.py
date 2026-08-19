"""Replay a Phase 0 recording: order-book reconstruction + GUMBO alignment.

The recorder writes raw streams (slugger/recorder/recorder.py). This turns
them back into the two series an analysis actually wants:

  * top of book over time, per market, rebuilt from orderbook_snapshot +
    orderbook_delta — not from the `ticker` channel, which is ~25x sparser
    and does not update on every quote change
  * game state over time, per game, with the WP anchor evaluated on it

Everything is keyed on `recv_ts`, our local receive time, because the
question these recordings exist to answer is what *we* could have traded
on, not what the exchange knew. Exchange time (`ts_ms`) is carried along
so latency can be measured against it.

A slate is ~18M lines / 6.4 GB, so the readers stream and pre-filter on raw
bytes before paying for json.loads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Kalshi quotes in dollars ("0.4800"); everything downstream is integer cents.
CENTS = 100


def _cents(price_dollars: str) -> int:
    return int(round(float(price_dollars) * CENTS))


# ─── Manifest ────────────────────────────────────────────────────────────────

@dataclass
class RecordedGame:
    game_pk: int
    away: str
    home: str
    game_datetime: str
    game_event_ticker: str
    total_event_ticker: str

    @property
    def home_market(self) -> str:
        """Ticker whose YES settles 1 iff the home team wins.

        The WP model predicts *home* win probability, so this is the market
        that can be compared to it without flipping signs.
        """
        return f"{self.game_event_ticker}-{self.home}"

    @property
    def away_market(self) -> str:
        return f"{self.game_event_ticker}-{self.away}"


@dataclass
class Manifest:
    date: str
    games: List[RecordedGame]
    markets: Dict[str, List[str]] = field(default_factory=dict)

    def by_pk(self) -> Dict[int, RecordedGame]:
        return {g.game_pk: g for g in self.games}


def load_manifest(kalshi_path: Path) -> Manifest:
    """Read the manifest record the recorder writes as its first line."""
    with Path(kalshi_path).open("rb") as f:
        for raw in f:
            # Probe on the bare word, not on '"type":"manifest"': the latter
            # is coupled to JsonlWriter's compact separators, and a reader
            # that silently finds nothing when they change is a bad trade
            # for the microseconds it saves.
            if b"manifest" not in raw:
                continue
            r = json.loads(raw)
            if r.get("type") != "manifest":
                continue
            return Manifest(
                date=r.get("date", ""),
                games=[
                    RecordedGame(
                        game_pk=g["game_pk"],
                        away=g["away"],
                        home=g["home"],
                        game_datetime=g.get("game_datetime", ""),
                        game_event_ticker=g.get("game_event_ticker") or "",
                        total_event_ticker=g.get("total_event_ticker") or "",
                    )
                    for g in r.get("games", [])
                    if g.get("game_event_ticker")
                ],
                markets=r.get("markets", {}),
            )
    raise ValueError(f"no manifest record in {kalshi_path}")


# ─── Order book ──────────────────────────────────────────────────────────────

@dataclass
class Quote:
    """Top of book at an instant, in cents. None where the side is empty."""
    recv_ts: float
    exch_ts_ms: Optional[int]
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def spread(self) -> Optional[int]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid


class OrderBook:
    """One market's two-sided ladder.

    Kalshi quotes both sides as bids: a resting NO bid at price p is an
    offer to sell YES at 100 - p. So the YES ask is 100 minus the best NO
    bid, which is why the raw feed never sends an 'ask'.
    """

    __slots__ = ("yes", "no")

    def __init__(self) -> None:
        self.yes: Dict[int, float] = {}
        self.no: Dict[int, float] = {}

    def apply_snapshot(self, msg: dict) -> None:
        self.yes = {_cents(p): float(s) for p, s in msg.get("yes_dollars_fp") or []}
        self.no = {_cents(p): float(s) for p, s in msg.get("no_dollars_fp") or []}

    def apply_delta(self, msg: dict) -> None:
        book = self.yes if msg.get("side") == "yes" else self.no
        price = _cents(msg["price_dollars"])
        size = book.get(price, 0.0) + float(msg.get("delta_fp") or 0.0)
        # Deltas can drive a level to (or fractionally below) zero. Dropping
        # empty levels keeps max() honest about where the top of book is.
        if size > 1e-9:
            book[price] = size
        else:
            book.pop(price, None)

    def top(self, recv_ts: float, exch_ts_ms: Optional[int] = None) -> Quote:
        yes_bid = max(self.yes) if self.yes else None
        no_bid = max(self.no) if self.no else None
        yes_ask = (CENTS - no_bid) if no_bid is not None else None
        return Quote(
            recv_ts=recv_ts,
            exch_ts_ms=exch_ts_ms,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            yes_bid_size=self.yes.get(yes_bid, 0.0) if yes_bid is not None else 0.0,
            yes_ask_size=self.no.get(no_bid, 0.0) if no_bid is not None else 0.0,
        )


@dataclass
class Trade:
    recv_ts: float
    exch_ts_ms: Optional[int]
    ticker: str
    yes_price: int
    count: float
    taker_side: str   # "yes" | "no"


def iter_quotes(
    kalshi_path: Path,
    tickers: Sequence[str],
    *,
    collect_trades: Optional[List[Trade]] = None,
) -> Iterator[Tuple[str, Quote]]:
    """Stream top-of-book changes for `tickers`, in receive order.

    Yields (ticker, Quote) only when the top of book actually moved — the
    great majority of deltas are deep-ladder noise that no strategy could
    have acted on.

    If `collect_trades` is given, trades on those tickers are appended to it
    during the same pass (the file is far too big to want a second one).
    """
    wanted: Set[str] = set(tickers)
    # Pre-filter on bytes: ~17.5M of 18.5M lines in a slate are deltas for
    # markets we did not ask about, and json.loads on all of them costs
    # minutes. The ticker name is the only probe — filtering on message type
    # too would couple this reader to JsonlWriter's exact separators for a
    # saving of a few hundred thousand parses.
    probes = [t.encode() for t in wanted]

    books: Dict[str, OrderBook] = {t: OrderBook() for t in wanted}
    last: Dict[str, Tuple] = {}

    with Path(kalshi_path).open("rb") as f:
        for raw in f:
            if not any(p in raw for p in probes):
                continue
            rec = json.loads(raw)
            msg = rec.get("msg") or {}
            ticker = msg.get("market_ticker")
            if ticker not in wanted:
                continue

            kind = rec.get("type")
            recv_ts = rec.get("recv_ts")
            exch_ts_ms = msg.get("ts_ms")

            if kind == "trade":
                if collect_trades is not None:
                    collect_trades.append(Trade(
                        recv_ts=recv_ts,
                        exch_ts_ms=exch_ts_ms,
                        ticker=ticker,
                        yes_price=_cents(msg["yes_price_dollars"]),
                        count=float(msg.get("count_fp") or 0.0),
                        taker_side=msg.get("taker_side", ""),
                    ))
                continue

            book = books[ticker]
            if kind == "orderbook_snapshot":
                book.apply_snapshot(msg)
            elif kind == "orderbook_delta":
                book.apply_delta(msg)
            else:
                continue

            q = book.top(recv_ts, exch_ts_ms)
            key = (q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size)
            if last.get(ticker) == key:
                continue
            last[ticker] = key
            yield ticker, q


# ─── GUMBO side ──────────────────────────────────────────────────────────────

@dataclass
class StateSample:
    recv_ts: float
    game_pk: int
    status: str
    state: dict
    wp: float          # home win probability under the WP anchor


@dataclass
class PlaySample:
    recv_ts: float
    game_pk: int
    at_bat_index: int
    inning: Optional[int]
    half: str
    end_time: str      # MLB's own timestamp for when the play completed
    event: str
    event_type: str
    description: str
    away_score: Optional[int]
    home_score: Optional[int]


def load_gumbo(
    gumbo_path: Path,
) -> Tuple[Dict[int, List[StateSample]], Dict[int, List[PlaySample]]]:
    """Load per-game state and play series, states scored by the WP anchor."""
    from slugger.wp import get_wp

    states: Dict[int, List[StateSample]] = {}
    plays: Dict[int, List[PlaySample]] = {}

    with Path(gumbo_path).open("rb") as f:
        for raw in f:
            r = json.loads(raw)
            pk = r.get("game_pk")
            kind = r.get("type")
            if kind == "gumbo_state":
                st = r.get("state") or {}
                # Pre-first-pitch snapshots have no innings played and no
                # meaningful WP; the market is trading pregame priors, which
                # this model does not represent.
                if r.get("status") != "Live":
                    continue
                states.setdefault(pk, []).append(StateSample(
                    recv_ts=r["recv_ts"],
                    game_pk=pk,
                    status=r.get("status", ""),
                    state=st,
                    wp=get_wp(st),
                ))
            elif kind == "gumbo_play":
                plays.setdefault(pk, []).append(PlaySample(
                    recv_ts=r["recv_ts"],
                    game_pk=pk,
                    at_bat_index=r.get("at_bat_index", -1),
                    inning=r.get("inning"),
                    half=r.get("half", ""),
                    end_time=r.get("end_time", "") or "",
                    event=r.get("event", "") or "",
                    event_type=r.get("event_type", "") or "",
                    description=r.get("description", "") or "",
                    away_score=r.get("away_score"),
                    home_score=r.get("home_score"),
                ))

    for series in states.values():
        series.sort(key=lambda s: s.recv_ts)
    for series in plays.values():
        series.sort(key=lambda p: p.recv_ts)
    return states, plays


# ─── Time-indexed lookup ─────────────────────────────────────────────────────

class Series:
    """As-of lookup over a time-ordered list of (ts, value).

    Strictly backward-looking: `at(t)` returns the last value observed at or
    before t, or None if nothing had been seen yet. That is the only lookup
    an honest replay can use — anything else leaks the future into a
    decision.
    """

    def __init__(self, samples: Iterable[Tuple[float, object]]):
        pairs = sorted(samples, key=lambda kv: kv[0])
        self.ts = [t for t, _ in pairs]
        self.vals = [v for _, v in pairs]

    def __len__(self) -> int:
        return len(self.ts)

    def at(self, t: float):
        import bisect
        i = bisect.bisect_right(self.ts, t) - 1
        return self.vals[i] if i >= 0 else None

    def next_after(self, t: float):
        import bisect
        i = bisect.bisect_right(self.ts, t)
        return self.vals[i] if i < len(self.vals) else None
