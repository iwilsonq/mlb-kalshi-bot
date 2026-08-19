"""Tests for Phase 0 recording replay (slugger/recorder/replay.py).

The overshoot analysis is only as trustworthy as the book reconstruction
underneath it, and a book rebuilt from deltas fails silently: it just drifts
into a slightly wrong price and every downstream number inherits the error.
"""
from __future__ import annotations

import json

import pytest

from slugger.recorder.replay import (
    OrderBook,
    Series,
    iter_quotes,
    load_gumbo,
    load_manifest,
)


def _snapshot(yes, no):
    return {
        "market_ticker": "T",
        "yes_dollars_fp": [[f"{p / 100:.4f}", f"{s:.2f}"] for p, s in yes],
        "no_dollars_fp": [[f"{p / 100:.4f}", f"{s:.2f}"] for p, s in no],
    }


def _delta(price_cents, delta, side):
    return {
        "market_ticker": "T",
        "price_dollars": f"{price_cents / 100:.4f}",
        "delta_fp": f"{delta:.2f}",
        "side": side,
    }


# ─── Order book ──────────────────────────────────────────────────────────────

def test_yes_ask_is_complement_of_best_no_bid():
    """Kalshi never sends an ask; both sides arrive as bids."""
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(47, 100), (48, 50)], no=[(50, 10), (51, 20)]))
    q = book.top(recv_ts=1.0)
    assert q.yes_bid == 48
    assert q.yes_ask == 49          # 100 - best no bid of 51
    assert q.spread == 1
    assert q.mid == 48.5
    assert q.yes_bid_size == 50
    assert q.yes_ask_size == 20     # size resting on the no side at 51


def test_delta_moves_top_of_book():
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(48, 50)], no=[(51, 20)]))
    book.apply_delta(_delta(49, 30, "yes"))
    assert book.top(1.0).yes_bid == 49


def test_exhausted_level_is_removed_not_left_at_zero():
    """A level drained to zero must stop counting as the top of book."""
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(48, 50), (49, 30)], no=[(51, 20)]))
    book.apply_delta(_delta(49, -30, "yes"))
    assert book.top(1.0).yes_bid == 48
    assert 49 not in book.yes


def test_negative_residual_size_is_treated_as_empty():
    """Fractional sizes mean a level can land marginally below zero."""
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(48, 50), (49, 30)], no=[(51, 20)]))
    book.apply_delta(_delta(49, -30.0001, "yes"))
    assert book.top(1.0).yes_bid == 48


def test_snapshot_replaces_rather_than_merges():
    """A mid-stream snapshot after a reconnect must not be merged into stale
    state, or the book keeps levels the exchange has already dropped."""
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(48, 50), (60, 5)], no=[(51, 20)]))
    book.apply_snapshot(_snapshot(yes=[(48, 50)], no=[(51, 20)]))
    assert book.top(1.0).yes_bid == 48


def test_empty_side_yields_no_mid():
    book = OrderBook()
    book.apply_snapshot(_snapshot(yes=[(48, 50)], no=[]))
    q = book.top(1.0)
    assert q.yes_ask is None
    assert q.mid is None
    assert q.spread is None


# ─── as-of Series ────────────────────────────────────────────────────────────

def test_series_is_strictly_backward_looking():
    s = Series([(10.0, "a"), (20.0, "b"), (30.0, "c")])
    assert s.at(9.9) is None        # nothing observed yet — no peeking
    assert s.at(10.0) == "a"        # inclusive at the observation instant
    assert s.at(19.9) == "a"
    assert s.at(20.0) == "b"
    assert s.at(999) == "c"
    assert s.next_after(20.0) == "c"
    assert s.next_after(30.0) is None


def test_series_sorts_unordered_input():
    s = Series([(30.0, "c"), (10.0, "a"), (20.0, "b")])
    assert s.at(25.0) == "b"


# ─── Manifest + file readers ────────────────────────────────────────────────

def _write_recording(tmp_path):
    kalshi = tmp_path / "kalshi.jsonl"
    lines = [
        {"src": "recorder", "type": "manifest", "date": "2026-08-18",
         "games": [{"game_pk": 1, "away": "NYY", "home": "BAL",
                    "game_datetime": "2026-08-18T22:35:00Z",
                    "game_event_ticker": "KXMLBGAME-26AUG181835NYYBAL",
                    "total_event_ticker": "KXMLBTOTAL-26AUG181835NYYBAL"}],
         "markets": {}},
        {"recv_ts": 1.0, "type": "orderbook_snapshot", "src": "kalshi",
         "msg": dict(_snapshot([(48, 50)], [(51, 20)]),
                     market_ticker="KXMLBGAME-26AUG181835NYYBAL-BAL")},
        # deep-ladder change: top of book unmoved, must not be yielded
        {"recv_ts": 2.0, "type": "orderbook_delta", "src": "kalshi",
         "msg": dict(_delta(10, 5, "yes"),
                     market_ticker="KXMLBGAME-26AUG181835NYYBAL-BAL")},
        {"recv_ts": 3.0, "type": "orderbook_delta", "src": "kalshi",
         "msg": dict(_delta(49, 5, "yes"),
                     market_ticker="KXMLBGAME-26AUG181835NYYBAL-BAL")},
        # a market we did not ask for
        {"recv_ts": 4.0, "type": "orderbook_delta", "src": "kalshi",
         "msg": dict(_delta(20, 5, "yes"),
                     market_ticker="KXMLBTOTAL-26AUG181835NYYBAL-8")},
    ]
    kalshi.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return kalshi


def test_load_manifest_and_market_naming(tmp_path):
    m = load_manifest(_write_recording(tmp_path))
    assert m.date == "2026-08-18"
    g = m.games[0]
    # YES on the home market settles 1 iff the home team wins, which is what
    # the WP model predicts — analysis compares them without a sign flip.
    assert g.home_market == "KXMLBGAME-26AUG181835NYYBAL-BAL"
    assert g.away_market == "KXMLBGAME-26AUG181835NYYBAL-NYY"


def test_load_manifest_raises_when_absent(tmp_path):
    p = tmp_path / "kalshi.jsonl"
    p.write_text('{"type":"ticker"}\n')
    with pytest.raises(ValueError):
        load_manifest(p)


def test_iter_quotes_only_emits_top_of_book_changes(tmp_path):
    kalshi = _write_recording(tmp_path)
    out = list(iter_quotes(kalshi, ["KXMLBGAME-26AUG181835NYYBAL-BAL"]))
    # snapshot (48/49) and the 49c bid (49/49); the deep 10c delta and the
    # unrelated totals market contribute nothing.
    assert [q.yes_bid for _, q in out] == [48, 49]
    assert [q.recv_ts for _, q in out] == [1.0, 3.0]


def test_iter_quotes_collects_trades_in_the_same_pass(tmp_path):
    kalshi = tmp_path / "kalshi.jsonl"
    kalshi.write_text("\n".join(json.dumps(x) for x in [
        {"recv_ts": 1.0, "type": "orderbook_snapshot", "src": "kalshi",
         "msg": dict(_snapshot([(48, 50)], [(51, 20)]), market_ticker="A")},
        {"recv_ts": 2.0, "type": "trade", "src": "kalshi",
         "msg": {"market_ticker": "A", "yes_price_dollars": "0.4800",
                 "count_fp": "3.00", "taker_side": "no", "ts_ms": 2000}},
        {"recv_ts": 3.0, "type": "trade", "src": "kalshi",
         "msg": {"market_ticker": "B", "yes_price_dollars": "0.5000",
                 "count_fp": "1.00", "taker_side": "yes"}},
    ]) + "\n")
    trades = []
    list(iter_quotes(kalshi, ["A"], collect_trades=trades))
    assert len(trades) == 1
    assert trades[0].yes_price == 48
    assert trades[0].taker_side == "no"


def test_load_gumbo_skips_pregame_states(tmp_path):
    """Preview snapshots carry no innings played; the WP anchor does not
    describe a pregame market and would emit a meaningless 'fair' value."""
    g = tmp_path / "gumbo.jsonl"
    live = {"inning": 3, "half": "Top", "is_top": True, "outs": 1,
            "home_runs": 2, "away_runs": 1,
            "on_first": False, "on_second": False, "on_third": False}
    g.write_text("\n".join(json.dumps(x) for x in [
        {"recv_ts": 1.0, "type": "gumbo_state", "game_pk": 1,
         "status": "Preview", "state": dict(live, inning=1)},
        {"recv_ts": 2.0, "type": "gumbo_state", "game_pk": 1,
         "status": "Live", "state": live},
        {"recv_ts": 2.0, "type": "gumbo_play", "game_pk": 1, "at_bat_index": 7,
         "inning": 3, "half": "top", "end_time": "2026-08-18T23:00:00.0Z",
         "event": "Single", "event_type": "single", "description": "d",
         "away_score": 1, "home_score": 2},
    ]) + "\n")
    states, plays = load_gumbo(g)
    assert [s.recv_ts for s in states[1]] == [2.0]
    assert 0.0 < states[1][0].wp < 1.0
    assert plays[1][0].event == "Single"


def test_load_gumbo_sorts_by_receive_time(tmp_path):
    g = tmp_path / "gumbo.jsonl"
    live = {"inning": 3, "half": "Top", "is_top": True, "outs": 1,
            "home_runs": 0, "away_runs": 0,
            "on_first": False, "on_second": False, "on_third": False}
    g.write_text("\n".join(json.dumps(x) for x in [
        {"recv_ts": 9.0, "type": "gumbo_state", "game_pk": 1,
         "status": "Live", "state": dict(live, outs=2)},
        {"recv_ts": 4.0, "type": "gumbo_state", "game_pk": 1,
         "status": "Live", "state": live},
    ]) + "\n")
    states, _ = load_gumbo(g)
    assert [s.recv_ts for s in states[1]] == [4.0, 9.0]
