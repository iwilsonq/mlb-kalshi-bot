"""Tests for slugger.signal_pipeline — market matching, threshold parsing, and pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from slugger.signal_pipeline import (
    MarketSpec,
    ModelResult,
    clear_unparsed_titles,
    evaluate_markets,
    parse_threshold_regex,
    unparsed_titles,
)


# ─── Threshold parsing ──────────────────────────────────────────────────────



class TestParseThresholdRegex:
    def test_basic(self):
        assert parse_threshold_regex("7+ strikeouts", r'(\d+)\s*\+') == 7

    def test_ceil(self):
        assert parse_threshold_regex("over 6.5 Ks", r'over\s+(\d+(?:\.\d+)?)', ceil=True) == 7

    def test_no_match(self):
        assert parse_threshold_regex("No match here", r'(\d+)\+\s*home\s*run') is None


# ─── Pipeline integration ────────────────────────────────────────────────────

def _make_market(ticker: str, title: str, yes_ask_dollars: str) -> dict:
    """Create a minimal Kalshi market dict."""
    return {
        "ticker": ticker,
        "title": title,
        "yes_ask_dollars": yes_ask_dollars,
    }


def _make_config(
    log_dir: str,
    min_edge: int = 3,
    max_position: float = 50.0,
    cost_buffer: int = 0,
) -> MagicMock:
    config = MagicMock()
    config.log_dir = log_dir
    config.min_edge_cents = min_edge
    config.edge_cost_buffer_cents = cost_buffer
    config.min_liquidity_dollars = 0
    # Mirror real Config: MagicMock would otherwise yield float()==1.0 here,
    # making the relative floor 100% of price.
    config.min_edge_frac_of_price = 0.20
    config.max_spread_cents = 40
    config.kelly_fraction = 0.25
    config.max_position_usd = max_position
    config.max_contracts_per_trade = 100
    return config


def _make_client(markets: list) -> MagicMock:
    client = MagicMock()
    client.get_event_markets.return_value = markets
    return client


class TestEvaluateMarkets:
    def test_basic_yes_signal(self, tmp_path):
        """Pipeline should produce a YES signal when model prob exceeds price + min_edge."""
        markets = [
            _make_market("KXMLBKS-TEST-SMITH-7", "Smith 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3)

        def model(title, threshold, price):
            return ModelResult(prob_pct=45, reason="test")

        spec = MarketSpec(
            event_ticker="KXMLBKS-TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)

        assert len(signals) == 1
        assert signals[0].side == "yes"
        assert signals[0].strategy == "pitcher_ks"
        # net edge = 45 - 30 - fee(30c)=2c - half-spread(0) - buffer(0) = 13
        assert signals[0].edge_cents == 13.0
        assert signals[0].ticker == "KXMLBKS-TEST-SMITH-7"

    def test_no_signal_when_no_edge(self, tmp_path):
        """Pipeline should produce no signals when model prob < price + min_edge."""
        markets = [
            _make_market("KXMLBKS-TEST-SMITH-7", "Smith 7+ strikeouts", "0.50"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3)

        def model(title, threshold, price):
            return ModelResult(prob_pct=50, reason="test")

        spec = MarketSpec(
            event_ticker="KXMLBKS-TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) == 0

    def test_player_name_filter(self, tmp_path):
        """Markets not matching player name should be filtered out."""
        markets = [
            _make_market("KXMLBKS-TEST-SMITH-7", "Smith 7+ strikeouts", "0.30"),
            _make_market("KXMLBKS-TEST-JONES-7", "Jones 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        def model(title, threshold, price):
            return ModelResult(prob_pct=50, reason="test")

        spec = MarketSpec(
            event_ticker="KXMLBKS-TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) == 1
        assert "SMITH" in signals[0].ticker

    def test_keyword_filter(self, tmp_path):
        """Markets not matching title keywords should be filtered out."""
        markets = [
            _make_market("TEST-1", "Smith 2+ hits", "0.40"),
            _make_market("TEST-2", "Smith home run", "0.10"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        def model(title, threshold, price):
            return ModelResult(prob_pct=60, reason="test")

        spec = MarketSpec(
            event_ticker="TEST-EVENT",
            strategy_name="player_hits",
            title_keywords=["hit"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
        )
        signals = evaluate_markets(spec, model, client, config)
        # Only "2+ hits" matches keyword "hit"
        assert len(signals) == 1
        assert signals[0].ticker == "TEST-1"

    def test_ticker_suffix_filter(self, tmp_path):
        """Only markets with the correct ticker suffix should match."""
        markets = [
            _make_market("KXMLBGAME-TEST-LAD", "Dodgers win", "0.30"),
            _make_market("KXMLBGAME-TEST-SF", "Giants win", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        def model(title, threshold, price):
            return ModelResult(prob_pct=55, reason="test")

        spec = MarketSpec(
            event_ticker="KXMLBGAME-TEST",
            strategy_name="game_winner",
            ticker_suffix="LAD",
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) == 1
        assert signals[0].ticker == "KXMLBGAME-TEST-LAD"

    def test_min_threshold_filter(self, tmp_path):
        """Markets below min_threshold should be skipped."""
        markets = [
            _make_market("TEST-4", "Smith 4+ strikeouts", "0.60"),
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        def model(title, threshold, price):
            return ModelResult(prob_pct=50, reason="test")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) == 1
        assert signals[0].ticker == "TEST-7"

    def test_max_signals_cap(self, tmp_path):
        """Should keep only the top N signals by edge when max_signals is set."""
        markets = [
            _make_market("TEST-6", "Smith 6+ strikeouts", "0.20"),
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.15"),
            _make_market("TEST-8", "Smith 8+ strikeouts", "0.10"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        probs = {"6": 40, "7": 35, "8": 30}

        def model(title, threshold, price):
            return ModelResult(prob_pct=probs.get(str(threshold), 10), reason="test")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
            max_signals=2,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) <= 2

    def test_no_side_trade(self, tmp_path):
        """NO-side should fire when model YES prob is very low but market prices it high."""
        markets = [
            _make_market("TEST-9", "Smith 9+ strikeouts", "0.25"),  # market says 25% YES
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3)

        def model(title, threshold, price):
            # Model says only 5% chance of 9+ Ks
            return ModelResult(prob_pct=5, reason="test low prob")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
            no_side=True,
            no_max_model_prob=10,
            no_min_edge_cents=5,
        )
        signals = evaluate_markets(spec, model, client, config)
        no_signals = [s for s in signals if s.side == "no"]
        assert len(no_signals) == 1
        assert no_signals[0].edge_cents == 20.0  # 25 - 5

    def test_model_returning_none_skips_market(self, tmp_path):
        """When model returns None, market should be silently skipped."""
        markets = [
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, lambda t, th, p: None, client, config)
        assert len(signals) == 0

    def test_signal_recording(self, tmp_path):
        """Pipeline should write signals to the journal for calibration."""
        markets = [
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path))

        def model(title, threshold, price):
            return ModelResult(prob_pct=45, reason="test reason")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        evaluate_markets(spec, model, client, config)

        signals_file = tmp_path / "signals.jsonl"
        assert signals_file.exists()
        lines = signals_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["strategy"] == "pitcher_ks"
        assert data["model_prob_pct"] == 45
        assert data["market_price_cents"] == 30

    def test_empty_event_ticker(self, tmp_path):
        """Empty event ticker should return no signals without calling client."""
        client = _make_client([])
        config = _make_config(str(tmp_path))

        spec = MarketSpec(event_ticker="", strategy_name="test")
        signals = evaluate_markets(spec, lambda t, th, p: None, client, config)
        assert len(signals) == 0
        client.get_event_markets.assert_not_called()

    def test_client_exception_returns_empty(self, tmp_path):
        """If client raises an exception, pipeline should return empty list."""
        client = MagicMock()
        client.get_event_markets.side_effect = Exception("API error")
        config = _make_config(str(tmp_path))

        spec = MarketSpec(event_ticker="TEST", strategy_name="test")
        signals = evaluate_markets(spec, lambda t, th, p: ModelResult(50, ""), client, config)
        assert len(signals) == 0

    def test_cost_buffer_blocks_marginal_edge(self, tmp_path):
        """Gross edge above floor but net edge below floor after buffer → no trade."""
        markets = [
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.30"),  # price 30¢
        ]
        client = _make_client(markets)
        # model 40 → gross edge 10; buffer 5 → net 5; floor 8 → no trade
        config = _make_config(str(tmp_path), min_edge=8, cost_buffer=5)

        def model(title, threshold, price):
            return ModelResult(prob_pct=40, reason="test")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert signals == []

        # Signal still journaled with gross edge
        data = json.loads((tmp_path / "signals.jsonl").read_text().strip().splitlines()[0])
        assert data["traded"] is False
        assert data["edge_cents"] == 10.0

    def test_cost_buffer_sizes_on_net_edge(self, tmp_path):
        """When trade fires, TradeSignal.edge_cents is net of cost buffer."""
        markets = [
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.30"),
        ]
        client = _make_client(markets)
        # model 50 → gross 20; buffer 5 → net 15; floor 10 → trade
        config = _make_config(str(tmp_path), min_edge=10, cost_buffer=5)

        def model(title, threshold, price):
            return ModelResult(prob_pct=50, reason="test")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert len(signals) == 1
        # net edge = 50 - 30 - fee(30c)=2c - buffer(5) = 13
        assert signals[0].edge_cents == 13.0

    def test_max_model_prob_band(self, tmp_path):
        """Probabilities above max_model_prob should not trade."""
        markets = [
            _make_market("TEST-7", "Smith 7+ strikeouts", "0.20"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3, cost_buffer=0)

        def model(title, threshold, price):
            return ModelResult(prob_pct=70, reason="too high")

        spec = MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
            min_model_prob=25,
            max_model_prob=55,
        )
        signals = evaluate_markets(spec, model, client, config)
        assert signals == []


# ─── Unparseable titles are recorded, not silently dropped ──────────────────

class TestUnparsedTitleTracking:
    """A market whose threshold will not parse is never priced at all.

    Real Kalshi titles observed so far all use the N+ form; this tracking is
    the tripwire that would justify resurrecting a richer parser from git.
    """

    def _spec(self):
        return MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
        )

    def test_records_title_the_live_pattern_cannot_parse(self, tmp_path):
        clear_unparsed_titles()
        markets = [
            _make_market("TEST-A", "Smith over 6.5 strikeouts", "0.20"),
            _make_market("TEST-B", "Smith at least 9 strikeouts", "0.20"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3, cost_buffer=0)

        signals = evaluate_markets(
            self._spec(),
            lambda title, threshold, price: ModelResult(prob_pct=45, reason="t"),
            client,
            config,
        )

        assert signals == []
        dropped = unparsed_titles()["pitcher_ks"]
        assert dropped == [
            "Smith at least 9 strikeouts",
            "Smith over 6.5 strikeouts",
        ]
        clear_unparsed_titles()

    def test_parseable_titles_are_not_recorded(self, tmp_path):
        clear_unparsed_titles()
        markets = [_make_market("TEST-7", "Smith 7+ strikeouts", "0.20")]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3, cost_buffer=0)

        evaluate_markets(
            self._spec(),
            lambda title, threshold, price: ModelResult(prob_pct=45, reason="t"),
            client,
            config,
        )
        assert unparsed_titles() == {}
        clear_unparsed_titles()

    def test_duplicate_titles_counted_once(self, tmp_path):
        clear_unparsed_titles()
        markets = [
            _make_market("TEST-A", "Smith over 6.5 strikeouts", "0.20"),
            _make_market("TEST-B", "Smith over 6.5 strikeouts", "0.21"),
        ]
        client = _make_client(markets)
        config = _make_config(str(tmp_path), min_edge=3, cost_buffer=0)

        evaluate_markets(
            self._spec(),
            lambda title, threshold, price: ModelResult(prob_pct=45, reason="t"),
            client,
            config,
        )
        assert unparsed_titles() == {
            "pitcher_ks": ["Smith over 6.5 strikeouts"],
        }
        clear_unparsed_titles()


# ─── Cost model: no half-spread double-count; relative floor; spread gate ────

class TestCostModelAndFloors:
    """gross_edge is measured at the ASK, so the spread is already paid.

    An earlier version subtracted half the spread on top, which double-counted
    and overcharged ~8c on live MLB props: median half-spread was 11c against a
    true ask-minus-fair distance of 2.8c, because de-vigged book fair value sits
    at 0.88 of the bid-ask range rather than the mid (mlb-kalshi-bot-4v6).
    """

    def _spec(self, min_edge=3, min_prob=0):
        return MarketSpec(
            event_ticker="TEST",
            strategy_name="pitcher_ks",
            title_keywords=["strikeout"],
            player_name="John Smith",
            threshold_pattern=r'(\d+)\s*\+',
            min_threshold=6,
            min_edge_cents=min_edge,
            min_model_prob=min_prob,
        )

    def _run(self, tmp_path, prob, ask, bid=None, **cfg):
        m = _make_market("TEST-7", "Smith 7+ strikeouts", f"{ask/100:.2f}")
        if bid is not None:
            m["yes_bid_dollars"] = f"{bid/100:.2f}"
        config = _make_config(str(tmp_path), min_edge=cfg.pop("min_edge", 3),
                              cost_buffer=cfg.pop("cost_buffer", 0))
        for k, v in cfg.items():
            setattr(config, k, v)
        return evaluate_markets(
            self._spec(min_edge=3),
            lambda t, th, p: ModelResult(prob_pct=prob, reason="t"),
            _make_client([m]), config,
        )

    def test_wide_spread_does_not_reduce_edge(self):
        """A wide spread must not be charged as a graduated cost."""
        import tempfile
        tight = self._run(tempfile.mkdtemp(), prob=45, ask=30, bid=29)
        wide = self._run(tempfile.mkdtemp(), prob=45, ask=30, bid=5)
        assert len(tight) == 1 and len(wide) == 1
        # Same ask, same model prob -> identical net edge regardless of the bid
        assert tight[0].edge_cents == wide[0].edge_cents == 13.0

    def test_spread_gate_rejects_unusable_quote(self, tmp_path):
        """bid 2c / ask 93c is not a market; Gate 0 saw it present as 37.7c edge."""
        sig = self._run(tmp_path, prob=90, ask=93, bid=2, max_spread_cents=40)
        assert sig == []

    def test_spread_gate_ignores_absent_bid(self, tmp_path):
        """No bid at all is not evidence of a bad quote, so do not reject on it."""
        sig = self._run(tmp_path, prob=45, ask=30, max_spread_cents=40)
        assert len(sig) == 1

    def test_relative_floor_matches_absolute_at_fifty_cents(self):
        """0.20 x 50c == the historical 10c floor, so behaviour is continuous."""
        import math as _m
        assert _m.ceil(0.20 * 50) == 10

    def test_relative_floor_tightens_at_high_prices(self, tmp_path):
        """10c on a 90c contract is an 11% return; the relative floor blocks it."""
        # prob 99, ask 90 -> gross 9, fee(90)=1, net 8. Absolute floor 3 would
        # pass; relative floor ceil(0.20*90)=18 must block.
        sig = self._run(tmp_path, prob=99, ask=90, bid=88,
                        min_edge_frac_of_price=0.20, max_spread_cents=40)
        assert sig == []
        # With the relative floor disabled it trades, proving that gate is the cause
        sig2 = self._run(tmp_path, prob=99, ask=90, bid=88,
                         min_edge_frac_of_price=0.0, max_spread_cents=40)
        assert len(sig2) == 1

    def test_absolute_floor_still_bans_longshots(self, tmp_path):
        """At 5c the relative floor is only 1c, so the absolute floor must bind.

        Fee drag was 7.7% of stake at 0-20c entries versus 0.6% at 60-80c, so
        keeping the absolute floor as a longshot ban is deliberate.
        """
        # prob 20, ask 5 -> gross 15, fee(5)=1, net 14. relative floor = 1.
        # Absolute floor of 20 must block it.
        sig = self._run(tmp_path, prob=20, ask=5, bid=4, min_edge=20,
                        min_edge_frac_of_price=0.20, max_spread_cents=40)
        assert sig == []
