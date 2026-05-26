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
    evaluate_markets,
    parse_threshold,
    parse_threshold_regex,
)


# ─── Threshold parsing ──────────────────────────────────────────────────────

class TestParseThreshold:
    def test_n_plus(self):
        assert parse_threshold("7+ strikeouts") == 7

    def test_over_half(self):
        assert parse_threshold("over 6.5 strikeouts") == 7

    def test_at_least(self):
        assert parse_threshold("at least 9 strikeouts") == 9

    def test_keyword_hit(self):
        assert parse_threshold("2+ hits?", keyword="hit") == 2

    def test_keyword_home_run(self):
        assert parse_threshold("1+ home runs", keyword="home run") == 1

    def test_no_match(self):
        assert parse_threshold("Will the Dodgers win?") is None

    def test_case_insensitive(self):
        assert parse_threshold("7+ STRIKEOUTS") == 7


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


def _make_config(log_dir: str, min_edge: int = 3, max_position: float = 50.0) -> MagicMock:
    config = MagicMock()
    config.log_dir = log_dir
    config.min_edge_cents = min_edge
    config.min_liquidity_dollars = 0
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
        assert signals[0].edge_cents == 15.0  # 45 - 30
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
