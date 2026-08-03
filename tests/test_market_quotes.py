"""Tests for market microstructure helpers and Phase 1 journal fields."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from slugger.game_processor import snapshot_pending_clv
from slugger.kalshi_client import (
    fill_price_cents_from_order,
    market_price,
    market_quotes,
)
from slugger.journal import load_journal, record_settlement, record_signal, record_trade
from slugger.types import MarketSpec, ModelResult
from slugger.signal_pipeline import evaluate_markets


def test_market_quotes_bid_ask_dollars():
    m = {
        "yes_bid_dollars": "0.28",
        "yes_ask_dollars": "0.32",
    }
    q = market_quotes(m)
    assert q["bid_cents"] == 28
    assert q["ask_cents"] == 32
    assert q["mid_cents"] == 30.0
    assert q["spread_cents"] == 4
    assert market_price(m) == 32


def test_market_quotes_ask_only():
    m = {"yes_ask_dollars": "0.40"}
    q = market_quotes(m)
    assert q["ask_cents"] == 40
    assert q["bid_cents"] == 0
    assert q["mid_cents"] == 40.0


def test_fill_price_from_avg_price_string():
    detail = {"avg_price": "0.33", "fill_count": "2"}
    assert fill_price_cents_from_order(detail, "yes", 35) == 33


def test_signal_records_microstructure(tmp_path):
    markets = [{
        "ticker": "TEST-7",
        "title": "Smith 7+ strikeouts",
        "yes_bid_dollars": "0.25",
        "yes_ask_dollars": "0.30",
    }]
    client = MagicMock()
    client.get_event_markets.return_value = markets
    config = MagicMock()
    config.log_dir = str(tmp_path)
    config.min_edge_cents = 3
    config.edge_cost_buffer_cents = 5
    config.min_liquidity_dollars = 0
    config.kelly_fraction = 0.25
    config.max_position_usd = 50.0
    config.max_contracts_per_trade = 100

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
    assert signals[0].bid_cents == 25
    assert signals[0].ask_cents == 30
    assert signals[0].spread_cents == 5
    assert signals[0].raw_model_prob_pct == 50
    assert signals[0].gross_edge_cents == 20.0  # 50 - 30
    # net = 20 - fee(30c)=2 - half-spread ceil(5/2)=3 - residual buffer 5 = 10.
    # The flat buffer used to stand in for fee+spread+adverse-selection; fee and
    # half-spread are now exact per contract (mlb-kalshi-bot-hyr).
    assert signals[0].edge_cents == 10.0

    data = json.loads((tmp_path / "signals.jsonl").read_text().strip().splitlines()[0])
    assert data["bid_cents"] == 25
    assert data["ask_cents"] == 30
    assert data["mid_cents"] == 27.5
    assert data["spread_cents"] == 5
    assert data["fee_cents"] == 2.0
    assert data["cost_buffer_cents"] == 10  # fee 2 + half-spread 3 + residual 5
    assert data["gross_edge_cents"] == 20.0
    assert data["net_edge_cents"] == 10.0
    assert data["calibrated_prob_pct"] == 50
    assert data["model_prob_pct"] == 50  # raw for calibration
    assert data["traded"] is True


def test_trade_and_settlement_measurement_fields(tmp_path):
    log_dir = str(tmp_path)
    record_trade(
        log_dir, "T-1", "pitcher_ks", "yes", 2, 30, 0.6, 15.0, "r", "oid-1",
        raw_model_prob_pct=48, model_prob_pct=45, gross_edge_cents=20,
        cost_buffer_cents=5, bid_cents=28, ask_cents=30, mid_cents=29,
        spread_cents=2, fill_price_cents=30, fill_count=2, fill_status="executed",
    )
    record_settlement(
        log_dir, "T-1", "yes", 2.0, 0.6, 0.02, "2026-01-01T00:00:00Z",
    )
    recs = load_journal(log_dir)
    trade = next(r for r in recs if r["type"] == "trade")
    sett = next(r for r in recs if r["type"] == "settlement")
    assert trade["fill_price_cents"] == 30
    assert trade["mid_cents"] == 29
    assert sett["settlement_price_cents"] == 100


def test_snapshot_pending_clv(tmp_path):
    log_dir = str(tmp_path)
    placed = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    # Manual trade row with placed_at in the past
    from slugger.journal import _append
    _append(log_dir, {
        "type": "trade",
        "placed_at": placed,
        "ticker": "CLV-TEST",
        "strategy": "pitcher_ks",
        "side": "yes",
        "count": 1,
        "price_cents": 30,
        "mid_cents": 29.0,
        "bid_cents": 28,
        "ask_cents": 30,
        "cost_usd": 0.3,
    })

    client = MagicMock()
    client.get_market.return_value = {
        "ticker": "CLV-TEST",
        "yes_bid_dollars": "0.34",
        "yes_ask_dollars": "0.38",
    }
    config = MagicMock()
    config.log_dir = log_dir

    n = snapshot_pending_clv(client, config, min_hours=1.0)
    assert n == 1
    clv = next(r for r in load_journal(log_dir) if r["type"] == "clv")
    assert clv["mid_cents"] == 36.0
    assert clv["entry_mid_cents"] == 29.0
    assert clv["clv_cents"] == 7.0  # mid moved up 7¢ toward YES
    # Idempotent
    assert snapshot_pending_clv(client, config, min_hours=1.0) == 0
