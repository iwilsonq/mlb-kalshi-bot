"""Tests for slugger.reporting — Brier/log-loss and ROI heatmaps."""
from __future__ import annotations

from slugger.reporting import (
    brier_score,
    build_report,
    format_report,
    log_loss,
    parse_threshold_from_ticker,
    price_band,
    roi_heatmap,
    score_probabilities,
)


def test_brier_perfect():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_worst():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_empty():
    assert brier_score([], []) is None


def test_log_loss_better_when_confident_correct():
    good = log_loss([0.9], [1])
    bad = log_loss([0.1], [1])
    assert good is not None and bad is not None
    assert good < bad


def test_price_band():
    assert price_band(23) == "20-29"
    assert price_band(5) == "0-9"


def test_parse_threshold():
    assert parse_threshold_from_ticker("KXMLBKS-26MAY-SMITH-7") == 7
    assert parse_threshold_from_ticker("KXMLBGAME-X-LAD") is None


def test_score_probabilities_model_beats_market():
    signals = [
        {
            "ticker": "A",
            "strategy": "pitcher_ks",
            "model_prob_pct": 60,
            "calibrated_prob_pct": 60,
            "mid_cents": 40,
            "ask_cents": 42,
            "traded": True,
        },
        {
            "ticker": "B",
            "strategy": "pitcher_ks",
            "model_prob_pct": 20,
            "calibrated_prob_pct": 20,
            "mid_cents": 40,
            "ask_cents": 42,
            "traded": True,
        },
    ]
    # A hits (model 60 > market 40), B misses (model 20 < market 40) — model better
    settlements = {
        "A": {"market_result": "yes", "pnl_usd": 1.0},
        "B": {"market_result": "no", "pnl_usd": -0.4},
    }
    rows, n = score_probabilities(signals, settlements, market_source="mid")
    assert n == 2
    overall = rows[0]
    assert overall.strategy == "overall"
    assert overall.model_brier is not None
    assert overall.market_brier is not None
    assert overall.model_brier < overall.market_brier
    assert overall.brier_edge is not None and overall.brier_edge > 0


def test_roi_heatmap_cells():
    trades = [
        {
            "type": "trade",
            "ticker": "KXMLBKS-X-SMITH-7",
            "strategy": "pitcher_ks",
            "price_cents": 25,
            "ask_cents": 25,
            "cost_usd": 1.0,
        },
        {
            "type": "trade",
            "ticker": "KXMLBKS-X-JONES-7",
            "strategy": "pitcher_ks",
            "price_cents": 25,
            "ask_cents": 25,
            "cost_usd": 1.0,
        },
    ]
    settlements = {
        "KXMLBKS-X-SMITH-7": {"market_result": "yes", "pnl_usd": 3.0},
        "KXMLBKS-X-JONES-7": {"market_result": "no", "pnl_usd": -1.0},
    }
    cells, n = roi_heatmap(trades, settlements)
    assert n == 2
    assert len(cells) == 1
    c = cells[0]
    assert c.strategy == "pitcher_ks"
    assert c.price_band == "20-29"
    assert c.threshold == "7+"
    assert c.n == 2
    assert c.wins == 1
    assert abs(c.pnl_usd - 2.0) < 1e-9
    assert c.roi_pct is not None and c.roi_pct == 100.0


def test_build_and_format_report():
    signals = [{
        "ticker": "T1",
        "strategy": "player_hits",
        "model_prob_pct": 40,
        "mid_cents": 35,
        "ask_cents": 36,
        "traded": True,
    }]
    journal = [
        {
            "type": "trade",
            "ticker": "T1",
            "strategy": "player_hits",
            "price_cents": 36,
            "ask_cents": 36,
            "cost_usd": 0.5,
        },
        {
            "type": "settlement",
            "ticker": "T1",
            "market_result": "no",
            "pnl_usd": -0.5,
        },
    ]
    report = build_report(signals, journal, market_source="mid")
    text = format_report(report, min_roi_n=1)
    assert "MODEL vs MARKET" in text
    assert "player_hits" in text
    assert "Brier" in text or "Brier_m" in text
    assert report.n_trades_scored == 1
