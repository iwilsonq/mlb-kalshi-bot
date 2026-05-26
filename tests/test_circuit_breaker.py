"""Tests for CircuitBreaker — risk management via settlement P&L tracking."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slugger.config import Config
from slugger.game_processor import CircuitBreaker


# ─── CircuitBreaker unit tests ────────────────────────────────────────────────


class TestCircuitBreakerSettlementTracking:
    """CircuitBreaker should trip based on settlement P&L, not placement cost."""

    def _make_breaker(self, max_loss: float = 10.0, max_consec: int = 3) -> CircuitBreaker:
        config = Config(cb_max_loss_usd=max_loss, cb_max_consecutive_losses=max_consec)
        return config, CircuitBreaker(config)

    def test_loss_increments_counters(self):
        """A negative P&L settlement should increment loss counters."""
        _, cb = self._make_breaker()
        cb.record_settlement(-3.50)
        assert cb.consec_losses == 1
        assert cb.total_loss == 3.50
        assert not cb.is_tripped()

    def test_win_resets_consecutive_losses(self):
        """A positive P&L settlement should reset consecutive loss counter."""
        _, cb = self._make_breaker()
        cb.record_settlement(-2.00)
        cb.record_settlement(-2.00)
        assert cb.consec_losses == 2
        cb.record_settlement(5.00)  # win
        assert cb.consec_losses == 0
        # total_loss should still reflect cumulative losses
        assert cb.total_loss == 4.00

    def test_trips_on_consecutive_losses(self):
        """Breaker should trip when consecutive losses reach the threshold."""
        _, cb = self._make_breaker(max_loss=100.0, max_consec=3)
        cb.record_settlement(-1.00)
        cb.record_settlement(-1.00)
        assert not cb.is_tripped()
        cb.record_settlement(-1.00)  # 3rd consecutive loss
        assert cb.is_tripped()

    def test_trips_on_total_loss(self):
        """Breaker should trip when total losses exceed the dollar threshold."""
        _, cb = self._make_breaker(max_loss=5.0, max_consec=100)
        cb.record_settlement(-3.00)
        cb.record_settlement(10.00)  # win resets consec, but total_loss stays
        cb.record_settlement(-3.00)
        assert cb.is_tripped()  # total_loss = 6.0 > 5.0

    def test_placement_cost_does_not_trip(self):
        """Passing a positive value (placement cost) must not trip the breaker."""
        _, cb = self._make_breaker(max_loss=1.0, max_consec=1)
        # Simulate what the old buggy code did — positive cost_usd
        cb.record_settlement(5.00)
        cb.record_settlement(5.00)
        cb.record_settlement(5.00)
        assert not cb.is_tripped()
        assert cb.consec_losses == 0
        assert cb.total_loss == 0.0


class TestSettlePendingFeedsCircuitBreaker:
    """settle_pending should feed settlement P&L into the circuit breaker."""

    def test_settle_pending_records_loss_to_circuit_breaker(self, tmp_path):
        """When a trade settles as a loss, the circuit breaker should be notified."""
        from slugger.game_processor import settle_pending

        # Set up a journal with one trade but no settlement
        log_dir = str(tmp_path)
        journal_file = tmp_path / "journal.jsonl"
        journal_file.write_text(json.dumps({
            "type": "trade",
            "ticker": "KXMLBKS-26MAY111810SFLAD-WEBB-7",
            "strategy": "pitcher_ks",
            "side": "yes",
            "count": 2,
            "price_cents": 35,
            "cost_usd": 0.70,
            "edge_cents": 8.0,
            "reason": "test",
            "order_id": "ord-123",
            "ts": "2026-05-11T18:00:00Z",
        }) + "\n")

        config = Config(
            log_dir=log_dir,
            cb_max_loss_usd=10.0,
            cb_max_consecutive_losses=3,
        )
        cb = CircuitBreaker(config)

        # Mock client returns a settlement with a loss
        client = MagicMock()
        client.get_settlements.return_value = [{
            "market_result": "no",
            "revenue": 0,             # 0 cents revenue (lost)
            "yes_total_cost_dollars": 0.70,
            "fee_cost": 0.02,
            "settled_time": "2026-05-11T22:00:00Z",
        }]

        n = settle_pending(client, config, circuit=cb)
        assert n == 1
        # P&L = revenue - cost - fee = 0 - 0.70 - 0.02 = -0.72
        assert cb.consec_losses == 1
        assert cb.total_loss == pytest.approx(0.72, abs=0.01)

    def test_settle_pending_records_win_to_circuit_breaker(self, tmp_path):
        """When a trade settles as a win, consecutive losses should reset."""
        from slugger.game_processor import settle_pending

        log_dir = str(tmp_path)
        journal_file = tmp_path / "journal.jsonl"
        journal_file.write_text(json.dumps({
            "type": "trade",
            "ticker": "KXMLBKS-26MAY111810SFLAD-WEBB-7",
            "strategy": "pitcher_ks",
            "side": "yes",
            "count": 2,
            "price_cents": 35,
            "cost_usd": 0.70,
            "edge_cents": 8.0,
            "reason": "test",
            "order_id": "ord-123",
            "ts": "2026-05-11T18:00:00Z",
        }) + "\n")

        config = Config(
            log_dir=log_dir,
            cb_max_loss_usd=10.0,
            cb_max_consecutive_losses=3,
        )
        cb = CircuitBreaker(config)
        # Pre-load one loss
        cb.record_settlement(-1.00)
        assert cb.consec_losses == 1

        # Mock client returns a settlement with a win
        client = MagicMock()
        client.get_settlements.return_value = [{
            "market_result": "yes",
            "revenue": 200,           # 200 cents = $2.00
            "yes_total_cost_dollars": 0.70,
            "fee_cost": 0.02,
            "settled_time": "2026-05-11T22:00:00Z",
        }]

        n = settle_pending(client, config, circuit=cb)
        assert n == 1
        # P&L = 2.00 - 0.70 - 0.02 = +1.28 (win)
        assert cb.consec_losses == 0

    def test_settle_pending_works_without_circuit_breaker(self, tmp_path):
        """settle_pending should still work when no circuit breaker is passed."""
        from slugger.game_processor import settle_pending

        log_dir = str(tmp_path)
        journal_file = tmp_path / "journal.jsonl"
        journal_file.write_text(json.dumps({
            "type": "trade",
            "ticker": "KXMLBKS-26MAY111810SFLAD-WEBB-7",
            "strategy": "pitcher_ks",
            "side": "yes",
            "count": 2,
            "price_cents": 35,
            "cost_usd": 0.70,
            "edge_cents": 8.0,
            "reason": "test",
            "order_id": "ord-123",
            "ts": "2026-05-11T18:00:00Z",
        }) + "\n")

        config = Config(log_dir=log_dir)

        client = MagicMock()
        client.get_settlements.return_value = [{
            "market_result": "no",
            "revenue": 0,
            "yes_total_cost_dollars": 0.70,
            "fee_cost": 0.02,
            "settled_time": "2026-05-11T22:00:00Z",
        }]

        # Should not raise
        n = settle_pending(client, config)
        assert n == 1
