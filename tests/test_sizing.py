"""Tests for slugger.sizing — binary Kelly and daily risk budget."""
from slugger.sizing import (
    DailyRiskBudget,
    binary_kelly_fraction,
    daily_spent_from_journal,
    kelly_count,
    legacy_edge_over_price_fraction,
)


def test_binary_kelly_known_example():
    """q=0.6, p=0.4 → f* = 0.2/0.6 = 1/3."""
    f = binary_kelly_fraction(0.6, 0.4)
    assert abs(f - (0.2 / 0.6)) < 1e-9


def test_binary_kelly_no_edge():
    assert binary_kelly_fraction(0.4, 0.4) == 0.0
    assert binary_kelly_fraction(0.3, 0.4) == 0.0


def test_kelly_count_uses_binary_formula():
    """f*=1/3, quarter-Kelly, $100 bankroll, p=40¢ → dollars=8.333 → 20 contracts."""
    # q=60%, p=40¢, edge=20¢
    count = kelly_count(
        edge_cents=20.0,
        price_cents=40,
        kelly_fraction=0.25,
        max_position_usd=100.0,
        model_prob_pct=60.0,
        bankroll_usd=100.0,
    )
    # f* = 1/3; frac Kelly = 1/12; dollars = 100/12 ≈ 8.333; contracts = 8.333/0.4 ≈ 20
    assert count == 20


def test_kelly_longshot_smaller_than_legacy():
    """Old edge/price overbets cheap contracts; binary Kelly does not."""
    edge, price = 10.0, 5  # 10¢ edge on a 5¢ longshot
    # Reconstruct q = 0.05 + 0.10 = 0.15
    new_c = kelly_count(
        edge, price, 1.0, 100.0, model_prob_pct=15.0, bankroll_usd=100.0,
    )
    # Legacy sizing (what the old function did)
    leg_f = legacy_edge_over_price_fraction(edge, price)
    legacy_count = int((1.0 * leg_f * 100.0 * 100) / price)
    assert legacy_count > new_c
    assert new_c > 0


def test_kelly_no_edge():
    assert kelly_count(0, 30, 0.25, 5.0) == 0
    assert kelly_count(-5.0, 30, 0.25, 5.0) == 0


def test_kelly_zero_price():
    assert kelly_count(10.0, 0, 0.25, 5.0) == 0


def test_kelly_max_contracts_cap():
    count = kelly_count(
        edge_cents=40.0,
        price_cents=50,
        kelly_fraction=1.0,
        max_position_usd=10000.0,
        max_contracts=3,
        model_prob_pct=90.0,
        bankroll_usd=10000.0,
    )
    assert count == 3


def test_kelly_daily_remaining_caps():
    count = kelly_count(
        edge_cents=20.0,
        price_cents=40,
        kelly_fraction=1.0,
        max_position_usd=100.0,
        model_prob_pct=60.0,
        bankroll_usd=100.0,
        remaining_daily_usd=0.40,  # only enough for 1 contract @ 40¢
    )
    assert count == 1


def test_kelly_daily_exhausted():
    assert kelly_count(
        20.0, 40, 1.0, 100.0, model_prob_pct=60.0, bankroll_usd=100.0,
        remaining_daily_usd=0.0,
    ) == 0


def test_daily_risk_budget():
    b = DailyRiskBudget(bankroll_usd=100.0, max_fraction=0.2, spent_usd=5.0)
    assert abs(b.cap_usd - 20.0) < 1e-9
    assert abs(b.remaining_usd - 15.0) < 1e-9
    assert b.can_spend(10.0)
    assert not b.can_spend(16.0)
    b.record(10.0)
    assert abs(b.remaining_usd - 5.0) < 1e-9


def test_daily_spent_from_journal():
    records = [
        {"type": "trade", "date": "2026-08-01", "cost_usd": 1.5},
        {"type": "trade", "date": "2026-08-01", "cost_usd": 2.0},
        {"type": "trade", "date": "2026-08-02", "cost_usd": 9.0},
        {"type": "settlement", "date": "2026-08-01", "cost_usd": 99.0},
    ]
    assert abs(daily_spent_from_journal(records, "2026-08-01") - 3.5) < 1e-9
