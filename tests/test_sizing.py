"""Tests for slugger.sizing — pure math, no I/O."""
from slugger.sizing import kelly_count


def test_kelly_basic():
    """Positive edge should produce non-zero count."""
    count = kelly_count(
        edge_cents=10.0,
        price_cents=30,
        kelly_fraction=0.25,
        max_position_usd=5.0,
    )
    assert count > 0


def test_kelly_no_edge():
    """Zero or negative edge should produce zero contracts."""
    assert kelly_count(0, 30, 0.25, 5.0) == 0
    assert kelly_count(-5.0, 30, 0.25, 5.0) == 0


def test_kelly_zero_price():
    """Zero price should produce zero contracts (division guard)."""
    assert kelly_count(10.0, 0, 0.25, 5.0) == 0


def test_kelly_max_contracts_cap():
    """Output should never exceed max_contracts."""
    count = kelly_count(
        edge_cents=50.0,
        price_cents=5,
        kelly_fraction=1.0,
        max_position_usd=10000.0,
        max_contracts=3,
    )
    assert count == 3


def test_kelly_scales_with_edge():
    """Higher edge should produce more contracts (all else equal)."""
    c1 = kelly_count(5.0, 30, 0.25, 5.0)
    c2 = kelly_count(15.0, 30, 0.25, 5.0)
    assert c2 >= c1


def test_kelly_scales_with_fraction():
    """Higher Kelly fraction should produce more contracts."""
    c1 = kelly_count(10.0, 30, 0.10, 5.0)
    c2 = kelly_count(10.0, 30, 0.50, 5.0)
    assert c2 >= c1


def test_kelly_identical_to_original():
    """Verify against the original _kelly_count formula from kalshi_client.py."""
    edge, price, frac, max_usd = 10.0, 30, 0.25, 5.0
    # Original formula:
    kelly_pct = edge / price
    expected = int((frac * kelly_pct * max_usd * 100) / price)
    expected = max(0, min(expected, 5000))
    assert kelly_count(edge, price, frac, max_usd) == expected
