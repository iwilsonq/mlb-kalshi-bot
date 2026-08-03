"""Tests for maker limit pricing and consensus gate."""
from slugger.consensus import consensus_allows_trade, load_consensus_prices
from slugger.execution import limit_price_cents, should_cancel_for_first_pitch
from slugger.strategies import RETIRED_STRATEGIES, STRATEGY_PIPELINE


def test_limit_price_never_above_ask_or_fair():
    # fair 40%, ask 38 → bid min(39, 38)=38 with buffer 1 → 39? fair-1=39, min(39,38)=38
    px = limit_price_cents(fair_prob_pct=40, ask_cents=38, buffer_cents=1)
    assert px == 38
    px2 = limit_price_cents(fair_prob_pct=40, ask_cents=50, buffer_cents=1)
    assert px2 == 39  # fair-1


def test_cancel_at_first_pitch():
    assert should_cancel_for_first_pitch(True) is True
    assert should_cancel_for_first_pitch(False) is False


def test_consensus_gate(tmp_path, monkeypatch):
    p = tmp_path / "cons.json"
    p.write_text('{"ABC": {"fair_cents": 40}}')
    monkeypatch.setenv("CONSENSUS_PRICES_PATH", str(p))
    c = load_consensus_prices()
    assert c["ABC"] == 40
    assert consensus_allows_trade("ABC", 30, c, min_edge_cents=5)
    assert not consensus_allows_trade("ABC", 38, c, min_edge_cents=5)
    assert consensus_allows_trade("OTHER", 99, c)  # no line


def test_retired_strategies_not_in_pipeline():
    names = {n for n, _ in STRATEGY_PIPELINE}
    for retired in RETIRED_STRATEGIES:
        assert retired not in names
    assert "pitcher_ks" in names
    assert "player_hits" in names
    assert "player_hr" not in names  # dark
