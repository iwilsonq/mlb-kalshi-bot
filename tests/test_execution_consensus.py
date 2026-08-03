"""Tests for maker limit pricing and consensus gate."""
from slugger.consensus import consensus_allows_trade, load_consensus_prices
from slugger.execution import (
    cancel_resting_orders_for_started_games,
    classify_fill_role,
    limit_price_cents,
    should_cancel_for_first_pitch,
)
from slugger.game_state import GameStateTracker
from slugger.strategies import RETIRED_STRATEGIES, STRATEGY_PIPELINE
from slugger.types import GameInfo


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
    assert "player_hr" in RETIRED_STRATEGIES


def test_retired_strategies_have_no_implementation_left():
    """Retired means deleted. A callable left behind is an invitation to re-add it."""
    import slugger.strategies as strat

    for retired in RETIRED_STRATEGIES:
        assert not hasattr(strat, f"strategy_{retired}"), retired
        assert not hasattr(strat, f"_run_{retired}"), retired
    # The legacy registries these used to live in are gone too
    assert not hasattr(strat, "STRATEGIES")
    assert not hasattr(strat, "BATTER_STRATEGIES")


def test_default_allowlist_is_a_subset_of_the_pipeline():
    """An ENABLED_STRATEGIES entry the pipeline does not register is inert.

    The live .env once listed player_hr, which silently did nothing and would
    have gone straight back to trading at −46% ROI the moment someone re-added
    the pipeline line.
    """
    from slugger.config import Config

    names = {n for n, _ in STRATEGY_PIPELINE}
    inert = set(Config().enabled_strategies) - names
    assert inert == set(), f"inert allowlist entries: {sorted(inert)}"


def test_cancel_resting_orders_for_started_games():
    cancelled = []
    orders = [
        {"order_id": "o1", "ticker": "KXMLBKS-STARTED", "status": "resting"},
        {"order_id": "o2", "ticker": "KXMLBKS-OPEN", "status": "resting"},
        {"order_id": "o3", "ticker": "KXMLBKS-DONE", "status": "executed"},
    ]
    n = cancel_resting_orders_for_started_games(
        orders,
        ticker_game_started=lambda t: t.endswith("STARTED"),
        cancel_fn=lambda oid: cancelled.append(oid) or True,
    )
    assert n == 1
    assert cancelled == ["o1"]


def test_classify_fill_role():
    assert classify_fill_role(40, 38, 40) == "maker"
    assert classify_fill_role(40, 40, 40) == "taker"


def test_sp_scratch_invalidates_game():
    tr = GameStateTracker()
    g1 = GameInfo(
        1, "A", "H", "AWY", "HOM", "1-1", "1-1", "P1", "P2", 10, 20,
        "2026-05-10T20:00:00Z", "V", {}, "Pre-Game",
    )
    assert tr.observe(g1) is None
    g2 = GameInfo(
        1, "A", "H", "AWY", "HOM", "1-1", "1-1", "P1b", "P2", 99, 20,
        "2026-05-10T20:00:00Z", "V", {}, "Pre-Game",
    )
    reason = tr.observe(g2)
    assert reason is not None
    assert tr.is_invalid(1)
