"""Tests for game-factor exposure and strategy health auto-disable."""
from slugger.risk import (
    GameFactorBudget,
    StrategyHealthMonitor,
    game_family_key,
    game_factor_key,
)


def test_game_family_groups_prop_types():
    ks = "KXMLBKS-26MAY231420HOUCHC-CHCCREA53-7"
    hit = "KXMLBHIT-26MAY231420HOUCHC-HOUPLAYER-2"
    assert game_family_key(ks) == game_family_key(hit)
    assert game_family_key(ks) == "26MAY231420HOUCHC"


def test_game_factor_budget_caps_same_family():
    b = GameFactorBudget(max_signals_per_game=2, max_exposure_usd=5.0)
    t1 = "KXMLBKS-26MAY231420HOUCHC-A-7"
    t2 = "KXMLBHIT-26MAY231420HOUCHC-B-2"
    t3 = "KXMLBHR-26MAY231420HOUCHC-C-1"
    assert b.can_place(t1, 1.0)
    b.record(t1, 1.0)
    assert b.can_place(t2, 1.0)
    b.record(t2, 1.0)
    assert not b.can_place(t3, 1.0)  # signal count
    # Different game still ok
    other = "KXMLBKS-26MAY241900NYYBOS-X-6"
    assert b.can_place(other, 1.0)


def test_game_factor_dollar_cap():
    b = GameFactorBudget(max_signals_per_game=10, max_exposure_usd=2.0)
    t = "KXMLBKS-26MAY231420HOUCHC-A-7"
    b.record(t, 1.5)
    assert not b.can_place(t, 1.0)


def test_strategy_health_disables_on_bad_roi():
    h = StrategyHealthMonitor(window_n=50, min_trades=10, min_roi_pct=-20.0)
    for _ in range(10):
        h.observe("pitcher_ks", pnl_usd=-1.0, cost_usd=1.0)
    assert "pitcher_ks" in h.disabled
    assert not h.is_enabled("pitcher_ks", ["pitcher_ks", "player_hits"])
    assert h.is_enabled("player_hits", ["pitcher_ks", "player_hits"])


def test_strategy_health_stays_enabled_when_ok():
    h = StrategyHealthMonitor(window_n=50, min_trades=10, min_roi_pct=-50.0)
    for i in range(10):
        h.observe("player_hits", pnl_usd=0.5 if i % 2 == 0 else -0.2, cost_usd=1.0)
    assert "player_hits" not in h.disabled


def test_strategy_health_brier_vs_market_kill_switch():
    """Model systematically worse Brier than market → auto-disable."""
    h = StrategyHealthMonitor(
        window_n=50, min_trades=10, min_roi_pct=-100.0, max_brier_deficit=0.01,
    )
    # ROI stays fine (break-even), but model always says 90% and is wrong vs 50¢ market
    for i in range(12):
        h.observe("pitcher_ks", pnl_usd=0.0, cost_usd=1.0)
        h.observe_probability("pitcher_ks", model_prob_pct=90, market_price_cents=50, outcome_yes=0)
    assert "pitcher_ks" in h.disabled


def test_load_from_journal():
    records = []
    for i in range(15):
        records.append({
            "type": "trade", "ticker": f"T{i}", "strategy": "bad", "cost_usd": 1.0,
        })
        records.append({
            "type": "settlement", "ticker": f"T{i}", "pnl_usd": -1.0,
        })
    h = StrategyHealthMonitor(window_n=50, min_trades=10, min_roi_pct=-10.0)
    h.load_from_journal(records)
    assert "bad" in h.disabled
