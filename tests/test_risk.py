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
    assert "rolling ROI" in h.disabled_reasons["bad"]


def _journal(strategy: str, outcomes: list) -> list:
    """outcomes: list of pnl_usd, each on a $1 cost trade, in file order."""
    records = []
    for i, pnl in enumerate(outcomes):
        records.append({
            "type": "trade", "ticker": f"{strategy}-{i}",
            "strategy": strategy, "cost_usd": 1.0,
        })
        records.append({
            "type": "settlement", "ticker": f"{strategy}-{i}", "pnl_usd": pnl,
        })
    return records


def test_seeding_judges_final_window_not_worst_ever():
    """A strategy that recovered must come back enabled after a restart.

    Regression: the disabled set is add-only, so replaying history with
    per-observation evaluation latched on the worst window in the file. A
    strategy whose current window is healthy stayed off forever.
    """
    # First 20 settle at a total loss, the next 20 at a profit. With window_n=20
    # the final window is entirely the profitable half.
    records = _journal("recovered", [-1.0] * 20 + [1.0] * 20)
    h = StrategyHealthMonitor(window_n=20, min_trades=10, min_roi_pct=-25.0)
    h.load_from_journal(records)

    assert h.disabled == set(), f"latched on a stale window: {h.disabled_reasons}"
    assert h.is_enabled("recovered", ["recovered"])
    assert h.health_reason("recovered") is None


def test_seeding_still_disables_when_current_window_is_bad():
    """The inverse: a strategy that started fine and then collapsed stays off."""
    records = _journal("collapsed", [1.0] * 20 + [-1.0] * 20)
    h = StrategyHealthMonitor(window_n=20, min_trades=10, min_roi_pct=-25.0)
    h.load_from_journal(records)

    assert "collapsed" in h.disabled
    assert not h.is_enabled("collapsed", ["collapsed"])


def test_seeding_evaluates_brier_from_final_window():
    """Brier kill-switch is also judged once, from the end state."""
    records = []
    # 20 settlements where the model was confidently wrong, then 20 where it was right
    for i, (result, prob) in enumerate([("no", 90)] * 20 + [("yes", 90)] * 20):
        records.append({
            "type": "trade", "ticker": f"B{i}", "strategy": "brier",
            "cost_usd": 1.0, "model_prob_pct": prob, "ask_cents": 50,
        })
        records.append({
            "type": "settlement", "ticker": f"B{i}",
            "pnl_usd": 1.0, "market_result": result,
        })
    h = StrategyHealthMonitor(
        window_n=20, min_trades=10, min_roi_pct=-100.0, max_brier_deficit=0.01,
    )
    h.load_from_journal(records)
    # Final Brier window is the accurate half, so no deficit to disable on
    assert "brier" not in h.disabled


def test_live_observations_still_latch():
    """Within a session the disable is sticky, so it cannot flap trade-by-trade."""
    h = StrategyHealthMonitor(window_n=10, min_trades=5, min_roi_pct=-25.0)
    for _ in range(5):
        h.observe("ks", pnl_usd=-1.0, cost_usd=1.0)
    assert "ks" in h.disabled

    # Recovering mid-session does not re-enable until the next restart
    for _ in range(10):
        h.observe("ks", pnl_usd=2.0, cost_usd=1.0)
    assert h.health_reason("ks") is None, "window itself should now look healthy"
    assert "ks" in h.disabled
    assert not h.is_enabled("ks", ["ks"])


def test_health_reason_does_not_mutate():
    h = StrategyHealthMonitor(window_n=10, min_trades=5, min_roi_pct=-25.0)
    h._seeding = True
    for _ in range(5):
        h.observe("ks", pnl_usd=-1.0, cost_usd=1.0)
    h._seeding = False
    assert h.health_reason("ks") is not None
    assert h.disabled == set()
    assert h.disabled_reasons == {}


def test_below_min_trades_is_never_disabled():
    """game_winner has 18 settled trades at -207% and must not trip min_trades=30."""
    records = _journal("thin", [-1.0] * 18)
    h = StrategyHealthMonitor(window_n=50, min_trades=30, min_roi_pct=-25.0)
    h.load_from_journal(records)
    assert "thin" not in h.disabled
    assert h.health_reason("thin") is None
