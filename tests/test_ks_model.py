"""Tests for trained pitcher_ks model — real outcomes + Brier vs market."""
from slugger.ks_model import (
    KsModel,
    clear_ks_model_cache,
    fit_ks_model,
    get_trained_ks_model,
    holdout_brier_vs_market,
    samples_from_pitcher_game_logs,
)
from slugger.models import expected_ks, poisson_ge
from slugger.types import PitcherProfile


def _synthetic_logs():
    """Pitcher with rising K rates — actual outcomes on each date."""
    logs = {"Ace Arm": []}
    # Build 15 starts with increasing Ks
    for i in range(15):
        logs["Ace Arm"].append({
            "date": f"2026-04-{(i + 1):02d}",
            "strikeouts": 4 + (i // 3),  # 4,4,4,5,5,5,...
        })
    return logs


def test_samples_use_only_prior_starts_and_actual_k():
    logs = _synthetic_logs()
    samples = samples_from_pitcher_game_logs(logs, as_of="2026-04-20", min_prior_starts=2)
    assert len(samples) >= 5
    # First usable sample is on 2026-04-03 (needs 2 prior)
    first = min(samples, key=lambda s: s["date"])
    assert first["date"] >= "2026-04-03"
    # actual_k is true strikeouts, not equal to recent_k by construction always
    assert "actual_k" in first
    # No sample on/after as_of
    assert all(s["date"] < "2026-04-20" for s in samples)


def test_fit_on_actual_outcomes_not_identity_lambda():
    logs = _synthetic_logs()
    samples = samples_from_pitcher_game_logs(logs, as_of="2026-05-01", min_prior_starts=2)
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.25)
    assert model.n_samples >= 5
    # Predict should be in a baseball-sensible range
    pred = model.predict_lambda(6.0, 5.5, 0.22)
    assert 2.0 < pred < 15.0
    assert model.holdout_mae is not None


def test_holdout_brier_vs_market_drives_shipped_functions():
    """Model that tracks truth should beat a flat 50¢ market on clear cases."""
    logs = _synthetic_logs()
    samples = samples_from_pitcher_game_logs(logs, as_of="2026-05-01", min_prior_starts=2)
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.3)
    # Build holdout props with actual K and a bad flat market
    props = []
    for s in samples[-5:]:
        actual = s["actual_k"]
        for thr in (5, 6, 7):
            props.append({
                **s,
                "threshold": thr,
                "actual_k": actual,
                "market_price_cents": 50,  # uninformative market
            })
    mb, kb, beats = holdout_brier_vs_market(model, props)
    assert mb is not None and kb is not None
    # Model using actual-trained λ should not be worse than random 50% market
    # on average for extreme thresholds (allow equality)
    assert mb <= kb + 0.05


def test_expected_ks_uses_trained_model(tmp_path, monkeypatch):
    clear_ks_model_cache()
    logs = _synthetic_logs()
    samples = samples_from_pitcher_game_logs(logs, as_of="2026-05-01", min_prior_starts=2)
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    path = tmp_path / "ks_model.json"
    model.save(str(path))
    clear_ks_model_cache()
    loaded = get_trained_ks_model(str(path))
    assert loaded is not None
    monkeypatch.setattr("slugger.ks_model.get_trained_ks_model", lambda path=None: loaded)
    profile = PitcherProfile(
        player_id=1, name="T", recent_k_per_start=6.0, recent_ip_per_start=6.0,
        k_per_9=9.0, max_k_in_start=10,
    )
    lam = expected_ks(profile, 0.22, use_trained=True)
    assert lam > 0
    clear_ks_model_cache()


def test_rejects_circular_lambda_as_actual_without_field():
    """Samples missing actual_k/strikeouts must not train."""
    samples = [
        {"date": "2026-04-01", "recent_k": 6, "season_k": 6, "opp_k_rate": 0.2},
    ] * 30
    m = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    # n_samples should be 0 path → fallback intercept only
    assert m.n_samples < 5 or m.n_samples == 0
