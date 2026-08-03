"""Tests for trained pitcher_ks model — walk-forward fit and predict."""
from slugger.ks_model import KsModel, clear_ks_model_cache, fit_ks_model, get_trained_ks_model
from slugger.models import expected_ks
from slugger.types import PitcherProfile


def test_fit_predict_learns_increasing_lambda():
    samples = []
    for i in range(40):
        recent = 4.0 + (i % 5)
        season = 5.0
        actual = recent + 0.5
        samples.append({
            "date": f"2026-04-{(i % 28) + 1:02d}",
            "recent_k": recent,
            "season_k": season,
            "opp_k_rate": 0.22,
            "actual_k": actual,
        })
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.2)
    assert model.n_samples >= 10
    lo = model.predict_lambda(4.0, 5.0, 0.22)
    hi = model.predict_lambda(9.0, 5.0, 0.22)
    assert hi > lo


def test_fit_as_of_excludes_future_samples():
    samples = [
        {"date": "2026-04-01", "recent_k": 5, "season_k": 5, "opp_k_rate": 0.2, "actual_k": 5},
        {"date": "2026-04-02", "recent_k": 6, "season_k": 5, "opp_k_rate": 0.2, "actual_k": 6},
        {"date": "2026-04-03", "recent_k": 7, "season_k": 5, "opp_k_rate": 0.2, "actual_k": 7},
        {"date": "2026-04-04", "recent_k": 8, "season_k": 5, "opp_k_rate": 0.2, "actual_k": 8},
        {"date": "2026-04-05", "recent_k": 9, "season_k": 5, "opp_k_rate": 0.2, "actual_k": 9},
        {"date": "2026-08-01", "recent_k": 20, "season_k": 20, "opp_k_rate": 0.3, "actual_k": 20},
    ] * 3
    m = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert m.as_of == "2026-05-01"
    # Future extreme should not pull lambda toward 20
    pred = m.predict_lambda(6.0, 5.0, 0.2)
    assert pred < 15


def test_expected_ks_uses_trained_model(tmp_path, monkeypatch):
    clear_ks_model_cache()
    samples = []
    for i in range(30):
        samples.append({
            "date": "2026-04-01",
            "recent_k": 6.0,
            "season_k": 6.0,
            "opp_k_rate": 0.22,
            "actual_k": 6.0 + (i % 3) * 0.1,
        })
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    path = tmp_path / "ks_model.json"
    model.save(str(path))
    clear_ks_model_cache()
    monkeypatch.setattr("slugger.ks_model.DEFAULT_MODEL_PATH", str(path))
    # Force load from our path
    loaded = get_trained_ks_model(str(path))
    assert loaded is not None
    # Patch get_trained_ks_model used inside expected_ks
    monkeypatch.setattr("slugger.ks_model.get_trained_ks_model", lambda path=None: loaded)
    profile = PitcherProfile(
        player_id=1, name="T", recent_k_per_start=6.0, recent_ip_per_start=6.0,
        k_per_9=9.0, max_k_in_start=10,
    )
    lam = expected_ks(profile, 0.22, use_trained=True)
    assert lam > 0
    clear_ks_model_cache()
