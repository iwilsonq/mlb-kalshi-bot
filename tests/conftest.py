"""Shared test fixtures.

Keeps the suite hermetic with respect to on-disk model artifacts.
"""
import pytest

import slugger.ks_model as ks_model


@pytest.fixture(autouse=True)
def isolate_trained_ks_model(tmp_path, monkeypatch):
    """Never let a test read the developer's real logs/ks_model.json.

    get_trained_ks_model() defaults to the relative path "logs/ks_model.json",
    resolved against the working directory, and memoises the result in a module
    global. So before this fixture existed, every test touching
    models.expected_ks silently depended on whether a model happened to be on
    disk — the suite was implicitly asserting "no trained model", and shipping a
    real artifact broke an unrelated backtest test.

    Tests that want a trained model should build one and inject it explicitly.
    """
    monkeypatch.setattr(
        ks_model, "DEFAULT_MODEL_PATH", str(tmp_path / "absent-ks_model.json")
    )
    ks_model.clear_ks_model_cache()
    yield
    ks_model.clear_ks_model_cache()
