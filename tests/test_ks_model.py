"""Tests for trained pitcher_ks — real outcomes, real markets, model-scored ROI."""
import math

import pytest

from slugger.ks_model import (
    LAMBDA_MAX,
    KsModel,
    build_holdout_props_from_signals,
    clear_ks_model_cache,
    fit_ks_model,
    get_trained_ks_model,
    holdout_brier_vs_market,
    journal_roi_for_strategy,
    model_roi_vs_phase0_baseline,
    samples_from_pitcher_game_logs,
)
from slugger.models import expected_ks
from slugger.types import PitcherProfile

# Phase-0 band the ROI harness enforces; fixtures must sit inside it.
BAND_LO, BAND_HI = 25.0, 55.0
COST_BUFFER = 5.0
MIN_EDGE = 20.0


def _synthetic_logs():
    logs = {"Paul Ace": []}
    for d in range(1, 21):
        logs["Paul Ace"].append({
            "date": f"2026-04-{d:02d}",
            "strikeouts": 5 + (d % 4),
        })
    return logs


def _multi_pitcher_logs():
    """Three pitchers at different K levels — gives the fit real variance to learn."""
    logs = {}
    for name, base in [("Paul Ace", 7.5), ("Mid Rotation", 5.5), ("Soft Toss", 3.5)]:
        logs[name] = [
            {"date": f"2026-04-{d:02d}", "strikeouts": max(0, int(base + ((d % 5) - 2)))}
            for d in range(1, 29)
        ]
    return logs


def _in_band_cells(
    model: KsModel,
    n: int = 15,
    *,
    threshold: int = 7,
    recent: float = 6.5,
    season: float = 6.5,
    opp: float = 0.25,
    win_every: int = 2,
) -> list:
    """Cells the Phase-0 gates actually fire on, priced off the model's own prob.

    Price is derived from model_pct so the fixture cannot silently drift out of
    the 25–55% band (the bug that made the previous fixture trade zero cells).
    Outcomes alternate rather than always winning, so ROI is earned, not rigged.
    win_every=0 makes every cell lose.
    """
    model_pct = model.prob_ge(threshold, recent, season, opp) * 100.0
    assert BAND_LO <= model_pct <= BAND_HI, (
        f"fixture prob {model_pct:.1f}% outside Phase-0 band [{BAND_LO},{BAND_HI}]"
    )
    price = model_pct - MIN_EDGE - COST_BUFFER - 1.0  # net edge = 21¢
    assert price > 0, f"derived price {price:.1f}¢ must be tradeable"

    cells = []
    for i in range(n):
        hit = win_every > 0 and (i % win_every) == 0
        cells.append({
            "date": f"2026-04-{10 + (i % 10):02d}",
            "threshold": threshold,
            "recent_k": recent,
            "season_k": season,
            "opp_k_rate": opp,
            "actual_k": float(threshold + 2) if hit else float(threshold - 3),
            "market_price_cents": price,
            "synthetic_market": False,
        })
    return cells


def _losing_journal(n: int = 20) -> list:
    records = []
    for i in range(n):
        records.append({
            "type": "trade", "ticker": f"B{i}", "strategy": "pitcher_ks",
            "cost_usd": 1.0, "date": "2026-05-01",
        })
        records.append({
            "type": "settlement", "ticker": f"B{i}",
            "market_result": "no", "pnl_usd": -1.0,
        })
    return records


# ── point-in-time sampling ────────────────────────────────────────────────────

def test_samples_use_only_prior_starts_and_actual_k():
    samples = samples_from_pitcher_game_logs(
        _synthetic_logs(), as_of="2026-04-20", min_prior_starts=2
    )
    assert len(samples) >= 5
    assert all(s["date"] < "2026-04-20" for s in samples)
    assert "actual_k" in samples[0]


# ── regularization / fit sanity ───────────────────────────────────────────────

def test_ridge_keeps_collinear_coefficients_bounded():
    """recent_k and season_k are collinear; the fit must not use cancelling weights.

    Regression test: an unstandardized 1e-3 penalty produced coef
    [15.4, -14.1, -0.19] and λ=17.7 Ks for an ordinary starter.
    """
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert model.n_samples >= 20
    assert all(abs(c) < 3.0 for c in model.coef), f"unbounded coef: {model.coef}"
    # Both K features should push λ the same direction, not fight each other
    assert model.coef[0] > 0
    assert model.coef[1] >= 0


def test_constant_feature_gets_zero_coefficient():
    """opp_k_rate is constant across training rows, so it must not absorb noise."""
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    assert len({s["opp_k_rate"] for s in samples}) == 1
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert model.coef[2] == 0.0


def test_fitted_lambda_is_plausible_and_monotone():
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    lams = [model.predict_lambda(k, k, 0.225) for k in (2.0, 4.0, 6.0, 8.0, 10.0)]
    assert lams == sorted(lams), f"λ not monotone in K rate: {lams}"
    assert all(0.0 < lam <= LAMBDA_MAX for lam in lams), lams
    # An average starter should land in a believable range
    mid = model.predict_lambda(6.0, 6.0, 0.225)
    assert 3.5 <= mid <= 9.0, f"λ={mid:.2f} implausible for a 6-K/start pitcher"


def test_lambda_clamped_on_extreme_extrapolation():
    wild = KsModel(intercept=50.0, coef=[5.0, 5.0, 5.0], n_samples=99)
    assert wild.predict_lambda(30.0, 30.0, 1.0) == pytest.approx(LAMBDA_MAX)
    assert wild.prob_ge(7, 30.0, 30.0, 1.0) <= 1.0


# ── holdout Brier vs real market ──────────────────────────────────────────────

def test_holdout_brier_uses_real_varied_market_prices():
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    props = []
    for i, s in enumerate(samples[-10:]):
        props.append({
            **s,
            "threshold": 6 + (i % 3),
            "market_price_cents": 20.0 + i * 3,  # varied, not a constant sentinel
            "synthetic_market": False,
        })
    model = fit_ks_model(
        samples, as_of="2026-05-01", holdout_frac=0.25, holdout_props=props
    )
    assert model.holdout_model_brier is not None
    assert model.holdout_market_brier is not None
    assert model.holdout_beats_market is (
        model.holdout_model_brier < model.holdout_market_brier
    )


def test_holdout_brier_rejects_synthetic_market_flag():
    """A synthetic price is not evidence — must yield no Brier verdict at all."""
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    props = [{
        "recent_k": 6.0, "season_k": 6.0, "opp_k_rate": 0.22,
        "threshold": 7, "actual_k": 8.0, "market_price_cents": 30.0,
        "synthetic_market": True,
    }] * 10
    mb, kb, beats = holdout_brier_vs_market(model, props)
    assert (mb, kb, beats) == (None, None, None)


def test_fit_without_holdout_props_no_synthetic_brier():
    samples = samples_from_pitcher_game_logs(
        _synthetic_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.25, holdout_props=None)
    assert model.holdout_beats_market is None
    assert model.holdout_model_brier is None


def test_holdout_props_carry_real_signal_price():
    """build_holdout_props_from_signals must propagate the signal's own price."""
    logs = {
        "Someone Ace": [
            {"date": f"2026-04-{d:02d}", "strikeouts": 6 + (d % 3)} for d in range(1, 15)
        ]
    }
    signals = [{
        "ticker": "KXMLBKS-26APR101900PITHOU-PITPACE30-7",
        "strategy": "pitcher_ks",
        "model_prob_pct": 40,
        "market_price_cents": 28,
        "ask_cents": 28,
        "date": "2026-04-10",
        "reason": "λ=6.0Ks  P(≥7)=40%",
    }]
    props = build_holdout_props_from_signals(signals, logs, as_of="2026-05-01")
    for p in props:
        assert p["market_price_cents"] == 28.0
        assert p["synthetic_market"] is False


# ── model-scored ROI vs Phase-0 baseline ──────────────────────────────────────

def test_model_roi_invokes_trained_probabilities():
    """The ROI gate must score cells with model.prob_ge, not journal edge_cents."""
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    cells = _in_band_cells(model, 15)

    calls = []
    orig = model.prob_ge

    def spy(thr, recent, season, opp=0.0):
        calls.append(thr)
        return orig(thr, recent, season, opp)

    model.prob_ge = spy  # type: ignore[method-assign]

    result = model_roi_vs_phase0_baseline(
        model, cells, _losing_journal(20), min_n=5, cost_buffer_cents=COST_BUFFER
    )

    assert len(calls) == 15, "prob_ge must be consulted for every candidate cell"
    assert result["model"]["n"] == 15
    assert result["status"] == "ok"
    assert result["model"]["roi_pct"] > 0
    assert result["baseline"]["roi_pct"] < 0
    assert result["not_worse_than_baseline"] is True


def test_model_roi_respects_phase0_gates():
    """Below-band, above-band, low-threshold and expensive cells must not trade."""
    model = KsModel(intercept=math.log(6.2), coef=[0.0, 0.0, 0.0], n_samples=50)
    base = {
        "recent_k": 6.5, "season_k": 6.5, "opp_k_rate": 0.25,
        "actual_k": 9.0, "synthetic_market": False,
    }
    model_pct = model.prob_ge(7, 6.5, 6.5, 0.25) * 100.0
    assert BAND_LO <= model_pct <= BAND_HI

    rejected = [
        {**base, "threshold": 5, "market_price_cents": 10.0},   # threshold < 6
        {**base, "threshold": 7, "market_price_cents": 30.0},   # net edge < 20¢
        {**base, "threshold": 7, "market_price_cents": 0.0},    # no market
        {**base, "threshold": 7, "market_price_cents": 10.0, "synthetic_market": True},
        {**base, "threshold": 12, "market_price_cents": 1.0},   # prob below band
    ]
    result = model_roi_vs_phase0_baseline(
        model, rejected, [], min_n=1, cost_buffer_cents=COST_BUFFER
    )
    assert result["model"]["n"] == 0
    assert result["status"] == "insufficient_data"


def test_model_roi_reports_worse_than_baseline():
    """A model that trades but loses must be reported worse, not waved through."""
    model = KsModel(intercept=math.log(6.2), coef=[0.0, 0.0, 0.0], n_samples=50)
    cells = _in_band_cells(model, 12, win_every=0)  # every cell loses
    winning_journal = [
        {"type": "trade", "ticker": "W0", "strategy": "pitcher_ks",
         "cost_usd": 1.0, "date": "2026-05-01"},
        {"type": "settlement", "ticker": "W0", "market_result": "yes", "pnl_usd": 1.0},
    ]
    result = model_roi_vs_phase0_baseline(
        model, cells, winning_journal, min_n=5, cost_buffer_cents=COST_BUFFER
    )
    assert result["model"]["n"] == 12
    assert result["model"]["roi_pct"] == -100.0
    assert result["status"] == "worse_than_baseline"
    assert result["not_worse_than_baseline"] is False


def test_model_roi_drops_in_sample_cells():
    """ROI may only be claimed on dates after the walk-forward holdout boundary."""
    model = KsModel(
        intercept=math.log(6.2), coef=[0.0, 0.0, 0.0], n_samples=50,
        holdout_from="2026-04-15",
    )
    cells = _in_band_cells(model, 12)
    for i, c in enumerate(cells):
        c["date"] = "2026-04-10" if i < 6 else "2026-04-20"

    result = model_roi_vs_phase0_baseline(
        model, cells, [], min_n=1, cost_buffer_cents=COST_BUFFER
    )
    assert result["n_in_sample_dropped"] == 6
    assert result["n_cells_scored"] == 6
    assert result["model"]["n"] == 6
    assert result["holdout_from"] == "2026-04-15"

    # Opting out scores everything — kept only for diagnostics, never for the gate
    unfiltered = model_roi_vs_phase0_baseline(
        model, cells, [], min_n=1, cost_buffer_cents=COST_BUFFER, holdout_only=False
    )
    assert unfiltered["n_in_sample_dropped"] == 0
    assert unfiltered["model"]["n"] == 12


def test_fit_records_holdout_boundary():
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.25)
    assert model.holdout_from is not None
    train_dates = sorted(s["date"] for s in samples)
    assert model.holdout_from in train_dates
    # Nothing on/after the boundary may be in the training count
    n_before = sum(1 for d in train_dates if d < model.holdout_from)
    assert model.n_samples == n_before

    # No holdout requested → no boundary to claim ROI from
    full = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert full.holdout_from is None


def test_single_date_samples_have_no_holdout_boundary():
    """If every start shares one date there is no clean split — claim no holdout."""
    samples = [
        {"date": "2026-04-10", "recent_k": 5.0 + i * 0.1, "season_k": 5.0,
         "opp_k_rate": 0.225, "actual_k": 5.0 + (i % 3)}
        for i in range(12)
    ]
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.25)
    assert model.holdout_from is None
    assert model.n_samples == 12


def test_holdout_boundary_survives_save_load(tmp_path):
    model = KsModel(
        intercept=1.2, coef=[0.8, 0.4, 0.0], n_samples=42, holdout_from="2026-04-15"
    )
    path = tmp_path / "ks_model.json"
    model.save(str(path))
    loaded = KsModel.load(str(path))
    assert loaded is not None
    assert loaded.holdout_from == "2026-04-15"


def test_broken_model_fails_not_worse():
    """Degenerate model never clears the gates → insufficient data, fail closed."""
    broken = KsModel(intercept=-10.0, coef=[0.0, 0.0, 0.0], n_samples=10, as_of="2026-05-01")
    good = KsModel(intercept=math.log(6.2), coef=[0.0, 0.0, 0.0], n_samples=50)
    cells = _in_band_cells(good, 15)
    result = model_roi_vs_phase0_baseline(
        broken, cells, _losing_journal(2), min_n=5, cost_buffer_cents=COST_BUFFER
    )
    assert result["model"]["n"] == 0
    assert result["not_worse_than_baseline"] is False
    assert result["status"] == "insufficient_data"


def test_empty_cells_not_worse_is_false():
    model = KsModel(intercept=1.0, coef=[0.5, 0.3, 0.0], n_samples=10)
    result = model_roi_vs_phase0_baseline(model, [], [], min_n=10)
    assert result["not_worse_than_baseline"] is False
    assert result["status"] == "insufficient_data"


def test_journal_roi_edge_filter():
    records = _losing_journal(5)
    for i in range(5):
        records.append({
            "type": "trade", "ticker": f"HI{i}", "strategy": "pitcher_ks",
            "cost_usd": 1.0, "edge_cents": 25.0, "date": "2026-05-02",
        })
        records.append({
            "type": "settlement", "ticker": f"HI{i}",
            "market_result": "yes", "pnl_usd": 2.0,
        })
    base = journal_roi_for_strategy(records, "pitcher_ks")
    gated = journal_roi_for_strategy(records, "pitcher_ks", min_edge_cents=20.0)
    assert base["n"] == 10
    assert gated["n"] == 5
    assert gated["roi_pct"] > base["roi_pct"]


def test_expected_ks_uses_trained_model(tmp_path, monkeypatch):
    clear_ks_model_cache()
    samples = samples_from_pitcher_game_logs(
        _synthetic_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    path = tmp_path / "ks_model.json"
    model.save(str(path))
    clear_ks_model_cache()
    loaded = get_trained_ks_model(str(path))
    monkeypatch.setattr("slugger.ks_model.get_trained_ks_model", lambda path=None: loaded)
    profile = PitcherProfile(
        player_id=1, name="T", recent_k_per_start=6.0, recent_ip_per_start=6.0,
        k_per_9=9.0, max_k_in_start=10,
    )
    lam = expected_ks(profile, 0.22, use_trained=True)
    assert lam > 0
    clear_ks_model_cache()
