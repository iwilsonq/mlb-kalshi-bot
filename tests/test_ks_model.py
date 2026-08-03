"""Tests for trained pitcher_ks — real outcomes, real markets, model-scored ROI."""
import math

import pytest

from slugger.ks_model import (
    LAMBDA_MAX,
    LEAGUE_AVG_K_RATE,
    KsModel,
    build_holdout_props_from_signals,
    clear_ks_model_cache,
    fit_and_save_ks_model,
    fit_ks_model,
    format_ks_fit_report,
    get_trained_ks_model,
    holdout_brier_vs_incumbent,
    holdout_brier_vs_market,
    journal_roi_for_strategy,
    model_roi_vs_phase0_baseline,
    samples_from_pitcher_game_logs,
    team_k_rate_as_of,
)
from slugger.models import expected_ks, poisson_ge
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
    """Without team logs opp_k_rate is constant, so it must not absorb noise."""
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    assert len({s["opp_k_rate"] for s in samples}) == 1
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert model.coef[2] == 0.0


# ── point-in-time opponent K% ─────────────────────────────────────────────────

def _team_logs(k_per_game: dict, days: int = 40) -> dict:
    """team → per-game K/PA logs at a fixed strikeout rate (38 PA per game)."""
    return {
        team: [
            {"date": f"2026-03-{d:02d}" if d <= 31 else f"2026-04-{d - 31:02d}",
             "strikeouts": ks, "plate_appearances": 38}
            for d in range(1, days + 1)
        ]
        for team, ks in k_per_game.items()
    }


def test_team_k_rate_uses_only_prior_games():
    logs = {
        "Whiffers": [
            {"date": "2026-04-01", "strikeouts": 15, "plate_appearances": 38},
            {"date": "2026-04-02", "strikeouts": 15, "plate_appearances": 38},
            {"date": "2026-04-03", "strikeouts": 15, "plate_appearances": 38},
            # Games on/after the as_of date must not count
            {"date": "2026-04-04", "strikeouts": 0, "plate_appearances": 38},
        ]
    }
    rate = team_k_rate_as_of(logs, "Whiffers", "2026-04-04", min_pa=100)
    assert rate == pytest.approx(45 / 114)
    # Same team, earlier cutoff, less data than min_pa → fall back
    assert team_k_rate_as_of(logs, "Whiffers", "2026-04-02", min_pa=100) == LEAGUE_AVG_K_RATE


def test_team_k_rate_falls_back_on_unknown_team():
    logs = _team_logs({"Known": 10})
    assert team_k_rate_as_of(logs, "", "2026-05-01") == LEAGUE_AVG_K_RATE
    assert team_k_rate_as_of(logs, "Nobody", "2026-05-01") == LEAGUE_AVG_K_RATE
    assert team_k_rate_as_of({}, "Known", "2026-05-01") == LEAGUE_AVG_K_RATE
    # Key matching tolerates case/whitespace drift between data sources
    assert team_k_rate_as_of(logs, "  known ", "2026-05-01") != LEAGUE_AVG_K_RATE


def test_samples_resolve_real_opponent_k_rate():
    """With team logs, each start carries its own opponent K% instead of 0.225."""
    team_logs = _team_logs({"Whiffers": 15, "Contact": 5})
    pitcher_logs = {
        "Paul Ace": [
            {"date": f"2026-04-{d:02d}", "strikeouts": 6,
             "opponent": "Whiffers" if d % 2 else "Contact"}
            for d in range(1, 21)
        ]
    }
    samples = samples_from_pitcher_game_logs(
        pitcher_logs, as_of="2026-05-01", min_prior_starts=2, team_game_logs=team_logs
    )
    rates = {round(s["opp_k_rate"], 6) for s in samples}
    assert len(rates) == 2, rates
    assert max(rates) == pytest.approx(15 / 38)
    assert min(rates) == pytest.approx(5 / 38)
    assert all(s["opponent"] in ("Whiffers", "Contact") for s in samples)


def test_unresolvable_opponent_falls_back_to_league_average():
    team_logs = _team_logs({"Whiffers": 15})
    pitcher_logs = {
        "Paul Ace": [
            {"date": f"2026-04-{d:02d}", "strikeouts": 6, "opponent": "Mystery"}
            for d in range(1, 21)
        ]
    }
    samples = samples_from_pitcher_game_logs(
        pitcher_logs, as_of="2026-05-01", min_prior_starts=2, team_game_logs=team_logs
    )
    assert samples
    assert all(s["opp_k_rate"] == LEAGUE_AVG_K_RATE for s in samples)


def test_opp_k_rate_becomes_a_live_feature_when_it_varies():
    """The point of iwt: with real variance the fit must actually use opp K%.

    Regression: samples_from_pitcher_game_logs hardcoded opp_k_rate for every
    row, so the column had zero variance and the ridge zeroed it — opponent
    strength had no effect on λ even though the live path passes a real value.
    """
    team_logs = _team_logs({"Whiffers": 15, "Contact": 5})
    # Same pitcher quality throughout; only the opponent differs, and actual Ks
    # track the opponent, so opp_k_rate is the only feature that can explain it.
    pitcher_logs = {}
    for p in range(4):
        games = []
        for d in range(1, 25):
            vs_whiffers = (d + p) % 2 == 0
            games.append({
                "date": f"2026-04-{d:02d}",
                "strikeouts": 9 if vs_whiffers else 4,
                "opponent": "Whiffers" if vs_whiffers else "Contact",
            })
        pitcher_logs[f"Pitcher {p}"] = games

    samples = samples_from_pitcher_game_logs(
        pitcher_logs, as_of="2026-05-01", min_prior_starts=2, team_game_logs=team_logs
    )
    assert len({round(s["opp_k_rate"], 6) for s in samples}) == 2
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)

    assert model.coef[2] != 0.0, "opp_k_rate still ignored by the fit"
    # Higher opponent strikeout rate must raise λ, not lower it
    assert model.coef[2] > 0
    lam_vs_whiffers = model.predict_lambda(6.5, 6.5, 15 / 38)
    lam_vs_contact = model.predict_lambda(6.5, 6.5, 5 / 38)
    assert lam_vs_whiffers > lam_vs_contact


def test_poisson_fit_is_unbiased_on_the_mean():
    """The fit must recover E[K], not the geometric mean.

    Regression: fitting OLS on log(max(actual, 0.5)) and exponentiating recovers
    the geometric mean. On real starts that understated λ by 0.57 Ks (predicted
    3.95 vs actual 4.52) — and a model biased low can never report YES edge,
    which is why the 20¢ gate admitted zero cells out of 449.
    """
    # Counts with a long-ish tail, where geometric and arithmetic means diverge
    counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 8
    samples = []
    for i, k in enumerate(counts):
        samples.append({
            "date": f"2026-04-{(i % 28) + 1:02d}",
            "recent_k": 4.0 + (k / 4.0),
            "season_k": 4.0 + (k / 5.0),
            "opp_k_rate": 0.20 + (i % 5) * 0.01,
            "actual_k": float(k),
        })
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    preds = [
        model.predict_lambda(s["recent_k"], s["season_k"], s["opp_k_rate"])
        for s in samples
    ]
    mean_pred = sum(preds) / len(preds)
    mean_actual = sum(s["actual_k"] for s in samples) / len(samples)
    geometric = math.exp(
        sum(math.log(max(s["actual_k"], 0.5)) for s in samples) / len(samples)
    )

    assert geometric < mean_actual * 0.9, "fixture must actually separate the means"
    # Poisson IRLS matches the arithmetic mean closely...
    assert mean_pred == pytest.approx(mean_actual, rel=0.05)
    # ...and is clearly not the geometric mean the old estimator returned
    assert abs(mean_pred - mean_actual) < abs(mean_pred - geometric)


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

    # Each cell must be attributed to the gate that actually rejected it,
    # otherwise a zero-trade result is indistinguishable from a broken model.
    gates = result["rejected_by_gate"]
    assert gates["threshold_below_min"] == 1
    assert gates["no_market_price"] == 1
    assert gates["synthetic_market"] == 1
    assert gates["edge_below_min"] == 1
    assert gates["prob_below_band"] == 1
    assert sum(gates.values()) == len(rejected)


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


# ── end-to-end fit orchestration (the path that writes ks_model.json) ─────────

def test_fit_and_save_writes_a_usable_model(tmp_path):
    """The whole calibrate --fit Ks path, with the network fetches injected.

    logs/ks_model.json is the only thing that puts a trained model in front of
    the live bot; without it models.expected_ks silently uses the hand-tuned
    fallback. This asserts the orchestration actually produces a loadable file.
    """
    team_logs = _team_logs({"Whiffers": 15, "Contact": 5})
    game_logs = {}
    for p in range(4):
        game_logs[f"Pitcher {p}"] = [
            {"date": f"2026-04-{d:02d}",
             "strikeouts": 9 if (d + p) % 2 == 0 else 4,
             "opponent": "Whiffers" if (d + p) % 2 == 0 else "Contact"}
            for d in range(1, 25)
        ]
    path = tmp_path / "ks_model.json"

    report = fit_and_save_ks_model(
        signals=[],
        journal_records=[],
        game_logs=game_logs,
        team_game_logs=team_logs,
        as_of="2026-05-01",
        model_path=str(path),
        cost_buffer_cents=5.0,
    )

    assert report["status"] == "ok"
    assert report["distinct_opp_k_rates"] == 2
    assert path.exists(), "no model file written"

    clear_ks_model_cache()
    loaded = KsModel.load(str(path))
    assert loaded is not None
    assert loaded.n_samples == report["model"].n_samples
    assert loaded.coef[2] != 0.0  # opponent K% survived the round trip
    assert 0.0 < loaded.predict_lambda(6.5, 6.5, 0.25) <= LAMBDA_MAX

    text = format_ks_fit_report(report)
    assert "Ks model saved to" in text
    assert "MODEL_ROI" in text
    # No real market prices were supplied, so no Brier verdict may be claimed
    assert loaded.holdout_beats_market is None
    assert "does not beat market Brier" in text


def test_fit_and_save_reports_no_samples_instead_of_writing_junk(tmp_path):
    path = tmp_path / "ks_model.json"
    report = fit_and_save_ks_model(
        signals=[],
        journal_records=[],
        game_logs={"Solo": [{"date": "2026-04-01", "strikeouts": 5}]},
        team_game_logs=None,
        as_of="2026-05-01",
        model_path=str(path),
    )
    assert report["status"] == "no_samples"
    assert not path.exists(), "must not persist a model it could not fit"
    assert "Not enough game-log samples" in format_ks_fit_report(report)


def test_fit_report_flags_a_model_that_loses_to_the_market():
    """The report must say so plainly when there is no demonstrated edge."""
    beaten = KsModel(
        intercept=1.0, coef=[0.5, 0.3, 0.0], n_samples=50,
        holdout_model_brier=0.30, holdout_market_brier=0.20,
        holdout_beats_market=False,
    )
    report = {
        "status": "ok", "model": beaten, "model_path": "x.json",
        "n_samples": 50, "distinct_opp_k_rates": 3, "n_holdout_props": 40,
        "roi": {
            "status": "worse_than_baseline", "n_cells_scored": 40,
            "n_in_sample_dropped": 0, "not_worse_than_baseline": False,
            "model": {"n": 12.0, "roi_pct": -30.0},
            "baseline": {"n": 100.0, "roi_pct": -15.0},
        },
    }
    text = format_ks_fit_report(report)
    assert "beats_market=False" in text
    assert "does not beat market Brier" in text
    assert "Do not re-enable" in text


def test_unknown_opponent_serves_league_average_not_zero(tmp_path, monkeypatch):
    """opp_k_rate=0.0 means "unknown", and the trained model must read it that way.

    Regression: strategies.py initialises opp_k_rate=0.0 and only overwrites it
    if get_team_profile succeeds. The hand model treats 0.0 as "skip the
    adjustment", but the trained model has a real positive coefficient on the
    feature, so a failed team fetch looked like a lineup that never strikes out
    and cost roughly 30% of λ.
    """
    import slugger.ks_model as ks_module

    trained = KsModel(
        intercept=-0.722, coef=[0.475, 0.543, 1.591], n_samples=2315,
    )
    path = tmp_path / "ks_model.json"
    trained.save(str(path))
    monkeypatch.setattr(ks_module, "DEFAULT_MODEL_PATH", str(path))
    clear_ks_model_cache()

    profile = PitcherProfile(
        player_id=1, name="T", recent_k_per_start=10.0, recent_ip_per_start=6.0,
        k_per_9=15.0, max_k_in_start=12,
    )
    lam_unknown = expected_ks(profile, 0.0, use_trained=True)
    lam_league = expected_ks(profile, LEAGUE_AVG_K_RATE, use_trained=True)
    assert lam_unknown == pytest.approx(lam_league), (
        "unknown opponent must be served as the league average, "
        "matching the fallback training used"
    )
    # And a real opponent value still moves λ
    lam_high = expected_ks(profile, 0.30, use_trained=True)
    assert lam_high > lam_league
    clear_ks_model_cache()


def test_get_trained_model_path_is_overridable(tmp_path, monkeypatch):
    """DEFAULT_MODEL_PATH must be resolved at call time, not frozen at import."""
    import slugger.ks_model as ks_module

    clear_ks_model_cache()
    monkeypatch.setattr(
        ks_module, "DEFAULT_MODEL_PATH", str(tmp_path / "nope.json")
    )
    assert get_trained_ks_model() is None

    model = KsModel(intercept=1.0, coef=[0.5, 0.3, 0.2], n_samples=99)
    real = tmp_path / "yes.json"
    model.save(str(real))
    clear_ks_model_cache()
    monkeypatch.setattr(ks_module, "DEFAULT_MODEL_PATH", str(real))
    loaded = get_trained_ks_model()
    assert loaded is not None and loaded.n_samples == 99
    clear_ks_model_cache()


def test_incumbent_brier_scores_the_live_fallback_formula():
    """The incumbent baseline must score models.fallback_ks_lambda itself."""
    from slugger.models import fallback_ks_lambda

    props = [
        {"recent_k": 6.0, "season_k": 6.0, "opp_k_rate": 0.225, "threshold": 6,
         "actual_k": 7.0, "market_price_cents": 40.0, "synthetic_market": False}
        for _ in range(10)
    ]
    brier, n = holdout_brier_vs_incumbent(props)
    assert n == 10
    lam = fallback_ks_lambda(6.0, 6.0, 0.225)
    expected = (poisson_ge(6, lam) - 1.0) ** 2
    assert brier == pytest.approx(expected)


def test_incumbent_brier_needs_enough_rows():
    props = [{"recent_k": 6.0, "season_k": 6.0, "opp_k_rate": 0.225,
              "threshold": 6, "actual_k": 7.0, "market_price_cents": 40.0}]
    assert holdout_brier_vs_incumbent(props) == (None, 1)


def test_fit_records_beats_incumbent_verdict():
    samples = samples_from_pitcher_game_logs(
        _multi_pitcher_logs(), as_of="2026-05-01", min_prior_starts=2
    )
    props = [
        {**s, "threshold": 6 + (i % 3), "market_price_cents": 20.0 + i * 3,
         "synthetic_market": False}
        for i, s in enumerate(samples[-10:])
    ]
    model = fit_ks_model(
        samples, as_of="2026-05-01", holdout_frac=0.25, holdout_props=props
    )
    assert model.holdout_incumbent_brier is not None
    assert model.holdout_beats_incumbent is (
        model.holdout_model_brier < model.holdout_incumbent_brier
    )
    # Verdict must survive the round trip so `calibrate --fit` output is auditable
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "m.json")
    model.save(p)
    reloaded = KsModel.load(p)
    assert reloaded.holdout_incumbent_brier == model.holdout_incumbent_brier
    assert reloaded.holdout_beats_incumbent == model.holdout_beats_incumbent


def test_report_says_delete_the_artifact_when_worse_than_incumbent():
    losing = KsModel(
        intercept=1.0, coef=[0.5, 0.3, 0.0], n_samples=50,
        holdout_model_brier=0.30, holdout_market_brier=0.20,
        holdout_beats_market=False,
        holdout_incumbent_brier=0.25, holdout_beats_incumbent=False,
    )
    report = {
        "status": "ok", "model": losing, "model_path": "logs/ks_model.json",
        "n_samples": 50, "distinct_opp_k_rates": 3, "n_holdout_props": 40,
        "roi": {
            "status": "insufficient_data", "n_cells_scored": 40,
            "n_in_sample_dropped": 0, "not_worse_than_baseline": False,
            "model": {"n": 0.0, "roi_pct": 0.0},
            "baseline": {"n": 512.0, "roi_pct": -15.2},
        },
    }
    text = format_ks_fit_report(report)
    assert "beats_incumbent=False" in text
    assert "worse than the hand-tuned fallback" in text
    assert "delete logs/ks_model.json" in text


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


def test_fit_estimates_dispersion_and_uses_it():
    """The fit must measure overdispersion, not assume Poisson.

    Overdispersed counts (var > mean) must yield dispersion > 1, and prob_ge must
    then put more weight in the tail than Poisson would.
    """
    from slugger.models import poisson_ge

    # Same mean everywhere, but wildly varying outcomes -> var >> mean
    samples = []
    for i in range(120):
        samples.append({
            "date": f"2026-04-{(i % 28) + 1:02d}",
            "recent_k": 6.0, "season_k": 6.0, "opp_k_rate": 0.225,
            "actual_k": float(0 if i % 2 else 12),
        })
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert model.dispersion > 1.5, f"dispersion not detected: {model.dispersion}"

    lam = model.predict_lambda(6.0, 6.0, 0.225)
    assert model.prob_ge(9, 6.0, 6.0, 0.225) > poisson_ge(9, lam)


def test_well_specified_poisson_keeps_dispersion_near_one():
    samples = []
    counts = [4, 5, 6, 5, 4, 6, 5, 5, 6, 4] * 12  # tight spread around 5
    for i, k in enumerate(counts):
        samples.append({
            "date": f"2026-04-{(i % 28) + 1:02d}",
            "recent_k": 5.0, "season_k": 5.0, "opp_k_rate": 0.225,
            "actual_k": float(k),
        })
    model = fit_ks_model(samples, as_of="2026-05-01", holdout_frac=0.0)
    assert model.dispersion < 1.2, f"spurious overdispersion: {model.dispersion}"


def test_dispersion_survives_save_load(tmp_path):
    m = KsModel(intercept=1.0, coef=[0.4, 0.4, 1.0], n_samples=99, dispersion=1.13)
    p = tmp_path / "m.json"
    m.save(str(p))
    loaded = KsModel.load(str(p))
    assert loaded is not None
    assert loaded.dispersion == pytest.approx(1.13)
    # And an older artifact without the field must default to Poisson
    import json
    d = json.loads(p.read_text())
    del d["dispersion"]
    p.write_text(json.dumps(d))
    assert KsModel.load(str(p)).dispersion == 1.0
