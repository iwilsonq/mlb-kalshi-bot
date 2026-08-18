"""Unit tests for the in-game win probability model (slugger.wp)."""
import math
import random

import pytest

from slugger.wp.fetch import extract_pa_states
from slugger.wp.model import WPModel, evaluate, get_wp, clear_wp_model_cache


# ─── Synthetic data ───────────────────────────────────────────────────────────

def _half_remaining(inning, is_top, outs):
    return max(1.0 / 3.0, (9 - inning) * 2 + (2 if is_top else 1) - outs / 3.0)


def _true_p(inning, is_top, outs, diff):
    z = 0.12 + 0.9 * diff / math.sqrt(_half_remaining(inning, is_top, outs))
    return 1.0 / (1.0 + math.exp(-z))


def _synthetic_rows(n=40000, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inning = rng.randint(1, 9)
        is_top = rng.random() < 0.5
        outs = rng.randint(0, 2)
        diff = rng.randint(-6, 6)
        if inning == 9 and not is_top and diff > 0:
            # Home leading in the bottom 9th never happens (walk-off ends game)
            diff = -diff
        rows.append({
            "inning": inning,
            "is_top": is_top,
            "outs": outs,
            "on1": rng.random() < 0.25,
            "on2": rng.random() < 0.15,
            "on3": rng.random() < 0.08,
            "score_diff": diff,
            "home_win": 1 if rng.random() < _true_p(inning, is_top, outs, diff) else 0,
            "date": "2025-06-15",
        })
    return rows


@pytest.fixture(scope="module")
def model():
    return WPModel.fit(_synthetic_rows())


# ─── Monotonicity / sanity ────────────────────────────────────────────────────

def test_leading_team_wp_above_half(model):
    # Home leading mid-game
    assert model.predict(5, True, 1, 2) > 0.5
    # Away leading mid-game → home WP below 0.5
    assert model.predict(5, True, 1, -2) < 0.5


def test_bigger_lead_higher_wp(model):
    ps = [model.predict(6, False, 1, d) for d in (0, 1, 2, 4, 6)]
    for a, b in zip(ps, ps[1:]):
        assert b > a


def test_later_lead_worth_more(model):
    early = model.predict(2, True, 0, 3)
    late = model.predict(8, True, 0, 3)
    assert late > early


def test_tie_game_inning1_near_home_field(model):
    p = model.predict(1, True, 0, 0)
    assert 0.50 < p < 0.60


def test_bottom_9th_walkoff_states_sane(model):
    # Down 1, bottom 9, 2 outs, bases empty: alive but unlikely
    p_down1 = model.predict(9, False, 2, -1)
    assert 0.0 < p_down1 < 0.5
    # Tied bottom 9 with runner on third, 0 outs: strong home advantage
    p_walkoff = model.predict(9, False, 0, 0, on3=True)
    assert p_walkoff > 0.5
    # Home up 3 entering top 9th: heavy favourite
    assert model.predict(9, True, 0, 3) > 0.8
    # Probabilities always clamped to (0, 1)
    assert 0.001 <= model.predict(9, False, 2, -8) <= 0.999


def test_extra_innings_and_clamps(model):
    # Inning >9 caps at 9; diff beyond ±8 clamps
    assert model.predict(12, True, 0, 0) == model.predict(9, True, 0, 0)
    assert model.predict(5, True, 0, 15) == model.predict(5, True, 0, 8)


# ─── Serialization ────────────────────────────────────────────────────────────

def test_serialization_round_trip(model, tmp_path):
    path = str(tmp_path / "wp_model.json")
    model.save(path)
    loaded = WPModel.load(path)
    assert loaded is not None
    assert loaded.n_train == model.n_train
    assert loaded.beta0 == pytest.approx(model.beta0)
    for state in [(1, True, 0, 0), (7, False, 2, -3), (9, True, 1, 2)]:
        assert loaded.predict(*state) == pytest.approx(model.predict(*state))

    # module-level get_wp uses the artifact
    clear_wp_model_cache()
    wp = get_wp({"inning": 7, "is_top": False, "outs": 2,
                 "score_diff": -3, "on1": True}, path=path)
    assert wp == pytest.approx(model.predict(7, False, 2, -3, on1=True))
    clear_wp_model_cache()


def test_load_missing_returns_none(tmp_path):
    assert WPModel.load(str(tmp_path / "nope.json")) is None


def test_evaluate_beats_constant_baseline(model):
    rows = _synthetic_rows(n=5000, seed=99)
    rep = evaluate(model, rows)
    assert rep["brier"] < rep["baseline_brier"]
    assert rep["logloss"] < rep["baseline_logloss"]


# ─── GUMBO state extraction ───────────────────────────────────────────────────

def _play(inning, half, outs_after, home, away, on1=False, on2=False, on3=False):
    return {
        "about": {"inning": inning, "halfInning": half, "isComplete": True},
        "count": {"outs": outs_after},
        "result": {"homeScore": home, "awayScore": away},
        "matchup": {
            "postOnFirst": {"id": 1} if on1 else None,
            "postOnSecond": {"id": 2} if on2 else None,
            "postOnThird": {"id": 3} if on3 else None,
        },
    }


def test_extract_pa_states_fixture():
    plays = [
        # Top 1: single (0 outs after), then double play (2 outs), then K (3 outs)
        _play(1, "top", 0, 0, 0, on1=True),
        _play(1, "top", 2, 0, 0),
        _play(1, "top", 3, 0, 0),
        # Bottom 1: HR (0 outs, 1-0 home), then out
        _play(1, "bottom", 0, 1, 0),
        _play(1, "bottom", 1, 1, 0),
        # Top 2 (fast-forward): away scores 2 on first play
        _play(2, "top", 0, 1, 2, on2=True),
        # ... final play: home walks it off in the 9th (2-run swing on the play)
        _play(9, "bottom", 1, 3, 2),
    ]
    rows = extract_pa_states(plays)
    assert rows is not None
    assert len(rows) == 7
    # All rows labelled home win (final 3-2)
    assert all(r["home_win"] == 1 for r in rows)

    # First PA of the game: clean slate
    assert rows[0] == {
        "inning": 1, "is_top": True, "outs": 0,
        "on1": False, "on2": False, "on3": False,
        "score_diff": 0, "home_win": 1,
    }
    # Second PA: runner on first from previous single, still 0 outs
    assert rows[1]["on1"] is True and rows[1]["outs"] == 0
    # Third PA: double play cleared bases, 2 outs
    assert rows[2]["outs"] == 2 and rows[2]["on1"] is False
    # Bottom 1 first PA: half-inning reset (outs=0, bases empty), score 0-0
    assert rows[3]["is_top"] is False and rows[3]["outs"] == 0
    assert rows[3]["score_diff"] == 0 and rows[3]["on1"] is False
    # Bottom 1 second PA: after the HR score is 1-0 home
    assert rows[4]["score_diff"] == 1
    # Top 2 first PA: reset again, score carries 1-0
    assert rows[5]["is_top"] is True and rows[5]["outs"] == 0
    assert rows[5]["score_diff"] == 1
    # 9th-inning PA: score before the walk-off play was 1-2 (home down 1)
    assert rows[6]["inning"] == 9 and rows[6]["score_diff"] == -1


def test_extract_pa_states_tie_or_empty_returns_none():
    assert extract_pa_states([]) is None
    tied = [_play(9, "bottom", 3, 4, 4)]
    assert extract_pa_states(tied) is None


def test_extract_pa_states_clamps_diff_and_inning():
    plays = [
        _play(10, "top", 0, 0, 0),          # extra innings → inning capped at 9
        _play(10, "top", 1, 12, 0),         # blowout by 12 → diff clamped
        _play(10, "top", 2, 12, 0),
    ]
    rows = extract_pa_states(plays)
    assert rows is not None
    assert rows[0]["inning"] == 9
    assert rows[2]["score_diff"] == 8
