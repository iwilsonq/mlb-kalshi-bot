"""Tests for mlb_data _apply_* helper functions.

These functions translate raw API / Statcast data into profile fields.
Tests use synthetic inputs so no network calls are made.
"""
import pandas as pd
import pytest

from slugger.mlb_data import _apply_batter_game_log, _apply_pitcher_statcast
from slugger.types import BatterProfile, PitcherProfile


# ── helpers ───────────────────────────────────────────────────────────────────

def _empty_batter() -> BatterProfile:
    return BatterProfile(player_id=1, name="Test", team="NYY")


def _empty_pitcher() -> PitcherProfile:
    return PitcherProfile(player_id=2, name="Test")


def _game(hits=0, ab=4, hr=0, doubles=0, triples=0, bb=0) -> dict:
    """Build a single game-log entry in the shape the MLB Stats API returns."""
    return {"stat": {
        "hits": str(hits),
        "atBats": str(ab),
        "homeRuns": str(hr),
        "doubles": str(doubles),
        "triples": str(triples),
        "baseOnBalls": str(bb),
    }}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cycle 1 — recent_ops is true OPS (OBP + SLG), not OBP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApplyBatterGameLog:
    def test_recent_ops_is_ops_not_obp(self):
        """recent_ops must be OBP + SLG, not (H+BB)/(AB+BB).

        Given: 2 games, each with 1 hit (a double), 3 AB, 1 BB
          OBP = (2H + 2BB) / (6AB + 2BB) = 4/8 = 0.500
          SLG = (2×2 doubles) / 6AB = 4/6 ≈ 0.667
          OPS = 0.500 + 0.667 ≈ 1.167

        The old (wrong) formula computed OBP = 0.500, not OPS = 1.167.
        """
        games = [_game(hits=1, ab=3, doubles=1, bb=1)] * 2
        profile = _empty_batter()
        _apply_batter_game_log(profile, games)

        # OBP = (2 + 2) / (6 + 2) = 0.500
        obp = (2 + 2) / (6 + 2)
        # SLG = 4 total bases / 6 AB = 0.6667
        slg = 4 / 6
        expected_ops = obp + slg

        assert profile.recent_ops == pytest.approx(expected_ops, abs=0.001), (
            f"recent_ops={profile.recent_ops:.3f} but expected OPS≈{expected_ops:.3f}; "
            "formula may be computing OBP instead of OPS"
        )

    def test_recent_ops_with_home_runs(self):
        """Home runs contribute 4 total bases each to SLG."""
        games = [_game(hits=1, ab=4, hr=1, bb=0)]  # solo HR, nothing else
        profile = _empty_batter()
        _apply_batter_game_log(profile, games)

        obp = 1 / 4         # H / AB (no BB)
        slg = 4 / 4         # 4 TB / 4 AB = 1.000
        expected_ops = obp + slg  # 1.250

        assert profile.recent_ops == pytest.approx(expected_ops, abs=0.001)

    def test_recent_avg_and_hr_still_correct(self):
        """Fixing OPS must not break recent_avg or recent_hr."""
        games = [
            _game(hits=2, ab=4, hr=1),
            _game(hits=0, ab=3, hr=0),
        ]
        profile = _empty_batter()
        _apply_batter_game_log(profile, games)

        assert profile.recent_avg == pytest.approx(2 / 7, abs=0.001)
        assert profile.recent_hr == 1

    def test_uses_last_7_games_only(self):
        """Only the last 7 games contribute to recent stats."""
        # 10 games: first 3 have HRs that should be ignored
        old_games = [_game(hits=0, ab=4, hr=1)] * 3
        recent_games = [_game(hits=1, ab=4, hr=0)] * 7
        profile = _empty_batter()
        _apply_batter_game_log(profile, old_games + recent_games)

        assert profile.recent_hr == 0, "HRs from games older than 7 should be excluded"
        assert profile.recent_avg == pytest.approx(7 / 28, abs=0.001)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cycle 2 — barrel_rate_against populated from pitcher Statcast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _pitcher_statcast_df(**col_overrides) -> pd.DataFrame:
    """Minimal Statcast DataFrame for a pitcher (pitch-level rows)."""
    base = {
        "pitch_type": ["FF"] * 10,
        "release_speed": [93.0] * 10,
        "description": ["called_strike"] * 10,
        "zone": [5] * 10,          # all in zone
        "launch_speed": [None] * 10,
        "launch_angle": [None] * 10,
        "barrel": [None] * 10,
        "estimated_woba_using_speedangle": [None] * 10,
    }
    base.update(col_overrides)
    return pd.DataFrame(base)


class TestApplyPitcherStatcastBarrel:
    def test_barrel_rate_against_populated_from_barrel_column(self):
        """When the Statcast barrel column is present, barrel_rate_against
        should be barrels / total batted balls."""
        # 10 pitches: 4 batted balls, 1 is a barrel
        launch_speeds = [95.0, 88.0, 102.0, 75.0] + [None] * 6
        barrels       = [0,    0,    1,     0]    + [None] * 6
        df = _pitcher_statcast_df(
            launch_speed=launch_speeds,
            barrel=barrels,
        )
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        # 1 barrel / 4 batted balls = 0.25
        assert profile.barrel_rate_against == pytest.approx(0.25, abs=0.001)

    def test_barrel_rate_against_zero_when_no_batted_balls(self):
        """No batted-ball data means barrel_rate_against stays 0."""
        df = _pitcher_statcast_df()  # all launch_speed = None
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        assert profile.barrel_rate_against == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cycle 3 — chase_rate populated from pitcher Statcast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApplyPitcherStatcastChaseRate:
    def test_chase_rate_is_swings_on_pitches_outside_zone(self):
        """chase_rate = swings on out-of-zone pitches / total out-of-zone pitches.

        Zones 11-14 are outside the strike zone.
        A swing outside the zone is any description that is not 'ball',
        'blocked_ball', or 'called_strike' — e.g. swinging_strike, foul, hit_into_play.

        Setup: 10 pitches total
          - 6 in zone  (zones 1-9): irrelevant for chase rate
          - 4 out of zone (zone 11): 2 swinging strikes (chases), 2 balls (no chase)
        Expected chase_rate = 2 / 4 = 0.50
        """
        zones       = [5] * 6 + [11] * 4
        description = (
            ["called_strike"] * 6 +
            ["swinging_strike", "swinging_strike", "ball", "ball"]
        )
        df = _pitcher_statcast_df(zone=zones, description=description)
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        assert profile.chase_rate == pytest.approx(0.50, abs=0.001)

    def test_chase_rate_zero_when_no_out_of_zone_pitches(self):
        """If all pitches are in the zone, chase_rate stays 0."""
        df = _pitcher_statcast_df(
            zone=[5] * 10,
            description=["called_strike"] * 10,
        )
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        assert profile.chase_rate == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cycle 4 — xera populated from pitcher Statcast
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestApplyPitcherStatcastXERA:
    def test_xera_derived_from_xwoba_on_contact(self):
        """xera is estimated from mean xwOBA on contact via a linear scale.

        The mapping used: xERA = (xwOBA / 0.320) * 4.00
        so xwOBA 0.320 → xERA 4.00 (league average),
           xwOBA 0.400 → xERA 5.00 (hitter-friendly),
           xwOBA 0.240 → xERA 3.00 (pitcher-dominant).

        Only pitches with a non-null estimated_woba_using_speedangle contribute
        (i.e. balls in play / contact pitches only).
        """
        # 4 contact pitches with xwOBA values; 6 non-contact (None)
        xwoba = [0.400, 0.200, 0.400, 0.200] + [None] * 6
        df = _pitcher_statcast_df(estimated_woba_using_speedangle=xwoba)
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        mean_xwoba = (0.400 + 0.200 + 0.400 + 0.200) / 4  # = 0.300
        expected_xera = (mean_xwoba / 0.320) * 4.00        # = 3.75

        assert profile.xera == pytest.approx(expected_xera, abs=0.01)

    def test_xera_zero_when_no_contact_data(self):
        """xera stays 0 when estimated_woba column is absent or all null."""
        df = _pitcher_statcast_df()  # column present but all None
        profile = _empty_pitcher()
        _apply_pitcher_statcast(profile, df)

        assert profile.xera == 0.0
