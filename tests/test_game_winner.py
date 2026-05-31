"""Tests for game_winner_probability model improvements."""
from slugger.models import game_winner_probability, pythagorean_win_pct
from slugger.types import PitcherProfile, TeamProfile


def _avg_team(**overrides) -> TeamProfile:
    defaults = dict(
        name="Test Team",
        abbrev="TST",
        team_id=100,
        runs_per_game=4.50,
        team_era=4.10,
        team_whip=1.28,
        bullpen_era=0.0,
        wins=40,
        losses=40,
        run_diff=0,
    )
    defaults.update(overrides)
    return TeamProfile(**defaults)


def _avg_pitcher(**overrides) -> PitcherProfile:
    defaults = dict(
        player_id=1,
        name="Test Pitcher",
        era=4.10,
    )
    defaults.update(overrides)
    return PitcherProfile(**defaults)


class TestBullpenERAImpact:
    """When bullpen_era is populated, it should meaningfully affect
    the game_winner probability. A team with a great bullpen (ERA 2.50)
    should be favored over one with a bad bullpen (ERA 5.50).
    """

    def test_good_bullpen_increases_win_prob(self):
        """A home team with an elite bullpen should get a higher win
        probability than the same team with a bad bullpen.
        """
        pitcher = _avg_pitcher()
        good_bp = _avg_team(bullpen_era=2.50)
        bad_bp = _avg_team(bullpen_era=5.50)
        neutral = _avg_team()

        prob_good, _ = game_winner_probability(pitcher, pitcher, good_bp, neutral)
        prob_bad, _ = game_winner_probability(pitcher, pitcher, bad_bp, neutral)

        assert prob_good > prob_bad, (
            f"Good bullpen prob={prob_good}% should be > "
            f"bad bullpen prob={prob_bad}%"
        )

    def test_bullpen_era_zero_falls_back_gracefully(self):
        """When bullpen_era is 0 (not populated), the model should
        still produce a reasonable result (bullpen factor = 1.0).
        """
        pitcher = _avg_pitcher()
        team_no_bp = _avg_team(bullpen_era=0.0)

        prob, _ = game_winner_probability(pitcher, pitcher, team_no_bp, team_no_bp)
        assert prob == 54, (
            f"Equal teams with no bullpen data should get ~54% home prob, got {prob}%"
        )


class TestPythagoreanWinPct:
    """pythagorean_win_pct should estimate true team quality from
    run differential, independent of actual win-loss record.
    """

    def test_positive_run_diff_above_500(self):
        """A team that outscores opponents should have a Pythagorean
        win% above .500.
        """
        # 80 games, +40 run diff (~0.5 runs/game better)
        pct = pythagorean_win_pct(runs_scored=4.7, runs_allowed=4.2)
        assert pct > 0.500, f"Expected >.500, got {pct:.3f}"

    def test_negative_run_diff_below_500(self):
        pct = pythagorean_win_pct(runs_scored=3.8, runs_allowed=4.5)
        assert pct < 0.500, f"Expected <.500, got {pct:.3f}"

    def test_equal_runs_equals_500(self):
        pct = pythagorean_win_pct(runs_scored=4.5, runs_allowed=4.5)
        assert abs(pct - 0.500) < 0.001, f"Expected ~.500, got {pct:.3f}"

    def test_used_in_game_winner(self):
        """The game_winner model should use Pythagorean win% instead
        of raw record when run_diff data is available.

        A team with a lucky 50-30 record but negative run differential
        should be rated lower than their record suggests.
        """
        pitcher = _avg_pitcher()
        # Lucky team: great record but negative run diff
        lucky = _avg_team(wins=50, losses=30, run_diff=-20, runs_per_game=4.0)
        # Unlucky team: bad record but positive run diff
        unlucky = _avg_team(wins=30, losses=50, run_diff=+20, runs_per_game=5.0)

        prob_lucky, _ = game_winner_probability(pitcher, pitcher, lucky, unlucky)
        # The unlucky team has better underlying quality (positive run diff
        # + higher runs/game), so the lucky team shouldn't get a huge edge
        # from its misleading record alone.
        assert prob_lucky < 60, (
            f"Lucky team home prob={prob_lucky}%; expected <60% since "
            f"Pythagorean record should temper the misleading W-L"
        )
