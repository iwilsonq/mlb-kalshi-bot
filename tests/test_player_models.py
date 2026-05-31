"""Tests for player_hits and player_hr model math.

Tests exercise the public model functions (expected_hits_lambda,
expected_hr_lambda) to verify that lambda deflation, adjustment
capping, and recent-form blending produce calibrated outputs.
"""
from slugger.models import expected_hits_lambda, expected_hr_lambda, poisson_ge
from slugger.types import BatterProfile, PitcherProfile


def _avg_batter(**overrides) -> BatterProfile:
    """Build a league-average batter profile for testing."""
    defaults = dict(
        player_id=1,
        name="Test Batter",
        team="NYY",
        avg=0.250,
        obp=0.320,
        slg=0.420,
        ops=0.740,
        hr=15,
        ab=300,
        hits=75,
        k_rate=0.22,
        bb_rate=0.08,
        hr_per_ab=0.017,
        recent_avg=0.260,
        recent_ops=0.750,
        recent_hr=2,
        avg_exit_velo=88.5,
        barrel_rate=0.065,
        hard_hit_rate=0.370,
        xba=0.250,
        xslg=0.420,
        vs_lhp_avg=0.260,
        vs_lhp_hr=5,
        vs_lhp_ab=100,
        vs_rhp_avg=0.245,
        vs_rhp_hr=10,
        vs_rhp_ab=200,
        batting_order=3,
    )
    defaults.update(overrides)
    return BatterProfile(**defaults)


def _avg_pitcher(**overrides) -> PitcherProfile:
    """Build a league-average pitcher profile for testing."""
    defaults = dict(
        player_id=2,
        name="Test Pitcher",
        era=4.10,
        whip=1.28,
        k_per_9=8.5,
        bb_per_9=3.0,
        hr_per_9=1.1,
        innings_pitched=100.0,
        throws="R",
        recent_k_per_start=0.0,
        recent_ip_per_start=0.0,
        whiff_rate=0.0,
        chase_rate=0.0,
        avg_fastball_velo=0.0,
        max_k_in_start=0,
        xera=0.0,
        recent_era=0.0,
        barrel_rate_against=0.0,
    )
    defaults.update(overrides)
    return PitcherProfile(**defaults)


class TestHitsLambdaDeflation:
    """The hits model must apply a deflator to lambda, analogous to
    KS_LAMBDA_DEFLATOR for the strikeout model.

    The calibration data shows the model over-predicts by ~2:1 (raw 25%
    → actual 12%), so we need a deflator of roughly 0.80 to bring
    predictions in line with observed hit rates.
    """

    def test_league_avg_batter_lambda_below_one(self):
        """A league-average batter facing a league-average pitcher at a
        neutral park should have a hits lambda below 1.0.

        Without deflation, lambda is ~1.0 (0.25 avg * 4.1 PA).
        With deflation, it should be ~0.80.

        The 2+ hits probability for such a batter should be well under
        30% (the market typically prices this around 20-25%).
        """
        batter = _avg_batter()
        pitcher = _avg_pitcher()
        lam = expected_hits_lambda(batter, pitcher, "ATL")  # neutral park
        two_plus = poisson_ge(2, lam) * 100

        assert lam < 1.0, (
            f"League-avg batter lambda={lam:.3f}; expected <1.0 after deflation"
        )
        assert two_plus < 30, (
            f"P(2+ hits)={two_plus:.1f}%; expected <30% for league-avg matchup"
        )


class TestHRLambdaAdjustmentCap:
    """The HR model multiplies 7 adjustment factors together (pitcher,
    park, barrel, EV, xSLG, pitcher BRA, temperature). Without a cap,
    a power hitter in a favorable spot can get a 2.5-3x combined
    multiplier, inflating lambda far beyond what's realistic.

    The total adjustment product must be capped to prevent runaway
    compounding.
    """

    def test_power_hitter_at_coors_capped(self):
        """A power hitter at Coors Field facing a HR-prone pitcher with
        elite Statcast metrics should still produce a realistic lambda.

        Without a cap, this scenario produces lambda > 0.30 (implying
        ~26% HR probability), which is absurd — even the best power
        hitters only homer ~7-8% of the time per game.

        With a cap, lambda should stay below 0.20 (~18% per-game HR
        probability, which is high but plausible for elite hitters on
        a hot streak at Coors).
        """
        batter = _avg_batter(
            hr=30, ab=400, hr_per_ab=0.075,  # 30 HR pace
            barrel_rate=0.12,       # elite (league avg 0.065)
            avg_exit_velo=93.0,     # elite (league avg 88.5)
            xslg=0.550,            # elite (league avg 0.400)
            batting_order=3,
        )
        pitcher = _avg_pitcher(
            hr_per_9=1.8,           # HR-prone
            barrel_rate_against=0.10,  # high BRA
            innings_pitched=100.0,
        )
        lam = expected_hr_lambda(batter, pitcher, "COL")  # Coors, park=1.38
        prob = poisson_ge(1, lam) * 100

        assert lam < 0.20, (
            f"Power hitter at Coors lambda={lam:.3f}; expected <0.20 "
            f"(cap should prevent multiplicative blowup)"
        )
        assert prob < 20, (
            f"P(1+ HR)={prob:.1f}%; expected <20% even for elite matchup"
        )

    def test_league_avg_matchup_realistic(self):
        """A league-average batter with no recent HR in a neutral matchup
        should have a HR probability around 3-5%.
        """
        batter = _avg_batter(recent_hr=0)
        pitcher = _avg_pitcher()
        lam = expected_hr_lambda(batter, pitcher, "ATL")
        prob = poisson_ge(1, lam) * 100

        assert 1 < prob < 8, (
            f"P(1+ HR)={prob:.1f}%; expected 1-8% for league-avg matchup"
        )

    def test_recent_hr_boosts_lambda(self):
        """A batter who hit 3 HR in his last 7 games should have a
        higher lambda than one with 0 recent HR.
        """
        hot = _avg_batter(recent_hr=3)
        cold = _avg_batter(recent_hr=0)
        pitcher = _avg_pitcher()

        lam_hot = expected_hr_lambda(hot, pitcher, "ATL")
        lam_cold = expected_hr_lambda(cold, pitcher, "ATL")

        assert lam_hot > lam_cold, (
            f"Hot HR hitter lambda={lam_hot:.3f} should be > "
            f"cold hitter lambda={lam_cold:.3f}"
        )


class TestHitsRecentFormBlend:
    """The hits model should weight recent form (last 10 games) when
    computing expected hits, similar to how the KS model blends
    recent K/start at 70/30 with season rate.

    A hot hitter (recent_avg >> season avg) should get a higher lambda
    than a cold hitter with the same season stats.
    """

    def test_hot_hitter_gets_higher_lambda(self):
        """A batter hitting .350 over the last 10 games should produce
        a higher lambda than the same batter hitting .150 recently,
        even with identical season stats.
        """
        base = dict(avg=0.250, hits=75, ab=300, xba=0.250)
        hot = _avg_batter(recent_avg=0.350, **base)
        cold = _avg_batter(recent_avg=0.150, **base)
        pitcher = _avg_pitcher()

        lam_hot = expected_hits_lambda(hot, pitcher, "ATL")
        lam_cold = expected_hits_lambda(cold, pitcher, "ATL")

        assert lam_hot > lam_cold, (
            f"Hot hitter lambda={lam_hot:.3f} should be > "
            f"cold hitter lambda={lam_cold:.3f}"
        )

    def test_recent_form_has_meaningful_impact(self):
        """The difference between a hot and cold hitter should be at
        least 5% of lambda (not a negligible adjustment).
        """
        base = dict(avg=0.250, hits=75, ab=300, xba=0.250)
        hot = _avg_batter(recent_avg=0.350, **base)
        cold = _avg_batter(recent_avg=0.150, **base)
        pitcher = _avg_pitcher()

        lam_hot = expected_hits_lambda(hot, pitcher, "ATL")
        lam_cold = expected_hits_lambda(cold, pitcher, "ATL")
        diff_pct = (lam_hot - lam_cold) / lam_cold * 100

        assert diff_pct > 5, (
            f"Hot vs cold lambda diff={diff_pct:.1f}%; expected >5% "
            f"(recent form should have meaningful impact)"
        )
