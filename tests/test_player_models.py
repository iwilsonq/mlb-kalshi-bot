"""Tests for probability-distribution and fee math in slugger.models."""

import pytest
from slugger.models import poisson_ge


class TestBinomialGe:
    """Hits in a game are Binomial(AB, avg): bounded trials, one Bernoulli each.

    Poisson is overdispersed at the same mean — too much mass at zero and too
    much in the tail — so it understated 1+ hit props and overstated 3+/4+.
    Measured on 7157 real markets: 1+ actual 62.2%, Poisson said 58.5%,
    binomial 62.5%; 4+ actual 1.2%, Poisson said 2.0%, binomial 0.6%.
    """

    def test_matches_closed_form_for_integer_trials(self):
        from slugger.models import binomial_ge
        # P(>=1 hit) with 4 AB at .270 = 1 - 0.73^4
        assert binomial_ge(1, 4, 0.270) == pytest.approx(1 - 0.73 ** 4, abs=1e-9)
        # P(>=4 of 4) = p^4
        assert binomial_ge(4, 4, 0.270) == pytest.approx(
            max(0.01, 0.270 ** 4), abs=1e-9
        )

    def test_fixes_poisson_bias_direction_at_each_threshold(self):
        from slugger.models import binomial_ge, poisson_ge
        n, p = 4, 0.270
        lam = n * p
        # Poisson understates the chance of at least one hit
        assert poisson_ge(1, lam) < binomial_ge(1, n, p)
        # ...and overstates the longshot tail, which is where phantom edge came from
        assert poisson_ge(3, lam) > binomial_ge(3, n, p)
        assert poisson_ge(4, lam) > binomial_ge(4, n, p)

    def test_monotone_decreasing_in_threshold(self):
        from slugger.models import binomial_ge
        vals = [binomial_ge(t, 4.3, 0.27) for t in (1, 2, 3, 4)]
        assert vals == sorted(vals, reverse=True)

    def test_monotone_increasing_in_p(self):
        from slugger.models import binomial_ge
        vals = [binomial_ge(2, 4.3, p) for p in (0.15, 0.20, 0.25, 0.30, 0.35)]
        assert vals == sorted(vals)

    def test_fractional_trials_interpolate(self):
        from slugger.models import binomial_ge
        lo = binomial_ge(2, 4, 0.27)
        hi = binomial_ge(2, 5, 0.27)
        mid = binomial_ge(2, 4.5, 0.27)
        assert lo < mid < hi
        assert mid == pytest.approx((lo + hi) / 2, abs=1e-9)

    def test_degenerate_inputs(self):
        from slugger.models import binomial_ge
        assert binomial_ge(1, 0, 0.3) == 0.01
        assert binomial_ge(1, 4, 0.0) == 0.01
        assert binomial_ge(0, 4, 0.3) == 0.99
        # Threshold above the trial count is impossible, so floor applies
        assert binomial_ge(9, 4, 0.3) == 0.01

    def test_clamped_like_poisson_ge(self):
        from slugger.models import binomial_ge
        assert 0.01 <= binomial_ge(1, 6, 0.99) <= 0.99


class TestNegbinomGe:
    """Strikeouts per start are overdispersed, so the tail needs more than Poisson.

    Measured on 649 holdout starts, conditioning on the model's own predicted
    lambda: weighted var/mean = 1.129. Poisson assumes 1.0 and therefore
    understates P(K >= threshold) at the 6+ thresholds pitcher_ks trades.
    """

    def test_reduces_to_poisson_when_not_overdispersed(self):
        from slugger.models import negbinom_ge, poisson_ge
        for thr in (1, 4, 7, 10):
            assert negbinom_ge(thr, 5.5, 1.0) == poisson_ge(thr, 5.5)
            assert negbinom_ge(thr, 5.5, 0.8) == poisson_ge(thr, 5.5)

    def test_fattens_the_tail_relative_to_poisson(self):
        from slugger.models import negbinom_ge, poisson_ge
        lam = 5.0
        # Above the mean the negative binomial must assign MORE probability
        for thr in (7, 8, 9, 10):
            assert negbinom_ge(thr, lam, 1.13) > poisson_ge(thr, lam)

    def test_mean_and_variance_match_the_parameterisation(self):
        """var/mean must equal the dispersion, i.e. quasi-Poisson (NB1)."""
        import math
        lam, phi = 6.0, 1.25
        p = 1.0 / phi
        r = lam / (phi - 1.0)
        pmf = []
        for k in range(0, 200):
            pmf.append(math.exp(
                math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                + r * math.log(p) + k * math.log1p(-p)
            ))
        mean = sum(k * v for k, v in enumerate(pmf))
        var = sum((k - mean) ** 2 * v for k, v in enumerate(pmf))
        assert mean == pytest.approx(lam, rel=1e-6)
        assert var / mean == pytest.approx(phi, rel=1e-6)

    def test_monotone_decreasing_in_threshold(self):
        from slugger.models import negbinom_ge
        vals = [negbinom_ge(t, 5.0, 1.13) for t in range(1, 12)]
        assert vals == sorted(vals, reverse=True)

    def test_monotone_increasing_in_lambda(self):
        from slugger.models import negbinom_ge
        vals = [negbinom_ge(7, lam, 1.13) for lam in (3.0, 4.0, 5.0, 6.0, 7.0)]
        assert vals == sorted(vals)

    def test_degenerate_inputs(self):
        from slugger.models import negbinom_ge
        assert negbinom_ge(7, 0.0, 1.13) == 0.01
        assert negbinom_ge(0, 5.0, 1.13) == 0.99


class TestKalshiFee:
    """Fee model verified against 724/1042 of our own settled fills exactly."""

    def test_matches_observed_fills(self):
        from slugger.models import kalshi_fee_cents_per_contract as fee
        # ceil(0.07 * P * (1-P) * 100) in cents
        assert fee(50) == 2   # ceil(1.75)
        assert fee(30) == 2   # ceil(1.47)
        assert fee(20) == 2   # ceil(1.12)
        assert fee(10) == 1   # ceil(0.63)
        assert fee(70) == 2   # symmetric with 30
        assert fee(95) == 1

    def test_symmetric_in_price(self):
        from slugger.models import kalshi_fee_cents_per_contract as fee
        for p in (5, 15, 25, 35, 45):
            assert fee(p) == fee(100 - p)

    def test_share_of_stake_is_what_kills_longshots(self):
        """Fee as % of stake is several times worse at 10c than at 80c.

        (10% vs 2.5% with ceiling rounding; the journal's realized 7.7% vs 0.6%
        gap is wider still because multi-contract fills round once, not per
        contract.)
        """
        from slugger.models import kalshi_fee_cents_per_contract as fee
        drag_10 = fee(10) / 10
        drag_80 = fee(80) / 80
        assert drag_10 >= 4 * drag_80

    def test_degenerate_prices(self):
        from slugger.models import kalshi_fee_cents_per_contract as fee
        assert fee(0) == 0
        assert fee(100) == 0
