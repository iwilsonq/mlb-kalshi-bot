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
    """Fee model verified against 390 of our own fills on 2026-08-19.

    See scripts/analyze_maker_fees.py for the audit and slugger/fees.py for
    why the per-series fee_multiplier is recorded but not applied.
    """

    def test_ceiling_applies_to_the_order_not_each_contract(self):
        """The bug this replaced: 10 contracts at 11c cost $0.0686, not $0.10.

        Kalshi ceilings 0.07 * P * (1-P) * count once, to $0.0001. Rounding
        per contract and multiplying reproduced 1 of 335 taker fills; this
        reproduces 287. Every case below is an actual charged fee.
        """
        from slugger.models import kalshi_fee_dollars
        assert kalshi_fee_dollars(11, 10) == pytest.approx(0.0686)
        assert kalshi_fee_dollars(58, 16.74) == pytest.approx(0.2855)
        assert kalshi_fee_dollars(35, 27.32) == pytest.approx(0.4351)
        assert kalshi_fee_dollars(26, 18.28) == pytest.approx(0.2462)
        assert kalshi_fee_dollars(3, 31.21) == pytest.approx(0.0636)

    def test_floating_point_noise_does_not_invent_a_fee(self):
        """0.07*0.40*0.60*10 is 0.16800000000000004 in binary floating point.

        Ceiling that unguarded bills $0.1681 for a $0.1680 fee, and cost 14
        of 335 fills their exact match.
        """
        from slugger.models import kalshi_fee_dollars
        assert kalshi_fee_dollars(40, 10) == pytest.approx(0.1680)

    def test_ceiling_still_rounds_up_when_it_should(self):
        from slugger.models import kalshi_fee_dollars
        # 0.07 * 0.11 * 0.89 * 10 = 0.068530 -> next $0.0001 up is 0.0686
        assert kalshi_fee_dollars(11, 10) > 0.06853

    def test_per_contract_fee_is_unrounded(self):
        """Edge math runs before sizing, so it cannot apply the order ceiling.

        A ceil-to-cent per contract overstated the fee by up to a full cent,
        which is material against a MIN_EDGE_CENTS of 5.
        """
        from slugger.models import kalshi_fee_cents_per_contract as fee
        assert fee(50) == pytest.approx(1.75)
        assert fee(30) == pytest.approx(1.47)
        assert fee(20) == pytest.approx(1.12)
        assert fee(10) == pytest.approx(0.63)

    def test_symmetric_in_price(self):
        from slugger.models import kalshi_fee_cents_per_contract as fee
        for p in (5, 15, 25, 35, 45):
            assert fee(p) == pytest.approx(fee(100 - p))

    def test_makers_are_free_on_quadratic_series(self):
        """50 of 50 maker fills on quadratic series were charged exactly $0."""
        from slugger.models import FEE_TYPE_QUADRATIC, kalshi_fee_dollars
        assert kalshi_fee_dollars(
            42, 20, maker=True, fee_type=FEE_TYPE_QUADRATIC) == 0.0

    def test_makers_pay_a_quarter_on_quadratic_with_maker_fees(self):
        """Observed 0.250-0.257 of the taker formula on 5 KXMLBGAME fills."""
        from slugger.models import (
            FEE_TYPE_QUADRATIC_WITH_MAKER_FEES, kalshi_fee_dollars,
        )
        taker = kalshi_fee_dollars(42, 20)
        maker = kalshi_fee_dollars(
            42, 20, maker=True, fee_type=FEE_TYPE_QUADRATIC_WITH_MAKER_FEES)
        assert maker == pytest.approx(0.0853)      # the actual charged fee
        assert maker / taker == pytest.approx(0.25, abs=0.002)

    def test_unknown_fee_type_charges_makers_the_taker_rate(self):
        """Guessing "makers are free" on an unrecognised series understates
        cost, and understated cost is what makes a bot trade at negative edge.
        """
        from slugger.models import (
            KALSHI_TAKER_FEE_RATE, kalshi_fee_dollars, maker_fee_rate,
        )
        assert maker_fee_rate("something_new") == KALSHI_TAKER_FEE_RATE
        assert kalshi_fee_dollars(42, 20, maker=True, fee_type="brand_new") \
            == kalshi_fee_dollars(42, 20)

    def test_share_of_stake_is_what_kills_longshots(self):
        """Fee as a share of stake is several times worse at 10c than at 80c.

        This is why price level dominated model quality in the journal: 7.7%
        of stake at 0-20c entries against 0.6% at 60-80c.
        """
        from slugger.models import kalshi_fee_cents_per_contract as fee
        assert fee(10) / 10 >= 4 * (fee(80) / 80)

    def test_degenerate_prices(self):
        from slugger.models import kalshi_fee_cents_per_contract as fee
        from slugger.models import kalshi_fee_dollars
        assert fee(0) == 0
        assert fee(100) == 0
        assert kalshi_fee_dollars(50, 0) == 0.0
