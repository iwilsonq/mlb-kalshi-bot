"""Tests for CalibrationLayer — fit, interpolate, and calibrate."""
from slugger.calibration import CalibrationLayer, _interpolate


class TestCalibrateDoesNotInflateLowProbabilities:
    """The calibration layer must never inflate a low raw probability
    above the actual observed win rate for that range.

    This guards against the clamping bug where _interpolate returned the
    first breakpoint's y-value for any input below the curve's domain,
    turning 1-5% raw probabilities into 18%.
    """

    def _build_layer_with_low_range_data(self):
        """Build a CalibrationLayer from synthetic signal/settlement data
        that spans the full 0-100% probability range, including a dense
        low-probability region with a ~6% actual win rate (matching our
        real pitcher_ks data).
        """
        signals = []
        settlements = {}

        # Low range: 0-5% raw prob, 6% actual win rate → 94 losses, 6 wins per 100
        for i in range(940):
            ticker = f"LOW-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 3})
            settlements[ticker] = {"market_result": "no"}
        for i in range(60):
            ticker = f"LOW-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 3})
            settlements[ticker] = {"market_result": "yes"}

        # Mid-low range: 5-10% raw prob, 10% actual win rate
        for i in range(180):
            ticker = f"MIDLOW-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 8})
            settlements[ticker] = {"market_result": "no"}
        for i in range(20):
            ticker = f"MIDLOW-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 8})
            settlements[ticker] = {"market_result": "yes"}

        # Mid range: 20-25% raw prob, 13% actual win rate
        for i in range(87):
            ticker = f"MID-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 22})
            settlements[ticker] = {"market_result": "no"}
        for i in range(13):
            ticker = f"MID-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 22})
            settlements[ticker] = {"market_result": "yes"}

        # High range: 65-70% raw prob, 98% actual win rate
        for i in range(2):
            ticker = f"HIGH-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 68})
            settlements[ticker] = {"market_result": "no"}
        for i in range(98):
            ticker = f"HIGH-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "pitcher_ks", "model_prob_pct": 68})
            settlements[ticker] = {"market_result": "yes"}

        return CalibrationLayer.fit(signals, settlements)

    def test_low_prob_not_inflated(self):
        """A 3% raw probability must not calibrate above ~10%.

        Before the fix, this returned 18% due to clamping at the first
        breakpoint.  With proper extrapolation and breakpoints in the
        low range, it should return roughly 6% (the observed win rate).
        """
        cal = self._build_layer_with_low_range_data()
        result = cal.calibrate("pitcher_ks", 3)
        assert result <= 10, (
            f"calibrate(pitcher_ks, 3) = {result}%; "
            f"expected ≤10% (actual win rate is ~6%)"
        )

    def test_monotonicity_low_to_high(self):
        """Calibrated probabilities must be monotonically non-decreasing."""
        cal = self._build_layer_with_low_range_data()
        prev = 0
        for raw in range(0, 100, 5):
            curr = cal.calibrate("pitcher_ks", raw)
            assert curr >= prev, (
                f"Monotonicity violated: calibrate({raw}%)={curr}% "
                f"< calibrate({raw-5}%)={prev}%"
            )
            prev = curr

    def test_fit_produces_low_range_breakpoints(self):
        """fit() must produce breakpoints below 20% when data exists there."""
        cal = self._build_layer_with_low_range_data()
        breakpoints = cal.curves.get("pitcher_ks", [])
        lowest_x = breakpoints[0][0] if breakpoints else 999
        assert lowest_x < 20, (
            f"Lowest breakpoint at {lowest_x}%; expected <20% "
            f"since we provided dense data in the 0-10% range"
        )


class TestInterpolateExtrapolatesBelowCurve:
    """_interpolate must extrapolate toward origin below the curve."""

    def test_zero_returns_zero(self):
        bp = [(20.0, 15.0), (50.0, 40.0)]
        assert _interpolate(bp, 0.0) == 0.0

    def test_below_first_breakpoint_scales_proportionally(self):
        bp = [(20.0, 10.0), (50.0, 40.0)]
        # At x=10 (half of 20), should return half of 10 = 5.0
        result = _interpolate(bp, 10.0)
        assert result == 5.0, f"Expected 5.0, got {result}"

    def test_below_curve_never_exceeds_first_breakpoint_y(self):
        bp = [(22.5, 17.56), (45.0, 17.83)]
        for x in range(0, 23):
            y = _interpolate(bp, float(x))
            assert y <= 17.56, (
                f"_interpolate({x}) = {y}, exceeds first breakpoint y=17.56"
            )

    def test_above_curve_clamps_to_last(self):
        bp = [(20.0, 15.0), (50.0, 40.0)]
        assert _interpolate(bp, 80.0) == 40.0
