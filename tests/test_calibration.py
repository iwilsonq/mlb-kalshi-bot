"""Tests for CalibrationLayer — fit, interpolate, and calibrate."""
import pytest

from slugger.calibration import (
    CalibrationLayer, _interpolate, _parse_hits_signal, _parse_ks_signal,
    backfill_outcomes, hits_outcomes_from_game_logs, parse_team_hitting_splits,
)


class TestParseTeamHittingSplits:
    """Per-game K/PA extraction for point-in-time opponent K% (mlb-kalshi-bot-iwt)."""

    def test_extracts_k_and_pa(self):
        games = parse_team_hitting_splits([
            {"date": "2026-04-01", "stat": {"strikeOuts": 10, "plateAppearances": 38}},
            {"date": "2026-04-02", "stat": {"strikeOuts": 7, "plateAppearances": 41}},
        ])
        assert games == [
            {"date": "2026-04-01", "strikeouts": 10, "plate_appearances": 38},
            {"date": "2026-04-02", "strikeouts": 7, "plate_appearances": 41},
        ]

    def test_skips_rows_that_cannot_anchor_a_rate(self):
        games = parse_team_hitting_splits([
            {"date": "", "stat": {"strikeOuts": 5, "plateAppearances": 38}},
            {"date": "2026-04-02", "stat": {"strikeOuts": 5, "plateAppearances": 0}},
            {"date": "2026-04-03", "stat": {}},
            {"date": "2026-04-04", "stat": {"strikeOuts": 9, "plateAppearances": 38}},
        ])
        assert [g["date"] for g in games] == ["2026-04-04"]

    def test_empty_input(self):
        assert parse_team_hitting_splits([]) == []


class TestParseKsSignal:
    """_parse_ks_signal extracts (pitcher_name, date, threshold, model_prob)
    from a pitcher_ks signal record, using the ticker and reason fields.
    """

    def test_standard_ticker(self):
        sig = {
            "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-7",
            "strategy": "pitcher_ks",
            "model_prob_pct": 42,
            "date": "2026-06-03",
            "reason": "λ=6.1Ks  P(≥7)=42%",
        }
        result = _parse_ks_signal(sig)
        assert result is not None
        name, date, threshold, prob = result
        assert name == "P Skenes"
        assert date == "2026-06-03"
        assert threshold == 7
        assert prob == 42

    def test_multi_word_last_name(self):
        sig = {
            "ticker": "KXMLBKS-26MAY131845PHIBOS-BOSCEARLY71-6",
            "strategy": "pitcher_ks",
            "model_prob_pct": 15,
            "date": "2026-05-13",
            "reason": "λ=4.0Ks",
        }
        result = _parse_ks_signal(sig)
        assert result is not None
        name, date, threshold, prob = result
        # The name should have first initial + rest as last name
        assert name == "C Early"
        assert threshold == 6

    def test_non_ks_signal_returns_none(self):
        sig = {
            "ticker": "KXMLBHIT-26JUN042140LADAZ-LADMBETTS50-2",
            "strategy": "player_hits",
            "model_prob_pct": 16,
            "date": "2026-06-04",
            "reason": "",
        }
        assert _parse_ks_signal(sig) is None


class TestBackfillOutcomes:
    """backfill_outcomes matches signals to MLB game log data to produce
    (model_prob, outcome) pairs without depending on Kalshi settlements.
    """

    def test_matches_signal_to_game_log(self):
        """Given a signal for 7+ Ks and a game log showing 8 Ks on that
        date, the outcome should be 1 (yes).
        """
        signals = [{
            "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-7",
            "strategy": "pitcher_ks",
            "model_prob_pct": 42,
            "date": "2026-06-03",
            "reason": "λ=6.1Ks",
        }]
        # Fake game log: Skenes threw 8 Ks on June 3
        game_logs = {
            "Paul Skenes": [
                {"date": "2026-06-03", "strikeouts": 8},
            ]
        }
        result = backfill_outcomes(signals, game_logs=game_logs)
        assert "pitcher_ks" in result
        assert len(result["pitcher_ks"]) == 1
        prob, outcome = result["pitcher_ks"][0]
        assert prob == 42.0
        assert outcome == 1  # 8 >= 7

    def test_threshold_not_met(self):
        """If actual Ks < threshold, outcome is 0."""
        signals = [{
            "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-9",
            "strategy": "pitcher_ks",
            "model_prob_pct": 18,
            "date": "2026-06-03",
            "reason": "λ=6.1Ks",
        }]
        game_logs = {
            "Paul Skenes": [
                {"date": "2026-06-03", "strikeouts": 8},
            ]
        }
        result = backfill_outcomes(signals, game_logs=game_logs)
        prob, outcome = result["pitcher_ks"][0]
        assert prob == 18.0
        assert outcome == 0  # 8 < 9

    def test_deduplicates_by_ticker(self):
        """Same ticker appearing multiple times (poll cycles) counts once."""
        sig = {
            "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-7",
            "strategy": "pitcher_ks",
            "model_prob_pct": 42,
            "date": "2026-06-03",
            "reason": "λ=6.1Ks",
        }
        signals = [sig, sig, sig]  # 3 poll cycles
        game_logs = {
            "Paul Skenes": [
                {"date": "2026-06-03", "strikeouts": 10},
            ]
        }
        result = backfill_outcomes(signals, game_logs=game_logs)
        assert len(result["pitcher_ks"]) == 1

    def test_multiple_thresholds_same_game(self):
        """Multiple thresholds for the same pitcher/game each get resolved."""
        signals = [
            {
                "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-5",
                "strategy": "pitcher_ks",
                "model_prob_pct": 73,
                "date": "2026-06-03",
                "reason": "",
            },
            {
                "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-7",
                "strategy": "pitcher_ks",
                "model_prob_pct": 42,
                "date": "2026-06-03",
                "reason": "",
            },
            {
                "ticker": "KXMLBKS-26JUN032010PITHOU-PITPSKENES30-10",
                "strategy": "pitcher_ks",
                "model_prob_pct": 9,
                "date": "2026-06-03",
                "reason": "",
            },
        ]
        game_logs = {
            "Paul Skenes": [
                {"date": "2026-06-03", "strikeouts": 7},
            ]
        }
        result = backfill_outcomes(signals, game_logs=game_logs)
        pairs = result["pitcher_ks"]
        assert len(pairs) == 3
        # Sort by prob descending for predictable order
        pairs_sorted = sorted(pairs, key=lambda p: -p[0])
        assert pairs_sorted[0] == (73.0, 1)   # 7 >= 5  → yes
        assert pairs_sorted[1] == (42.0, 1)   # 7 >= 7  → yes
        assert pairs_sorted[2] == (9.0, 0)    # 7 < 10  → no

    def test_skips_non_pitcher_ks_signals(self):
        """Only pitcher_ks signals are processed."""
        signals = [{
            "ticker": "KXMLBHIT-26JUN042140LADAZ-LADMBETTS50-2",
            "strategy": "player_hits",
            "model_prob_pct": 16,
            "date": "2026-06-04",
            "reason": "",
        }]
        result = backfill_outcomes(signals, game_logs={})
        assert result == {}

    def test_deduplicates_correlated_wins_within_bin(self):
        """When a pitcher beats multiple thresholds in the same probability bin,
        only one observation per (pitcher, game, bin) is kept.

        This prevents a single dominant performance (e.g. 10 Ks when model
        said 1% for 6+, 7+, 8+, 9+) from inflating the bin win rate 4x.
        """
        # Pitcher gets 10 Ks. Model assigned 1% to thresholds 6, 7, 8, 9 —
        # all land in the 0-5% bin. Without dedup: 4 wins. With dedup: 1.
        signals = [
            {
                "ticker": "KXMLBKS-26MAY272005NYYHOU-NYYGCOLE45-6",
                "strategy": "pitcher_ks",
                "model_prob_pct": 1,
                "date": "2026-05-27",
                "reason": "",
            },
            {
                "ticker": "KXMLBKS-26MAY272005NYYHOU-NYYGCOLE45-7",
                "strategy": "pitcher_ks",
                "model_prob_pct": 1,
                "date": "2026-05-27",
                "reason": "",
            },
            {
                "ticker": "KXMLBKS-26MAY272005NYYHOU-NYYGCOLE45-8",
                "strategy": "pitcher_ks",
                "model_prob_pct": 1,
                "date": "2026-05-27",
                "reason": "",
            },
            {
                "ticker": "KXMLBKS-26MAY272005NYYHOU-NYYGCOLE45-9",
                "strategy": "pitcher_ks",
                "model_prob_pct": 1,
                "date": "2026-05-27",
                "reason": "",
            },
        ]
        game_logs = {"Gerrit Cole": [{"date": "2026-05-27", "strikeouts": 10}]}
        result = backfill_outcomes(signals, game_logs=game_logs)
        pairs = result["pitcher_ks"]
        # Without fix: 4 wins. With fix: 1 win per (pitcher, game, bin).
        assert len(pairs) == 1, (
            f"Expected 1 observation (one per game per bin), got {len(pairs)}"
        )


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


class TestInterpolateExtrapolatesOutsideCurve:
    """_interpolate must extrapolate toward certainty at both ends.

    Outside the fitted domain there are no observations, so flattening asserts
    something the data cannot support. Below, flattening inflated sub-5% signals
    into phantom edge. Above, it collapsed every raw value over the last
    breakpoint to one number (mlb-kalshi-bot-dbr).
    """

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

    def test_one_hundred_returns_one_hundred(self):
        bp = [(20.0, 15.0), (50.0, 40.0)]
        assert _interpolate(bp, 100.0) == 100.0

    def test_above_last_breakpoint_scales_toward_certainty(self):
        bp = [(20.0, 15.0), (50.0, 40.0)]
        # x=80 is 60% of the way from 50 to 100, so y is 60% of the way 40 -> 100
        assert _interpolate(bp, 80.0) == pytest.approx(76.0)

    def test_above_curve_stays_monotone_and_bounded(self):
        bp = [(15.0, 5.56), (25.0, 14.59), (32.5, 19.57), (37.5, 31.25)]
        ys = [_interpolate(bp, float(x)) for x in range(0, 101)]
        assert ys == sorted(ys), "calibration must stay monotone"
        assert all(0.0 <= y <= 100.0 for y in ys)

    def test_high_raw_probabilities_are_distinguishable(self):
        """The player_hits curve must stop reporting one number for every batter.

        Regression: with the real fitted curve, Ohtani (raw 67%) and Betts
        (raw 46%) both calibrated to 31% against markets of 67¢ and 64¢.
        """
        bp = [(15.0, 5.56), (25.0, 14.59), (32.5, 19.57), (37.5, 31.25)]
        ohtani = _interpolate(bp, 67.0)
        betts = _interpolate(bp, 46.0)
        assert ohtani > betts + 10, (
            f"raw 67% and 46% must differ materially, got {ohtani:.1f} vs {betts:.1f}"
        )
        # And a ~65% event must no longer be priced at half its value
        assert ohtani > 50.0


class TestFitDeduplicatesByTicker:
    """fit() must deduplicate signals by ticker before computing bin
    win rates.  The signal pipeline records a signal on every poll
    cycle (~60s), so the same market can have 100+ duplicate signals.
    Without dedup, a single winning ticker inflates the bin win rate
    by contributing 100+ "wins" instead of 1.
    """

    def test_duplicate_signals_not_counted(self):
        """If the same ticker appears 100 times and wins, it should
        only count as 1 win in the sample_counts, not 100.
        """
        signals = []
        settlements = {}

        # Low bin (0-5%): 1 ticker duplicated 100 times (won) + 9 unique losers
        # Without dedup: 109 samples
        # With dedup: 10 samples
        for i in range(100):
            signals.append({"ticker": "DUPE-WIN", "strategy": "test_strat", "model_prob_pct": 3})
        settlements["DUPE-WIN"] = {"market_result": "yes"}
        for i in range(9):
            ticker = f"LOW-LOSS-{i}"
            signals.append({"ticker": ticker, "strategy": "test_strat", "model_prob_pct": 3})
            settlements[ticker] = {"market_result": "no"}

        # High bin (60-65%): 35 unique winners + 15 unique losers (70% rate)
        for i in range(35):
            ticker = f"HIGH-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "test_strat", "model_prob_pct": 62})
            settlements[ticker] = {"market_result": "yes"}
        for i in range(15):
            ticker = f"HIGH-LOSS-{i}"
            signals.append({"ticker": ticker, "strategy": "test_strat", "model_prob_pct": 62})
            settlements[ticker] = {"market_result": "no"}

        cal = CalibrationLayer.fit(signals, settlements, min_samples=5)

        # With dedup: 10 unique low + 50 unique high = 60 samples
        # Without dedup: 109 low + 50 high = 159 samples
        assert cal.sample_counts.get("test_strat", 0) == 60, (
            f"Expected 60 unique samples after dedup, "
            f"got {cal.sample_counts.get('test_strat', 0)}"
        )

    def test_dedup_preserves_unique_signals(self):
        """Unique signals (different tickers) should all count normally."""
        signals = []
        settlements = {}

        # 50 unique tickers in 0-5% bin, 5 winners (10% rate)
        for i in range(45):
            ticker = f"LOSS-{i}"
            signals.append({"ticker": ticker, "strategy": "s", "model_prob_pct": 3})
            settlements[ticker] = {"market_result": "no"}
        for i in range(5):
            ticker = f"WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "s", "model_prob_pct": 3})
            settlements[ticker] = {"market_result": "yes"}

        # 50 unique tickers in 60-65% bin, 35 winners (70% rate)
        for i in range(15):
            ticker = f"HIGH-LOSS-{i}"
            signals.append({"ticker": ticker, "strategy": "s", "model_prob_pct": 62})
            settlements[ticker] = {"market_result": "no"}
        for i in range(35):
            ticker = f"HIGH-WIN-{i}"
            signals.append({"ticker": ticker, "strategy": "s", "model_prob_pct": 62})
            settlements[ticker] = {"market_result": "yes"}

        cal = CalibrationLayer.fit(signals, settlements, min_samples=5)
        low = cal.calibrate("s", 3)
        high = cal.calibrate("s", 62)

        assert low <= 15, f"Low bin calibrated to {low}%, expected ~10%"
        assert high >= 60, f"High bin calibrated to {high}%, expected ~70%"


class TestWalkForwardNoLeakage:
    """Walk-forward fit must not use records on or after as_of."""

    def test_filter_records_before(self):
        from slugger.calibration import filter_records_before
        signals = [
            {"ticker": "A", "date": "2026-06-01", "strategy": "s", "model_prob_pct": 30},
            {"ticker": "B", "date": "2026-06-10", "strategy": "s", "model_prob_pct": 30},
        ]
        settlements = {
            "A": {"market_result": "yes", "settled_at": "2026-06-01T12:00:00+00:00"},
            "B": {"market_result": "no", "settled_at": "2026-06-10T12:00:00+00:00"},
        }
        sigs, setts = filter_records_before(signals, settlements, "2026-06-05")
        assert len(sigs) == 1 and sigs[0]["ticker"] == "A"
        assert "A" in setts and "B" not in setts

    def test_fit_as_of_excludes_future(self):
        """Future settlements must not change sample_counts when as_of cuts them."""
        signals = []
        settlements = {}
        # Past: 40 unique in two bins so fit can succeed with min_samples=10
        for i in range(20):
            t = f"PAST-LO-{i}"
            signals.append({
                "ticker": t, "strategy": "wf", "model_prob_pct": 10,
                "date": "2026-05-01", "timestamp": "2026-05-01T12:00:00+00:00",
            })
            settlements[t] = {
                "market_result": "no" if i < 18 else "yes",
                "settled_at": "2026-05-01T20:00:00+00:00",
            }
        for i in range(20):
            t = f"PAST-HI-{i}"
            signals.append({
                "ticker": t, "strategy": "wf", "model_prob_pct": 70,
                "date": "2026-05-01", "timestamp": "2026-05-01T12:00:00+00:00",
            })
            settlements[t] = {
                "market_result": "yes" if i < 14 else "no",
                "settled_at": "2026-05-01T20:00:00+00:00",
            }
        # Future flood that would dominate without as_of
        for i in range(200):
            t = f"FUT-{i}"
            signals.append({
                "ticker": t, "strategy": "wf", "model_prob_pct": 10,
                "date": "2026-08-01", "timestamp": "2026-08-01T12:00:00+00:00",
            })
            settlements[t] = {
                "market_result": "yes",
                "settled_at": "2026-08-01T20:00:00+00:00",
            }

        full = CalibrationLayer.fit(signals, settlements, min_samples=10)
        wf = CalibrationLayer.fit(signals, settlements, min_samples=10, as_of="2026-06-01")
        assert full.sample_counts.get("wf", 0) == 240
        assert wf.sample_counts.get("wf", 0) == 40
        assert wf.as_of == "2026-06-01"

    def test_fit_walk_forward_sets_lag(self):
        signals = []
        settlements = {}
        for i in range(40):
            t = f"T-{i}"
            signals.append({
                "ticker": t, "strategy": "s2", "model_prob_pct": 20 if i < 20 else 60,
                "date": "2026-04-01", "timestamp": "2026-04-01T00:00:00+00:00",
            })
            settlements[t] = {
                "market_result": "yes" if i % 2 == 0 else "no",
                "settled_at": "2026-04-01T12:00:00+00:00",
            }
        cal = CalibrationLayer.fit_walk_forward(
            signals, settlements, lag_days=1, as_of="2026-05-01", min_samples=10,
        )
        assert cal.lag_days == 1
        assert cal.as_of == "2026-05-01"
        assert cal.sample_counts.get("s2", 0) == 40


class TestHitsBackfillRemovesSelectionBias:
    """player_hits calibration must be fit from every LISTED market, not settlements.

    Kalshi settlements only exist for markets the bot traded, and it traded the
    ones showing the largest apparent edge — i.e. the ones it most overestimated.
    Fitting on that subsample taught the curve the model overpredicts far more
    than it does; applying it to the whole population was worse than no
    calibration at all (walk-forward Brier 0.18679 vs 0.17746 raw).
    """

    def _sig(self, ticker, name, date, prob, reason_extra="  4H/20AB(vsR)"):
        return {
            "type": "signal", "strategy": "player_hits", "ticker": ticker,
            "date": date, "model_prob_pct": prob,
            "reason": f"{name}{reason_extra}  λ=1.10  P(2+H)={prob}%",
        }

    def test_parses_name_date_threshold(self):
        sig = self._sig("KXMLBHIT-26MAY191610ATLMIA-MIAXEDWARDS9-2",
                        "Xavier Edwards", "2026-05-19", 22)
        assert _parse_hits_signal(sig) == ("Xavier Edwards", "2026-05-19", 2)

    def test_rejects_other_strategies_and_malformed(self):
        assert _parse_hits_signal({"strategy": "pitcher_ks", "ticker": "X-7"}) is None
        assert _parse_hits_signal(
            {"strategy": "player_hits", "ticker": "no-threshold-suffix"}
        ) is None
        assert _parse_hits_signal(
            {"strategy": "player_hits", "ticker": "X-2", "reason": "", "date": "2026-05-19"}
        ) is None

    def test_outcomes_come_from_game_logs_not_settlements(self):
        signals = [
            self._sig("T-A-1", "Ann Batter", "2026-05-01", 60),
            self._sig("T-B-3", "Ann Batter", "2026-05-02", 8),
        ]
        logs = {"Ann Batter": [
            {"date": "2026-05-01", "hits": 2, "ab": 4},   # 1+ -> win
            {"date": "2026-05-02", "hits": 1, "ab": 4},   # 3+ -> loss
        ]}
        pairs = hits_outcomes_from_game_logs(signals, logs)
        assert sorted(pairs) == [(8.0, 0), (60.0, 1)]

    def test_untraded_markets_are_included(self):
        """The whole point: a market with no settlement still yields an outcome."""
        signals = [self._sig("T-C-2", "Ann Batter", "2026-05-03", 30)]
        logs = {"Ann Batter": [{"date": "2026-05-03", "hits": 3, "ab": 5}]}
        # No settlements dict is consulted at all
        assert hits_outcomes_from_game_logs(signals, logs) == [(30.0, 1)]

    def test_deduplicates_poll_cycle_repeats_by_ticker(self):
        signals = [self._sig("T-D-2", "Ann Batter", "2026-05-04", 30) for _ in range(50)]
        logs = {"Ann Batter": [{"date": "2026-05-04", "hits": 2, "ab": 4}]}
        assert len(hits_outcomes_from_game_logs(signals, logs)) == 1

    def test_deduplicates_same_batter_day_within_a_bin(self):
        """One 4-hit game must not stuff a probability bin with wins."""
        signals = [
            self._sig("T-E-1", "Ann Batter", "2026-05-05", 61),
            self._sig("T-E-2", "Ann Batter", "2026-05-05", 62),
            self._sig("T-E-3", "Ann Batter", "2026-05-05", 63),
        ]
        logs = {"Ann Batter": [{"date": "2026-05-05", "hits": 4, "ab": 4}]}
        pairs = hits_outcomes_from_game_logs(signals, logs)
        assert len(pairs) == 1, f"expected one row per (batter, day, bin), got {pairs}"

    def test_skips_batters_without_a_game_log(self):
        signals = [self._sig("T-F-2", "Ghost Player", "2026-05-06", 30)]
        assert hits_outcomes_from_game_logs(signals, {}) == []

    def test_backfill_outcomes_covers_player_hits(self):
        signals = [
            self._sig("T-G-1", "Ann Batter", "2026-05-07", 60),
            self._sig("T-G-3", "Ann Batter", "2026-05-08", 7),
        ]
        logs = {"Ann Batter": [
            {"date": "2026-05-07", "hits": 1, "ab": 4},
            {"date": "2026-05-08", "hits": 0, "ab": 4},
        ]}
        out = backfill_outcomes(signals, game_logs={}, batter_logs=logs)
        assert "player_hits" in out
        assert sorted(out["player_hits"]) == [(7.0, 0), (60.0, 1)]
