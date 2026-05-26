"""Probability calibration for strategy model outputs.

Uses isotonic regression (pool adjacent violators algorithm) to map
raw model probabilities to well-calibrated probabilities based on
historical signal outcomes.

No external dependencies — implements PAVA from scratch.

Usage:
    # Fit calibration from historical data
    cal = CalibrationLayer.fit(signals, settlements)
    cal.save("logs/calibration.json")

    # Load and apply
    cal = CalibrationLayer.load("logs/calibration.json")
    calibrated_prob = cal.calibrate("pitcher_ks", raw_prob_pct=35)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Minimum settled signals per strategy before trusting calibration.
# Below this threshold, the calibration is too noisy to help.
_MIN_SAMPLES = 30


# ─── Isotonic regression (PAVA) ──────────────────────────────────────────────

def _isotonic_regression(x: List[float], y: List[float]) -> List[Tuple[float, float]]:
    """Fit isotonic regression using the pool adjacent violators algorithm.

    Given (x, y) pairs sorted by x, produces a monotonically non-decreasing
    step function mapping x → calibrated_y.

    Args:
        x: Sorted input values (model probabilities).
        y: Observed outcomes (0 or 1 for binary, or averages per bin).

    Returns:
        List of (x_value, calibrated_y) breakpoints defining the step function.
        To interpolate: find the two nearest breakpoints and linearly interpolate.
    """
    if not x or not y or len(x) != len(y):
        return []

    # Sort by x (should already be sorted, but ensure it)
    pairs = sorted(zip(x, y), key=lambda p: p[0])

    # PAVA: merge adjacent blocks that violate monotonicity
    blocks: List[List[Tuple[float, float]]] = [[p] for p in pairs]

    i = 0
    while i < len(blocks) - 1:
        # Compute block averages
        avg_curr = sum(p[1] for p in blocks[i]) / len(blocks[i])
        avg_next = sum(p[1] for p in blocks[i + 1]) / len(blocks[i + 1])

        if avg_curr > avg_next:
            # Merge: pool adjacent violators
            blocks[i] = blocks[i] + blocks[i + 1]
            blocks.pop(i + 1)
            # Step back to check if merge created a new violation
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build breakpoints: each block's mean x → mean y
    breakpoints: List[Tuple[float, float]] = []
    for block in blocks:
        mean_x = sum(p[0] for p in block) / len(block)
        mean_y = sum(p[1] for p in block) / len(block)
        breakpoints.append((mean_x, mean_y))

    return breakpoints


def _interpolate(breakpoints: List[Tuple[float, float]], x: float) -> float:
    """Linearly interpolate a calibrated value from PAVA breakpoints.

    Clamps to the range of the breakpoints (no extrapolation).
    """
    if not breakpoints:
        return x

    # Clamp to endpoints
    if x <= breakpoints[0][0]:
        return breakpoints[0][1]
    if x >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    # Find the two surrounding breakpoints
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return breakpoints[-1][1]


# ─── Calibration layer ───────────────────────────────────────────────────────

@dataclass
class CalibrationLayer:
    """Per-strategy isotonic calibration curves.

    Each strategy gets its own set of breakpoints mapping raw model
    probability → calibrated probability.  Strategies without enough
    data use an identity mapping (pass-through).
    """
    # strategy_name → list of (x, y) breakpoints
    curves: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    # strategy_name → number of samples used to fit
    sample_counts: Dict[str, int] = field(default_factory=dict)

    def calibrate(self, strategy: str, raw_prob_pct: int) -> int:
        """Apply calibration to a raw model probability.

        Args:
            strategy:     Strategy name (e.g. "pitcher_ks").
            raw_prob_pct: Raw model probability as integer percentage (0-100).

        Returns:
            Calibrated probability as integer percentage (0-100).
            Returns raw_prob_pct unchanged if no calibration data exists.
        """
        if strategy not in self.curves:
            return raw_prob_pct

        calibrated = _interpolate(self.curves[strategy], float(raw_prob_pct))
        return max(0, min(100, round(calibrated)))

    def has_calibration(self, strategy: str) -> bool:
        """Return True if calibration data exists for this strategy."""
        return strategy in self.curves

    def save(self, path: str) -> None:
        """Save calibration curves to a JSON file."""
        data = {
            "curves": {k: [[x, y] for x, y in v] for k, v in self.curves.items()},
            "sample_counts": self.sample_counts,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        log.info("Saved calibration to %s (%d strategies)", path, len(self.curves))

    @classmethod
    def load(cls, path: str) -> CalibrationLayer:
        """Load calibration curves from a JSON file.

        Returns an empty (pass-through) layer if the file doesn't exist
        or can't be parsed.
        """
        p = Path(path)
        if not p.exists():
            log.debug("No calibration file at %s — using uncalibrated", path)
            return cls()
        try:
            data = json.loads(p.read_text())
            curves = {
                k: [(x, y) for x, y in v]
                for k, v in data.get("curves", {}).items()
            }
            sample_counts = data.get("sample_counts", {})
            log.info(
                "Loaded calibration from %s: %s",
                path,
                ", ".join(f"{k}({sample_counts.get(k, '?')} samples)" for k in curves),
            )
            return cls(curves=curves, sample_counts=sample_counts)
        except Exception as exc:
            log.warning("Could not load calibration from %s: %s", path, exc)
            return cls()

    @classmethod
    def fit(
        cls,
        signals: List[dict],
        settlements: Dict[str, dict],
        min_samples: int = _MIN_SAMPLES,
    ) -> CalibrationLayer:
        """Fit calibration curves from historical signal and settlement data.

        Args:
            signals:      List of signal records from load_signals().
            settlements:  Dict of ticker → settlement record from load_journal().
            min_samples:  Minimum settled signals per strategy to fit calibration.

        Returns:
            CalibrationLayer with per-strategy isotonic regression curves.
        """
        # Group (model_prob, outcome) pairs by strategy
        strategy_data: Dict[str, List[Tuple[float, int]]] = {}

        for sig in signals:
            ticker = sig.get("ticker", "")
            strategy = sig.get("strategy", "")
            prob = sig.get("model_prob_pct", 0)

            settlement = settlements.get(ticker)
            if not settlement:
                continue

            result = settlement.get("market_result", "")
            if result == "void":
                continue

            outcome = 1 if result == "yes" else 0

            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append((float(prob), outcome))

        # Fit isotonic regression per strategy using binned data.
        # Binning is critical: raw binary outcomes (0/1) are too noisy for
        # isotonic regression. We group signals into 5%-wide probability bins,
        # compute the actual win rate per bin, then fit PAVA on (bin_midpoint,
        # actual_win_rate) pairs. This produces smooth, meaningful calibration.
        _BIN_WIDTH = 5  # percentage points per bin

        curves: Dict[str, List[Tuple[float, float]]] = {}
        sample_counts: Dict[str, int] = {}

        for strategy, data in strategy_data.items():
            sample_counts[strategy] = len(data)

            if len(data) < min_samples:
                log.info(
                    "Calibration: %s has only %d samples (need %d) — skipping",
                    strategy, len(data), min_samples,
                )
                continue

            # Bin by model probability
            bins: Dict[int, List[int]] = {}  # bin_midpoint → [outcomes]
            for prob, outcome in data:
                bin_idx = int(prob // _BIN_WIDTH)
                midpoint = bin_idx * _BIN_WIDTH + _BIN_WIDTH / 2
                if midpoint not in bins:
                    bins[midpoint] = []
                bins[midpoint].append(outcome)

            # Compute (midpoint, actual_win_rate) for bins with enough samples
            _MIN_BIN_SIZE = 5
            bin_points: List[Tuple[float, float]] = []
            for midpoint in sorted(bins.keys()):
                outcomes = bins[midpoint]
                if len(outcomes) >= _MIN_BIN_SIZE:
                    win_rate = sum(outcomes) / len(outcomes) * 100  # as pct
                    bin_points.append((midpoint, win_rate))

            if len(bin_points) < 2:
                log.info(
                    "Calibration: %s — %d samples but only %d usable bins — skipping",
                    strategy, len(data), len(bin_points),
                )
                continue

            # Fit isotonic regression on binned data
            x_vals = [p[0] for p in bin_points]
            y_vals = [p[1] for p in bin_points]

            breakpoints = _isotonic_regression(x_vals, y_vals)
            if breakpoints:
                curves[strategy] = breakpoints
                # Log calibration summary
                low_x, low_y = breakpoints[0]
                high_x, high_y = breakpoints[-1]
                log.info(
                    "Calibration: %s — %d samples, %d bins, %d breakpoints  "
                    "[%.0f%%→%.0f%%, %.0f%%→%.0f%%]",
                    strategy, len(data), len(bin_points), len(breakpoints),
                    low_x, low_y, high_x, high_y,
                )

        return cls(curves=curves, sample_counts=sample_counts)

    def format_report(self) -> str:
        """Format a human-readable calibration report."""
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append("  CALIBRATION CURVES")
        lines.append(f"{'=' * 70}")

        if not self.curves:
            lines.append("  No calibration data available.")
            lines.append(f"{'=' * 70}")
            return "\n".join(lines)

        for strategy in sorted(self.curves.keys()):
            breakpoints = self.curves[strategy]
            n = self.sample_counts.get(strategy, 0)
            lines.append(f"\n  {strategy}  ({n} samples, {len(breakpoints)} breakpoints)")
            lines.append(f"  {'-' * 60}")
            lines.append(f"  {'Model':>8}  {'Calibrated':>10}  {'Shift':>8}")
            lines.append(f"  {'-' * 60}")

            # Sample the curve at regular intervals
            for pct in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]:
                cal = _interpolate(breakpoints, float(pct))
                shift = cal - pct
                if abs(shift) >= 0.5:  # only show meaningful shifts
                    lines.append(
                        f"  {pct:>7}%  {cal:>9.1f}%  {shift:>+7.1f}%"
                    )

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)
