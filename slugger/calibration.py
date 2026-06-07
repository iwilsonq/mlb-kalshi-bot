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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Minimum settled signals per strategy before trusting calibration.
# Below this threshold, the calibration is too noisy to help.
_MIN_SAMPLES = 30


# ─── Ticker parsing (pitcher_ks signals) ─────────────────────────────────────

# Ticker format: KXMLBKS-26JUN032010PITHOU-PITPSKENES30-7
#   segment[0] = KXMLBKS         (product)
#   segment[1] = 26JUN032010PITHOU  (11-char date + away_team + home_team)
#   segment[2] = PITPSKENES30    (team + first_initial + LASTNAME + jersey#)
#   segment[3] = 7               (threshold)

_NAME_RE = re.compile(r"^([A-Z])([A-Z]+?)(\d+)$")


def _parse_ks_signal(sig: dict) -> Optional[Tuple[str, str, int, float]]:
    """Extract (pitcher_name, date, threshold, model_prob) from a pitcher_ks signal.

    Returns None if the signal is not a pitcher_ks signal or can't be parsed.
    The pitcher_name is formatted as "F Lastname" (first initial + title-cased last name).
    """
    if sig.get("strategy") != "pitcher_ks":
        return None

    ticker = sig.get("ticker", "")
    parts = ticker.split("-")
    if len(parts) < 4:
        return None

    try:
        threshold = int(parts[-1])
    except (ValueError, IndexError):
        return None

    # Extract teams from event segment to determine team prefix length.
    # Event segment: 11-char date + teams (e.g. "26JUN032010PITHOU").
    event = parts[1]
    teams_str = event[11:]      # e.g. "PITHOU", "AZTEX", "LADAZ"
    pitcher_part = parts[2]     # e.g. "PITPSKENES30"

    # Try all valid (2,3) × (2,3) team splits; pick the longest match.
    team = None
    candidates = []
    for t1_len in (2, 3):
        for t2_len in (2, 3):
            if t1_len + t2_len != len(teams_str):
                continue
            t1 = teams_str[:t1_len]
            t2 = teams_str[t1_len:]
            if pitcher_part.startswith(t1):
                candidates.append(t1)
            if pitcher_part.startswith(t2):
                candidates.append(t2)
    if not candidates:
        return None
    team = max(candidates, key=len)

    rest = pitcher_part[len(team):]
    m = _NAME_RE.match(rest)
    if not m:
        return None

    first_initial, last_raw, _jersey = m.groups()
    name = f"{first_initial} {last_raw.title()}"

    date = sig.get("date", "")
    prob = float(sig.get("model_prob_pct", 0))

    return name, date, threshold, prob


_BIN_WIDTH = 5  # must match the value used in CalibrationLayer.fit()


def backfill_outcomes(
    signals: List[dict],
    game_logs: Optional[Dict[str, List[dict]]] = None,
) -> Dict[str, List[Tuple[float, int]]]:
    """Produce (model_prob, outcome) pairs for pitcher_ks signals using MLB game logs.

    Instead of relying on Kalshi settlement data, this function matches each
    signal's pitcher + date + threshold against actual K counts from MLB game
    logs to determine outcomes directly.

    Deduplication is applied at two levels:
      1. By ticker — poll-cycle duplicates of the same market are collapsed.
      2. By (pitcher, date, probability bin) — when a pitcher beats multiple
         thresholds that all fall in the same bin (e.g. model said 1% for 6+,
         7+, 8+, 9+ and the pitcher threw 10 Ks), only the one with the
         closest probability to the bin midpoint is kept.  This prevents a
         single dominant start from inflating the bin win rate.

    Args:
        signals:    List of signal records (from load_signals).
        game_logs:  Dict mapping pitcher full name → list of
                    {"date": "YYYY-MM-DD", "strikeouts": int}.
                    If None, fetches live from the MLB Stats API.

    Returns:
        Dict mapping strategy name → list of (model_prob_pct, outcome) pairs.
        Currently only produces data for "pitcher_ks".
    """
    if game_logs is None:
        game_logs = _fetch_all_game_logs(signals)

    # Build a fast lookup: (pitcher_name_lower, date) → strikeouts
    ks_by_game: Dict[Tuple[str, str], int] = {}
    for name, games in game_logs.items():
        for g in games:
            key = (name.lower(), g["date"])
            ks_by_game[key] = g["strikeouts"]

    # First pass: deduplicate by ticker, then collect all (name, date, prob, outcome).
    seen_ticker: set = set()
    candidates: List[Tuple[str, str, float, int]] = []  # (name, date, prob, outcome)

    for sig in signals:
        parsed = _parse_ks_signal(sig)
        if parsed is None:
            continue

        ticker = sig.get("ticker", "")
        if ticker in seen_ticker:
            continue
        seen_ticker.add(ticker)

        name, date, threshold, prob = parsed

        actual_ks = _lookup_ks(name, date, ks_by_game)
        if actual_ks is None:
            continue

        outcome = 1 if actual_ks >= threshold else 0
        candidates.append((name, date, prob, outcome))

    # Second pass: deduplicate within each (name, date, bin).
    # Keep the candidate whose probability is closest to the bin midpoint.
    # This ensures one observation per game per calibration bin.
    seen_game_bin: Dict[Tuple[str, str, float], Tuple[float, int]] = {}
    for name, date, prob, outcome in candidates:
        bin_idx = int(prob // _BIN_WIDTH)
        bin_mid = bin_idx * _BIN_WIDTH + _BIN_WIDTH / 2
        key = (name, date, bin_mid)
        if key not in seen_game_bin:
            seen_game_bin[key] = (prob, outcome)
        else:
            existing_prob, _ = seen_game_bin[key]
            if abs(prob - bin_mid) < abs(existing_prob - bin_mid):
                seen_game_bin[key] = (prob, outcome)

    pairs = [(prob, outcome) for (prob, outcome) in seen_game_bin.values()]
    return {"pitcher_ks": pairs} if pairs else {}


def _lookup_ks(
    short_name: str,
    date: str,
    ks_by_game: Dict[Tuple[str, str], int],
) -> Optional[int]:
    """Look up actual strikeout count for a pitcher on a given date.

    short_name is "F Lastname" format (e.g. "P Skenes").
    ks_by_game keys are (full_name_lower, date).
    We match by checking if the full name ends with the last name
    and the first name starts with the initial.
    """
    initial = short_name[0].lower()
    last = short_name.split(" ", 1)[1].lower() if " " in short_name else short_name.lower()

    for (full_name_lower, game_date), ks in ks_by_game.items():
        if game_date != date:
            continue
        # Match: full name's last part matches and first char matches initial
        name_parts = full_name_lower.split()
        if len(name_parts) < 2:
            continue
        full_first = name_parts[0]
        full_last = " ".join(name_parts[1:])
        if full_first.startswith(initial) and full_last == last:
            return ks

    return None


def _fetch_all_game_logs(
    signals: List[dict],
) -> Dict[str, List[dict]]:
    """Fetch MLB game logs for all pitchers referenced in pitcher_ks signals.

    Returns dict mapping pitcher full name → list of
    {"date": "YYYY-MM-DD", "strikeouts": int} entries.
    """
    # Collect unique pitcher short names
    pitcher_names: set = set()
    for sig in signals:
        parsed = _parse_ks_signal(sig)
        if parsed:
            pitcher_names.add(parsed[0])  # short name

    if not pitcher_names:
        return {}

    log.info("Backfill: fetching game logs for %d pitchers", len(pitcher_names))

    # Import here to avoid circular dependency and keep module lightweight
    import statsapi
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import date

    MLB_API = "https://statsapi.mlb.com/api/v1"
    season = date.today().year

    def _lookup_and_fetch(short_name: str) -> Tuple[str, List[dict]]:
        """Look up pitcher by name and fetch their game log."""
        initial = short_name[0]
        last = short_name.split(" ", 1)[1] if " " in short_name else short_name

        try:
            results = statsapi.lookup_player(last)
        except Exception as exc:
            log.debug("Player lookup failed for %s: %s", short_name, exc)
            return short_name, []

        # Find matching pitcher
        match = None
        for r in results:
            full = r.get("fullName", "")
            pos = r.get("primaryPosition", {}).get("abbreviation", "")
            if pos == "P" and full[0].upper() == initial.upper():
                match = r
                break

        if not match:
            # Fallback: try any matching first initial
            for r in results:
                full = r.get("fullName", "")
                if full and full[0].upper() == initial.upper():
                    match = r
                    break

        if not match:
            log.debug("No MLB match for %s", short_name)
            return short_name, []

        player_id = match["id"]
        full_name = match["fullName"]

        try:
            url = f"{MLB_API}/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}"
            resp = requests.get(url, timeout=10)
            splits = resp.json().get("stats", [{}])[0].get("splits", [])
        except Exception as exc:
            log.debug("Game log fetch failed for %s (%d): %s", full_name, player_id, exc)
            return full_name, []

        games = []
        for g in splits:
            stat = g.get("stat", {})
            game_date = g.get("date", "")
            ks = int(stat.get("strikeOuts", 0))
            games.append({"date": game_date, "strikeouts": ks})

        log.debug("Backfill: %s — %d starts", full_name, len(games))
        return full_name, games

    game_logs: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_lookup_and_fetch, name): name
            for name in pitcher_names
        }
        for fut in as_completed(futures):
            full_name, games = fut.result()
            if games:
                game_logs[full_name] = games

    log.info(
        "Backfill: fetched game logs for %d/%d pitchers",
        len(game_logs), len(pitcher_names),
    )
    return game_logs


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

    Below the curve's domain: linearly extrapolate from the first two
    breakpoints toward the origin (0, 0).  This prevents low raw
    probabilities from being inflated to the first breakpoint's y-value,
    which was causing phantom edge on sub-5% signals.

    Above the curve's domain: clamp to the last breakpoint (unchanged).
    """
    if not breakpoints:
        return x

    # Below the curve: extrapolate toward the origin.
    # The calibration curve has no data below its first breakpoint, so we
    # cannot trust it to be accurate there.  The safest assumption is a
    # straight line from (0, 0) to the first breakpoint — a raw 0% should
    # always calibrate to 0%, and values in between scale proportionally.
    if x <= breakpoints[0][0]:
        x0, y0 = breakpoints[0]
        if x0 <= 0:
            return y0
        return y0 * (x / x0)

    # Above the curve: clamp to the last breakpoint.
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
        mlb_outcomes: Optional[Dict[str, List[Tuple[float, int]]]] = None,
    ) -> CalibrationLayer:
        """Fit calibration curves from historical signal and settlement data.

        Args:
            signals:       List of signal records from load_signals().
            settlements:   Dict of ticker → settlement record from load_journal().
            min_samples:   Minimum settled signals per strategy to fit calibration.
            mlb_outcomes:  Optional dict of strategy → [(model_prob, outcome)]
                           from backfill_outcomes().  For strategies present here,
                           these outcomes replace Kalshi settlement data (they are
                           more complete — every game, not just traded markets).

        Returns:
            CalibrationLayer with per-strategy isotonic regression curves.
        """
        # Group (model_prob, outcome) pairs by strategy.
        # DEDUP by (strategy, ticker): the signal pipeline records a signal
        # on every poll cycle, so the same market can appear 100+ times.
        # Without dedup, one winning ticker inflates the bin win rate by
        # contributing 100+ "wins" instead of 1.
        strategy_data: Dict[str, List[Tuple[float, int]]] = {}
        seen: set = set()  # (strategy, ticker) pairs already processed

        # Strategies with MLB backfill data use that instead of settlements.
        mlb_strategies = set(mlb_outcomes.keys()) if mlb_outcomes else set()

        for sig in signals:
            ticker = sig.get("ticker", "")
            strategy = sig.get("strategy", "")
            prob = sig.get("model_prob_pct", 0)

            # Skip strategies that have MLB backfill data — we'll use that instead.
            if strategy in mlb_strategies:
                continue

            # Skip duplicate signals for the same market
            dedup_key = (strategy, ticker)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

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

        # Merge in MLB backfill data (already deduplicated by backfill_outcomes).
        if mlb_outcomes:
            for strategy, pairs in mlb_outcomes.items():
                strategy_data[strategy] = pairs

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
