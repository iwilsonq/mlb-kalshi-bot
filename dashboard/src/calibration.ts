/**
 * Calibration layer for the Slugger dashboard.
 *
 * Loads the same calibration.json file produced by `python3 main.py calibrate --fit`
 * and applies isotonic regression interpolation to map raw model probabilities
 * to calibrated probabilities.
 *
 * This lets the dashboard show both the raw model output and the calibrated
 * value that actually drives trading decisions.
 */

import { readFileSync, existsSync } from "node:fs"

// ── Types ────────────────────────────────────────────────────────────────

type Breakpoint = [number, number] // [x, y]

interface CalibrationData {
  curves: Record<string, Breakpoint[]>
  sample_counts: Record<string, number>
}

// ── Interpolation ────────────────────────────────────────────────────────

/**
 * Linearly interpolate a calibrated value from PAVA breakpoints.
 * Clamps to the range of the breakpoints (no extrapolation).
 * Mirrors slugger/calibration.py _interpolate().
 */
function interpolate(breakpoints: Breakpoint[], x: number): number {
  if (breakpoints.length === 0) return x

  // Clamp to endpoints
  if (x <= breakpoints[0][0]) return breakpoints[0][1]
  if (x >= breakpoints[breakpoints.length - 1][0]) {
    return breakpoints[breakpoints.length - 1][1]
  }

  // Find the two surrounding breakpoints
  for (let i = 0; i < breakpoints.length - 1; i++) {
    const [x0, y0] = breakpoints[i]
    const [x1, y1] = breakpoints[i + 1]
    if (x0 <= x && x <= x1) {
      if (x1 === x0) return y0
      const t = (x - x0) / (x1 - x0)
      return y0 + t * (y1 - y0)
    }
  }

  return breakpoints[breakpoints.length - 1][1]
}

// ── CalibrationLayer ─────────────────────────────────────────────────────

export class CalibrationLayer {
  private curves: Map<string, Breakpoint[]>
  private sampleCounts: Map<string, number>

  constructor(
    curves: Map<string, Breakpoint[]> = new Map(),
    sampleCounts: Map<string, number> = new Map(),
  ) {
    this.curves = curves
    this.sampleCounts = sampleCounts
  }

  /**
   * Apply calibration to a raw model probability.
   * Returns the calibrated probability (0-100), or the raw value if
   * no calibration exists for this strategy.
   */
  calibrate(strategy: string, rawProbPct: number): number {
    const breakpoints = this.curves.get(strategy)
    if (!breakpoints) return rawProbPct
    const calibrated = interpolate(breakpoints, rawProbPct)
    return Math.max(0, Math.min(100, Math.round(calibrated)))
  }

  /**
   * Check if calibration data exists for a strategy.
   */
  hasCalibration(strategy: string): boolean {
    return this.curves.has(strategy)
  }

  /**
   * Get the number of samples used to fit calibration for a strategy.
   */
  sampleCount(strategy: string): number {
    return this.sampleCounts.get(strategy) ?? 0
  }

  /**
   * Get all strategy names that have calibration curves.
   */
  get strategies(): string[] {
    return [...this.curves.keys()]
  }

  /**
   * Load calibration from the JSON file produced by `main.py calibrate --fit`.
   * Returns an empty (pass-through) layer if the file doesn't exist.
   */
  static load(logsDir: string): CalibrationLayer {
    const path = `${logsDir}/calibration.json`
    if (!existsSync(path)) {
      return new CalibrationLayer()
    }

    try {
      const data: CalibrationData = JSON.parse(readFileSync(path, "utf-8"))
      const curves = new Map<string, Breakpoint[]>()
      const sampleCounts = new Map<string, number>()

      for (const [strategy, breakpoints] of Object.entries(data.curves ?? {})) {
        curves.set(strategy, breakpoints)
      }
      for (const [strategy, count] of Object.entries(data.sample_counts ?? {})) {
        sampleCounts.set(strategy, count)
      }

      return new CalibrationLayer(curves, sampleCounts)
    } catch {
      return new CalibrationLayer()
    }
  }
}
