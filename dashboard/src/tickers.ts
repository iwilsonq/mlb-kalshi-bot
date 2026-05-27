/**
 * Ticker parsing utilities for the Slugger dashboard.
 *
 * Kalshi MLB ticker format:
 *   KXMLBKS-26MAY271840LAADET-LAAJSORIANO59-6
 *   PREFIX  -DATEINFO         -TEAMPLAYERNUM  -THRESHOLD
 *
 * This module extracts human-readable descriptions from tickers:
 *   "LAA J. Soriano 6+ Ks"
 */

const TWO_CHAR_TEAMS = new Set(["SD", "SF", "KC", "TB"])

/** Market type extracted from the ticker prefix. */
function marketType(prefix: string): string {
  if (prefix.includes("KS")) return "Ks"
  if (prefix.includes("HIT")) return "hits"
  if (prefix.includes("HRR")) return "H+R+RBI"
  if (prefix.includes("HR")) return "HRs"
  if (prefix.includes("GAME")) return "winner"
  return ""
}

/** Parse a player name from a ticker segment like "LAAJSORIANO59". */
function parsePlayer(segment: string): { team: string; name: string } | null {
  const prefix2 = segment.slice(0, 2)
  if (TWO_CHAR_TEAMS.has(prefix2)) {
    const m = segment.slice(2).match(/^([A-Z])([A-Z]+?)(\d+)$/)
    if (m) {
      return {
        team: prefix2,
        name: m[1] + ". " + m[2].charAt(0).toUpperCase() + m[2].slice(1).toLowerCase(),
      }
    }
  }

  const m = segment.match(/^([A-Z]{3})([A-Z])([A-Z]+?)(\d+)$/)
  if (m) {
    return {
      team: m[1],
      name: m[2] + ". " + m[3].charAt(0).toUpperCase() + m[3].slice(1).toLowerCase(),
    }
  }

  return null
}

/**
 * Parse a Kalshi MLB ticker into a human-readable description.
 *
 * Examples:
 *   "KXMLBKS-26MAY271840LAADET-LAAJSORIANO59-6"  → "LAA J. Soriano 6+ Ks"
 *   "KXMLBHR-26MAY271340STLMIL-MILJBAUERS9-4"    → "MIL J. Bauers 4+ HRs"
 *   "KXMLBGAME-26MAY131840COLPIT-PIT"             → null (no player)
 *
 * Returns null if the ticker can't be parsed into a player description.
 */
export function describeTicker(ticker: string): string | null {
  const parts = ticker.split("-")
  if (parts.length < 3) return null

  const prefix = parts[0]
  const mType = marketType(prefix)

  // Threshold from last segment (if purely numeric)
  const lastPart = parts[parts.length - 1]
  const threshold = /^\d+$/.test(lastPart) ? `${lastPart}+` : null

  // Player from the appropriate segment
  const playerIdx = threshold ? parts.length - 2 : parts.length - 1
  const player = parsePlayer(parts[playerIdx])
  if (!player) return null

  let display = `${player.team} ${player.name}`
  if (threshold) display += ` ${threshold}`
  if (mType) display += ` ${mType}`

  return display
}

/**
 * Get just the player name from a ticker (without team, threshold, or market type).
 * Useful for compact displays.
 */
export function playerFromTicker(ticker: string): string | null {
  const parts = ticker.split("-")
  if (parts.length < 3) return null

  const lastPart = parts[parts.length - 1]
  const playerIdx = /^\d+$/.test(lastPart) ? parts.length - 2 : parts.length - 1
  const player = parsePlayer(parts[playerIdx])
  return player ? player.name : null
}
