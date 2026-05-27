/**
 * Domain types for the Slugger dashboard.
 *
 * These mirror the Python dataclasses in slugger/journal.py and slugger/types.py,
 * matching the exact JSON shapes written to journal.jsonl and signals.jsonl.
 */

// ── Journal record types ─────────────────────────────────────────────────────

/** Discriminated union tag for journal records. */
export type JournalRecordType = "trade" | "settlement"

/** Written immediately after a successful order placement. */
export interface TradeRecord {
  type: "trade"
  placed_at: string // ISO 8601 UTC
  date: string // YYYY-MM-DD
  ticker: string
  strategy: string
  side: "yes" | "no"
  count: number
  price_cents: number
  cost_usd: number
  edge_cents: number
  reason: string
  order_id: string
  model_version?: string
}

/** Written by cmd_settle once Kalshi resolves a market we traded. */
export interface SettlementRecord {
  type: "settlement"
  settled_at: string // ISO 8601 UTC
  ticker: string
  market_result: "yes" | "no" | "void" | "scalar"
  revenue_usd: number
  yes_cost_usd: number
  fee_usd: number
  pnl_usd: number
}

export type JournalRecord = TradeRecord | SettlementRecord

// ── Signal record types ──────────────────────────────────────────────────────

/** Every signal the model evaluates (traded or not), for calibration. */
export interface SignalRecord {
  type: "signal"
  timestamp: string // ISO 8601 UTC
  date: string // YYYY-MM-DD
  ticker: string
  strategy: string
  model_prob_pct: number // 0-100
  market_price_cents: number // 1-99
  edge_cents: number
  traded: boolean
  reason: string
  model_version?: string
}

// ── Placed ledger ────────────────────────────────────────────────────────────

/** Daily dedup ledger: logs/placed_YYYY-MM-DD.json is a string array of tickers. */
export type PlacedLedger = string[]

// ── Computed stats (mirrors slugger/journal.py StrategyStats) ─────────────

export interface StrategyStats {
  strategy: string
  bets: number
  settled: number
  wins: number
  voids: number
  totalCostUsd: number
  totalRevenueUsd: number
  totalFeeUsd: number
  totalPnlUsd: number
  /** settled - wins - voids */
  losses: number
  /** wins / (settled - voids), or null if no decided bets */
  winRate: number | null
  /** (totalPnlUsd / totalCostUsd) * 100, or null */
  roiPct: number | null
  /** bets - settled */
  pending: number
}

// ── Kalshi API types (subset needed for dashboard) ───────────────────────────

export interface KalshiPosition {
  ticker: string
  event_ticker: string
  market_title?: string
  /** Number of YES contracts held (positive = long YES) */
  yes_count: number
  /** Number of NO contracts held */
  no_count: number
  /** Average price paid in cents */
  avg_price_yes?: number
  avg_price_no?: number
  /** Total cost in dollars */
  total_cost?: number
}

export interface KalshiMarket {
  ticker: string
  event_ticker: string
  title: string
  status: string
  yes_ask?: number
  yes_bid?: number
  no_ask?: number
  no_bid?: number
  volume?: number
  open_interest?: number
  /** Dollar string like "0.35" */
  yes_ask_dollars?: string
  last_price?: number
}
