/**
 * Position enrichment for the Slugger dashboard.
 *
 * Takes raw Kalshi positions and enriches them with:
 *   - Entry price (from journal trades or API fills)
 *   - Current market price (from markets API)
 *   - Unrealized P&L (current - entry) * quantity
 *   - Strategy attribution (from journal trades)
 *   - Aggregate exposure and unrealized P&L
 */

import type { KalshiClient, KalshiPosition, KalshiMarketData } from "./kalshi.js"
import type { JournalReader } from "./journal.js"
import type { TradeRecord } from "./types.js"

// ── Enriched position ────────────────────────────────────────────────────

export interface EnrichedPosition {
  ticker: string
  eventTicker: string
  side: "yes" | "no"
  quantity: number
  /** Entry price in cents (from journal or fills) */
  entryCents: number
  /** Current market mid price in cents */
  currentCents: number
  /** Unrealized P&L in USD: (current - entry) * qty / 100 for YES, inverse for NO */
  unrealizedPnl: number
  /** Total cost in USD */
  costUsd: number
  /** Current market value in USD */
  marketValueUsd: number
  /** Strategy from journal, or "unknown" */
  strategy: string
  /** Market title if available */
  title: string
}

export interface PortfolioSummary {
  /** Total cost basis of open positions */
  totalExposureUsd: number
  /** Sum of unrealized P&L across all positions */
  totalUnrealizedPnl: number
  /** Current market value of all positions */
  totalMarketValueUsd: number
  /** Enriched positions */
  positions: EnrichedPosition[]
}

// ── Circuit breaker state ────────────────────────────────────────────────

export interface CircuitBreakerState {
  /** Current streak of consecutive losses (resets on win/void) */
  consecutiveLosses: number
  /** Total realized loss in USD for the day */
  todayLossUsd: number
  /** Whether the breaker is tripped */
  tripped: boolean
  /** Configured max consecutive losses */
  maxConsecutiveLosses: number
  /** Configured max loss USD */
  maxLossUsd: number
}

// ── Enrichment logic ─────────────────────────────────────────────────────

/**
 * Enrich positions with market prices, entry prices, and strategy attribution.
 */
export async function enrichPositions(
  rawPositions: KalshiPosition[],
  client: KalshiClient,
  journal: JournalReader,
): Promise<PortfolioSummary> {
  if (rawPositions.length === 0) {
    return {
      totalExposureUsd: 0,
      totalUnrealizedPnl: 0,
      totalMarketValueUsd: 0,
      positions: [],
    }
  }

  // Build journal trade lookup: ticker -> TradeRecord
  const tradeLookup = new Map<string, TradeRecord>()
  for (const trade of journal.trades) {
    tradeLookup.set(trade.ticker, trade)
  }

  // Fetch current market prices for all position tickers (parallel, best-effort)
  const marketPromises = rawPositions.map((pos) =>
    client.getMarket(pos.ticker).catch(() => null),
  )
  const markets = await Promise.all(marketPromises)
  const marketMap = new Map<string, KalshiMarketData>()
  for (const m of markets) {
    if (m) marketMap.set(m.ticker, m)
  }

  const enriched: EnrichedPosition[] = []

  for (const pos of rawPositions) {
    const qty = pos.position ?? 0
    if (qty === 0) continue

    const side: "yes" | "no" = qty > 0 ? "yes" : "no"
    const absQty = Math.abs(qty)

    // Entry price from journal trade
    const journalTrade = tradeLookup.get(pos.ticker)
    let entryCents = 0
    let strategy = "unknown"

    if (journalTrade) {
      entryCents = journalTrade.price_cents
      strategy = journalTrade.strategy
    }

    // Current market price
    const market = marketMap.get(pos.ticker)
    let currentCents = 0
    let title = pos.ticker

    if (market) {
      title = market.title || pos.ticker

      if (side === "yes") {
        // For YES positions, use the YES bid (what we could sell at)
        const bidDollars = parseFloat(market.yes_bid_dollars ?? "0")
        const askDollars = parseFloat(market.yes_ask_dollars ?? "0")
        // Use mid if both available, else bid, else ask
        if (bidDollars > 0 && askDollars > 0) {
          currentCents = Math.round(((bidDollars + askDollars) / 2) * 100)
        } else if (bidDollars > 0) {
          currentCents = Math.round(bidDollars * 100)
        } else if (askDollars > 0) {
          currentCents = Math.round(askDollars * 100)
        }
      } else {
        // For NO positions, use NO bid
        const bidDollars = parseFloat(market.no_bid_dollars ?? "0")
        const askDollars = parseFloat(market.no_ask_dollars ?? "0")
        if (bidDollars > 0 && askDollars > 0) {
          currentCents = Math.round(((bidDollars + askDollars) / 2) * 100)
        } else if (bidDollars > 0) {
          currentCents = Math.round(bidDollars * 100)
        } else if (askDollars > 0) {
          currentCents = Math.round(askDollars * 100)
        }
      }
    }

    // Compute P&L
    const costUsd = (entryCents * absQty) / 100
    const marketValueUsd = (currentCents * absQty) / 100
    const unrealizedPnl = marketValueUsd - costUsd

    enriched.push({
      ticker: pos.ticker,
      eventTicker: pos.event_ticker,
      side,
      quantity: absQty,
      entryCents,
      currentCents,
      unrealizedPnl,
      costUsd,
      marketValueUsd,
      strategy,
      title,
    })
  }

  // Sort by absolute unrealized P&L descending (biggest movers first)
  enriched.sort((a, b) => Math.abs(b.unrealizedPnl) - Math.abs(a.unrealizedPnl))

  const totalExposureUsd = enriched.reduce((sum, p) => sum + p.costUsd, 0)
  const totalUnrealizedPnl = enriched.reduce((sum, p) => sum + p.unrealizedPnl, 0)
  const totalMarketValueUsd = enriched.reduce((sum, p) => sum + p.marketValueUsd, 0)

  return {
    totalExposureUsd,
    totalUnrealizedPnl,
    totalMarketValueUsd,
    positions: enriched,
  }
}

// ── Circuit breaker ──────────────────────────────────────────────────────

/**
 * Compute circuit breaker state from today's journal data.
 *
 * Mirrors the logic in slugger/game_processor.py CircuitBreaker:
 *   - Counts consecutive losses from the most recent settlements
 *   - Sums total loss for the day
 *   - Trips if either threshold is exceeded
 */
export function computeCircuitBreaker(
  journal: JournalReader,
  maxConsecutiveLosses: number = 3,
  maxLossUsd: number = 10,
): CircuitBreakerState {
  const today = new Date().toISOString().slice(0, 10)

  // Get today's trades and their settlement outcomes
  const todayTrades = journal.trades.filter((t) => t.date === today)
  const settlementMap = new Map(
    journal.settlements.map((s) => [s.ticker, s]),
  )

  // Walk today's trades in order, track consecutive losses
  let consecutiveLosses = 0
  let todayLossUsd = 0

  for (const trade of todayTrades) {
    const settlement = settlementMap.get(trade.ticker)
    if (!settlement) continue // Not yet settled

    if (settlement.market_result === "void") {
      // Voids don't count
      continue
    }

    if (settlement.pnl_usd < 0) {
      consecutiveLosses += 1
      todayLossUsd += Math.abs(settlement.pnl_usd)
    } else {
      consecutiveLosses = 0 // Reset on win
    }
  }

  const tripped =
    consecutiveLosses >= maxConsecutiveLosses ||
    todayLossUsd >= maxLossUsd

  return {
    consecutiveLosses,
    todayLossUsd,
    tripped,
    maxConsecutiveLosses,
    maxLossUsd,
  }
}
