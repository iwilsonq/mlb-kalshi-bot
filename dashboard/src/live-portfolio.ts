/**
 * Live portfolio powered by Kalshi WebSocket streams.
 *
 * Combines three WebSocket channels into a single real-time portfolio view:
 *   - ticker: real-time bid/ask prices for position tickers
 *   - market_positions: position changes (fills, settlements)
 *   - fill: individual fill events (new trades)
 *
 * Replaces the REST polling approach (GET /portfolio/positions + GET /markets)
 * with push-based updates. The REST client is still used for initial bootstrap
 * (loading current positions on startup), after which the WebSocket takes over.
 *
 * Usage:
 *   const lp = new LivePortfolio(config, journal)
 *   lp.onChange = () => render()
 *   await lp.start()
 *   // lp.positions, lp.prices, lp.balance are always up to date
 */

import type { DashboardConfig } from "./config.js"
import type { JournalReader } from "./journal.js"
import type { TradeRecord } from "./types.js"
import { KalshiClient } from "./kalshi.js"
import { KalshiWS, type WSStatus } from "./kalshi-ws.js"

// ── Types ────────────────────────────────────────────────────────────────

export interface LivePrice {
  yesBidDollars: number
  yesAskDollars: number
  lastPriceDollars: number
  /** Mid price in cents, computed from bid/ask */
  midCents: number
}

export interface LivePosition {
  ticker: string
  /** Number of contracts (from position_fp) */
  quantity: number
  /** Cost basis in USD (from position_cost_dollars) */
  costUsd: number
  /** Realized P&L in USD */
  realizedPnlUsd: number
  /** Fees paid in USD */
  feesPaidUsd: number
}

export interface LiveEnrichedPosition {
  ticker: string
  quantity: number
  costUsd: number
  /** Entry price in cents (cost / qty * 100) */
  entryCents: number
  /** Current mid price in cents */
  currentCents: number
  /** Unrealized P&L: market value - cost */
  unrealizedPnl: number
  /** Current market value */
  marketValueUsd: number
  /** Strategy from journal, or "unknown" */
  strategy: string
}

export interface LivePortfolioSummary {
  totalExposureUsd: number
  totalUnrealizedPnl: number
  totalMarketValueUsd: number
  positions: LiveEnrichedPosition[]
}

// ── LivePortfolio ────────────────────────────────────────────────────────

export class LivePortfolio {
  private config: DashboardConfig
  private journal: JournalReader
  private restClient: KalshiClient
  private ws: KalshiWS

  /** In-memory price cache: ticker -> LivePrice */
  private prices = new Map<string, LivePrice>()

  /** In-memory position cache: ticker -> LivePosition */
  private positionMap = new Map<string, LivePosition>()

  /** WS subscription IDs for cleanup */
  private tickerSid: number | null = null
  private positionSid: number | null = null
  private fillSid: number | null = null

  /** Callback when any data changes */
  onChange?: () => void

  /** Current balance in USD (refreshed periodically via REST) */
  balance = 0

  constructor(config: DashboardConfig, journal: JournalReader) {
    this.config = config
    this.journal = journal
    this.restClient = new KalshiClient(config)
    this.ws = new KalshiWS(config)

    // Wire up WS message handlers
    this.ws.on("ticker", (msg) => this.handleTicker(msg))
    this.ws.on("market_position", (msg) => this.handlePosition(msg))
    this.ws.on("fill", (msg) => this.handleFill(msg))
  }

  // ── Public API ────────────────────────────────────────────────────────

  get wsStatus(): WSStatus {
    return this.ws.status
  }

  get restClientInstance(): KalshiClient {
    return this.restClient
  }

  /**
   * Bootstrap: load current positions via REST, then connect WS and subscribe.
   */
  async start(): Promise<void> {
    // 1. Load initial balance and positions via REST
    try {
      this.balance = await this.restClient.getBalance()
    } catch {
      // Continue without balance
    }

    await this.bootstrapPositions()

    // 2. Connect WebSocket
    try {
      await this.ws.connect()
    } catch {
      // WS will auto-reconnect; continue with REST data
      return
    }

    // 3. Subscribe to position + fill channels (all markets, no filter)
    try {
      const posSids = await this.ws.subscribe(["market_positions"])
      if (posSids.length > 0) this.positionSid = posSids[0]
    } catch {
      // Best-effort
    }

    try {
      const fillSids = await this.ws.subscribe(["fill"])
      if (fillSids.length > 0) this.fillSid = fillSids[0]
    } catch {
      // Best-effort
    }

    // 4. Subscribe to ticker for all position tickers
    await this.subscribeToPositionTickers()
  }

  /**
   * Get the current enriched portfolio snapshot.
   */
  getPortfolio(): LivePortfolioSummary {
    const tradeLookup = new Map<string, TradeRecord>()
    for (const trade of this.journal.trades) {
      tradeLookup.set(trade.ticker, trade)
    }

    const enriched: LiveEnrichedPosition[] = []

    for (const [ticker, pos] of this.positionMap) {
      if (pos.quantity === 0) continue
      if (!ticker.startsWith("KXMLB")) continue

      const journalTrade = tradeLookup.get(ticker)
      const strategy = journalTrade?.strategy ?? "unknown"

      const entryCents =
        pos.costUsd > 0 && pos.quantity > 0
          ? Math.round((pos.costUsd / pos.quantity) * 100)
          : journalTrade?.price_cents ?? 0

      const price = this.prices.get(ticker)
      const currentCents = price?.midCents ?? 0

      const marketValueUsd = (currentCents * pos.quantity) / 100
      const unrealizedPnl = currentCents > 0 ? marketValueUsd - pos.costUsd : 0

      enriched.push({
        ticker,
        quantity: pos.quantity,
        costUsd: pos.costUsd,
        entryCents,
        currentCents,
        unrealizedPnl,
        marketValueUsd,
        strategy,
      })
    }

    // Sort by absolute unrealized P&L descending
    enriched.sort((a, b) => Math.abs(b.unrealizedPnl) - Math.abs(a.unrealizedPnl))

    const priced = enriched.filter((p) => p.currentCents > 0)
    const totalExposureUsd = enriched.reduce((s, p) => s + p.costUsd, 0)
    const totalUnrealizedPnl = priced.reduce((s, p) => s + p.unrealizedPnl, 0)
    const totalMarketValueUsd = priced.reduce((s, p) => s + p.marketValueUsd, 0)

    return {
      totalExposureUsd,
      totalUnrealizedPnl,
      totalMarketValueUsd,
      positions: enriched,
    }
  }

  /**
   * Get a price for a specific ticker, or null if not tracked.
   */
  getPrice(ticker: string): LivePrice | null {
    return this.prices.get(ticker) ?? null
  }

  /**
   * Refresh balance via REST (cheap call, do periodically).
   */
  async refreshBalance(): Promise<void> {
    try {
      this.balance = await this.restClient.getBalance()
    } catch {
      // Keep previous balance
    }
  }

  /**
   * Destroy WS connection and clean up.
   */
  destroy(): void {
    this.ws.destroy()
  }

  // ── WS Handlers ──────────────────────────────────────────────────────

  private handleTicker(msg: Record<string, unknown>): void {
    const ticker = msg.market_ticker as string
    if (!ticker) return

    const yesBid = parseFloat((msg.yes_bid_dollars as string) ?? "0")
    const yesAsk = parseFloat((msg.yes_ask_dollars as string) ?? "0")
    const lastPrice = parseFloat((msg.price_dollars as string) ?? "0")

    let midCents = 0
    if (yesBid > 0 && yesAsk > 0) {
      midCents = Math.round(((yesBid + yesAsk) / 2) * 100)
    } else if (yesBid > 0) {
      midCents = Math.round(yesBid * 100)
    } else if (yesAsk > 0) {
      midCents = Math.round(yesAsk * 100)
    } else if (lastPrice > 0) {
      midCents = Math.round(lastPrice * 100)
    }

    this.prices.set(ticker, { yesBidDollars: yesBid, yesAskDollars: yesAsk, lastPriceDollars: lastPrice, midCents })
    this.onChange?.()
  }

  private handlePosition(msg: Record<string, unknown>): void {
    const ticker = msg.market_ticker as string
    if (!ticker) return

    const quantity = parseFloat((msg.position_fp as string) ?? "0")
    const costUsd = parseFloat((msg.position_cost_dollars as string) ?? "0")
    const realizedPnlUsd = parseFloat((msg.realized_pnl_dollars as string) ?? "0")
    const feesPaidUsd = parseFloat((msg.fees_paid_dollars as string) ?? "0")

    if (quantity === 0) {
      // Position closed (settled or sold)
      this.positionMap.delete(ticker)
    } else {
      this.positionMap.set(ticker, { ticker, quantity, costUsd, realizedPnlUsd, feesPaidUsd })
    }

    this.onChange?.()
  }

  private handleFill(msg: Record<string, unknown>): void {
    const ticker = msg.market_ticker as string
    if (!ticker) return

    // If this is a new ticker we're not tracking prices for, subscribe
    if (!this.prices.has(ticker) && ticker.startsWith("KXMLB")) {
      this.addTickerSubscription(ticker)
    }

    this.onChange?.()
  }

  // ── Subscription management ──────────────────────────────────────────

  private async bootstrapPositions(): Promise<void> {
    try {
      const rawPositions = await this.restClient.getPositions()
      for (const pos of rawPositions) {
        if (!pos.ticker.startsWith("KXMLB")) continue
        const quantity = parseFloat(pos.position_fp ?? "0")
        if (quantity === 0) continue
        this.positionMap.set(pos.ticker, {
          ticker: pos.ticker,
          quantity,
          costUsd: parseFloat(pos.total_traded_dollars ?? "0"),
          realizedPnlUsd: parseFloat(pos.realized_pnl_dollars ?? "0"),
          feesPaidUsd: parseFloat(pos.fees_paid_dollars ?? "0"),
        })
      }
    } catch {
      // Continue without initial positions
    }
  }

  private async subscribeToPositionTickers(): Promise<void> {
    const tickers = [...this.positionMap.keys()]
    if (tickers.length === 0) return

    try {
      const sids = await this.ws.subscribe(["ticker"], {
        market_tickers: tickers,
        send_initial_snapshot: true,
      })
      if (sids.length > 0) this.tickerSid = sids[0]
    } catch {
      // Best-effort
    }
  }

  private async addTickerSubscription(ticker: string): Promise<void> {
    if (this.tickerSid === null) {
      // No ticker subscription yet, create one
      try {
        const sids = await this.ws.subscribe(["ticker"], {
          market_tickers: [ticker],
          send_initial_snapshot: true,
        })
        if (sids.length > 0) this.tickerSid = sids[0]
      } catch {
        // Best-effort
      }
    } else {
      // Add to existing subscription
      try {
        await this.ws.updateSubscription({
          sid: this.tickerSid,
          action: "add_markets",
          market_tickers: [ticker],
          send_initial_snapshot: true,
        })
      } catch {
        // Best-effort
      }
    }
  }
}
