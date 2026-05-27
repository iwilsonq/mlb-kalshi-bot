/**
 * Read-only Kalshi API client for the Slugger dashboard.
 *
 * Ports the RSA-PSS authentication from slugger/kalshi_client.py
 * and exposes only the read endpoints needed for dashboard display:
 *   - get_balance()
 *   - get_positions()
 *   - get_market(ticker)
 *   - get_event_markets(event_ticker)
 *   - get_settlements()
 *
 * No order placement — this client is intentionally read-only.
 */

import { createSign } from "node:crypto"
import { readFileSync } from "node:fs"
import type { DashboardConfig } from "./config.js"

// ── Auth ─────────────────────────────────────────────────────────────────────

function loadPrivateKey(keyPath: string): string {
  return readFileSync(keyPath, "utf-8").trim()
}

/**
 * Create RSA-PSS signature for a Kalshi API request.
 *
 * Mirrors slugger/kalshi_client.py _sign_request():
 *   message = f"{timestamp}{method}{full_path}"
 *   RSA-PSS with SHA-256, salt length = digest length (32)
 */
function signRequest(
  privateKeyPem: string,
  timestamp: string,
  method: string,
  path: string,
  baseUrl: string,
): string {
  // Strip query params before signing
  const pathNoQuery = path.split("?")[0]
  // Build the full URL path from the API root
  const url = new URL(baseUrl + pathNoQuery)
  const fullPath = url.pathname

  const message = `${timestamp}${method}${fullPath}`

  const signer = createSign("SHA256")
  signer.update(message)
  signer.end()

  const signature = signer.sign(
    {
      key: privateKeyPem,
      padding: 6, // RSA_PKCS1_PSS_PADDING
      saltLength: 32, // DIGEST_LENGTH for SHA-256
    },
    "base64",
  )

  return signature
}

function authHeaders(
  apiKeyId: string,
  privateKeyPem: string,
  method: string,
  path: string,
  baseUrl: string,
): Record<string, string> {
  const timestamp = String(Date.now())
  const signature = signRequest(privateKeyPem, timestamp, method, path, baseUrl)
  return {
    "KALSHI-ACCESS-KEY": apiKeyId,
    "KALSHI-ACCESS-SIGNATURE": signature,
    "KALSHI-ACCESS-TIMESTAMP": timestamp,
  }
}

// ── Client ───────────────────────────────────────────────────────────────────

export class KalshiClient {
  private apiKeyId: string
  private privateKeyPem: string
  private baseUrl: string

  constructor(config: DashboardConfig) {
    this.apiKeyId = config.kalshiApiKeyId
    this.privateKeyPem = loadPrivateKey(config.kalshiPrivateKeyPath)

    this.baseUrl = config.useDemo
      ? "https://external-api.demo.kalshi.co/trade-api/v2"
      : "https://external-api.kalshi.com/trade-api/v2"
  }

  // ── HTTP helpers ─────────────────────────────────────────────────────

  private async get<T = Record<string, unknown>>(
    path: string,
    params?: Record<string, string | number>,
  ): Promise<T> {
    let queryString = ""
    if (params) {
      const searchParams = new URLSearchParams()
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null) {
          searchParams.set(k, String(v))
        }
      }
      queryString = `?${searchParams.toString()}`
    }

    const fullPath = path + queryString
    const headers = authHeaders(
      this.apiKeyId,
      this.privateKeyPem,
      "GET",
      fullPath,
      this.baseUrl,
    )

    const resp = await fetch(this.baseUrl + fullPath, {
      method: "GET",
      headers: {
        ...headers,
        Accept: "application/json",
      },
    })

    if (!resp.ok) {
      const body = await resp.text().catch(() => "")
      throw new Error(`Kalshi API error ${resp.status}: ${body}`)
    }

    return (await resp.json()) as T
  }

  // ── Account ──────────────────────────────────────────────────────────

  /**
   * Get available balance in USD.
   */
  async getBalance(): Promise<number> {
    const data = await this.get<{ balance: number }>("/portfolio/balance")
    return (data.balance ?? 0) / 100.0
  }

  /**
   * Get all open market positions.
   */
  async getPositions(): Promise<KalshiPosition[]> {
    const data = await this.get<{
      market_positions: KalshiPosition[]
      event_positions: KalshiEventPosition[]
    }>("/portfolio/positions", { limit: 200 })
    return data.market_positions ?? []
  }

  /**
   * Get event-level position summaries.
   */
  async getEventPositions(): Promise<KalshiEventPosition[]> {
    const data = await this.get<{
      market_positions: KalshiPosition[]
      event_positions: KalshiEventPosition[]
    }>("/portfolio/positions", { limit: 200 })
    return data.event_positions ?? []
  }

  // ── Market queries ───────────────────────────────────────────────────

  /**
   * Get a single market by ticker.
   */
  async getMarket(ticker: string): Promise<KalshiMarketData | null> {
    try {
      const data = await this.get<{ market: KalshiMarketData }>(
        `/markets/${ticker}`,
      )
      return data.market ?? null
    } catch (e) {
      if (e instanceof Error && e.message.includes("404")) return null
      throw e
    }
  }

  /**
   * Get markets for an event ticker.
   */
  async getEventMarkets(
    eventTicker: string,
    status: string = "open",
  ): Promise<KalshiMarketData[]> {
    const data = await this.get<{ markets: KalshiMarketData[] }>("/markets", {
      event_ticker: eventTicker,
      limit: 100,
      status,
    })
    return data.markets ?? []
  }

  /**
   * Get settlement records.
   */
  async getSettlements(
    limit: number = 200,
    ticker?: string,
  ): Promise<KalshiSettlement[]> {
    const params: Record<string, string | number> = { limit }
    if (ticker) params.ticker = ticker
    const data = await this.get<{ settlements: KalshiSettlement[] }>(
      "/portfolio/settlements",
      params,
    )
    return data.settlements ?? []
  }

  /**
   * Get fill records (matched order legs).
   */
  async getFills(
    limit: number = 200,
    ticker?: string,
  ): Promise<KalshiFill[]> {
    const params: Record<string, string | number> = { limit }
    if (ticker) params.ticker = ticker
    const data = await this.get<{ fills: KalshiFill[] }>(
      "/portfolio/fills",
      params,
    )
    return data.fills ?? []
  }

  /**
   * Check API connectivity by fetching balance.
   * Returns true if successful, false otherwise.
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.getBalance()
      return true
    } catch {
      return false
    }
  }
}

// ── Response types ───────────────────────────────────────────────────────────

export interface KalshiPosition {
  ticker: string
  /** Dollar string: total cost of position */
  total_traded_dollars?: string
  /** Dollar string: current market exposure */
  market_exposure_dollars?: string
  /** Dollar string: realized P&L */
  realized_pnl_dollars?: string
  /** Dollar string: fees paid */
  fees_paid_dollars?: string
  /** Float string: number of contracts held (e.g. "10.00") */
  position_fp?: string
  /** Number of resting orders */
  resting_orders_count?: number
  last_updated_ts?: string
}

export interface KalshiEventPosition {
  event_ticker: string
  /** Dollar string: total cost across all markets in this event */
  total_cost_dollars?: string
  /** Dollar string: event-level exposure */
  event_exposure_dollars?: string
  /** Dollar string: realized P&L */
  realized_pnl_dollars?: string
  /** Dollar string: fees paid */
  fees_paid_dollars?: string
  /** Float string: total shares */
  total_cost_shares_fp?: string
}

export interface KalshiMarketData {
  ticker: string
  event_ticker: string
  title: string
  subtitle?: string
  status: string
  /** Dollar strings like "0.35" */
  yes_ask_dollars?: string
  yes_bid_dollars?: string
  no_ask_dollars?: string
  no_bid_dollars?: string
  last_price_dollars?: string
  volume_fp?: string
  open_interest_fp?: string
}

export interface KalshiFill {
  ticker: string
  market_ticker: string
  order_id: string
  fill_id: string
  side: string
  action: string
  book_side: string
  /** Dollar strings */
  yes_price_dollars: string
  no_price_dollars: string
  count_fp: string
  fee_cost: string
  created_time: string
  ts: number
  outcome_side?: string
  is_taker?: boolean
}

export interface KalshiSettlement {
  ticker: string
  market_result: string
  revenue: number
  /** Dollar strings */
  yes_total_cost_dollars?: string
  no_total_cost_dollars?: string
  fee_cost?: string
  settled_time?: string
  yes_count_fp?: string
  no_count_fp?: string
  value?: number
}
