/**
 * Kalshi WebSocket client for the Slugger dashboard.
 *
 * Provides authenticated, auto-reconnecting WebSocket connection to the
 * Kalshi streaming API. Handles:
 *   - RSA-PSS auth in the handshake headers
 *   - Command ID tracking for subscribe/unsubscribe correlation
 *   - Auto-reconnect with exponential backoff
 *   - Message routing to channel-specific handlers
 *   - Subscription management (subscribe, unsubscribe, update)
 *
 * Usage:
 *   const ws = new KalshiWS(config)
 *   ws.on("ticker", (msg) => { ... })
 *   ws.on("fill", (msg) => { ... })
 *   ws.on("market_position", (msg) => { ... })
 *   await ws.connect()
 *   ws.subscribe(["ticker"], { market_tickers: [...], send_initial_snapshot: true })
 */

import { createSign } from "node:crypto"
import { readFileSync } from "node:fs"
import type { DashboardConfig } from "./config.js"

// ── Types ────────────────────────────────────────────────────────────────

export type WSChannel =
  | "ticker"
  | "trade"
  | "fill"
  | "market_positions"
  | "orderbook_delta"
  | "market_lifecycle_v2"
  | "multivariate_market_lifecycle"
  | "user_orders"
  | "order_group_updates"
  | "communications"

export type WSStatus = "disconnected" | "connecting" | "connected" | "reconnecting"

export interface SubscribeParams {
  channels: WSChannel[]
  market_ticker?: string
  market_tickers?: string[]
  send_initial_snapshot?: boolean
}

export interface UpdateSubscriptionParams {
  sids?: number[]
  sid?: number
  action: "add_markets" | "delete_markets" | "get_snapshot"
  market_ticker?: string
  market_tickers?: string[]
  send_initial_snapshot?: boolean
}

/** Subscription confirmed by server */
export interface Subscription {
  channel: WSChannel
  sid: number
}

/** Any message from the WebSocket */
export interface WSMessage {
  type: string
  sid?: number
  id?: number
  seq?: number
  msg?: Record<string, unknown>
}

type MessageHandler = (msg: Record<string, unknown>) => void
type StatusHandler = (status: WSStatus) => void

// ── Auth ─────────────────────────────────────────────────────────────────

function signRequest(
  privateKeyPem: string,
  timestamp: string,
  method: string,
  path: string,
): string {
  const message = `${timestamp}${method}${path}`
  const signer = createSign("SHA256")
  signer.update(message)
  signer.end()
  return signer.sign(
    {
      key: privateKeyPem,
      padding: 6, // RSA_PKCS1_PSS_PADDING
      saltLength: 32,
    },
    "base64",
  )
}

// ── KalshiWS ─────────────────────────────────────────────────────────────

export class KalshiWS {
  private config: DashboardConfig
  private privateKeyPem: string
  private wsUrl: string
  private wsPath: string
  private ws: WebSocket | null = null
  private _status: WSStatus = "disconnected"

  // Command tracking
  private nextCmdId = 1
  private pendingCommands = new Map<
    number,
    { resolve: (value: WSMessage) => void; reject: (err: Error) => void }
  >()

  // Subscription tracking
  private subscriptions = new Map<number, Subscription>() // sid -> subscription
  private resubscribeQueue: Array<{
    params: SubscribeParams
    resolve: (sids: number[]) => void
  }> = []

  // Message handlers by message type
  private handlers = new Map<string, Set<MessageHandler>>()
  private statusHandlers = new Set<StatusHandler>()

  // Reconnection
  private reconnectAttempts = 0
  private maxReconnectDelay = 30000
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private shouldReconnect = true
  private destroyed = false

  constructor(config: DashboardConfig) {
    this.config = config
    this.privateKeyPem = readFileSync(config.kalshiPrivateKeyPath, "utf-8").trim()

    this.wsPath = "/trade-api/ws/v2"
    if (config.useDemo) {
      this.wsUrl = `wss://external-api-ws.demo.kalshi.co${this.wsPath}`
    } else {
      this.wsUrl = `wss://external-api-ws.kalshi.com${this.wsPath}`
    }
  }

  // ── Public API ────────────────────────────────────────────────────────

  get status(): WSStatus {
    return this._status
  }

  /**
   * Connect to the WebSocket. Resolves when connected and ready.
   */
  async connect(): Promise<void> {
    if (this._status === "connected" || this._status === "connecting") return
    this.destroyed = false
    this.shouldReconnect = true
    return this.doConnect()
  }

  /**
   * Subscribe to one or more channels.
   * Returns the server-assigned subscription IDs (sids).
   */
  async subscribe(
    channels: WSChannel[],
    opts: {
      market_ticker?: string
      market_tickers?: string[]
      send_initial_snapshot?: boolean
    } = {},
  ): Promise<number[]> {
    const params: SubscribeParams = { channels, ...opts }

    if (this._status !== "connected") {
      // Queue for resubscribe after reconnect
      return new Promise((resolve) => {
        this.resubscribeQueue.push({ params, resolve })
      })
    }

    return this.doSubscribe(params)
  }

  /**
   * Unsubscribe by subscription IDs.
   */
  async unsubscribe(sids: number[]): Promise<void> {
    if (this._status !== "connected" || sids.length === 0) return

    const id = this.nextCmdId++
    this.send({
      id,
      cmd: "unsubscribe",
      params: { sids },
    })

    try {
      await this.waitForCommand(id, 5000)
    } catch {
      // Best-effort
    }

    for (const sid of sids) {
      this.subscriptions.delete(sid)
    }
  }

  /**
   * Update an existing subscription (add/remove markets).
   */
  async updateSubscription(params: UpdateSubscriptionParams): Promise<void> {
    if (this._status !== "connected") return

    const id = this.nextCmdId++
    this.send({
      id,
      cmd: "update_subscription",
      params,
    })

    try {
      await this.waitForCommand(id, 5000)
    } catch {
      // Best-effort
    }
  }

  /**
   * Register a handler for a message type.
   * Common types: "ticker", "fill", "market_position", "trade",
   *   "orderbook_snapshot", "orderbook_delta", "subscribed", "error"
   */
  on(type: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set())
    }
    this.handlers.get(type)!.add(handler)
    return () => this.handlers.get(type)?.delete(handler)
  }

  /**
   * Register a handler for connection status changes.
   */
  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler)
    return () => this.statusHandlers.delete(handler)
  }

  /**
   * Get all active subscription IDs.
   */
  getSubscriptions(): Map<number, Subscription> {
    return new Map(this.subscriptions)
  }

  /**
   * Disconnect and stop reconnecting.
   */
  destroy(): void {
    this.destroyed = true
    this.shouldReconnect = false

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    // Reject all pending commands
    for (const [, pending] of this.pendingCommands) {
      pending.reject(new Error("WebSocket destroyed"))
    }
    this.pendingCommands.clear()

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    this.setStatus("disconnected")
  }

  // ── Internal ──────────────────────────────────────────────────────────

  private setStatus(status: WSStatus): void {
    if (this._status === status) return
    this._status = status
    for (const handler of this.statusHandlers) {
      try {
        handler(status)
      } catch {
        // Don't let handler errors break the WS
      }
    }
  }

  private async doConnect(): Promise<void> {
    this.setStatus(this.reconnectAttempts > 0 ? "reconnecting" : "connecting")

    // Build auth headers for the handshake
    const timestamp = String(Date.now())
    const signature = signRequest(
      this.privateKeyPem,
      timestamp,
      "GET",
      this.wsPath,
    )

    return new Promise<void>((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl, {
          headers: {
            "KALSHI-ACCESS-KEY": this.config.kalshiApiKeyId,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
          },
        } as any)

        this.ws.onopen = () => {
          this.reconnectAttempts = 0
          this.setStatus("connected")

          // Re-subscribe any queued subscriptions
          this.processResubscribeQueue()

          resolve()
        }

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data as string)
        }

        this.ws.onclose = (event) => {
          this.ws = null
          this.setStatus("disconnected")

          if (this.shouldReconnect && !this.destroyed) {
            this.scheduleReconnect()
          }
        }

        this.ws.onerror = (event) => {
          // onerror is always followed by onclose, so reconnect logic
          // is handled there. Only reject the initial connect promise.
          if (this._status === "connecting") {
            reject(new Error("WebSocket connection failed"))
          }
        }
      } catch (err) {
        this.setStatus("disconnected")
        reject(err)
      }
    })
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer || this.destroyed) return

    const delay = Math.min(
      1000 * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay,
    )
    this.reconnectAttempts++

    this.reconnectTimer = setTimeout(async () => {
      this.reconnectTimer = null
      if (!this.shouldReconnect || this.destroyed) return

      try {
        await this.doConnect()
        // Re-subscribe to all previous subscriptions
        await this.resubscribeAll()
      } catch {
        // doConnect failure will trigger another scheduleReconnect via onclose
      }
    }, delay)
  }

  private async resubscribeAll(): Promise<void> {
    // Re-subscribe to channels that were active before disconnect
    const prevSubs = [...this.subscriptions.values()]
    this.subscriptions.clear()

    // Group by channel
    const channelMap = new Map<WSChannel, string[]>()
    for (const sub of prevSubs) {
      // We don't track market_tickers per subscription here,
      // so callers should re-subscribe with their own state.
      if (!channelMap.has(sub.channel)) {
        channelMap.set(sub.channel, [])
      }
    }

    // Re-subscribe channel by channel
    for (const [channel] of channelMap) {
      try {
        await this.doSubscribe({ channels: [channel] })
      } catch {
        // Best-effort
      }
    }
  }

  private async processResubscribeQueue(): Promise<void> {
    const queue = this.resubscribeQueue.splice(0)
    for (const { params, resolve } of queue) {
      try {
        const sids = await this.doSubscribe(params)
        resolve(sids)
      } catch {
        resolve([])
      }
    }
  }

  private async doSubscribe(params: SubscribeParams): Promise<number[]> {
    const id = this.nextCmdId++
    this.send({
      id,
      cmd: "subscribe",
      params,
    })

    const resp = await this.waitForCommand(id, 10000)
    // Response may be a "subscribed" message with sid
    const sids: number[] = []
    if (resp.msg?.sid !== undefined) {
      const sid = resp.msg.sid as number
      const channel = (resp.msg.channel as WSChannel) ?? params.channels[0]
      sids.push(sid)
      this.subscriptions.set(sid, { channel, sid })
    }
    return sids
  }

  private send(data: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.ws.send(JSON.stringify(data))
  }

  private waitForCommand(id: number, timeoutMs: number): Promise<WSMessage> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingCommands.delete(id)
        reject(new Error(`Command ${id} timed out`))
      }, timeoutMs)

      this.pendingCommands.set(id, {
        resolve: (msg) => {
          clearTimeout(timer)
          this.pendingCommands.delete(id)
          resolve(msg)
        },
        reject: (err) => {
          clearTimeout(timer)
          this.pendingCommands.delete(id)
          reject(err)
        },
      })
    })
  }

  private handleMessage(raw: string): void {
    let msg: WSMessage
    try {
      msg = JSON.parse(raw) as WSMessage
    } catch {
      return
    }

    // Route command responses (subscribed, unsubscribed, ok, error with id)
    if (msg.id !== undefined && msg.id > 0) {
      const pending = this.pendingCommands.get(msg.id)
      if (pending) {
        if (msg.type === "error") {
          pending.reject(
            new Error(`WS error: ${(msg.msg as any)?.msg ?? "unknown"}`),
          )
        } else {
          pending.resolve(msg)
        }
        // Don't return -- also dispatch to type handlers
      }
    }

    // Also resolve for "subscribed" messages that may use a different
    // correlation pattern (sid-based rather than id-based)
    if (msg.type === "subscribed" && msg.id !== undefined && msg.id > 0) {
      const pending = this.pendingCommands.get(msg.id)
      if (pending) {
        pending.resolve(msg)
      }
    }

    // Dispatch to type handlers
    const type = msg.type
    if (type && msg.msg) {
      const handlers = this.handlers.get(type)
      if (handlers) {
        for (const handler of handlers) {
          try {
            handler(msg.msg as Record<string, unknown>)
          } catch {
            // Don't let handler errors break the WS
          }
        }
      }
    }
  }
}
