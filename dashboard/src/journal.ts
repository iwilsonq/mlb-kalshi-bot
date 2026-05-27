/**
 * JSONL file reader for Slugger's journal and signal logs.
 *
 * Reads append-only JSONL files with support for:
 *   - Full load (read entire file, parse all lines)
 *   - Incremental tail (track byte offset, only parse new lines)
 *   - Graceful handling of partial lines at EOF
 *   - Periodic polling via setInterval
 *
 * The Python bot writes one JSON object per line using open/append/close,
 * so concurrent reads are safe (no file locks needed).
 */

import { readFile, stat } from "node:fs/promises"
import type {
  JournalRecord,
  TradeRecord,
  SettlementRecord,
  SignalRecord,
  StrategyStats,
  PlacedLedger,
} from "./types.js"

// ── JSONL Parsing ────────────────────────────────────────────────────────────

/**
 * Parse a JSONL string into typed records, skipping malformed lines.
 */
function parseJsonl<T>(content: string): T[] {
  const records: T[] = []
  for (const line of content.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      records.push(JSON.parse(trimmed) as T)
    } catch {
      // Skip malformed lines (partial write at EOF)
    }
  }
  return records
}

// ── File reader with tail support ────────────────────────────────────────────

export interface TailState {
  /** Byte offset of last successful read */
  offset: number
  /** File size at last read */
  size: number
}

/**
 * Read new lines from a file since the last read.
 * Returns the parsed records and updated tail state.
 */
async function readTail<T>(
  filePath: string,
  prevState: TailState,
): Promise<{ records: T[]; state: TailState }> {
  let fileSize: number
  try {
    const s = await stat(filePath)
    fileSize = s.size
  } catch {
    // File doesn't exist yet
    return { records: [], state: prevState }
  }

  if (fileSize <= prevState.offset) {
    // No new data (or file was truncated/replaced — reset)
    if (fileSize < prevState.offset) {
      return { records: [], state: { offset: 0, size: fileSize } }
    }
    return { records: [], state: prevState }
  }

  // Read only the new bytes
  const file = Bun.file(filePath)
  const blob = file.slice(prevState.offset, fileSize)
  const newContent = await blob.text()

  const records = parseJsonl<T>(newContent)
  return {
    records,
    state: { offset: fileSize, size: fileSize },
  }
}

/**
 * Read an entire file and parse all JSONL records.
 */
async function readAll<T>(filePath: string): Promise<T[]> {
  try {
    const content = await readFile(filePath, "utf-8")
    return parseJsonl<T>(content)
  } catch {
    return []
  }
}

// ── Journal Reader ───────────────────────────────────────────────────────────

export class JournalReader {
  private journalPath: string
  private signalsPath: string
  private logsDir: string

  private journalState: TailState = { offset: 0, size: 0 }
  private signalsState: TailState = { offset: 0, size: 0 }

  /** All journal records accumulated so far */
  trades: TradeRecord[] = []
  settlements: SettlementRecord[] = []
  signals: SignalRecord[] = []

  /** Callback for new data */
  onUpdate?: () => void

  private pollTimer?: ReturnType<typeof setInterval>

  constructor(logsDir: string) {
    this.logsDir = logsDir
    this.journalPath = `${logsDir}/journal.jsonl`
    this.signalsPath = `${logsDir}/signals.jsonl`
  }

  /**
   * Full initial load of all records.
   */
  async load(): Promise<void> {
    const journalRecords = await readAll<JournalRecord>(this.journalPath)
    for (const rec of journalRecords) {
      if (rec.type === "trade") {
        this.trades.push(rec as TradeRecord)
      } else if (rec.type === "settlement") {
        this.settlements.push(rec as SettlementRecord)
      }
    }

    this.signals = await readAll<SignalRecord>(this.signalsPath)

    // Set tail state to end of files so next poll only gets new data
    try {
      const js = await stat(this.journalPath)
      this.journalState = { offset: js.size, size: js.size }
    } catch {
      /* file may not exist */
    }
    try {
      const ss = await stat(this.signalsPath)
      this.signalsState = { offset: ss.size, size: ss.size }
    } catch {
      /* file may not exist */
    }
  }

  /**
   * Poll for new records appended since last read.
   * Returns true if any new records were found.
   */
  async poll(): Promise<boolean> {
    let changed = false

    const journalResult = await readTail<JournalRecord>(
      this.journalPath,
      this.journalState,
    )
    if (journalResult.records.length > 0) {
      for (const rec of journalResult.records) {
        if (rec.type === "trade") {
          this.trades.push(rec as TradeRecord)
        } else if (rec.type === "settlement") {
          this.settlements.push(rec as SettlementRecord)
        }
      }
      changed = true
    }
    this.journalState = journalResult.state

    const signalsResult = await readTail<SignalRecord>(
      this.signalsPath,
      this.signalsState,
    )
    if (signalsResult.records.length > 0) {
      this.signals.push(...signalsResult.records)
      changed = true
    }
    this.signalsState = signalsResult.state

    if (changed && this.onUpdate) {
      this.onUpdate()
    }

    return changed
  }

  /**
   * Start polling at the given interval (milliseconds).
   */
  startPolling(intervalMs: number = 5000): void {
    this.stopPolling()
    this.pollTimer = setInterval(() => this.poll(), intervalMs)
  }

  /**
   * Stop polling.
   */
  stopPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer)
      this.pollTimer = undefined
    }
  }

  /**
   * Load today's placed ledger (dedup file).
   */
  async loadPlacedLedger(date?: string): Promise<PlacedLedger> {
    const d = date ?? new Date().toISOString().slice(0, 10)
    const path = `${this.logsDir}/placed_${d}.json`
    try {
      const content = await readFile(path, "utf-8")
      return JSON.parse(content) as PlacedLedger
    } catch {
      return []
    }
  }

  // ── Computed stats ───────────────────────────────────────────────────

  /**
   * Get today's trades.
   */
  todaysTrades(): TradeRecord[] {
    const today = new Date().toISOString().slice(0, 10)
    return this.trades.filter((t) => t.date === today)
  }

  /**
   * Get today's signals.
   */
  todaysSignals(): SignalRecord[] {
    const today = new Date().toISOString().slice(0, 10)
    return this.signals.filter((s) => s.date === today)
  }

  /**
   * Compute per-strategy stats from journal records.
   * Mirrors slugger/journal.py get_stats().
   */
  computeStats(dateFilter?: string): {
    overall: StrategyStats
    perStrategy: Map<string, StrategyStats>
  } {
    const trades = dateFilter
      ? this.trades.filter((t) => t.date === dateFilter)
      : this.trades

    // Build settlement lookup: ticker -> SettlementRecord
    const settlementMap = new Map<string, SettlementRecord>()
    for (const s of this.settlements) {
      settlementMap.set(s.ticker, s)
    }

    const perStrategy = new Map<string, StrategyStats>()
    const overall = makeEmptyStats("overall")

    for (const trade of trades) {
      const stratName = trade.strategy || "unknown"
      if (!perStrategy.has(stratName)) {
        perStrategy.set(stratName, makeEmptyStats(stratName))
      }
      const s = perStrategy.get(stratName)!

      s.bets += 1
      s.totalCostUsd += trade.cost_usd
      overall.bets += 1
      overall.totalCostUsd += trade.cost_usd

      const settlement = settlementMap.get(trade.ticker)
      if (settlement) {
        s.settled += 1
        s.totalRevenueUsd += settlement.revenue_usd
        s.totalFeeUsd += settlement.fee_usd
        s.totalPnlUsd += settlement.pnl_usd
        overall.settled += 1
        overall.totalRevenueUsd += settlement.revenue_usd
        overall.totalFeeUsd += settlement.fee_usd
        overall.totalPnlUsd += settlement.pnl_usd

        if (settlement.market_result === "yes") {
          s.wins += 1
          overall.wins += 1
        } else if (settlement.market_result === "void") {
          s.voids += 1
          overall.voids += 1
        }
      }
    }

    // Compute derived fields
    for (const s of [overall, ...perStrategy.values()]) {
      s.losses = s.settled - s.wins - s.voids
      s.pending = s.bets - s.settled
      const decided = s.settled - s.voids
      s.winRate = decided > 0 ? s.wins / decided : null
      s.roiPct = s.totalCostUsd > 0 ? (s.totalPnlUsd / s.totalCostUsd) * 100 : null
    }

    return { overall, perStrategy }
  }
}

function makeEmptyStats(strategy: string): StrategyStats {
  return {
    strategy,
    bets: 0,
    settled: 0,
    wins: 0,
    voids: 0,
    totalCostUsd: 0,
    totalRevenueUsd: 0,
    totalFeeUsd: 0,
    totalPnlUsd: 0,
    losses: 0,
    winRate: null,
    roiPct: null,
    pending: 0,
  }
}
