import { createCliRenderer, Box, Text, t, bold, fg, dim, type KeyEvent } from "@opentui/core"
import { loadConfig } from "./config.js"
import { JournalReader } from "./journal.js"
import { KalshiClient, type KalshiPosition } from "./kalshi.js"
import type { TradeRecord, SignalRecord, StrategyStats } from "./types.js"

// ── Resolve paths and load config ────────────────────────────────────────
const REPO_ROOT = new URL("../../", import.meta.url).pathname
const config = loadConfig(REPO_ROOT)

// ── Initialize data sources ──────────────────────────────────────────────
const journal = new JournalReader(config.logsDir)
await journal.load()

let kalshiClient: KalshiClient | null = null
let balance = 0
let positions: KalshiPosition[] = []
let apiConnected = false

try {
  kalshiClient = new KalshiClient(config)
  apiConnected = await kalshiClient.healthCheck()
  if (apiConnected) {
    balance = await kalshiClient.getBalance()
    positions = await kalshiClient.getPositions()
  }
} catch {
  // API unavailable — dashboard still works with journal data
}

// ── Bootstrap renderer ───────────────────────────────────────────────────
const renderer = await createCliRenderer({
  exitOnCtrlC: true,
  targetFps: 10,
})

// ── Color palette ────────────────────────────────────────────────────────
const c = {
  bg: "#0D1117",
  bgPanel: "#161B22",
  border: "#30363D",
  borderActive: "#58A6FF",
  text: "#C9D1D9",
  muted: "#8B949E",
  green: "#3FB950",
  red: "#F85149",
  yellow: "#D29922",
  blue: "#58A6FF",
  purple: "#BC8CFF",
}

// ── Helpers ──────────────────────────────────────────────────────────────
function pnlColor(val: number): string {
  return val > 0 ? c.green : val < 0 ? c.red : c.muted
}

function fmtUsd(val: number): string {
  const sign = val >= 0 ? "+" : ""
  return `${sign}$${val.toFixed(2)}`
}

function fmtPct(val: number | null): string {
  if (val === null) return "--"
  return `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" })
  } catch {
    return "--:--"
  }
}

function truncate(s: string, maxLen: number): string {
  return s.length <= maxLen ? s : s.slice(0, maxLen - 1) + "\u2026"
}

// ── Header ───────────────────────────────────────────────────────────────
function Header() {
  const now = new Date()
  const time = now.toLocaleTimeString("en-US", { hour12: false })
  const date = now.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  })
  const statusDot = apiConnected ? fg(c.green)("\u25CF") : fg(c.red)("\u25CF")
  const balStr = apiConnected ? fg(c.green)(`$${balance.toFixed(2)}`) : fg(c.muted)("--")

  return Box(
    {
      width: "100%",
      height: 3,
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      paddingX: 2,
      backgroundColor: c.bgPanel,
      borderStyle: "single",
      borderColor: c.border,
    },
    Text({
      content: t`${bold(fg(c.blue)("Slugger"))} ${fg(c.muted)("v0.2.0")}  ${statusDot} ${balStr}`,
    }),
    Text({
      content: t`${fg(c.muted)(`${date}  ${time}`)}`,
    }),
  )
}

// ── Portfolio panel ──────────────────────────────────────────────────────
function PortfolioPanel() {
  const { overall } = journal.computeStats()
  const todayStats = journal.computeStats(new Date().toISOString().slice(0, 10))

  const lines: string[] = []

  if (apiConnected) {
    lines.push(`Balance:    $${balance.toFixed(2)}`)
    lines.push(`Positions:  ${positions.length} open`)
  } else {
    lines.push("API: disconnected")
  }
  lines.push("")
  lines.push(`Today:  ${todayStats.overall.bets} trades  P&L ${fmtUsd(todayStats.overall.totalPnlUsd)}`)
  lines.push(`All:    ${overall.bets} trades  ${overall.settled} settled  ${overall.pending} pending`)
  lines.push(`Win:    ${overall.winRate !== null ? (overall.winRate * 100).toFixed(1) + "%" : "--"}  ROI: ${fmtPct(overall.roiPct)}  P&L: ${fmtUsd(overall.totalPnlUsd)}`)

  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: c.border,
      title: " Portfolio ",
      padding: 1,
      backgroundColor: c.bgPanel,
      flexDirection: "column",
    },
    ...lines.map((line) =>
      Text({ content: t`${fg(c.text)(line)}` }),
    ),
  )
}

// ── Positions panel ──────────────────────────────────────────────────────
function PositionsPanel() {
  if (!apiConnected || positions.length === 0) {
    const msg = apiConnected ? "No open positions" : "API disconnected"
    return Box(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Open Positions ",
        padding: 1,
        backgroundColor: c.bgPanel,
      },
      Text({ content: t`${fg(c.muted)(msg)}` }),
    )
  }

  const rows = positions.slice(0, 15).map((pos) => {
    const ticker = truncate(pos.ticker, 40)
    const qty = pos.position ?? 0
    const side = qty > 0 ? "YES" : qty < 0 ? "NO" : "---"
    const sideColor = qty > 0 ? c.green : qty < 0 ? c.red : c.muted
    return Text({
      content: t`${fg(sideColor)(side.padEnd(4))} ${fg(c.text)(String(Math.abs(qty)).padStart(3))} ${fg(c.muted)(ticker)}`,
    })
  })

  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: c.border,
      title: ` Open Positions (${positions.length}) `,
      padding: 1,
      backgroundColor: c.bgPanel,
      flexDirection: "column",
    },
    Text({
      content: t`${fg(c.muted)("SIDE QTY TICKER")}`,
    }),
    ...rows,
  )
}

// ── Today's trades panel ─────────────────────────────────────────────────
function TodaysTradesPanel() {
  const trades = journal.todaysTrades()

  if (trades.length === 0) {
    return Box(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Today's Trades ",
        padding: 1,
        backgroundColor: c.bgPanel,
      },
      Text({ content: t`${fg(c.muted)("No trades today")}` }),
    )
  }

  // Show most recent first, limit to fit
  const recent = [...trades].reverse().slice(0, 15)
  const rows = recent.map((tr) => {
    const time = fmtTime(tr.placed_at)
    const side = tr.side.toUpperCase().padEnd(3)
    const sideColor = tr.side === "yes" ? c.green : c.red
    const strat = truncate(tr.strategy, 12).padEnd(12)
    const price = `${tr.price_cents}\u00A2`.padStart(4)
    const edge = `${tr.edge_cents > 0 ? "+" : ""}${tr.edge_cents.toFixed(0)}\u00A2`
    const ticker = truncate(tr.ticker, 30)
    return Text({
      content: t`${fg(c.muted)(time)} ${fg(sideColor)(side)} ${fg(c.purple)(strat)} ${fg(c.text)(price)} ${fg(c.yellow)(edge.padStart(5))} ${fg(c.muted)(ticker)}`,
    })
  })

  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: c.border,
      title: ` Today's Trades (${trades.length}) `,
      padding: 1,
      backgroundColor: c.bgPanel,
      flexDirection: "column",
    },
    Text({
      content: t`${fg(c.muted)("TIME  SIDE STRATEGY     PRICE  EDGE TICKER")}`,
    }),
    ...rows,
  )
}

// ── Signal feed panel ────────────────────────────────────────────────────
function SignalFeedPanel() {
  const signals = journal.todaysSignals()

  if (signals.length === 0) {
    return Box(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Signal Feed ",
        padding: 1,
        backgroundColor: c.bgPanel,
      },
      Text({ content: t`${fg(c.muted)("No signals today")}` }),
    )
  }

  // Most recent first, limit display
  const recent = [...signals].reverse().slice(0, 15)
  const rows = recent.map((sig) => {
    const time = fmtTime(sig.timestamp)
    const traded = sig.traded ? fg(c.green)("\u2713") : fg(c.muted)("\u00B7")
    const strat = truncate(sig.strategy, 12).padEnd(12)
    const prob = `${sig.model_prob_pct}%`.padStart(4)
    const mkt = `${sig.market_price_cents}\u00A2`.padStart(4)
    const edge = `${sig.edge_cents > 0 ? "+" : ""}${sig.edge_cents.toFixed(0)}\u00A2`
    const edgeColor = sig.edge_cents > 0 ? c.green : c.red
    return Text({
      content: t`${fg(c.muted)(time)} ${traded} ${fg(c.purple)(strat)} ${fg(c.text)(prob)} ${fg(c.muted)(mkt)} ${fg(edgeColor)(edge.padStart(5))}`,
    })
  })

  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: c.border,
      title: ` Signals (${signals.length} today, ${signals.filter((s) => s.traded).length} traded) `,
      padding: 1,
      backgroundColor: c.bgPanel,
      flexDirection: "column",
    },
    Text({
      content: t`${fg(c.muted)("TIME  T  STRATEGY     MODEL  MKT   EDGE")}`,
    }),
    ...rows,
  )
}

// ── Footer ───────────────────────────────────────────────────────────────
function Footer() {
  return Box(
    {
      width: "100%",
      height: 1,
      flexDirection: "row",
      justifyContent: "center",
      gap: 3,
      backgroundColor: c.bgPanel,
    },
    Text({
      content: t`${fg(c.blue)("r")} ${fg(c.muted)("refresh")}  ${fg(c.blue)("q")} ${fg(c.muted)("quit")}`,
    }),
  )
}

// ── Main layout ──────────────────────────────────────────────────────────
function render() {
  for (const child of renderer.root.getChildren()) {
    child.destroy()
  }

  renderer.root.add(
    Box(
      {
        width: "100%",
        height: "100%",
        flexDirection: "column",
        backgroundColor: c.bg,
      },
      Header(),
      Box(
        {
          flexGrow: 1,
          flexDirection: "row",
          gap: 1,
          padding: 1,
        },
        // Left column
        Box(
          { flexGrow: 1, flexDirection: "column", gap: 1 },
          PortfolioPanel(),
          PositionsPanel(),
        ),
        // Right column
        Box(
          { flexGrow: 1, flexDirection: "column", gap: 1 },
          TodaysTradesPanel(),
          SignalFeedPanel(),
        ),
      ),
      Footer(),
    ),
  )
}

// ── Refresh data and re-render ───────────────────────────────────────────
async function refresh() {
  await journal.poll()

  if (kalshiClient) {
    try {
      balance = await kalshiClient.getBalance()
      positions = await kalshiClient.getPositions()
      apiConnected = true
    } catch {
      apiConnected = false
    }
  }

  render()
}

// ── Keyboard handling ────────────────────────────────────────────────────
renderer.keyInput.on("keypress", (key: KeyEvent) => {
  if (key.name === "q" && !key.ctrl && !key.meta) {
    journal.stopPolling()
    renderer.destroy()
  }
  if (key.name === "r" && !key.ctrl && !key.meta) {
    refresh()
  }
})

// ── Auto-refresh on interval ─────────────────────────────────────────────
journal.onUpdate = () => render()
journal.startPolling(5000)

// Refresh API data every 15 seconds
setInterval(() => refresh(), 15000)

// ── Initial render ───────────────────────────────────────────────────────
render()
