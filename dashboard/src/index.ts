import {
  createCliRenderer,
  Box,
  Text,
  ScrollBox,
  t,
  bold,
  fg,
  type KeyEvent,
} from "@opentui/core"
import { loadConfig } from "./config.js"
import { JournalReader } from "./journal.js"
import { KalshiClient, type KalshiPosition } from "./kalshi.js"
import {
  enrichPositions,
  computeCircuitBreaker,
  type PortfolioSummary,
  type CircuitBreakerState,
} from "./positions.js"
import { BotManager } from "./bot.js"
import type { StrategyStats } from "./types.js"

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
let portfolio: PortfolioSummary = {
  totalExposureUsd: 0,
  totalUnrealizedPnl: 0,
  totalMarketValueUsd: 0,
  positions: [],
}
let circuitBreaker: CircuitBreakerState = computeCircuitBreaker(
  journal,
  config.cbMaxConsecutiveLosses,
  config.cbMaxLossUsd,
)

try {
  kalshiClient = new KalshiClient(config)
  apiConnected = await kalshiClient.healthCheck()
  if (apiConnected) {
    balance = await kalshiClient.getBalance()
    positions = await kalshiClient.getPositions()
    portfolio = await enrichPositions(positions, kalshiClient, journal)
  }
} catch {
  // API unavailable — dashboard still works with journal data
}

// ── Bot manager ──────────────────────────────────────────────────────────
const bot = new BotManager(config)

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
  cyan: "#56D4DD",
}

// ── Tab state ────────────────────────────────────────────────────────────
const TABS = ["Portfolio", "Trades", "Signals", "Stats", "Bot Log"] as const
type TabName = (typeof TABS)[number]
let activeTab: TabName = "Portfolio"

// ── Helpers ──────────────────────────────────────────────────────────────
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
    return d.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
    })
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
  const apiDot = apiConnected ? fg(c.green)("\u25CF") : fg(c.red)("\u25CF")
  const balStr = apiConnected
    ? fg(c.green)(`$${balance.toFixed(2)}`)
    : fg(c.muted)("--")

  // Bot status
  let botText: string
  let botColor: string
  switch (bot.status) {
    case "running":
      botText = `BOT \u25B6 ${bot.uptime}`
      botColor = c.green
      break
    case "stopping":
      botText = "BOT \u23F8"
      botColor = c.yellow
      break
    case "crashed":
      botText = `BOT \u2717 exit=${bot.exitCode ?? "?"}`
      botColor = c.red
      break
    default:
      botText = "BOT \u25A0"
      botColor = c.muted
  }

  // Mode badge
  const modeText = config.dryRun ? "[DRY RUN]" : "[LIVE]"
  const modeColor = config.dryRun ? c.yellow : c.red

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
      content: t`${bold(fg(c.blue)("Slugger"))} ${fg(c.muted)("v0.2.0")}  ${apiDot} ${balStr}  ${fg(botColor)(botText)}  ${fg(modeColor)(modeText)}`,
    }),
    Text({
      content: t`${fg(c.muted)(`${date}  ${time}`)}`,
    }),
  )
}

// ── Tab bar ──────────────────────────────────────────────────────────────
function TabBar() {
  const children = TABS.flatMap((name, i) => {
    const num = `${i + 1}`
    const isActive = name === activeTab
    const items: ReturnType<typeof Text>[] = []

    if (i > 0) {
      items.push(Text({ content: t`${fg(c.border)("  \u2502  ")}` }))
    }

    if (isActive) {
      items.push(
        Text({
          content: t`${bold(fg(c.blue)(`[${num}]`))} ${bold(fg(c.text)(name))}`,
        }),
      )
    } else {
      items.push(
        Text({
          content: t`${fg(c.muted)(`[${num}]`)} ${fg(c.muted)(name)}`,
        }),
      )
    }

    return items
  })

  return Box(
    {
      width: "100%",
      height: 1,
      flexDirection: "row",
      paddingX: 2,
      backgroundColor: c.bg,
      alignItems: "center",
    },
    ...children,
  )
}

// ── Portfolio view (2-column overview) ───────────────────────────────────
function PortfolioPanel() {
  const { overall } = journal.computeStats()
  const todayStats = journal.computeStats(new Date().toISOString().slice(0, 10))
  const cb = circuitBreaker

  const children: ReturnType<typeof Text>[] = []

  if (apiConnected) {
    children.push(
      Text({ content: t`${fg(c.muted)("Balance:")}    ${fg(c.green)(`$${balance.toFixed(2)}`)}` }),
    )
    if (portfolio.positions.length > 0) {
      children.push(
        Text({
          content: t`${fg(c.muted)("Positions:")}  ${fg(c.text)(`${portfolio.positions.length} open`)}  ${fg(c.muted)("Exposure:")} ${fg(c.yellow)(`$${portfolio.totalExposureUsd.toFixed(2)}`)}`,
        }),
      )
      const upnlColor = portfolio.totalUnrealizedPnl >= 0 ? c.green : c.red
      children.push(
        Text({
          content: t`${fg(c.muted)("Mkt Value:")}  ${fg(c.text)(`$${portfolio.totalMarketValueUsd.toFixed(2)}`)}  ${fg(c.muted)("Unreal P&L:")} ${fg(upnlColor)(fmtUsd(portfolio.totalUnrealizedPnl))}`,
        }),
      )
    } else {
      children.push(Text({ content: t`${fg(c.muted)("Positions:")}  ${fg(c.text)("none")}` }))
    }
  } else {
    children.push(Text({ content: t`${fg(c.red)("API: disconnected")}` }))
  }

  children.push(Text({ content: "" }))

  // Circuit breaker status
  const cbIcon = cb.tripped ? fg(c.red)("\u26A0 TRIPPED") : fg(c.green)("\u2713 Armed")
  children.push(
    Text({
      content: t`${fg(c.muted)("Breaker:")}   ${cbIcon}  ${fg(c.muted)("streak:")} ${fg(cb.consecutiveLosses >= cb.maxConsecutiveLosses ? c.red : c.text)(`${cb.consecutiveLosses}/${cb.maxConsecutiveLosses}`)}  ${fg(c.muted)("loss:")} ${fg(cb.todayLossUsd >= cb.maxLossUsd ? c.red : c.text)(`$${cb.todayLossUsd.toFixed(2)}/$${cb.maxLossUsd.toFixed(2)}`)}`,
    }),
  )

  children.push(Text({ content: "" }))

  // Today + overall stats
  const todayPnlColor = todayStats.overall.totalPnlUsd >= 0 ? c.green : c.red
  children.push(
    Text({
      content: t`${fg(c.muted)("Today:")}     ${fg(c.text)(`${todayStats.overall.bets} trades`)}  ${fg(c.muted)("P&L:")} ${fg(todayPnlColor)(fmtUsd(todayStats.overall.totalPnlUsd))}`,
    }),
  )
  children.push(
    Text({
      content: t`${fg(c.muted)("All:")}       ${fg(c.text)(`${overall.bets} trades  ${overall.settled} settled  ${overall.pending} pending`)}`,
    }),
  )
  const overallPnlColor = overall.totalPnlUsd >= 0 ? c.green : c.red
  children.push(
    Text({
      content: t`${fg(c.muted)("Win:")}       ${fg(c.text)(overall.winRate !== null ? (overall.winRate * 100).toFixed(1) + "%" : "--")}  ${fg(c.muted)("ROI:")} ${fg(c.yellow)(fmtPct(overall.roiPct))}  ${fg(c.muted)("P&L:")} ${fg(overallPnlColor)(fmtUsd(overall.totalPnlUsd))}`,
    }),
  )

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
    ...children,
  )
}

function PositionsPanel() {
  if (!apiConnected || portfolio.positions.length === 0) {
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

  const rows = portfolio.positions.slice(0, 15).map((pos) => {
    const side = pos.side.toUpperCase().padEnd(3)
    const sideColor = pos.side === "yes" ? c.green : c.red
    const qty = String(pos.quantity).padStart(3)
    const entry = pos.entryCents > 0 ? `${pos.entryCents}\u00A2`.padStart(4) : "  --"
    const current = pos.currentCents > 0 ? `${pos.currentCents}\u00A2`.padStart(4) : "  --"
    const pnl = pos.unrealizedPnl !== 0 ? fmtUsd(pos.unrealizedPnl).padStart(7) : "     --"
    const pnlColor = pos.unrealizedPnl > 0 ? c.green : pos.unrealizedPnl < 0 ? c.red : c.muted
    const strat = truncate(pos.strategy, 10).padEnd(10)
    const ticker = truncate(pos.ticker, 34)
    return Text({
      content: t`${fg(sideColor)(side)} ${fg(c.text)(qty)} ${fg(c.text)(entry)} ${fg(c.cyan)(current)} ${fg(pnlColor)(pnl)} ${fg(c.purple)(strat)} ${fg(c.muted)(ticker)}`,
    })
  })

  const upnlColor = portfolio.totalUnrealizedPnl >= 0 ? c.green : c.red

  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: c.border,
      title: ` Open Positions (${portfolio.positions.length}) `,
      padding: 1,
      backgroundColor: c.bgPanel,
      flexDirection: "column",
    },
    Text({
      content: t`${fg(c.muted)("SIDE QTY  BUY  NOW    P&L  STRATEGY   TICKER")}`,
    }),
    ...rows,
    Text({ content: "" }),
    Text({
      content: t`${fg(c.muted)("Total:")} ${fg(c.yellow)(`$${portfolio.totalExposureUsd.toFixed(2)} exposed`)}  ${fg(upnlColor)(fmtUsd(portfolio.totalUnrealizedPnl) + " unrealized")}`,
    }),
  )
}

function TodaysTradesMini() {
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

  const recent = [...trades].reverse().slice(0, 10)
  const rows = recent.map((tr) => {
    const time = fmtTime(tr.placed_at)
    const side = tr.side.toUpperCase().padEnd(3)
    const sideColor = tr.side === "yes" ? c.green : c.red
    const strat = truncate(tr.strategy, 12).padEnd(12)
    const price = `${tr.price_cents}\u00A2`.padStart(4)
    const edge = `${tr.edge_cents > 0 ? "+" : ""}${tr.edge_cents.toFixed(0)}\u00A2`
    return Text({
      content: t`${fg(c.muted)(time)} ${fg(sideColor)(side)} ${fg(c.purple)(strat)} ${fg(c.text)(price)} ${fg(c.yellow)(edge.padStart(5))}`,
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
      content: t`${fg(c.muted)("TIME  SIDE STRATEGY     PRICE  EDGE")}`,
    }),
    ...rows,
  )
}

function SignalFeedMini() {
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

  const recent = [...signals].reverse().slice(0, 10)
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

function PortfolioView() {
  return Box(
    {
      flexGrow: 1,
      flexDirection: "row",
      gap: 1,
      padding: 1,
    },
    Box(
      { flexGrow: 1, flexDirection: "column", gap: 1 },
      PortfolioPanel(),
      PositionsPanel(),
    ),
    Box(
      { flexGrow: 1, flexDirection: "column", gap: 1 },
      TodaysTradesMini(),
      SignalFeedMini(),
    ),
  )
}

// ── Trades view (full-width scrollable) ──────────────────────────────────
function TradesView() {
  const trades = journal.todaysTrades()

  if (trades.length === 0) {
    return Box(
      {
        flexGrow: 1,
        padding: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Today's Trades ",
        backgroundColor: c.bgPanel,
        margin: 1,
      },
      Text({ content: t`${fg(c.muted)("No trades today")}` }),
    )
  }

  const recent = [...trades].reverse()
  const rows = recent.map((tr) => {
    const time = fmtTime(tr.placed_at)
    const side = tr.side.toUpperCase().padEnd(3)
    const sideColor = tr.side === "yes" ? c.green : c.red
    const strat = truncate(tr.strategy, 14).padEnd(14)
    const cnt = String(tr.count).padStart(3)
    const price = `${tr.price_cents}\u00A2`.padStart(4)
    const edge = `${tr.edge_cents > 0 ? "+" : ""}${tr.edge_cents.toFixed(0)}\u00A2`
    const cost = `$${tr.cost_usd.toFixed(2)}`.padStart(7)
    const ticker = truncate(tr.ticker, 44)
    return Text({
      content: t`${fg(c.muted)(time)} ${fg(sideColor)(side)} ${fg(c.purple)(strat)} ${fg(c.text)(cnt)} ${fg(c.text)(price)} ${fg(c.yellow)(edge.padStart(5))} ${fg(c.cyan)(cost)} ${fg(c.muted)(ticker)}`,
    })
  })

  return Box(
    {
      flexGrow: 1,
      flexDirection: "column",
      margin: 1,
    },
    Box(
      {
        width: "100%",
        height: 1,
        paddingX: 1,
      },
      Text({
        content: t`${fg(c.muted)(`Today's Trades (${trades.length})`)}  ${fg(c.muted)("TIME  SIDE STRATEGY       QTY PRICE  EDGE    COST TICKER")}`,
      }),
    ),
    ScrollBox(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        backgroundColor: c.bgPanel,
        stickyScroll: true,
        stickyStart: "top",
        viewportCulling: true,
      },
      ...rows,
    ),
  )
}

// ── Signals view (full-width scrollable) ─────────────────────────────────
function SignalsView() {
  const signals = journal.todaysSignals()

  if (signals.length === 0) {
    return Box(
      {
        flexGrow: 1,
        padding: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Signal Feed ",
        backgroundColor: c.bgPanel,
        margin: 1,
      },
      Text({ content: t`${fg(c.muted)("No signals today")}` }),
    )
  }

  const tradedCount = signals.filter((s) => s.traded).length
  const recent = [...signals].reverse()
  const rows = recent.map((sig) => {
    const time = fmtTime(sig.timestamp)
    const traded = sig.traded ? fg(c.green)("\u2713") : fg(c.muted)("\u00B7")
    const strat = truncate(sig.strategy, 14).padEnd(14)
    const prob = `${sig.model_prob_pct}%`.padStart(4)
    const mkt = `${sig.market_price_cents}\u00A2`.padStart(4)
    const edge = `${sig.edge_cents > 0 ? "+" : ""}${sig.edge_cents.toFixed(0)}\u00A2`
    const edgeColor = sig.edge_cents > 0 ? c.green : c.red
    const reason = truncate(sig.reason, 50)
    return Text({
      content: t`${fg(c.muted)(time)} ${traded} ${fg(c.purple)(strat)} ${fg(c.text)(prob)} ${fg(c.muted)(mkt)} ${fg(edgeColor)(edge.padStart(5))}  ${fg(c.muted)(reason)}`,
    })
  })

  return Box(
    {
      flexGrow: 1,
      flexDirection: "column",
      margin: 1,
    },
    Box(
      {
        width: "100%",
        height: 1,
        paddingX: 1,
      },
      Text({
        content: t`${fg(c.muted)(`Signals (${signals.length} today, ${tradedCount} traded)`)}  ${fg(c.muted)("TIME  T  STRATEGY       MODEL  MKT   EDGE  REASON")}`,
      }),
    ),
    ScrollBox(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        backgroundColor: c.bgPanel,
        stickyScroll: true,
        stickyStart: "top",
        viewportCulling: true,
      },
      ...rows,
    ),
  )
}

// ── Stats view (per-strategy breakdown) ──────────────────────────────────
function StatsView() {
  const { overall, perStrategy } = journal.computeStats()
  const todayStats = journal.computeStats(new Date().toISOString().slice(0, 10))

  function statsRow(s: StrategyStats, label?: string): ReturnType<typeof Text> {
    const name = (label ?? s.strategy).padEnd(16)
    const bets = String(s.bets).padStart(5)
    const settled = String(s.settled).padStart(5)
    const wins = String(s.wins).padStart(4)
    const losses = String(s.losses).padStart(4)
    const wr =
      s.winRate !== null ? `${(s.winRate * 100).toFixed(1)}%`.padStart(6) : "   -- "
    const roi = fmtPct(s.roiPct).padStart(8)
    const pnl = fmtUsd(s.totalPnlUsd).padStart(9)
    const pnlC = s.totalPnlUsd > 0 ? c.green : s.totalPnlUsd < 0 ? c.red : c.muted
    const pending = s.pending > 0 ? ` (${s.pending}p)` : ""

    return Text({
      content: t`${fg(c.text)(name)} ${fg(c.muted)(bets)} ${fg(c.muted)(settled)} ${fg(c.green)(wins)} ${fg(c.red)(losses)} ${fg(c.text)(wr)} ${fg(c.yellow)(roi)} ${fg(pnlC)(pnl)}${fg(c.muted)(pending)}`,
    })
  }

  const header = Text({
    content: t`${fg(c.muted)("STRATEGY         BETS  STTL WINS LOSS    WR      ROI       P&L")}`,
  })

  const separator = Text({
    content: t`${fg(c.border)("\u2500".repeat(72))}`,
  })

  const strategyRows = [...perStrategy.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, s]) => statsRow(s))

  // Today's summary
  const todayHeader = Text({
    content: t`\n${bold(fg(c.blue)("Today"))}`,
  })

  const todayStratRows = [...todayStats.perStrategy.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, s]) => statsRow(s))

  return Box(
    {
      flexGrow: 1,
      flexDirection: "column",
      margin: 1,
    },
    Box(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Strategy Stats ",
        padding: 1,
        backgroundColor: c.bgPanel,
        flexDirection: "column",
      },
      Text({ content: t`${bold(fg(c.blue)("All Time"))}` }),
      Text({ content: "" }),
      header,
      separator,
      statsRow(overall, "OVERALL"),
      separator,
      ...strategyRows,
      todayHeader,
      Text({ content: "" }),
      Text({
        content: t`${fg(c.muted)("STRATEGY         BETS  STTL WINS LOSS    WR      ROI       P&L")}`,
      }),
      Text({
        content: t`${fg(c.border)("\u2500".repeat(72))}`,
      }),
      statsRow(todayStats.overall, "TODAY TOTAL"),
      Text({
        content: t`${fg(c.border)("\u2500".repeat(72))}`,
      }),
      ...todayStratRows,
    ),
  )
}

// ── Bot Log view ─────────────────────────────────────────────────────────
function BotLogView() {
  const lines = bot.logLines

  if (lines.length === 0) {
    const hint =
      bot.status === "stopped"
        ? 'Press "s" to start the bot'
        : "Waiting for output..."
    return Box(
      {
        flexGrow: 1,
        padding: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        title: " Bot Log ",
        backgroundColor: c.bgPanel,
        margin: 1,
      },
      Text({ content: t`${fg(c.muted)(hint)}` }),
    )
  }

  const rows = lines.map((line) => {
    const time = line.timestamp.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
    const streamColor = line.stream === "stderr" ? c.red : c.muted
    const textColor = line.stream === "stderr" ? c.red : c.text
    return Text({
      content: t`${fg(streamColor)(time)} ${fg(textColor)(line.text)}`,
    })
  })

  // Status bar at top
  let statusLine: string
  switch (bot.status) {
    case "running":
      statusLine = `\u25B6 Running (PID ${bot.pid}, uptime ${bot.uptime})`
      break
    case "stopping":
      statusLine = "\u23F8 Stopping..."
      break
    case "crashed":
      statusLine = `\u2717 Crashed (exit=${bot.exitCode ?? "?"})`
      break
    default:
      statusLine = "\u25A0 Stopped"
  }
  const statusColor =
    bot.status === "running"
      ? c.green
      : bot.status === "crashed"
        ? c.red
        : c.muted

  return Box(
    {
      flexGrow: 1,
      flexDirection: "column",
      margin: 1,
    },
    Box(
      {
        width: "100%",
        height: 1,
        paddingX: 1,
        flexDirection: "row",
        justifyContent: "space-between",
      },
      Text({ content: t`${fg(statusColor)(statusLine)}` }),
      Text({
        content: t`${fg(c.muted)(`${lines.length} lines`)}  ${fg(c.blue)("s")} ${fg(c.muted)(bot.status === "running" ? "stop" : "start")}`,
      }),
    ),
    ScrollBox(
      {
        flexGrow: 1,
        borderStyle: "rounded",
        borderColor: c.border,
        backgroundColor: c.bgPanel,
        stickyScroll: true,
        stickyStart: "bottom",
        viewportCulling: true,
      },
      ...rows,
    ),
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
      gap: 2,
      backgroundColor: c.bgPanel,
    },
    Text({
      content: t`${fg(c.blue)("1-5")} ${fg(c.muted)("tabs")}  ${fg(c.blue)("tab")} ${fg(c.muted)("next")}  ${fg(c.blue)("s")} ${fg(c.muted)("bot")}  ${fg(c.blue)("r")} ${fg(c.muted)("refresh")}  ${fg(c.blue)("q")} ${fg(c.muted)("quit")}`,
    }),
  )
}

// ── Main layout ──────────────────────────────────────────────────────────
function render() {
  for (const child of renderer.root.getChildren()) {
    child.destroy()
  }

  let content: ReturnType<typeof Box>
  switch (activeTab) {
    case "Portfolio":
      content = PortfolioView()
      break
    case "Trades":
      content = TradesView()
      break
    case "Signals":
      content = SignalsView()
      break
    case "Stats":
      content = StatsView()
      break
    case "Bot Log":
      content = BotLogView()
      break
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
      TabBar(),
      content,
      Footer(),
    ),
  )
}

// ── Refresh data and re-render ───────────────────────────────────────────
async function refresh() {
  await journal.poll()
  circuitBreaker = computeCircuitBreaker(
    journal,
    config.cbMaxConsecutiveLosses,
    config.cbMaxLossUsd,
  )

  if (kalshiClient) {
    try {
      balance = await kalshiClient.getBalance()
      positions = await kalshiClient.getPositions()
      portfolio = await enrichPositions(positions, kalshiClient, journal)
      apiConnected = true
    } catch {
      apiConnected = false
    }
  }

  render()
}

// ── Tab navigation ───────────────────────────────────────────────────────
function switchTab(tab: TabName) {
  if (activeTab !== tab) {
    activeTab = tab
    render()
  }
}

function nextTab() {
  const idx = TABS.indexOf(activeTab)
  switchTab(TABS[(idx + 1) % TABS.length])
}

function prevTab() {
  const idx = TABS.indexOf(activeTab)
  switchTab(TABS[(idx - 1 + TABS.length) % TABS.length])
}

// ── Keyboard handling ────────────────────────────────────────────────────
renderer.keyInput.on("keypress", (key: KeyEvent) => {
  // Quit — kill bot child first
  if (key.name === "q" && !key.ctrl && !key.meta) {
    journal.stopPolling()
    bot.destroy()
    renderer.destroy()
    return
  }

  // Refresh
  if (key.name === "r" && !key.ctrl && !key.meta) {
    refresh()
    return
  }

  // Bot toggle
  if (key.name === "s" && !key.ctrl && !key.meta) {
    bot.toggle()
    return
  }

  // Tab switching by number
  if (key.name === "1" && !key.ctrl && !key.meta) {
    switchTab("Portfolio")
    return
  }
  if (key.name === "2" && !key.ctrl && !key.meta) {
    switchTab("Trades")
    return
  }
  if (key.name === "3" && !key.ctrl && !key.meta) {
    switchTab("Signals")
    return
  }
  if (key.name === "4" && !key.ctrl && !key.meta) {
    switchTab("Stats")
    return
  }
  if (key.name === "5" && !key.ctrl && !key.meta) {
    switchTab("Bot Log")
    return
  }

  // Tab / Shift+Tab
  if (key.name === "tab" && !key.shift) {
    nextTab()
    return
  }
  if (key.name === "tab" && key.shift) {
    prevTab()
    return
  }
})

// ── Bot log updates trigger re-render ────────────────────────────────────
bot.onUpdate = () => render()

// ── Auto-refresh on interval ─────────────────────────────────────────────
journal.onUpdate = () => render()
journal.startPolling(5000)

// Refresh API data every 15 seconds
setInterval(() => refresh(), 15000)

// ── Initial render ───────────────────────────────────────────────────────
render()
