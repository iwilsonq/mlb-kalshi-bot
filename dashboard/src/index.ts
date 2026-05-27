import { createCliRenderer, Box, Text, t, bold, fg, type KeyEvent } from "@opentui/core"

// ── Resolve paths relative to the repo root ──────────────────────────────
const REPO_ROOT = new URL("../../", import.meta.url).pathname
const LOGS_DIR = `${REPO_ROOT}logs`
const ENV_PATH = `${REPO_ROOT}.env`

// ── Bootstrap renderer ───────────────────────────────────────────────────
const renderer = await createCliRenderer({
  exitOnCtrlC: true,
  targetFps: 10,
})

// ── Color palette ────────────────────────────────────────────────────────
const colors = {
  bg: "#0D1117",
  bgPanel: "#161B22",
  border: "#30363D",
  borderActive: "#58A6FF",
  text: "#C9D1D9",
  textMuted: "#8B949E",
  green: "#3FB950",
  red: "#F85149",
  yellow: "#D29922",
  blue: "#58A6FF",
  purple: "#BC8CFF",
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

  return Box(
    {
      width: "100%",
      height: 3,
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      paddingX: 2,
      backgroundColor: colors.bgPanel,
      borderStyle: "single",
      borderColor: colors.border,
    },
    Text({
      content: t`${bold(fg(colors.blue)("Slugger"))} ${fg(colors.textMuted)("v0.2.0")}`,
    }),
    Text({
      content: t`${fg(colors.textMuted)(`${date}  ${time}`)}`,
    }),
  )
}

// ── Placeholder panel ────────────────────────────────────────────────────
function Panel(title: string, content: string) {
  return Box(
    {
      flexGrow: 1,
      borderStyle: "rounded",
      borderColor: colors.border,
      title: ` ${title} `,
      padding: 1,
      backgroundColor: colors.bgPanel,
      flexDirection: "column",
      gap: 1,
    },
    Text({
      content: t`${fg(colors.textMuted)(content)}`,
    }),
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
      backgroundColor: colors.bgPanel,
    },
    Text({
      content: t`${fg(colors.blue)("r")} ${fg(colors.textMuted)("refresh")}  ${fg(colors.blue)("tab")} ${fg(colors.textMuted)("switch panel")}  ${fg(colors.blue)("q")} ${fg(colors.textMuted)("quit")}`,
    }),
  )
}

// ── Main layout ──────────────────────────────────────────────────────────
function render() {
  // Clear existing children
  for (const child of renderer.root.getChildren()) {
    child.destroy()
  }

  renderer.root.add(
    Box(
      {
        width: "100%",
        height: "100%",
        flexDirection: "column",
        backgroundColor: colors.bg,
      },
      // Header
      Header(),

      // Main content area
      Box(
        {
          flexGrow: 1,
          flexDirection: "row",
          gap: 1,
          padding: 1,
        },
        // Left column
        Box(
          {
            flexGrow: 1,
            flexDirection: "column",
            gap: 1,
          },
          Panel("Portfolio", "Balance, exposure, P&L — awaiting Kalshi client"),
          Panel("Open Positions", "Active contracts — awaiting Kalshi client"),
        ),
        // Right column
        Box(
          {
            flexGrow: 1,
            flexDirection: "column",
            gap: 1,
          },
          Panel("Today's Trades", "Trade log — awaiting JSONL reader"),
          Panel("Signal Feed", "Live signals — awaiting JSONL reader"),
        ),
      ),

      // Footer
      Footer(),
    ),
  )
}

// ── Keyboard handling ────────────────────────────────────────────────────
renderer.keyInput.on("keypress", (key: KeyEvent) => {
  if (key.name === "q" && !key.ctrl && !key.meta) {
    renderer.destroy()
  }
  if (key.name === "r" && !key.ctrl && !key.meta) {
    render()
  }
})

// ── Initial render ───────────────────────────────────────────────────────
render()
