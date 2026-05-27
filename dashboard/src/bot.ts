/**
 * Bot process manager for the Slugger dashboard.
 *
 * Spawns and manages the Python bot (main.py run) as a child process.
 * Captures stdout/stderr into a ring buffer for display in the dashboard.
 * Provides start/stop/status controls.
 *
 * Safety:
 *   - No auto-restart on crash (trading bot -- user must review and restart)
 *   - Clean SIGTERM on stop, escalates to SIGKILL after timeout
 *   - Dashboard quit always kills child process
 */

import { spawn, type ChildProcess } from "node:child_process"
import type { DashboardConfig } from "./config.js"

// ── Types ────────────────────────────────────────────────────────────────

export type BotStatus = "stopped" | "running" | "crashed" | "stopping"

export interface LogLine {
  timestamp: Date
  stream: "stdout" | "stderr"
  text: string
}

// ── Ring buffer for log lines ────────────────────────────────────────────

const MAX_LOG_LINES = 500

// ── BotManager ───────────────────────────────────────────────────────────

export class BotManager {
  private config: DashboardConfig
  private proc: ChildProcess | null = null
  private _status: BotStatus = "stopped"
  private _exitCode: number | null = null
  private _exitSignal: string | null = null
  private _startedAt: Date | null = null
  private _stoppedAt: Date | null = null

  /** Ring buffer of captured log lines */
  logLines: LogLine[] = []

  /** Called when status changes or new log lines arrive */
  onUpdate?: () => void

  constructor(config: DashboardConfig) {
    this.config = config
  }

  // ── Getters ──────────────────────────────────────────────────────────

  get status(): BotStatus {
    return this._status
  }

  get exitCode(): number | null {
    return this._exitCode
  }

  get exitSignal(): string | null {
    return this._exitSignal
  }

  get startedAt(): Date | null {
    return this._startedAt
  }

  get uptime(): string {
    if (!this._startedAt || this._status !== "running") return "--"
    const ms = Date.now() - this._startedAt.getTime()
    const secs = Math.floor(ms / 1000)
    const mins = Math.floor(secs / 60)
    const hrs = Math.floor(mins / 60)
    if (hrs > 0) return `${hrs}h${mins % 60}m`
    if (mins > 0) return `${mins}m${secs % 60}s`
    return `${secs}s`
  }

  get pid(): number | null {
    return this.proc?.pid ?? null
  }

  // ── Start ────────────────────────────────────────────────────────────

  start(args: string[] = []): void {
    if (this._status === "running" || this._status === "stopping") return

    this._exitCode = null
    this._exitSignal = null
    this._startedAt = new Date()
    this._stoppedAt = null
    this._status = "running"

    const cmdArgs = ["main.py", "run", ...args]

    this.addLog("stdout", `$ python3 ${cmdArgs.join(" ")}`)

    this.proc = spawn("python3", cmdArgs, {
      cwd: this.config.repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1", // Force unbuffered output for real-time logs
      },
    })

    this.proc.stdout?.on("data", (data: Buffer) => {
      this.handleOutput("stdout", data)
    })

    this.proc.stderr?.on("data", (data: Buffer) => {
      this.handleOutput("stderr", data)
    })

    this.proc.on("exit", (code, signal) => {
      this._exitCode = code
      this._exitSignal = signal?.toString() ?? null
      this._stoppedAt = new Date()

      if (this._status === "stopping") {
        // Expected stop
        this._status = "stopped"
        this.addLog("stdout", `Bot stopped (code=${code})`)
      } else {
        // Unexpected exit
        this._status = code === 0 ? "stopped" : "crashed"
        const reason = signal ? `signal=${signal}` : `code=${code}`
        this.addLog("stderr", `Bot exited unexpectedly (${reason})`)
      }

      this.proc = null
      this.onUpdate?.()
    })

    this.proc.on("error", (err) => {
      this._status = "crashed"
      this._stoppedAt = new Date()
      this.addLog("stderr", `Failed to start: ${err.message}`)
      this.proc = null
      this.onUpdate?.()
    })

    this.onUpdate?.()
  }

  // ── Stop ─────────────────────────────────────────────────────────────

  stop(): void {
    if (!this.proc || this._status !== "running") return

    this._status = "stopping"
    this.addLog("stdout", "Stopping bot (SIGTERM)...")
    this.proc.kill("SIGTERM")

    // Escalate to SIGKILL if still running after 5 seconds
    const killTimer = setTimeout(() => {
      if (this.proc && !this.proc.killed) {
        this.addLog("stderr", "Bot did not stop, sending SIGKILL")
        this.proc.kill("SIGKILL")
      }
    }, 5000)

    // Clear the kill timer if process exits cleanly
    this.proc.once("exit", () => {
      clearTimeout(killTimer)
    })

    this.onUpdate?.()
  }

  // ── Toggle ───────────────────────────────────────────────────────────

  toggle(): void {
    if (this._status === "running") {
      this.stop()
    } else if (this._status === "stopped" || this._status === "crashed") {
      this.start()
    }
    // Do nothing if "stopping" -- wait for it to finish
  }

  // ── Cleanup (called on dashboard quit) ────────────────────────────────

  destroy(): void {
    if (this.proc && !this.proc.killed) {
      this.proc.kill("SIGTERM")
      // Give it a moment, then force kill
      setTimeout(() => {
        if (this.proc && !this.proc.killed) {
          this.proc.kill("SIGKILL")
        }
      }, 2000)
    }
  }

  // ── Internal ─────────────────────────────────────────────────────────

  private handleOutput(stream: "stdout" | "stderr", data: Buffer): void {
    const text = data.toString("utf-8")
    // Split on newlines, handle partial lines
    for (const line of text.split("\n")) {
      const trimmed = line.trimEnd()
      if (trimmed) {
        this.addLog(stream, trimmed)
      }
    }
    this.onUpdate?.()
  }

  private addLog(stream: "stdout" | "stderr", text: string): void {
    this.logLines.push({ timestamp: new Date(), stream, text })
    // Ring buffer: drop oldest lines
    if (this.logLines.length > MAX_LOG_LINES) {
      this.logLines = this.logLines.slice(-MAX_LOG_LINES)
    }
  }
}
