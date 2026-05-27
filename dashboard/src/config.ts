/**
 * Configuration loader for the Slugger dashboard.
 *
 * Reads the same .env file as the Python bot so both processes
 * share credentials and settings.
 */

import { readFileSync, existsSync } from "node:fs"
import { resolve } from "node:path"

export interface DashboardConfig {
  // Kalshi auth
  kalshiApiKeyId: string
  kalshiPrivateKeyPath: string
  useDemo: boolean

  // Paths
  logsDir: string
  repoRoot: string

  // Dashboard-specific
  pollIntervalMs: number
}

/**
 * Parse a .env file into a key-value map.
 * Does NOT override existing process.env values.
 */
function parseEnvFile(envPath: string): Map<string, string> {
  const vars = new Map<string, string>()
  if (!existsSync(envPath)) return vars

  const content = readFileSync(envPath, "utf-8")
  for (const line of content.split("\n")) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue
    const eqIdx = trimmed.indexOf("=")
    const key = trimmed.slice(0, eqIdx).trim()
    let val = trimmed.slice(eqIdx + 1).trim()
    // Strip surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1)
    }
    if (key) vars.set(key, val)
  }
  return vars
}

function getEnv(vars: Map<string, string>, key: string, fallback: string = ""): string {
  return process.env[key] ?? vars.get(key) ?? fallback
}

/**
 * Load dashboard config from .env file at the repo root.
 */
export function loadConfig(repoRoot?: string): DashboardConfig {
  const root = repoRoot ?? resolve(new URL("../../", import.meta.url).pathname)
  const envPath = resolve(root, ".env")
  const vars = parseEnvFile(envPath)

  let keyPath = getEnv(vars, "KALSHI_PRIVATE_KEY_PATH")
  if (keyPath.startsWith("~")) {
    keyPath = keyPath.replace("~", process.env.HOME ?? "")
  }

  return {
    kalshiApiKeyId: getEnv(vars, "KALSHI_API_KEY_ID", getEnv(vars, "KALSHI_KEY_ID")),
    kalshiPrivateKeyPath: keyPath,
    useDemo: getEnv(vars, "USE_DEMO", "false").toLowerCase() === "true",
    logsDir: resolve(root, getEnv(vars, "LOG_DIR", "logs")),
    repoRoot: root,
    pollIntervalMs: parseInt(getEnv(vars, "POLL_INTERVAL_SEC", "10")) * 1000,
  }
}
