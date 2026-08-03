# Slugger — MLB Kalshi Trading Bot

An automated trading bot that analyzes MLB games and places trades on the [Kalshi prediction market](https://kalshi.com).

## What It Does

- Fetches live MLB data (schedules, pitcher stats, batter stats, weather)
- Analyzes games with the strategies registered in `STRATEGY_PIPELINE`
- Places limit orders on Kalshi markets (or dry-runs by default)
- Manages risk with Kelly sizing, per-game/daily caps, rolling strategy
  health auto-disable, and circuit breakers

## Strategies

The live registry is `STRATEGY_PIPELINE` in `slugger/strategies.py`; the
`ENABLED_STRATEGIES` allowlist can only narrow it, never extend it.

| Strategy | Status | Model |
|----------|--------|-------|
| `player_hits` | live | Binomial over expected AB; per-AB probability from shrunk splits × WHIP/hard-hit/park, calibrated on game-log outcomes |
| `pitcher_ks` | auto-disabled (rolling ROI) | Poisson GLM λ with negative-binomial tail, trained walk-forward on game logs (`logs/ks_model.json`) |

Retired strategies (`game_winner`, `total_runs`, `player_hr`,
`player_hr_rbis`, `combo`, `pitcher_er`) were deleted; the journal
evidence for each lives in `RETIRED_STRATEGIES` in `slugger/strategies.py`.

Honest status: neither model beats the market's Brier score on holdout,
so the bot currently finds no edge and places few or no trades. The
measurement/reporting pipeline (`calibrate --fit`, `report`) is the part
that earns its keep.

## Prerequisites

- Python 3.9+
- Kalshi API credentials (get them at https://kalshi.com/account/api)

## Setup

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Configure your API keys

Create a `.env` file in the project root from the template:

```bash
cp .env.example .env
```

Edit `.env` and fill in:
- `KALSHI_API_KEY_ID` — your API key ID from Kalshi
- `KALSHI_PRIVATE_KEY_PATH` — path to your PEM private key file

Also set your trading preferences:
- `DRY_RUN=true` — test mode, no real orders (default)
- `USE_DEMO=true` — use Kalshi's demo environment (default)
- `MAX_POSITION_USD=5` — max per trade
- `KELLY_FRACTION=0.25` — quarter-Kelly sizing
- `MIN_EDGE_CENTS=5` — minimum *cost-adjusted* edge to trigger a trade
- `EDGE_COST_BUFFER_CENTS=2` — residual adverse-selection haircut (the exact
  Kalshi fee and half-spread are computed per contract automatically)
- `MAX_EXPOSURE_PER_GAME_USD=16` — dollar cap per game (~2× max position)
- `ENABLED_STRATEGIES=pitcher_ks,player_hits` — allowlist (subset of the pipeline)

### 3. Generate an API key (if you haven't)

1. Go to [kalshi.com/account/api](https://kalshi.com/account/api)
2. Generate a new API key
3. Download the private key PEM file
4. Set the paths in `.env`

## Run

### Check API connection
```bash
python3 main.py check
```

### View today's games and markets
```bash
python3 main.py status
```

### Model vs market report
```bash
python3 main.py report           # Brier/log-loss + ROI heatmaps
python3 main.py report --ask     # use ask instead of mid for market implied
python3 main.py report --min-n 10
```

### Start the bot (dry-run)
```bash
python3 main.py run
```

The bot will:
1. Fetch today's MLB games
2. Query Kalshi for relevant markets
3. Run enabled strategies for each game
4. Log signals (no real orders in dry-run mode)

### Start live trading
Set `DRY_RUN=false` in `.env`, then:
```bash
python3 main.py run
```

### Use specific strategies
Edit `ENABLED_STRATEGIES` in `.env`. Entries not registered in
`STRATEGY_PIPELINE` are inert (a test enforces the default allowlist is a
subset of the pipeline):
```bash
ENABLED_STRATEGIES=pitcher_ks,player_hits
```

### Retrain models and calibration (daily)
```bash
python3 main.py calibrate --fit
```
Fits walk-forward calibration curves from game-log outcomes (not just
traded settlements), retrains the Ks model, and reports holdout Brier vs
the market — with an explicit warning when the model has no demonstrated
edge.

## CLI Options

```
python3 main.py run [-h] [--env ENV] [--verbose]

Commands:
  run      Start the bot loop
  status   Show today's games and market status
  check    Test Kalshi API connection

Options:
  --env ENV       Path to .env file (default: .env)
  --verbose, -v   Enable debug logging
```

## Project Structure

```
mlb-kalshi-bot/
├── slugger/
│   ├── config.py           # Configuration from .env
│   ├── mlb_data.py         # MLB Stats API + Statcast data
│   ├── kalshi_client.py    # Kalshi API client (auth, orders, markets)
│   ├── strategies.py       # Strategy fns + STRATEGY_PIPELINE registry
│   ├── signal_pipeline.py  # Market matching, edge/fee math, signal recording
│   ├── models.py           # Distributions and probability models
│   ├── ks_model.py         # Trained strikeout model (walk-forward)
│   ├── calibration.py      # Isotonic calibration from game-log outcomes
│   ├── risk.py             # Rolling strategy health, exposure budgets
│   ├── execution.py        # Limit pricing, order cancellation
│   ├── game_processor.py   # Main loop
│   └── backtest/           # Point-in-time replay harness
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── .env.example            # Example config file
```

## How the Bot Works

1. **Data Collection**: Fetches MLB schedules, pitcher profiles, batter stats from the Stats API and Statcast
2. **Market Scanning**: Queries Kalshi for open markets matching MLB games
3. **Signal Generation**: Each strategy analyzes the data and generates trade signals when a positive edge is detected
4. **Position Sizing**: Uses fractional Kelly criterion to determine how many contracts to buy
5. **Execution**: Places limit orders on Kalshi (or logs in dry-run mode)
6. **Risk Management**: Circuit breakers stop trading after consecutive losses or total loss threshold

## Customization

- **Add strategies**: Create the strategy function in `slugger/strategies.py`, wrap it with the uniform `StrategyFn` signature, and register it in `STRATEGY_PIPELINE` (the `STRATEGIES` dict no longer exists)
- **Adjust sizing**: Modify `kelly_fraction`, `max_position_usd`, or `min_edge_cents` in `.env`
- **Add sportsbooks**: The bot currently only supports Kalshi — adding others would require new API clients in `kalshi_client.py`