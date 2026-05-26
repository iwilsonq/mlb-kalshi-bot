# Slugger — MLB Kalshi Trading Bot

An automated trading bot that analyzes MLB games and places trades on the [Kalshi prediction market](https://kalshi.com).

## What It Does

- Fetches live MLB data (schedules, pitcher stats, batter stats, weather)
- Analyzes games using 4 built-in strategies
- Places limit orders on Kalshi markets (or dry-runs by default)
- Manages risk with Kelly criterion sizing and circuit breakers

## Strategies

| Strategy | Description |
|----------|-------------|
| `game_winner` | Predicts home team wins based on pitcher ERA |
| `pitcher_ks` | Predicts pitcher strikeout props |
| `player_hr` | Predicts player home run props |
| `total_runs` | Predicts over/under on total runs |

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
- `MIN_EDGE_CENTS=3` — minimum edge to trigger a trade

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
Edit `ENABLED_STRATEGIES` in `.env`:
```bash
ENABLED_STRATEGIES=game_winner,pitcher_ks
```

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
│   ├── __init__.py      # Package init, version
│   ├── config.py        # Configuration from .env
│   ├── mlb_data.py      # MLB Stats API + Statcast data
│   ├── kalshi_client.py # Kalshi API client (auth, orders, markets)
│   └── strategies.py    # Trading strategies
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
└── .env.example         # Example config file
```

## How the Bot Works

1. **Data Collection**: Fetches MLB schedules, pitcher profiles, batter stats from the Stats API and Statcast
2. **Market Scanning**: Queries Kalshi for open markets matching MLB games
3. **Signal Generation**: Each strategy analyzes the data and generates trade signals when a positive edge is detected
4. **Position Sizing**: Uses fractional Kelly criterion to determine how many contracts to buy
5. **Execution**: Places limit orders on Kalshi (or logs in dry-run mode)
6. **Risk Management**: Circuit breakers stop trading after consecutive losses or total loss threshold

## Customization

- **Add strategies**: Create a new function in `slugger/strategies.py` following the `TradeSignal` return pattern, then register it in `STRATEGIES`
- **Adjust sizing**: Modify `kelly_fraction`, `max_position_usd`, or `min_edge_cents` in `.env`
- **Add sportsbooks**: The bot currently only supports Kalshi — adding others would require new API clients in `kalshi_client.py`