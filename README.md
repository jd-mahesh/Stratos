# Stratos

Stratos is a cloud-deployed algorithmic trading system. Trading strategies are
backtested on historical data and then paper-traded live against Alpaca, packaged as
three Docker containers that run on AWS.

The point of the project is the plumbing as much as the strategy: one strategy
function shared by backtest and live code, containers that run the same on a laptop
and on Lambda, credentials in Secrets Manager, and a scheduler that can fire twice
without placing a trade twice.

**Paper trading only.** The code connects to Alpaca's paper environment and has no
setting for real money. Nothing here is investment advice.

## How it fits together

```
strategy.py ──imported by──> backtester image ──┐
            └─imported by──> live-trader image ─┼─docker push─> ECR
                             dashboard image ───┘

historical bars ──> backtester (Lambda, on demand) ──backtest_* tables──┐
EventBridge (every 5 min, market hours) ──> live-trader (Lambda) ─────┼──> Postgres (RDS)
     Secrets Manager ──keys──┘      │  ▲                              │
                                    ▼  │ poll prices / submit orders   ▼
                                   Alpaca paper API          dashboard (ECS Express Mode)
```

| Folder | What it is |
|---|---|
| `trader_core/` | Shared code: the strategy, price-data providers, database schema, settings |
| `backtester/` | Bar-by-bar backtest engine, performance metrics, CLI and Lambda handler |
| `live_trader/` | One trading "tick": read account, ask the strategy, place orders, record everything |
| `dashboard/` | Streamlit app that reads only the database |
| `docker/` | One Dockerfile per service |
| `tests/` | pytest suite (no network or API keys needed) |
| `docs/DEPLOY.md` | Step-by-step AWS deployment with the CLI |
| `scripts/check_symbols.py` | Checks every ticker in `SYMBOLS` against Alpaca: tradable, fractional, how much price history |
| `ci/ci.yml` | GitHub Actions workflow (tests against Postgres, builds all three images). Move it to `.github/workflows/ci.yml` when you put the repo on GitHub |

## Quick start (no API keys needed)

You need Python 3.11+ (macOS's built-in 3.9 is too old for current libraries;
`brew install python@3.12` or the python.org installer both work).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt

pytest -q                                      # runs in about a second, no network needed
python -m backtester --provider synthetic      # backtest on fake prices, saved to trading.db
streamlit run dashboard/app.py --server.port 8502  # open http://localhost:8502
```

`--provider synthetic` generates random-walk prices so you can run everything offline.
The dashboard flags those runs so they can't be mistaken for real results.

## With real data

1. Make a free account at [alpaca.markets](https://alpaca.markets), switch to the **Paper** account,
   and generate API keys.
2. `cp .env.example .env` and paste the keys in.
3. Run a real backtest:

   ```bash
   python -m backtester --start 2021-01-01
   python -m backtester --symbols SPY,QQQ --fast 10 --slow 30 --provider yfinance
   ```

4. Before trading a new list of tickers, check them:

   ```bash
   python scripts/check_symbols.py
   ```

   It flags tickers Alpaca doesn't carry, ones that can't be bought in fractional
   shares, and recent listings without enough price history yet.

5. Run one live tick without sending orders (works when the market is closed too):

   ```bash
   python -m live_trader --dry-run --force
   ```

## With Docker (the same way it runs on AWS)

```bash
docker compose up -d db dashboard                        # Postgres + dashboard on :8502
docker compose run --rm backtester --provider synthetic  # one-shot jobs
docker compose run --rm live-trader --dry-run --force
docker compose down                                      # add -v to wipe the database
```

The backtester and live-trader images are built on AWS's Lambda base image, which
includes an emulator of the Lambda API. You can call the real Lambda handler locally:

```bash
docker compose build backtester
docker run --rm -p 9000:8080 -e DATA_PROVIDER=synthetic stratos-backtester
# in another terminal:
curl -s localhost:9000/2015-03-31/functions/function/invocations -d '{"save": false}'
```

## Strategies

Pick one with `STRATEGY` in `.env` (or `--strategy` when backtesting). All of them
answer "what fraction of the account should each symbol be?", and the backtester and
live trader size and place orders with the same code (`trader_core/portfolio.py`).

| Strategy | Rule | Checked |
|---|---|---|
| `ma_crossover` | Own a symbol while its 20-day average is above its 50-day average | every run |
| `trend` | Own a symbol while its price is above its 200-day average | monthly |
| `momentum` | Hold the 3 symbols with the best 12-month return, only if that return is positive | monthly |

`trend` and `momentum` use the standard settings from published research (Faber's
10-month trend filter, 12-month "dual momentum"), not values tuned on recent data.
Change them with `TREND_WINDOW`, `MOMENTUM_LOOKBACK` and `MOMENTUM_TOP`.

Monthly strategies rebalance on the first run of each month (the bot remembers the month
in the `bot_state` table), trimming or topping up positions that drifted more than 25%
from their target. The other runs that month just record the account value.

### Live prices (`SIGNAL_MODE`)

- `close`: decisions use completed daily closes, so they can only change once a day.
- `intraday`: the live price right now is added as the newest point in the history, so a
  sharp move today counts immediately. The backtester simulates this by deciding with the
  opening price at the open and the closing price at the close (daily bars can't show every
  5-minute tick, so it's an approximation).

Run the same backtest with `--signal-mode close` and `--signal-mode intraday` to see whether
reacting to live prices helps or just adds whipsaw.

### Crash mode (`CRASH_SWITCH`)

The normal strategy is built for normal markets. Crashes are handled separately:

- **Detector** (`trader_core/regime.py`), checked on every run: crash mode starts when QQQ is
  below its 200-day average **and** at least 10% below its 1-year high, and ends when QQQ
  closes back above its 200-day average. The thresholds are standard conventions (10% is the
  textbook "correction"), not values fitted to any crash.
- **Backup** while in crash mode: hold whichever of BIL (T-bills), IEF, TLT (Treasuries) or
  GLD (gold) did best over the last 3 months, or BIL if none beat it. `CRASH_MODE=cash` holds
  plain cash instead.
- A change of mode rebalances immediately, not at the next month start.
- **Confirmation** (`CRASH_CONFIRM_DAYS`, default 1): with N above 1 the mode only changes
  after the rule has held N daily closes in a row, and the check uses finished closes only
  (not the live price). This stops the flip-flopping the first crash report showed.
  Try `--crash-confirm 5`.

Settings: `CRASH_SWITCH`, `CRASH_INDEX`, `CRASH_DRAWDOWN` (percent), `CRASH_WINDOW`,
`CRASH_CONFIRM_DAYS`, `CRASH_MODE`, `CRASH_ASSETS`. Try it in a backtest with `--crash-switch on`.

**Crash report.** Finds every fall of 15% or more in QQQ in the data (no hand-picked list)
and compares the strategy with and without crash mode in each one: losses, every stretch
crash mode was on, recovery times, and false alarms in normal markets.

```bash
python -m backtester --crash-report --strategy momentum --universe sectors --provider yfinance --start 1999-06-01
```

**Tuning for normal markets.** `--normal-only` makes the sweep score settings on
normal-market days only, so crashes don't shape the normal strategy:

```bash
python -m backtester --strategy momentum --sweep --normal-only --universe tech2020 --provider yfinance --start 2021-01-01
```

The whole system should still be judged over full history, crashes included, since that's
what it will live through; the crash report shows that side.

### Choosing momentum settings (`--sweep`)

```bash
python -m backtester --strategy momentum --sweep --universe sectors --provider yfinance --start 2007-01-01 --cash-rate 1.5
```

This tries a short, fixed grid (3, 6 and 12-month lookbacks; hold the top 1, 2, 3 or 5)
and splits the period in half. A setting passes only if it beats buy & hold in **both**
halves. Prefer a setting whose neighbours pass too; a lone winner is more likely luck.
Sweep runs aren't saved to the dashboard.

**Rebalancing more often.** Momentum rebalances monthly by default. `--every N` makes a
backtest rebalance every N trading days instead (5 = weekly), and in a sweep it takes a
list to compare, with `0` meaning monthly. `--lookbacks` and `--tops` set the rest of the
grid (in trading days and number of symbols):

```bash
python -m backtester --strategy momentum --sweep --universe sectors --provider yfinance \
  --start 1999-06-01 --lookbacks 21,63,126 --tops 5 --every 2,5,10,0
```

Faster rebalancing reacts sooner but trades much more; the per-trade cost (`--slippage-bps`,
default 5) is charged on every trade, so the comparison includes that. This is a
backtester-only setting for now: the live trader still rebalances monthly.

**Skip-month momentum.** Stocks that spiked in the last few weeks often give some of it
back. `--skip N` measures each stock's return up to N trading days ago instead of up to
today (`--skip 21` ignores the most recent month), the way the classic research version
of momentum works. In a sweep it takes a list, e.g. `--skip 0,21`:

```bash
python -m backtester --strategy momentum --sweep --universe sectors --provider yfinance \
  --start 1999-06-01 --lookbacks 126,252 --tops 5 --skip 0,21
```

Also backtester-only for now.

**Volatility scaling.** Momentum's worst crashes tend to come in stretches of extreme
volatility. `--vol-scale N` compares how volatile the picks were over their last N trading
days with their last year, and invests less when recent volatility is higher: twice as
volatile as usual means half invested, the rest in cash. It never invests more than 100%.
Based on Barroso & Santa-Clara (2015) and Moreira & Muir (2017); `21` is the research
version. In a sweep, e.g. `--vol-scale 0,21,63` (0 = off):

```bash
python -m backtester --strategy momentum --sweep --universe sectors --provider yfinance \
  --start 1999-06-01 --lookbacks 126 --tops 5 --vol-scale 0,21,63
```

Unlike the options above, this one is also a live setting: `MOMENTUM_VOL_SCALE=21` in `.env`
turns it on for the live trader (0 = off). In backtests from 1999 (sector funds), 2007
(tech2020) and 2016 (the watchlist) it cut the worst drop on every list, by up to 9 points,
at about the same return. It's checked at each monthly rebalance, so a crash that
starts mid-month isn't caught until the next one. Turning it on (or changing it) counts
as a new strategy setting, so the next run rebalances right away.

### Adding a strategy

Everything else (backtester, live trader, dashboard) works with any strategy, so a new one
is three small edits in `trader_core/strategy.py`:

1. Write a class with a `decide(history, slots)` method that returns a target weight per
   symbol (see `Momentum` for a cross-sectional example, `Trend` for a per-symbol one).
2. Add it to `STRATEGIES`.
3. Give it defaults in `make_strategy` (and settings in `config.py` if it needs any).

Then backtest it exactly like the others before setting `STRATEGY` to it.

### Does it work beyond 2021–2026?

Five years is one market regime. Yahoo Finance has decades of history, so test across
2008, 2011, 2015–16, 2018, the 2020 crash and 2022:

```bash
python -m backtester --strategy momentum --universe sectors --provider yfinance --start 2007-01-01
python -m backtester --strategy trend --universe assets --provider yfinance --start 2007-01-01
```

Better still, develop on one period and confirm on another the strategy has never seen:
`--start 2007-01-01 --end 2018-12-31`, then `--start 2019-01-01`. (Skip `mega2020` for
periods before 2021: it's a list of 2020's winners.)

`--cash-rate 3` credits idle cash with 3% a year, closer to what a money-market fund paid
in recent years; the "same exposure" benchmark gets the same rate.

## Safety

Stratos is built to the standard of a real-money account even while it paper trades.
Every live run goes through `trader_core/safeguards.py`; every problem is logged and sent
as an alert (`live_trader/alerts.py`: emailed through AWS SNS when `ALERT_TOPIC_ARN` is set,
at most once a day per problem). All limits are settings in `.env` (see `.env.example`).

| Safeguard | What it does | Default |
|---|---|---|
| Kill switch | `TRADING_HALTED=true`: record balances, place no orders | off |
| Settings check | Refuse to start if the strategy's normal targets would break the limits (e.g. `MOMENTUM_TOP=2` = 50% per stock) | on |
| Order limits | Check the whole plan before sending anything; any breach cancels the run | 25% per buy, 30% per position, 20 orders per run |
| Short-sale guard | Never sell more than is held | on |
| Price sanity | Skip a symbol whose live price is missing, whose data is stale, or that moved implausibly far from its last close; retry next run | 40% move, 5 days old |
| Circuit breaker | Halt after a big loss in a day or from the peak, and stay halted until a person runs `python -m live_trader --resume` (or invokes the Lambda with `{"resume": true}`) | 15% in a day, 60% from peak |

The defaults sit beyond anything the strategy does normally (its worst backtested drop was
about 40%), so they only trip when something is broken: bad data, a bug, a misconfiguration.
Deposits or withdrawals look like gains or losses to the circuit breaker. The dashboard shows
a red banner while trading is halted.

## Reading backtest results honestly

Every backtest prints three columns:

- **Strategy**: what the crossover rule did.
- **Buy & hold**: buying every symbol on day one and never trading.
- **Same exposure**: buy & hold scaled down to the strategy's *average* amount invested
  (say 60%), with the rest in cash.

The strategy usually makes less than buy & hold and has smaller drawdowns, which on its
own proves nothing: holding less stock does that automatically. The real test is whether
it beats **same exposure** (more return, or a smaller drawdown for the same return) and
whether its **Sharpe** is higher than buy & hold's. Scaling a portfolio down doesn't change
its Sharpe, so a higher Sharpe means better return per unit of risk, not just less risk.
The backtest prints a one-line verdict and a year-by-year table.

**Watch out for hindsight.** A watchlist you made recently is full of stocks that already
did well, so backtesting it over past years flatters both the strategy and buy & hold.
For an honest read, test on lists that could have been chosen at the start:

```bash
python -m backtester --universe sectors --start 2021-01-01    # the 11 S&P sector ETFs
python -m backtester --universe assets --start 2021-01-01     # stocks, bonds, gold, commodities, real estate
python -m backtester --universe mega2020 --start 2021-01-01   # ~30 largest US companies at the end of 2020
python -m backtester --universe tech2020 --start 2021-01-01   # ~25 largest US tech and chip companies at the end of 2020
```

The presets live in `trader_core/universes.py`. You can use them in `.env` too
(`SYMBOLS=@sectors`) or mix them with tickers (`SYMBOLS=@etf4,NVDA`).

## Design notes

**No lookahead.** In close mode the backtester computes each signal from closes up to
day *t* and fills at day *t+1*'s open; in intraday mode it decides with the price at that
moment and fills at that same price. The tests check both, because a backtest that trades
on a price it couldn't have seen yet looks great and means nothing.

**Targets, not events.** The strategy answers "how much should I hold?" rather than
"buy now". Asking the same question twice gives the same answer, so running the live
trader every five minutes doesn't stack up duplicate positions. On top of that,
symbols with unfilled orders are skipped, and each order's `client_order_id` is derived
from the scheduled run time so a retried Lambda invocation is rejected by Alpaca
instead of trading twice.

**Backtest and live are meant to agree.** Both modes are simulated by the backtester,
and both programs use the same strategy code and the same order planner.

**Sizing.** Each position's target is a fraction of the account (1/number of symbols for
`ma_crossover` and `trend`, 1/3 for `momentum`). With `FRACTIONAL_SHARES=true` (the
default) buys spend their whole target, even on a $1,000 stock with a $1,900 slot. The bot
only spends the account's actual cash, never Alpaca's margin "buying power", so it can't
borrow. Symbols don't need matching histories: a recent listing waits until it has enough
prices. Positions in symbols that aren't in `SYMBOLS` are left alone.

**Beating the market isn't the goal; measuring honestly is.** A strategy is only worth
running if it beats "same exposure" and has a better Sharpe than buy & hold on lists
chosen without hindsight, over more than one market regime.

## Ideas for later

- Deploy from GitHub Actions on merge to `main` (OIDC role instead of stored AWS keys)
- Describe the AWS resources in Terraform or AWS CDK instead of CLI commands
- A parameter sweep (run the backtester Lambda for many window pairs in parallel)
- Volatility-based sizing (smaller positions in more volatile symbols)
