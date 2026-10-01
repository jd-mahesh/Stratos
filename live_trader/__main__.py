"""Run one live-trader tick from the command line.

    python -m live_trader              # one tick against your Alpaca paper account
    python -m live_trader --dry-run    # decide and record, but don't send orders
    python -m live_trader --force      # run even if the market is closed
    python -m live_trader --resume     # clear a circuit-breaker halt (shows why it stopped), then exit
"""
import argparse
import json
import logging

from trader_core import db, safeguards
from trader_core.config import Settings
from trader_core.data import make_provider

from .broker import AlpacaBroker
from .trader import run_tick

parser = argparse.ArgumentParser(description="Run one live paper-trading tick.")
parser.add_argument("--dry-run", action="store_true", help="don't send orders")
parser.add_argument("--force", action="store_true", help="run even when the market is closed")
parser.add_argument("--resume", action="store_true",
                    help="clear a circuit-breaker halt after you've checked what happened, then exit")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
settings = Settings.from_env()
if args.resume:
    engine = db.get_engine(settings.database_url)
    db.init_db(engine)
    previous = safeguards.clear_halt(engine)
    if previous:
        print(f"Resumed. Trading was halted since {previous.get('since')} because: {previous.get('reason')}")
    else:
        print("Trading wasn't halted by the circuit breaker; nothing to resume.")
    if settings.trading_halted:
        print("Note: the kill switch is still on (TRADING_HALTED=true), so no orders will be placed until it's off.")
    raise SystemExit(0)
settings.dry_run = settings.dry_run or args.dry_run
settings.force_run = settings.force_run or args.force
settings.require_alpaca()

engine = db.get_engine(settings.database_url)
db.init_db(engine)
result = run_tick(
    settings,
    AlpacaBroker(settings.alpaca_key_id, settings.alpaca_secret_key),
    make_provider(settings, "alpaca"),
    engine,
)
print(json.dumps(result, indent=2, default=str))
