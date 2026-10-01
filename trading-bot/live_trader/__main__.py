"""Run one live-trader tick from the command line.

    python -m live_trader              # one tick against your Alpaca paper account
    python -m live_trader --dry-run    # decide and record, but don't send orders
    python -m live_trader --force      # run even if the market is closed
"""
import argparse
import json
import logging

from trader_core import db
from trader_core.config import Settings
from trader_core.data import make_provider

from .broker import AlpacaBroker
from .trader import run_tick

parser = argparse.ArgumentParser(description="Run one live paper-trading tick.")
parser.add_argument("--dry-run", action="store_true", help="don't send orders")
parser.add_argument("--force", action="store_true", help="run even when the market is closed")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
settings = Settings.from_env()
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
