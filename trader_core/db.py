"""Database schema and helpers (SQLAlchemy Core).

The same code talks to SQLite (local dev, tests) and Postgres (docker compose,
RDS). ``init_db`` creates any missing tables, which is enough for a project
this size; a bigger one would use migrations (e.g. Alembic).

Tables
    backtest_runs      one row per backtest: parameters and summary metrics
    backtest_equity    daily equity curve of each run, plus buy-and-hold benchmark
    backtest_trades    round-trip trades of each run
    decisions          every live-trader decision, including "hold"
    equity_history     paper-account equity snapshot per live-trader tick
    positions          current paper positions (replaced every tick)
    bot_state          small key/value facts the live trader must remember,
                       e.g. which month a monthly strategy last rebalanced
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Mapping, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

backtest_runs = Table(
    "backtest_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("strategy", String(64), nullable=False),
    Column("params", JSON, nullable=False),
    Column("symbols", Text, nullable=False),
    Column("data_provider", String(32), nullable=False),
    Column("start_date", Date, nullable=False),
    Column("end_date", Date, nullable=False),
    Column("initial_capital", Float, nullable=False),
    Column("final_equity", Float, nullable=False),
    Column("total_return", Float, nullable=False),
    Column("cagr", Float),
    Column("max_drawdown", Float, nullable=False),
    Column("sharpe", Float),
    Column("win_rate", Float),
    Column("num_trades", Integer, nullable=False),
    Column("benchmark_return", Float, nullable=False),
    # Added later; nullable so older databases can be upgraded in place (see _add_missing_columns).
    Column("universe", String(64)),
    Column("avg_exposure", Float),
    Column("benchmark_cagr", Float),
    Column("benchmark_max_drawdown", Float),
    Column("benchmark_sharpe", Float),
    Column("matched_return", Float),
    Column("matched_cagr", Float),
    Column("matched_max_drawdown", Float),
    Column("signal_mode", String(16)),  # close | intraday
    Column("cash_rate", Float),  # yearly interest assumed on cash
)

backtest_equity = Table(
    "backtest_equity",
    metadata,
    Column("run_id", Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), primary_key=True),
    Column("ts", Date, primary_key=True),
    Column("equity", Float, nullable=False),
    Column("benchmark_equity", Float, nullable=False),
    Column("matched_equity", Float),  # buy & hold at the strategy's average exposure
    Column("exposure", Float),  # fraction of equity invested at the close
    Column("mode", String(8)),  # normal | crash, for strategies with a crash mode
)

backtest_trades = Table(
    "backtest_trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("symbol", String(16), nullable=False),
    Column("entry_ts", Date, nullable=False),
    Column("entry_price", Float, nullable=False),
    Column("exit_ts", Date),  # NULL while still open at the end of the test
    Column("exit_price", Float),
    Column("qty", Float, nullable=False),
    Column("pnl", Float),
    Column("return_pct", Float),
)

decisions = Table(
    "decisions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("symbol", String(16), nullable=False),
    Column("price", Float),
    Column("fast_ma", Float),
    Column("slow_ma", Float),
    Column("target", Float),
    Column("current_qty", Float, nullable=False),
    Column("target_qty", Float),
    Column("action", String(16), nullable=False),  # buy | sell | hold | skip | error
    Column("order_qty", Float),
    Column("order_id", String(64)),
    Column("reason", Text, nullable=False),
    Column("dry_run", Boolean, nullable=False, default=False),
)

equity_history = Table(
    "equity_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("equity", Float, nullable=False),
    Column("cash", Float, nullable=False),
    Column("buying_power", Float, nullable=False),
    # CAPITAL_RESERVE in force when the snapshot was taken (NULL = 0, before the setting
    # existed). Stratos's trading budget at that moment is equity - reserve.
    Column("reserve", Float),
)

positions = Table(
    "positions",
    metadata,
    Column("symbol", String(16), primary_key=True),
    Column("qty", Float, nullable=False),
    Column("avg_entry_price", Float),
    Column("market_value", Float),
    Column("unrealized_pl", Float),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


bot_state = Table(
    "bot_state",
    metadata,
    Column("key", String(200), primary_key=True),
    Column("value", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def get_state(engine: Engine, key: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(select(bot_state.c.value).where(bot_state.c.key == key)).first()
    return row[0] if row else None


def set_state(engine: Engine, key: str, value: str) -> None:
    with engine.begin() as conn:
        conn.execute(delete(bot_state).where(bot_state.c.key == key))
        conn.execute(insert(bot_state).values(key=key, value=value, updated_at=utcnow()))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Bring an older database up to date by adding any columns it doesn't have yet.

    ``create_all`` only creates missing *tables*. New columns are always
    nullable, so they can be added to tables that already hold rows.
    A real project would use a migration tool such as Alembic for this.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            have = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in have:
                    ddl = column.type.compile(dialect=engine.dialect)
                    conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}"))


def plain(value):
    """numpy scalars -> Python scalars. SQLite accepts numpy floats; Postgres (psycopg2) does not."""
    return value.item() if hasattr(value, "item") and not isinstance(value, (str, bytes)) else value


def insert_rows(engine: Engine, table: Table, rows: Iterable[Mapping]) -> None:
    rows = [{k: plain(v) for k, v in row.items()} for row in rows]
    if rows:
        with engine.begin() as conn:
            conn.execute(insert(table), rows)


def replace_positions(engine: Engine, rows: List[Mapping]) -> None:
    """Swap the positions table for the latest snapshot in one transaction."""
    rows = [{k: plain(v) for k, v in row.items()} for row in rows]
    with engine.begin() as conn:
        conn.execute(delete(positions))
        if rows:
            conn.execute(insert(positions), rows)
