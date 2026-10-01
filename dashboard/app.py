"""Streamlit dashboard. Reads the database only; it never calls Alpaca.

    streamlit run dashboard/app.py

Set DASHBOARD_PASSWORD to put a simple password prompt in front of it
(worth doing once it has a public App Runner URL).
"""
from __future__ import annotations

import hmac
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make trader_core importable

from trader_core import db  # noqa: E402

st.set_page_config(page_title="Stratos", layout="wide")


def check_password() -> bool:
    expected = os.environ.get("DASHBOARD_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return True
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, expected):
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False


@st.cache_resource
def get_engine():
    engine = db.get_engine(os.environ.get("DATABASE_URL", "sqlite:///trading.db"))
    db.init_db(engine)
    return engine


@st.cache_data(ttl=60)
def query(sql: str, **params) -> pd.DataFrame:
    engine = get_engine()
    if not inspect(engine).get_table_names():
        return pd.DataFrame()
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params)


def pct(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.2f}%"


def money(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"${v:,.2f}"


def paper_tab() -> None:
    equity = query("SELECT ts, equity, cash FROM equity_history ORDER BY ts")
    if equity.empty:
        st.info(
            "No live-trader runs recorded yet. Run `python -m live_trader --dry-run --force` "
            "locally, or wait for the scheduled Lambda to fire during market hours."
        )
        return
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True).dt.tz_convert("America/New_York")
    latest, first = equity.iloc[-1], equity.iloc[0]
    today = latest["ts"].date()
    before_today = equity[equity["ts"].dt.date < today]
    day_base = before_today.iloc[-1]["equity"] if not before_today.empty else first["equity"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", money(latest["equity"]))
    c2.metric("Today", pct(latest["equity"] / day_base - 1))
    c3.metric("Since first run", pct(latest["equity"] / first["equity"] - 1))
    c4.metric("Last update (ET)", latest["ts"].strftime("%b %d, %H:%M"))

    mode = query("SELECT value, updated_at FROM bot_state WHERE key LIKE 'mode:%' ORDER BY updated_at DESC LIMIT 1")
    if not mode.empty:
        if mode.iloc[0]["value"] == "crash":
            st.error("Crash mode: the market looks crash-like, so the bot is holding its backup (safe havens or cash).")
        else:
            st.success("Normal mode: the crash detector sees normal market conditions.")

    st.subheader("Equity")
    st.line_chart(equity.set_index("ts")[["equity"]], height=280)

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Positions")
        pos = query("SELECT symbol, qty, avg_entry_price, market_value, unrealized_pl FROM positions ORDER BY symbol")
        if pos.empty:
            st.caption("All cash.")
        else:
            st.dataframe(pos, hide_index=True, width="stretch")
    with right:
        st.subheader("Decisions")
        dec = query(
            "SELECT ts, symbol, action, order_qty, price, fast_ma, slow_ma, reason, dry_run "
            "FROM decisions ORDER BY ts DESC LIMIT 500"
        )
        if dec.empty:
            st.caption("No decisions yet.")
            return
        dec["ts"] = pd.to_datetime(dec["ts"], utc=True).dt.tz_convert("America/New_York")
        only_trades = st.toggle("Only buys, sells and errors", value=True)
        if only_trades:
            dec = dec[dec["action"].isin(["buy", "sell", "error"])]
        st.dataframe(dec, hide_index=True, width="stretch", height=320)


def backtest_tab() -> None:
    runs = query("SELECT * FROM backtest_runs ORDER BY id DESC")
    if runs.empty:
        st.info("No backtests yet. Run `python -m backtester --provider synthetic` to create one.")
        return

    def label(r) -> str:
        params = r["params"] if isinstance(r["params"], dict) else json.loads(r["params"])
        n = len(str(r["symbols"]).split(","))
        universe = r.get("universe") if isinstance(r.get("universe"), str) and r.get("universe") else f"{n} symbols"
        shown = {k: v for k, v in params.items() if k != "absolute" and v not in ("", None)}
        if not params.get("market"):
            shown.pop("market_window", None)
        settings = " ".join(f"{k} {v}" for k, v in shown.items())
        mode = r.get("signal_mode") if isinstance(r.get("signal_mode"), str) else "close"
        return (f"#{r['id']}  {r['strategy']} ({settings})  {universe}  {mode}  "
                f"{r['start_date']} to {r['end_date']}  ({r['data_provider']})")

    options = {label(r): int(r["id"]) for _, r in runs.iterrows()}
    run_id = options[st.selectbox("Run", list(options))]
    run = runs[runs["id"] == run_id].iloc[0]

    if run["data_provider"] == "synthetic":
        st.warning("This run used synthetic prices, not real market data.")
    universe = run.get("universe")
    if isinstance(universe, str) and universe.startswith("custom"):
        st.caption("Custom symbol list: if it was chosen recently, both the strategy and buy & hold "
                   "benefit from hindsight. Compare with a preset universe (see README).")

    def num(v):
        return "n/a" if v is None or pd.isna(v) else f"{v:.2f}"

    has_matched = "matched_return" in run and not pd.isna(run["matched_return"])
    table = pd.DataFrame(
        {
            "Strategy": [pct(run["total_return"]), pct(run["cagr"]), pct(run["max_drawdown"]), num(run["sharpe"])],
            "Buy & hold": [pct(run["benchmark_return"]), pct(run.get("benchmark_cagr")),
                           pct(run.get("benchmark_max_drawdown")), num(run.get("benchmark_sharpe"))],
        },
        index=["Total return", "Per year", "Max drawdown", "Sharpe"],
    )
    if has_matched:
        table["Same exposure"] = [pct(run["matched_return"]), pct(run["matched_cagr"]),
                                  pct(run["matched_max_drawdown"]), num(run.get("benchmark_sharpe"))]
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(table, width="stretch")
    with right:
        st.metric("Trades", int(run["num_trades"]))
        st.metric("Win rate", "n/a" if pd.isna(run["win_rate"]) else f"{run['win_rate'] * 100:.0f}%")
        if has_matched:
            st.metric("Average invested", f"{run['avg_exposure'] * 100:.0f}%")
    if has_matched:
        st.caption("Same exposure = buy & hold scaled down to the strategy's average amount invested, rest in cash. "
                   "If the strategy can't beat it, its timing isn't adding anything beyond holding less.")

    curve = query("SELECT * FROM backtest_equity WHERE run_id = :run_id ORDER BY ts", run_id=run_id)
    curve["ts"] = pd.to_datetime(curve["ts"])
    if "mode" in curve and curve["mode"].notna().any():
        share = (curve["mode"] == "crash").mean()
        switches = int((curve["mode"] != curve["mode"].shift()).sum() - 1)
        st.caption(f"Crash mode was on {share * 100:.0f}% of days ({switches} switches between normal and crash).")
    lines = curve.set_index("ts").rename(columns={"equity": "strategy", "benchmark_equity": "buy & hold",
                                                  "matched_equity": "same exposure"})
    lines = lines[[c for c in ("strategy", "buy & hold", "same exposure") if c in lines and lines[c].notna().any()]]
    st.subheader("Growth of the account")
    st.line_chart(lines, height=300)

    st.subheader("By year")
    yearly = []
    for year, chunk in lines.groupby(lines.index.year):
        before = lines[lines.index < chunk.index[0]]
        base = before.iloc[-1] if not before.empty else chunk.iloc[0]
        yearly.append({"year": str(year), **{c: chunk[c].iloc[-1] / base[c] - 1 for c in lines.columns}})
    st.dataframe(
        pd.DataFrame(yearly),
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(c, format="percent") for c in lines.columns},
    )

    st.subheader("Trades")
    trades = query("SELECT symbol, entry_ts, entry_price, exit_ts, exit_price, qty, pnl, return_pct "
                   "FROM backtest_trades WHERE run_id = :run_id ORDER BY entry_ts", run_id=run_id)
    money_col = st.column_config.NumberColumn(format="dollar")
    st.dataframe(
        trades,
        hide_index=True,
        width="stretch",
        height=300,
        column_config={
            "entry_price": money_col,
            "exit_price": money_col,
            "pnl": money_col,
            "return_pct": st.column_config.NumberColumn("return", format="percent"),
        },
    )

    with st.expander("All runs"):
        cols = ["id", "created_at", "strategy", "params", "universe", "signal_mode", "cash_rate",
                "start_date", "end_date", "total_return",
                "benchmark_return", "matched_return", "max_drawdown", "sharpe", "benchmark_sharpe",
                "num_trades", "data_provider"]
        st.dataframe(runs[[c for c in cols if c in runs]], hide_index=True, width="stretch")


if check_password():
    st.title("Stratos")
    paper, backtests = st.tabs(["Paper account", "Backtests"])
    with paper:
        paper_tab()
    with backtests:
        backtest_tab()
