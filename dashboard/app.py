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


# --- status bar ----------------------------------------------------------------

NY = "America/New_York"
STALE_MINUTES = 15  # same rule as the watchdog (live_trader/status.py)
# Status colors (good / warning / serious / critical); every pill also carries an icon and a label,
# so the color never carries the meaning alone.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
          "neutral": "#8a8a85"}
ICON = {"good": "●", "warning": "▲", "serious": "◆", "critical": "■", "neutral": "○"}

PILL_CSS = """
<style>
.stratos-pills {display:flex; flex-wrap:wrap; gap:8px; margin:-0.5rem 0 1rem 0;}
.stratos-pill {display:inline-flex; align-items:center; gap:6px; padding:3px 12px; border-radius:999px;
  border:1px solid rgba(128,128,128,0.35); font-size:0.85rem; line-height:1.4; white-space:nowrap;}
.stratos-pill .icon {font-size:0.8rem;}
.stratos-pill .muted {opacity:0.7;}
</style>
"""


def pill(kind: str, label: str, detail: str = "") -> str:
    extra = f' <span class="muted">· {detail}</span>' if detail else ""
    return (f'<span class="stratos-pill"><span class="icon" style="color:{STATUS[kind]}">{ICON[kind]}</span>'
            f"{label}{extra}</span>")


def _state(key: str):
    row = query("SELECT value FROM bot_state WHERE key = :key", key=key)
    if row.empty or not row.iloc[0]["value"]:
        return None
    try:
        return json.loads(row.iloc[0]["value"])
    except ValueError:
        return row.iloc[0]["value"]


def _ts(value):
    """A stored timestamp as UTC (SQLite returns them without a time zone; they're saved in UTC)."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    t = pd.Timestamp(value)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def _clock_time(t: pd.Timestamp, now: pd.Timestamp) -> str:
    local, today = t.tz_convert(NY), now.tz_convert(NY).date()
    text_time = local.strftime("%I:%M %p").lstrip("0")
    return text_time if local.date() == today else f"{local:%a} {text_time}"


def market_state(clock, now: pd.Timestamp):
    """(is_open, pill html) from the market clock the trader saved, or regular hours if there's none yet."""
    if clock:
        is_open, nopen, nclose = clock.get("is_open"), _ts(clock.get("next_open")), _ts(clock.get("next_close"))
        if (is_open and nclose is not None and now < nclose) or \
                (not is_open and nopen is not None and nclose is not None and nopen <= now < nclose):
            return True, pill("good", "Market open", f"closes {_clock_time(nclose, now)}")
        if nopen is not None and now < nopen:
            return False, pill("neutral", "Market closed", f"opens {_clock_time(nopen, now)}")
        return False, pill("neutral", "Market closed")
    local = now.tz_convert(NY)
    open_now = local.weekday() < 5 and (9, 30) <= (local.hour, local.minute) < (16, 0)
    return open_now, pill("good" if open_now else "neutral", "Market open" if open_now else "Market closed",
                          "regular hours, not yet confirmed by Stratos")


def last_run_pill(last, market_open: bool, now: pd.Timestamp) -> str:
    if last is None:
        return pill("neutral", "No runs yet")
    minutes = (now - last).total_seconds() / 60
    if not market_open:
        return pill("neutral", "Last run", _clock_time(last, now))
    ago = "just now" if minutes < 1 else f"{int(minutes)} min ago"
    if minutes > STALE_MINUTES:
        return pill("warning", "Last run", f"{ago}, should run every 5 min")
    return pill("good", "Last run", ago)


def first_trading_day(year: int, month: int):
    """First weekday of the month, skipping New Year's Day (and its Monday observance) and Labor Day."""
    from datetime import date, timedelta

    d = date(year, month, 1)
    while True:
        holiday = (month == 1 and (d.day == 1 or (d.day == 2 and d.weekday() == 0))) or \
                  (month == 9 and d.weekday() == 0 and d.day <= 7)
        if d.weekday() < 5 and not holiday:
            return d
        d += timedelta(days=1)


def next_rebalance_pill(settings, now: pd.Timestamp) -> str:
    if settings and settings.get("rebalance") not in (None, "monthly"):
        return pill("neutral", "Rebalance", f"every {settings['params'].get('every')} trading days")
    local = now.tz_convert(NY).date()
    month = local.strftime("%Y-%m")
    done = query("SELECT value FROM bot_state WHERE key LIKE 'last_rebalance:%' AND value = :m", m=month)
    if done.empty:
        first = first_trading_day(local.year, local.month)
        if local <= first:
            return pill("neutral", "Next rebalance", "today" if local == first else f"{first:%a, %b} {first.day}")
        return pill("warning", "Rebalance pending", "this month's hasn't run yet")
    y, m = (local.year + 1, 1) if local.month == 12 else (local.year, local.month + 1)
    nxt = first_trading_day(y, m)
    return pill("neutral", "Next rebalance", f"{nxt:%a, %b} {nxt.day}")


@st.fragment(run_every="60s")
def status_bar() -> None:
    now = pd.Timestamp.now(tz="UTC")
    settings = _state("dashboard:settings") or {}
    is_open, market = market_state(_state("dashboard:clock"), now)
    last = query("SELECT MAX(ts) AS ts FROM equity_history")
    last_ts = _ts(last.iloc[0]["ts"]) if not last.empty and last.iloc[0]["ts"] is not None else None

    if not settings.get("crash_switch"):
        mode = pill("neutral", "Crash switch off")
    else:
        m = query("SELECT value FROM bot_state WHERE key LIKE 'mode:%' ORDER BY updated_at DESC LIMIT 1")
        crash = not m.empty and m.iloc[0]["value"] == "crash"
        mode = pill("serious", "Crash mode") if crash else pill("good", "Normal mode")

    if _state("halt"):
        activity = pill("critical", "Halted", "circuit breaker")
    elif settings.get("trading_halted"):
        activity = pill("critical", "Halted", "kill switch on")
    elif settings.get("dry_run"):
        activity = pill("warning", "Dry run", "no orders sent")
    elif settings:
        activity = pill("good", "Trading active")
    else:
        activity = pill("neutral", "Status unknown", "waiting for the next run")

    pills = [market, last_run_pill(last_ts, is_open, now), mode, activity, next_rebalance_pill(settings, now)]
    st.markdown(PILL_CSS + '<div class="stratos-pills">' + "".join(pills) + "</div>", unsafe_allow_html=True)


# --- momentum ranking -------------------------------------------------------------

def _period(settings) -> str:
    lookback = (settings or {}).get("params", {}).get("lookback")
    if not lookback:
        return "Lookback"
    months = round(lookback / 21)
    return f"{months}-month" if 1 <= months <= 24 else f"{lookback}-day"


def ranking_status(row) -> str:
    cat, held = row["category"], bool(row["held"])
    if cat == "top":
        return "Held" if held else "Top pick · bought at the next rebalance"
    if held:
        return "Held · drops out at the next rebalance"
    if cat == "next":
        return f"Next up · {row['note']}" if row["note"] else "Next up"
    if cat == "negative":
        return "Excluded · return not positive"
    if cat == "no_data":
        return f"Not ranked · {row['note'] or 'not enough history'}"
    return ""


def ranking_section() -> None:
    snaps = query("SELECT ts, MAX(CASE WHEN rebalance THEN 1 ELSE 0 END) AS rebalance FROM rankings "
                  "GROUP BY ts ORDER BY ts DESC LIMIT 90")
    st.subheader("Momentum ranking")
    if snaps.empty:
        st.caption("No ranking saved yet. Stratos saves one each trading day on its last run, and at every rebalance.")
        return
    settings = _state("dashboard:settings") or {}
    now = pd.Timestamp.now(tz="UTC")
    snaps["when"] = pd.to_datetime(snaps["ts"], utc=True)
    labels = [t.tz_convert(NY).strftime("%a %b %d, %I:%M %p ET") + (" · rebalance" if r else "")
              for t, r in zip(snaps["when"], snaps["rebalance"])]
    choice = st.selectbox("Snapshot", range(len(labels)), format_func=lambda i: labels[i], label_visibility="collapsed")
    chosen = snaps.iloc[choice]
    rows = query("SELECT symbol, rank, momentum, meter, category, held, gap, note FROM rankings WHERE ts = :ts",
                 ts=chosen["ts"])
    rows["held"] = rows["held"].astype(bool)
    rows = rows.sort_values(["rank", "symbol"], na_position="last")
    top = rows[rows["category"] == "top"]
    rows["Status"] = [ranking_status(r) for _, r in rows.iterrows()]

    period = _period(settings)
    held = ", ".join(rows.loc[rows["held"], "symbol"]) or "nothing"
    nxt = rows[rows["category"] == "next"]
    closest = f"{nxt.iloc[0]['symbol']} ({nxt.iloc[0]['note']})" if not nxt.empty and nxt.iloc[0]["note"] else "none"
    ranked = int(rows["rank"].notna().sum())
    st.caption(f"Ranks {ranked} of {len(rows)} stocks by {period.lower()} return. "
               f"Stratos holds the top {settings.get('top') or len(top)} with a positive return. "
               f"Held: {held}. Closest to swapping in: {closest}. "
               "Momentum meter = percentile in the list (100 = strongest). Swaps only happen at a rebalance.")

    show_all = st.toggle("Show every stock", value=False)
    view = rows if show_all else rows[(rows["rank"] <= 10) | rows["held"]]
    table = pd.DataFrame({
        "#": view["rank"].astype("Int64"),
        "Symbol": view["symbol"],
        f"{period} return": view["momentum"] * 100,
        "Momentum": view["meter"],
        "Status": view["Status"],
    })
    try:  # one neutral hue for magnitude (newer Streamlit); red would read as "bad"
        meter = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f", color="blue")
    except TypeError:
        meter = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={f"{period} return": st.column_config.NumberColumn(format="%+.1f%%"), "Momentum": meter},
    )


def paper_tab() -> None:
    equity = query("SELECT ts, equity, cash, reserve FROM equity_history ORDER BY ts")
    if equity.empty:
        st.info(
            "No live-trader runs recorded yet. Run `python -m live_trader --dry-run --force` "
            "locally, or wait for the scheduled Lambda to fire during market hours."
        )
        return
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True).dt.tz_convert("America/New_York")
    equity["reserve"] = pd.to_numeric(equity["reserve"]).fillna(0.0)
    reserve = float(equity["reserve"].iloc[-1])
    whole_account = float(equity["equity"].iloc[-1])
    # With CAPITAL_RESERVE set, Stratos trades only what's above the reserve: show that budget,
    # counted from when the current reserve was set.
    equity = equity[(equity["reserve"] - reserve).abs() < 0.005].copy()
    equity["value"] = equity["equity"] - equity["reserve"]
    latest, first = equity.iloc[-1], equity.iloc[0]
    today = latest["ts"].date()
    before_today = equity[equity["ts"].dt.date < today]
    day_base = before_today.iloc[-1]["value"] if not before_today.empty else first["value"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading budget" if reserve > 0 else "Equity", money(latest["value"]))
    c2.metric("Today", pct(latest["value"] / day_base - 1))
    c3.metric("Since budget was set" if reserve > 0 else "Since first run", pct(latest["value"] / first["value"] - 1))
    c4.metric("Last update (ET)", latest["ts"].strftime("%b %d, %H:%M"))
    if reserve > 0:
        # "\$": a pair of dollar signs would otherwise be rendered as a math formula
        st.caption(f"Stratos trades only what's above a {money(reserve)} reserve (CAPITAL_RESERVE), "
                   f"as if that were the whole account. Whole paper account: {money(whole_account)}."
                   .replace("$", "\\$"))

    halt = query("SELECT value FROM bot_state WHERE key = 'halt'")
    if not halt.empty and halt.iloc[0]["value"]:
        try:
            info = json.loads(halt.iloc[0]["value"])
        except ValueError:
            info = {"reason": halt.iloc[0]["value"], "since": "unknown"}
        st.error(f"Trading is halted by the circuit breaker (since {info.get('since')}): {info.get('reason')}. "
                 "No orders are placed until it's resumed with `python -m live_trader --resume`.")

    mode = query("SELECT value, updated_at FROM bot_state WHERE key LIKE 'mode:%' ORDER BY updated_at DESC LIMIT 1")
    if not mode.empty:
        if mode.iloc[0]["value"] == "crash":
            st.error("Crash mode: the market looks crash-like, so the bot is holding its backup (safe havens or cash).")
        else:
            st.success("Normal mode: the crash detector sees normal market conditions.")

    st.subheader("Trading budget" if reserve > 0 else "Equity")
    st.line_chart(equity.set_index("ts")[["value"]].rename(columns={"value": "budget" if reserve > 0 else "equity"}),
                  height=280)

    ranking_section()

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Positions")
        pos = query("SELECT symbol, qty, avg_entry_price, market_value, unrealized_pl FROM positions ORDER BY symbol")
        if pos.empty:
            st.caption("All cash.")
        else:
            st.dataframe(pos, hide_index=True, width="stretch")
    with right:
        dec = query(
            "SELECT ts, symbol, action, order_qty, price, fast_ma, slow_ma, reason, dry_run "
            "FROM decisions ORDER BY ts DESC LIMIT 500"
        )
        # the toggle sits on the heading's line, so this table starts level with the Positions table
        head, toggle = st.columns([3, 2], vertical_alignment="bottom")
        head.subheader("Decisions")
        if dec.empty:
            st.caption("No decisions yet.")
            return
        with toggle:
            only_trades = st.toggle("Only buys, sells and errors", value=True)
        dec["ts"] = pd.to_datetime(dec["ts"], utc=True).dt.tz_convert("America/New_York")
        if only_trades:
            dec = dec[dec["action"].isin(["buy", "sell", "error"])]
        if dec.empty:
            st.caption("No buys, sells or errors yet.")
            return
        # sized to its rows like the Positions table (no empty rows), up to the old 320px
        st.dataframe(dec, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(dec)))


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
    status_bar()
    paper, backtests = st.tabs(["Paper account", "Backtests"])
    with paper:
        paper_tab()
    with backtests:
        backtest_tab()
