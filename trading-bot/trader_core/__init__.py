"""Shared code imported by the backtester, the live trader and the dashboard.

Anything that must behave identically in backtest and live (the strategy,
the database schema, settings) lives here so the two can't drift apart.
"""
