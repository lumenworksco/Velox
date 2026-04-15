"""V10 Database Models — SQLAlchemy Table definitions matching existing schema.

These models are designed to be compatible with both SQLite and PostgreSQL.
The existing database.py functions continue to work unchanged; these models
provide the foundation for the SQLAlchemy migration path.
"""

from sqlalchemy import (
    MetaData, Table, Column, Integer, Float, Text, String, Boolean,
    Index, DateTime,
)

metadata = MetaData()

trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(10), nullable=False),
    Column("strategy", String(30), nullable=False),
    Column("side", String(4), nullable=False),
    Column("entry_price", Float),
    Column("exit_price", Float),
    Column("qty", Float),
    Column("entry_time", Text),
    Column("exit_time", Text),
    Column("exit_reason", Text),
    Column("pnl", Float),
    Column("pnl_pct", Float),
    Index("idx_trades_symbol", "symbol"),
    Index("idx_trades_strategy", "strategy"),
    Index("idx_trades_exit_time", "exit_time"),
)

signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("symbol", String(10), nullable=False),
    Column("strategy", String(30), nullable=False),
    Column("signal_type", String(4), nullable=False),
    Column("acted_on", Integer, default=0),
    Column("skip_reason", Text),
    Index("idx_signals_timestamp", "timestamp"),
    Index("idx_signals_strategy", "strategy"),
)

open_positions = Table(
    "open_positions", metadata,
    Column("symbol", String(10), primary_key=True),
    Column("strategy", String(30), nullable=False),
    Column("side", String(4), nullable=False),
    Column("entry_price", Float),
    Column("qty", Float),
    Column("entry_time", Text),
    Column("take_profit", Float),
    Column("stop_loss", Float),
    Column("alpaca_order_id", Text),
    Column("hold_type", Text, default="day"),
    Column("time_stop", Text),
    Column("max_hold_date", Text),
    Column("pair_id", Text, default=""),
    Column("partial_exits", Integer, default=0),
    Column("highest_price_seen", Float, default=0.0),
    Column("lowest_price_seen", Float, default=0.0),
    Column("entry_atr", Float, default=0.0),
    # BUG-FIX (2026-04-15): real OU params for adaptive-exit math
    Column("entry_mu", Float, default=0.0),
    Column("entry_sigma", Float, default=0.0),
    Column("entry_half_life_hours", Float, default=0.0),
)

daily_snapshots = Table(
    "daily_snapshots", metadata,
    Column("date", Text, primary_key=True),
    Column("portfolio_value", Float),
    Column("cash", Float),
    Column("day_pnl", Float),
    Column("day_pnl_pct", Float),
    Column("total_trades", Integer),
    Column("win_rate", Float),
    Column("sharpe_rolling", Float),
)

backtest_results = Table(
    "backtest_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_date", Text),
    Column("strategy", String(30)),
    Column("total_return", Float),
    Column("annualized_return", Float),
    Column("sharpe_ratio", Float),
    Column("win_rate", Float),
    Column("profit_factor", Float),
    Column("max_drawdown", Float),
    Column("total_trades", Integer),
    Column("avg_hold_minutes", Float),
)

model_performance = Table(
    "model_performance", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("strategy", String(30), nullable=False),
    Column("train_samples", Integer),
    Column("test_precision", Float),
    Column("test_recall", Float),
    Column("test_f1", Float),
    Column("features_used", Text),
    Column("model_version", Text),
)

optimization_history = Table(
    "optimization_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("strategy", String(30), nullable=False),
    Column("old_params", Text),
    Column("new_params", Text),
    Column("old_sharpe", Float),
    Column("new_sharpe", Float),
    Column("applied", Integer, default=0),
)

allocation_history = Table(
    "allocation_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", Text, nullable=False),
    Column("weights", Text, nullable=False),
)

shadow_trades = Table(
    "shadow_trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(10), nullable=False),
    Column("strategy", String(30), nullable=False),
    Column("side", String(4), nullable=False),
    Column("entry_price", Float),
    Column("qty", Float),
    Column("entry_time", Text),
    Column("take_profit", Float),
    Column("stop_loss", Float),
    Column("time_stop", Text),
    Column("exit_price", Float),
    Column("exit_time", Text),
    Column("exit_reason", Text),
    Column("pnl", Float),
    Column("pnl_pct", Float),
    Column("status", Text, default="open"),
    Index("idx_shadow_status", "status"),
)


kalman_pairs = Table(
    "kalman_pairs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol1", Text, nullable=False),
    Column("symbol2", Text, nullable=False),
    Column("hedge_ratio", Float),
    Column("spread_mean", Float),
    Column("spread_std", Float),
    Column("correlation", Float),
    Column("coint_pvalue", Float),
    Column("half_life", Float),
    Column("sector_group", Text),
    Column("active", Integer, default=1),
    Column("last_updated", Text),
    Index("idx_kalman_active", "active"),
)

kelly_params = Table(
    "kelly_params", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("strategy", String(30), nullable=False),
    Column("win_rate", Float),
    Column("avg_win_loss", Float),
    Column("kelly_f", Float),
    Column("half_kelly_f", Float),
    Column("sample_size", Integer),
    Column("computed_at", Text),
    Index("idx_kelly_strategy", "strategy"),
)

consistency_log = Table(
    "consistency_log", metadata,
    Column("date", Text, primary_key=True),
    Column("consistency_score", Float),
    Column("pct_positive_days", Float),
    Column("sharpe", Float),
    Column("max_drawdown", Float),
    Column("vol_scalar_avg", Float),
    Column("beta_avg", Float),
)

execution_analytics = Table(
    "execution_analytics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("order_id", Text),
    Column("symbol", Text),
    Column("strategy", Text),
    Column("side", Text),
    Column("expected_price", Float),
    Column("filled_price", Float),
    Column("slippage_pct", Float),
    Column("submitted_at", Text),
    Column("filled_at", Text),
    Column("latency_ms", Integer),
    Column("qty_requested", Integer),
    Column("qty_filled", Integer),
    Column("fill_rate", Float),
    Index("idx_exec_strategy", "strategy"),
)

monte_carlo_results = Table(
    "monte_carlo_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Text, nullable=False),
    Column("var_95", Float),
    Column("var_99", Float),
    Column("cvar_95", Float),
    Column("cvar_99", Float),
    Column("horizon_days", Integer),
    Column("simulations", Integer),
    Column("computed_at", Text),
    Index("idx_mc_date", "date"),
)

ou_parameters = Table(
    "ou_parameters", metadata,
    Column("symbol", Text, primary_key=True),
    Column("date", Text, primary_key=True),
    Column("kappa", Float),
    Column("mu", Float),
    Column("sigma", Float),
    Column("half_life", Float),
    Column("hurst", Float),
    Column("adf_pvalue", Float),
)

watchdog_ping = Table(
    "_watchdog_ping", metadata,
    Column("ts", Text),
)


def create_all_tables(engine):
    """Create all tables that don't exist yet. Safe to call multiple times."""
    metadata.create_all(engine)
