"""SQLite database — replaces state.json for persistence and adds trade/signal logging."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_FILE, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Create all tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            entry_time TEXT,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            pnl_pct REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            acted_on INTEGER DEFAULT 0,
            skip_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS open_positions (
            symbol TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            qty REAL,
            entry_time TEXT,
            take_profit REAL,
            stop_loss REAL,
            alpaca_order_id TEXT,
            hold_type TEXT DEFAULT 'day',
            time_stop TEXT,
            max_hold_date TEXT,
            pair_id TEXT DEFAULT '',
            partial_exits INTEGER DEFAULT 0,
            highest_price_seen REAL DEFAULT 0.0,
            entry_atr REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            portfolio_value REAL,
            cash REAL,
            day_pnl REAL,
            day_pnl_pct REAL,
            total_trades INTEGER,
            win_rate REAL,
            sharpe_rolling REAL
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            strategy TEXT,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            avg_hold_minutes REAL
        );

        -- V3: ML model performance history
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            train_samples INTEGER,
            test_precision REAL,
            test_recall REAL,
            test_f1 REAL,
            features_used TEXT,
            model_version TEXT
        );

        -- V3: Parameter optimization history
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            old_params TEXT,
            new_params TEXT,
            old_sharpe REAL,
            new_sharpe REAL,
            applied INTEGER DEFAULT 0
        );

        -- V3: Capital allocation weight history
        CREATE TABLE IF NOT EXISTS allocation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            weights TEXT NOT NULL
        );

        -- V5: Shadow trades (paper-simulated trades for strategy evaluation)
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            qty REAL,
            entry_time TEXT,
            take_profit REAL,
            stop_loss REAL,
            time_stop TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            pnl_pct REAL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
        CREATE INDEX IF NOT EXISTS idx_signals_acted ON signals(acted_on);
        CREATE INDEX IF NOT EXISTS idx_shadow_strategy ON shadow_trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_shadow_entry_time ON shadow_trades(entry_time);

        -- V6: OU process parameters for mean reversion universe
        CREATE TABLE IF NOT EXISTS ou_parameters (
            symbol TEXT,
            date TEXT,
            kappa REAL,
            mu REAL,
            sigma REAL,
            half_life REAL,
            hurst REAL,
            adf_pvalue REAL,
            PRIMARY KEY (symbol, date)
        );

        -- V6: Kalman pairs for pairs trading
        CREATE TABLE IF NOT EXISTS kalman_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol1 TEXT NOT NULL,
            symbol2 TEXT NOT NULL,
            hedge_ratio REAL,
            spread_mean REAL,
            spread_std REAL,
            correlation REAL,
            coint_pvalue REAL,
            half_life REAL,
            sector_group TEXT,
            active INTEGER DEFAULT 1,
            last_updated TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kalman_active ON kalman_pairs(active);

        -- V6: Daily consistency metrics
        CREATE TABLE IF NOT EXISTS consistency_log (
            date TEXT PRIMARY KEY,
            consistency_score REAL,
            pct_positive_days REAL,
            sharpe REAL,
            max_drawdown REAL,
            vol_scalar_avg REAL,
            beta_avg REAL
        );

        -- V8: Kelly criterion parameters
        CREATE TABLE IF NOT EXISTS kelly_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            win_rate REAL,
            avg_win_loss REAL,
            kelly_f REAL,
            half_kelly_f REAL,
            sample_size INTEGER,
            computed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kelly_strategy ON kelly_params(strategy);
    """)
    conn.commit()

    # V4: Add new columns to open_positions if they don't exist (migration)
    try:
        cursor = conn.execute("PRAGMA table_info(open_positions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        v4_cols = {
            "pair_id": "TEXT DEFAULT ''",
            "partial_exits": "INTEGER DEFAULT 0",
            "highest_price_seen": "REAL DEFAULT 0.0",
            "entry_atr": "REAL DEFAULT 0.0",
        }
        for col, col_type in v4_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE open_positions ADD COLUMN {col} {col_type}")
                logger.info(f"Added column {col} to open_positions")
        conn.commit()
    except Exception as e:
        logger.warning(f"V4 schema migration note: {e}")

    logger.info("Database initialized")


# --- Trade Logging ---

def log_trade(symbol: str, strategy: str, side: str, entry_price: float,
              exit_price: float, qty: float, entry_time: datetime,
              exit_time: datetime, exit_reason: str, pnl: float, pnl_pct: float):
    """Log a completed trade."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO trades (symbol, strategy, side, entry_price, exit_price,
           qty, entry_time, exit_time, exit_reason, pnl, pnl_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, strategy, side, entry_price, exit_price, qty,
         entry_time.isoformat(), exit_time.isoformat(), exit_reason, pnl, pnl_pct),
    )
    conn.commit()


# --- Signal Logging ---

def log_signal(timestamp: datetime, symbol: str, strategy: str,
               signal_type: str, acted_on: bool, skip_reason: str = ""):
    """Log a signal (whether or not it was acted on)."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signals (timestamp, symbol, strategy, signal_type,
           acted_on, skip_reason) VALUES (?, ?, ?, ?, ?, ?)""",
        (timestamp.isoformat(), symbol, strategy, signal_type,
         1 if acted_on else 0, skip_reason),
    )
    conn.commit()


# --- Open Positions (replaces state.json) ---

def save_open_positions(open_trades: dict):
    """Replace all open_positions rows with current state."""
    conn = _get_conn()
    conn.execute("DELETE FROM open_positions")
    for symbol, trade in open_trades.items():
        conn.execute(
            """INSERT INTO open_positions (symbol, strategy, side, entry_price,
               qty, entry_time, take_profit, stop_loss, alpaca_order_id,
               hold_type, time_stop, max_hold_date,
               pair_id, partial_exits, highest_price_seen, entry_atr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, trade.strategy, trade.side, trade.entry_price,
             trade.qty, trade.entry_time.isoformat(), trade.take_profit,
             trade.stop_loss, trade.order_id,
             getattr(trade, 'hold_type', 'day'),
             trade.time_stop.isoformat() if trade.time_stop else None,
             trade.max_hold_date.isoformat() if getattr(trade, 'max_hold_date', None) else None,
             getattr(trade, 'pair_id', ''),
             getattr(trade, 'partial_exits', 0),
             getattr(trade, 'highest_price_seen', 0.0),
             getattr(trade, 'entry_atr', 0.0)),
        )
    conn.commit()


def load_open_positions() -> list[dict]:
    """Load open positions from DB. Returns list of dicts."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM open_positions").fetchall()
    return [dict(row) for row in rows]


# --- Daily Snapshots ---

def save_daily_snapshot(date: str, portfolio_value: float, cash: float,
                        day_pnl: float, day_pnl_pct: float,
                        total_trades: int, win_rate: float, sharpe_rolling: float):
    """Insert or replace daily snapshot."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_snapshots
           (date, portfolio_value, cash, day_pnl, day_pnl_pct,
            total_trades, win_rate, sharpe_rolling)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (date, portfolio_value, cash, day_pnl, day_pnl_pct,
         total_trades, win_rate, sharpe_rolling),
    )
    conn.commit()


# --- Analytics Queries ---

def get_recent_trades(days: int = 7) -> list[dict]:
    """Get trades from the last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time DESC",
        (cutoff,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_trades() -> list[dict]:
    """Get all trades ever."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM trades ORDER BY exit_time DESC").fetchall()
    return [dict(row) for row in rows]


def get_daily_snapshots(days: int = 30) -> list[dict]:
    """Get daily snapshots for the last N days."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_daily_pnl_series(days: int = 30) -> list[float]:
    """Get list of daily P&L percentages for analytics."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT day_pnl_pct FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["day_pnl_pct"] for row in reversed(rows)]


def get_portfolio_values(days: int = 30) -> list[float]:
    """Get list of portfolio values for drawdown calculation."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT portfolio_value FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["portfolio_value"] for row in reversed(rows)]


def get_signal_stats_today() -> dict:
    """Get today's signal statistics."""
    conn = _get_conn()
    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ?",
        (f"{today}%",),
    ).fetchone()["cnt"]
    acted = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ? AND acted_on = 1",
        (f"{today}%",),
    ).fetchone()["cnt"]
    return {"total": total, "acted": acted, "skipped": total - acted}


# --- Backtest Results ---

def save_backtest_result(run_date: str, strategy: str, total_return: float,
                         annualized_return: float, sharpe_ratio: float,
                         win_rate: float, profit_factor: float,
                         max_drawdown: float, total_trades: int,
                         avg_hold_minutes: float):
    """Save backtest results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO backtest_results
           (run_date, strategy, total_return, annualized_return, sharpe_ratio,
            win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_date, strategy, total_return, annualized_return, sharpe_ratio,
         win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes),
    )
    conn.commit()


# --- Migration ---

def migrate_from_json():
    """One-time migration from state.json to SQLite."""
    json_path = Path(config.STATE_FILE)
    if not json_path.exists():
        return

    try:
        data = json.loads(json_path.read_text())
        # Migrate open trades
        for symbol, td in data.get("open_trades", {}).items():
            conn = _get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (symbol, strategy, side, entry_price, qty, entry_time,
                    take_profit, stop_loss, alpaca_order_id, hold_type, time_stop)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, td["strategy"], td["side"], td["entry_price"],
                 td["qty"], td["entry_time"], td["take_profit"],
                 td["stop_loss"], td.get("order_id", ""), "day",
                 td.get("time_stop")),
            )
        conn = _get_conn()
        conn.commit()

        # Rename json file so migration doesn't run again
        backup = json_path.with_suffix(".json.bak")
        json_path.rename(backup)
        logger.info(f"Migrated state.json to SQLite, backup at {backup}")

    except Exception as e:
        logger.error(f"Migration from state.json failed: {e}")


# =============================================================================
# V3 ADDITIONS
# =============================================================================

# --- Strategy-specific trade queries ---

def get_recent_trades_by_strategy(strategy: str, days: int = 20) -> list[dict]:
    """Get trades for a specific strategy from the last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM trades WHERE strategy = ? AND exit_time >= ? ORDER BY exit_time",
        (strategy, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def get_signals_with_outcomes() -> list[dict]:
    """JOIN signals that were acted on with their trade outcomes for ML training.

    Returns rows with signal fields + trade pnl/pnl_pct (if trade exists).
    Only returns signals that were acted on (acted_on=1) and have a matching trade.
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            s.timestamp, s.symbol, s.strategy, s.signal_type,
            t.entry_price, t.exit_price, t.qty,
            t.entry_time, t.exit_time, t.exit_reason,
            t.pnl, t.pnl_pct,
            CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END as profitable
        FROM signals s
        INNER JOIN trades t
            ON s.symbol = t.symbol
            AND s.strategy = t.strategy
            AND s.acted_on = 1
            AND ABS(julianday(s.timestamp) - julianday(t.entry_time)) < 0.01
        ORDER BY s.timestamp
    """).fetchall()
    return [dict(row) for row in rows]


def get_trade_count_by_strategy(strategy: str) -> int:
    """Get total number of completed trades for a strategy."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE strategy = ?",
        (strategy,),
    ).fetchone()
    return row["cnt"]


# --- ML Model Performance ---

def log_model_performance(strategy: str, train_samples: int,
                          test_precision: float, test_recall: float,
                          test_f1: float, features_used: list[str],
                          model_version: str):
    """Log ML model training results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO model_performance
           (timestamp, strategy, train_samples, test_precision, test_recall,
            test_f1, features_used, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(config.ET).isoformat(), strategy, train_samples,
         test_precision, test_recall, test_f1,
         json.dumps(features_used), model_version),
    )
    conn.commit()


# --- Optimization History ---

def log_optimization(strategy: str, old_params: dict, new_params: dict,
                     old_sharpe: float, new_sharpe: float, applied: bool):
    """Log a parameter optimization run."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO optimization_history
           (timestamp, strategy, old_params, new_params, old_sharpe, new_sharpe, applied)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(config.ET).isoformat(), strategy,
         json.dumps(old_params), json.dumps(new_params),
         old_sharpe, new_sharpe, 1 if applied else 0),
    )
    conn.commit()


# --- Allocation History ---

def log_allocation_weights(weights: dict):
    """Log capital allocation weight change."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO allocation_history (timestamp, weights) VALUES (?, ?)",
        (datetime.now(config.ET).isoformat(), json.dumps(weights)),
    )
    conn.commit()


# --- Signal Statistics (extended for web dashboard) ---

def get_signals_by_date(date_str: str) -> list[dict]:
    """Get all signals for a specific date."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE timestamp LIKE ? ORDER BY timestamp",
        (f"{date_str}%",),
    ).fetchall()
    return [dict(row) for row in rows]


def get_signal_skip_reasons(days: int = 7) -> dict:
    """Get breakdown of signal skip reasons over last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT skip_reason, COUNT(*) as cnt
           FROM signals
           WHERE timestamp >= ? AND acted_on = 0 AND skip_reason != ''
           GROUP BY skip_reason
           ORDER BY cnt DESC""",
        (cutoff,),
    ).fetchall()
    return {row["skip_reason"]: row["cnt"] for row in rows}


def get_trades_paginated(limit: int = 100, offset: int = 0,
                         strategy: str | None = None) -> list[dict]:
    """Get trades with pagination and optional strategy filter."""
    conn = _get_conn()
    if strategy:
        rows = conn.execute(
            "SELECT * FROM trades WHERE strategy = ? ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (strategy, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


# =============================================================================
# V5 ADDITIONS — Shadow Trades
# =============================================================================

def log_shadow_entry(symbol, strategy, side, entry_price, qty, entry_time,
                     take_profit, stop_loss, time_stop=None):
    """Record a shadow trade entry (no real order submitted)."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO shadow_trades
           (symbol, strategy, side, entry_price, qty, entry_time,
            take_profit, stop_loss, time_stop)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, strategy, side, entry_price, qty,
         entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
         take_profit, stop_loss,
         time_stop.isoformat() if time_stop and hasattr(time_stop, 'isoformat') else str(time_stop) if time_stop else None),
    )
    conn.commit()


def get_open_shadow_trades():
    """Get all shadow trades that haven't been closed yet."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, symbol, strategy, side, entry_price, qty, entry_time, "
        "take_profit, stop_loss, time_stop "
        "FROM shadow_trades WHERE exit_time IS NULL"
    ).fetchall()
    cols = ["id", "symbol", "strategy", "side", "entry_price", "qty",
            "entry_time", "take_profit", "stop_loss", "time_stop"]
    return [dict(zip(cols, row)) for row in rows]


def close_shadow_trade(trade_id, exit_price, exit_time, exit_reason):
    """Close a shadow trade with simulated exit."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT entry_price, qty, side FROM shadow_trades WHERE id = ?",
        (trade_id,)
    ).fetchone()
    if not row:
        return
    entry_price, qty, side = row["entry_price"], row["qty"], row["side"]
    if side == "buy":
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0
    else:
        pnl = (entry_price - exit_price) * qty
        pnl_pct = (entry_price - exit_price) / entry_price if entry_price else 0
    conn.execute(
        """UPDATE shadow_trades
           SET exit_price = ?, exit_time = ?, exit_reason = ?, pnl = ?, pnl_pct = ?
           WHERE id = ?""",
        (exit_price,
         exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time),
         exit_reason, round(pnl, 2), round(pnl_pct, 4), trade_id),
    )
    conn.commit()


def get_shadow_performance(strategy=None, days=14):
    """Get shadow trade performance stats."""
    conn = _get_conn()
    query = """SELECT strategy, COUNT(*) as trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as total_pnl, AVG(pnl_pct) as avg_pnl_pct
               FROM shadow_trades WHERE exit_time IS NOT NULL
               AND entry_time >= datetime('now', ?)"""
    params = [f"-{days} days"]
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    query += " GROUP BY strategy"
    rows = conn.execute(query, params).fetchall()
    cols = ["strategy", "trades", "wins", "total_pnl", "avg_pnl_pct"]
    return [dict(zip(cols, row)) for row in rows]


# =============================================================================
# V5 ADDITIONS — Trade Analysis
# =============================================================================

def get_exit_reason_breakdown(days=7):
    """Return exit reason breakdown for recent trades."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT exit_reason, COUNT(*) as count,
           AVG(pnl) as avg_pnl, AVG(pnl_pct) as avg_pnl_pct
           FROM trades
           WHERE exit_time >= datetime('now', ?)
           AND exit_reason IS NOT NULL
           GROUP BY exit_reason
           ORDER BY count DESC""",
        (f"-{days} days",)
    ).fetchall()
    cols = ["exit_reason", "count", "avg_pnl", "avg_pnl_pct"]
    return [dict(zip(cols, row)) for row in rows]


def get_filter_block_summary(date=None):
    """Return skip reason counts for signals."""
    conn = _get_conn()
    if date is None:
        query = """SELECT skip_reason, COUNT(*) as count
                   FROM signals WHERE acted_on = 0 AND skip_reason IS NOT NULL
                   AND timestamp >= datetime('now', '-1 day')
                   GROUP BY skip_reason ORDER BY count DESC"""
        rows = conn.execute(query).fetchall()
    else:
        query = """SELECT skip_reason, COUNT(*) as count
                   FROM signals WHERE acted_on = 0 AND skip_reason IS NOT NULL
                   AND DATE(timestamp) = ?
                   GROUP BY skip_reason ORDER BY count DESC"""
        rows = conn.execute(query, (str(date),)).fetchall()
    return {row["skip_reason"]: row["count"] for row in rows}


# =============================================================================
# V6: OU Parameters
# =============================================================================

def save_ou_parameters(symbol: str, date: str, kappa: float, mu: float,
                       sigma: float, half_life: float, hurst: float,
                       adf_pvalue: float):
    """Save OU process parameters for a symbol."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO ou_parameters
           (symbol, date, kappa, mu, sigma, half_life, hurst, adf_pvalue)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, date, kappa, mu, sigma, half_life, hurst, adf_pvalue),
    )
    conn.commit()


def get_ou_parameters(symbol: str, date: str) -> dict | None:
    """Get OU parameters for a symbol on a specific date."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM ou_parameters WHERE symbol = ? AND date = ?",
        (symbol, date),
    ).fetchone()
    return dict(row) if row else None


def get_mr_universe(date: str) -> list[dict]:
    """Get all symbols that passed MR universe filter on a date."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM ou_parameters WHERE date = ? ORDER BY hurst ASC",
        (date,),
    ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# V6: Kalman Pairs
# =============================================================================

def save_kalman_pair(symbol1: str, symbol2: str, hedge_ratio: float,
                     spread_mean: float, spread_std: float, correlation: float,
                     coint_pvalue: float, half_life: float, sector_group: str):
    """Save or update a Kalman pair."""
    conn = _get_conn()
    # Check if pair exists
    existing = conn.execute(
        "SELECT id FROM kalman_pairs WHERE symbol1 = ? AND symbol2 = ? AND active = 1",
        (symbol1, symbol2),
    ).fetchone()

    now_str = __import__('datetime').datetime.now().isoformat()

    if existing:
        conn.execute(
            """UPDATE kalman_pairs SET hedge_ratio=?, spread_mean=?, spread_std=?,
               correlation=?, coint_pvalue=?, half_life=?, sector_group=?,
               last_updated=? WHERE id=?""",
            (hedge_ratio, spread_mean, spread_std, correlation, coint_pvalue,
             half_life, sector_group, now_str, existing['id']),
        )
    else:
        conn.execute(
            """INSERT INTO kalman_pairs
               (symbol1, symbol2, hedge_ratio, spread_mean, spread_std,
                correlation, coint_pvalue, half_life, sector_group, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol1, symbol2, hedge_ratio, spread_mean, spread_std,
             correlation, coint_pvalue, half_life, sector_group, now_str),
        )
    conn.commit()


def get_active_kalman_pairs() -> list[dict]:
    """Get all active Kalman pairs."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM kalman_pairs WHERE active = 1 ORDER BY coint_pvalue ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def deactivate_kalman_pair(pair_id: int):
    """Deactivate a Kalman pair."""
    conn = _get_conn()
    conn.execute("UPDATE kalman_pairs SET active = 0 WHERE id = ?", (pair_id,))
    conn.commit()


def deactivate_all_kalman_pairs():
    """Deactivate all pairs (before weekly re-selection)."""
    conn = _get_conn()
    conn.execute("UPDATE kalman_pairs SET active = 0")
    conn.commit()


# =============================================================================
# V6: Consistency Log
# =============================================================================

def save_consistency_log(date: str, consistency_score: float,
                         pct_positive: float, sharpe: float,
                         max_drawdown: float, vol_scalar_avg: float = 0.0,
                         beta_avg: float = 0.0):
    """Save daily consistency metrics."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO consistency_log
           (date, consistency_score, pct_positive_days, sharpe, max_drawdown,
            vol_scalar_avg, beta_avg)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (date, consistency_score, pct_positive, sharpe, max_drawdown,
         vol_scalar_avg, beta_avg),
    )
    conn.commit()


def get_consistency_log(days: int = 30) -> list[dict]:
    """Get recent consistency log entries."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM consistency_log ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [dict(r) for r in rows]


# --- V7 additions ---

def get_trades_by_strategy(strategy: str, days: int = 30) -> list[dict]:
    """Get closed trades for a specific strategy within the last N days."""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE strategy = ? AND exit_time > ? ORDER BY exit_time DESC",
        (strategy, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


def get_signals_by_strategy(strategy: str, days: int = 7) -> list[dict]:
    """Get signal records for a specific strategy within the last N days."""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE strategy = ? AND timestamp > ? ORDER BY timestamp DESC",
        (strategy, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# V8: Kelly Criterion Parameters
# =============================================================================

def save_kelly_params(strategy: str, win_rate: float, avg_win_loss: float,
                      kelly_f: float, half_kelly_f: float, sample_size: int):
    """Save Kelly criterion parameters for a strategy."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO kelly_params
           (strategy, win_rate, avg_win_loss, kelly_f, half_kelly_f, sample_size, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (strategy, win_rate, avg_win_loss, kelly_f, half_kelly_f, sample_size,
         datetime.now(config.ET).isoformat()),
    )
    conn.commit()
