"""MON-002: Performance Metrics Pipeline — rolling risk/return metrics with alert triggers.

Computes and stores metrics hourly:
  - Rolling Sharpe (1-day, 1-week, 1-month, 3-month, inception)
  - Rolling Sortino, max drawdown, drawdown duration
  - Rolling win rate and profit factor per strategy
  - Rolling alpha and beta vs SPY
  - Rolling information ratio
  - Hit rate by signal confidence bucket
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all computed metrics."""

    timestamp: datetime

    # Sharpe by window
    sharpe_1d: float = 0.0
    sharpe_1w: float = 0.0
    sharpe_1m: float = 0.0
    sharpe_3m: float = 0.0
    sharpe_inception: float = 0.0

    # Sortino by window
    sortino_1d: float = 0.0
    sortino_1w: float = 0.0
    sortino_1m: float = 0.0
    sortino_3m: float = 0.0
    sortino_inception: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # Per-strategy performance
    strategy_win_rates: dict = field(default_factory=dict)
    strategy_profit_factors: dict = field(default_factory=dict)

    # Market-relative
    alpha_vs_spy: float = 0.0
    beta_vs_spy: float = 0.0
    information_ratio: float = 0.0

    # Signal quality
    hit_rate_by_confidence: dict = field(default_factory=dict)

    # Metadata
    total_trades: int = 0
    computation_time_ms: float = 0.0


@dataclass
class MetricThreshold:
    """Alert threshold for a single metric."""

    metric_name: str
    warning_level: float
    critical_level: float
    comparison: str = "below"  # "below" or "above"


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = [
    MetricThreshold("sharpe_1m", warning_level=0.3, critical_level=0.0, comparison="below"),
    MetricThreshold("sortino_1m", warning_level=0.5, critical_level=0.0, comparison="below"),
    MetricThreshold("max_drawdown", warning_level=-0.05, critical_level=-0.10, comparison="below"),
    MetricThreshold("current_drawdown", warning_level=-0.03, critical_level=-0.07, comparison="below"),
    MetricThreshold("drawdown_duration_days", warning_level=10, critical_level=20, comparison="above"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray, risk_free: float = 0.045,
                    periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = np.std(returns, ddof=1)
    if std < 1e-8:
        return 0.0
    return float((np.mean(excess) / std) * np.sqrt(periods_per_year))


def _compute_sortino(returns: np.ndarray, risk_free: float = 0.045,
                     periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 10.0
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else np.sqrt(np.mean(downside ** 2))
    if down_std < 1e-8:
        return 10.0
    return float((np.mean(excess) / down_std) * np.sqrt(periods_per_year))


def _compute_drawdown(equity_curve: np.ndarray) -> tuple[float, float, int]:
    """Return (max_drawdown, current_drawdown, drawdown_duration_days).

    All drawdown values are negative (e.g. -0.05 = 5% drawdown).
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd))
    current_dd = float(dd[-1])

    # Duration: how many bars since last peak
    duration = 0
    if current_dd < -1e-6:
        for i in range(len(dd) - 1, -1, -1):
            if dd[i] >= -1e-6:
                break
            duration += 1
    return max_dd, current_dd, duration


def _compute_alpha_beta(portfolio_returns: np.ndarray,
                        benchmark_returns: np.ndarray) -> tuple[float, float]:
    """OLS alpha and beta vs benchmark."""
    if len(portfolio_returns) < 5 or len(benchmark_returns) < 5:
        return 0.0, 1.0
    n = min(len(portfolio_returns), len(benchmark_returns))
    pr = portfolio_returns[-n:]
    br = benchmark_returns[-n:]
    cov = np.cov(pr, br)
    var_b = cov[1, 1]
    if var_b < 1e-12:
        return 0.0, 0.0
    beta = float(cov[0, 1] / var_b)
    alpha = float(np.mean(pr) - beta * np.mean(br)) * 252  # annualized
    return alpha, beta


def _compute_information_ratio(portfolio_returns: np.ndarray,
                               benchmark_returns: np.ndarray) -> float:
    """Annualized information ratio."""
    n = min(len(portfolio_returns), len(benchmark_returns))
    if n < 5:
        return 0.0
    active = portfolio_returns[-n:] - benchmark_returns[-n:]
    te = np.std(active, ddof=1)
    if te < 1e-8:
        return 0.0
    return float((np.mean(active) / te) * np.sqrt(252))


# ---------------------------------------------------------------------------
# MetricsPipeline
# ---------------------------------------------------------------------------

class MetricsPipeline:
    """Computes and stores rolling performance metrics.

    Usage:
        pipeline = MetricsPipeline()
        snapshot = pipeline.compute_metrics()
        history = pipeline.get_metric_history("sharpe_1m", lookback=30)
        alerts = pipeline.check_thresholds(snapshot)
    """

    def __init__(self, thresholds: list[MetricThreshold] | None = None,
                 alert_callback=None):
        """
        Args:
            thresholds: List of MetricThreshold objects. Uses DEFAULT_THRESHOLDS if None.
            alert_callback: Optional callable(level: str, message: str, source: str)
                            for alert integration.
        """
        self._thresholds = thresholds or list(DEFAULT_THRESHOLDS)
        self._alert_callback = alert_callback
        self._history: list[MetricsSnapshot] = []
        self._last_compute: float = 0.0
        self._min_interval_sec: float = 300  # don't recompute more often than 5 min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_metrics(self) -> MetricsSnapshot:
        """Compute a full metrics snapshot from trade history and equity curve.

        Returns a MetricsSnapshot. Never raises — returns a default snapshot on error.
        """
        start = time.monotonic()
        try:
            snapshot = self._compute_inner()
        except Exception as e:
            logger.error(f"MetricsPipeline.compute_metrics failed: {e}")
            snapshot = MetricsSnapshot(timestamp=datetime.now(config.ET))

        snapshot.computation_time_ms = (time.monotonic() - start) * 1000
        self._history.append(snapshot)
        # Keep last 720 snapshots (~30 days at hourly)
        if len(self._history) > 720:
            self._history = self._history[-720:]
        self._last_compute = time.monotonic()

        # Check thresholds and fire alerts
        try:
            self.check_thresholds(snapshot)
        except Exception as e:
            logger.error(f"MetricsPipeline threshold check failed: {e}")

        return snapshot

    def get_metric_history(self, metric_name: str, lookback: int = 30) -> list[tuple[datetime, float]]:
        """Return time-series of a specific metric from cached snapshots.

        Args:
            metric_name: Attribute name on MetricsSnapshot (e.g. "sharpe_1m").
            lookback: Number of most recent snapshots to return.

        Returns:
            List of (timestamp, value) tuples.
        """
        result = []
        for snap in self._history[-lookback:]:
            val = getattr(snap, metric_name, None)
            if val is not None and not isinstance(val, (dict,)):
                result.append((snap.timestamp, float(val)))
        return result

    def check_thresholds(self, snapshot: MetricsSnapshot) -> list[dict]:
        """Check snapshot against configured thresholds and fire alerts.

        Returns list of triggered alerts: [{"metric", "level", "value", "threshold"}]
        """
        triggered = []
        for th in self._thresholds:
            value = getattr(snapshot, th.metric_name, None)
            if value is None:
                continue

            level = None
            if th.comparison == "below":
                if value < th.critical_level:
                    level = "CRITICAL"
                elif value < th.warning_level:
                    level = "WARNING"
            else:  # "above"
                if value > th.critical_level:
                    level = "CRITICAL"
                elif value > th.warning_level:
                    level = "WARNING"

            if level:
                alert_info = {
                    "metric": th.metric_name,
                    "level": level,
                    "value": value,
                    "threshold": th.critical_level if level == "CRITICAL" else th.warning_level,
                }
                triggered.append(alert_info)

                msg = (f"Metric {th.metric_name} = {value:.4f} "
                       f"breached {level} threshold "
                       f"({th.critical_level if level == 'CRITICAL' else th.warning_level:.4f})")
                logger.warning(f"METRICS ALERT [{level}]: {msg}")

                if self._alert_callback:
                    try:
                        self._alert_callback(level, msg, "metrics_pipeline")
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

        return triggered

    @property
    def last_snapshot(self) -> Optional[MetricsSnapshot]:
        """Return the most recent snapshot, or None."""
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[MetricsSnapshot]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_inner(self) -> MetricsSnapshot:
        """Load data from database and compute all metrics."""
        import database

        now = datetime.now(config.ET)
        snapshot = MetricsSnapshot(timestamp=now)

        # Load closed trades
        trades = self._load_trades()
        if not trades:
            return snapshot

        snapshot.total_trades = len(trades)

        # Build daily returns from trade PnL
        daily_returns = self._build_daily_returns(trades)
        if len(daily_returns) == 0:
            return snapshot

        returns_arr = np.array(daily_returns, dtype=float)

        # Build equity curve (cumulative)
        equity_curve = np.cumprod(1.0 + returns_arr)

        # Sharpe by window
        snapshot.sharpe_1d = _compute_sharpe(returns_arr[-1:]) if len(returns_arr) >= 1 else 0.0
        snapshot.sharpe_1w = _compute_sharpe(returns_arr[-5:]) if len(returns_arr) >= 5 else 0.0
        snapshot.sharpe_1m = _compute_sharpe(returns_arr[-21:]) if len(returns_arr) >= 5 else 0.0
        snapshot.sharpe_3m = _compute_sharpe(returns_arr[-63:]) if len(returns_arr) >= 10 else 0.0
        snapshot.sharpe_inception = _compute_sharpe(returns_arr)

        # Sortino by window
        snapshot.sortino_1d = _compute_sortino(returns_arr[-1:]) if len(returns_arr) >= 1 else 0.0
        snapshot.sortino_1w = _compute_sortino(returns_arr[-5:]) if len(returns_arr) >= 5 else 0.0
        snapshot.sortino_1m = _compute_sortino(returns_arr[-21:]) if len(returns_arr) >= 5 else 0.0
        snapshot.sortino_3m = _compute_sortino(returns_arr[-63:]) if len(returns_arr) >= 10 else 0.0
        snapshot.sortino_inception = _compute_sortino(returns_arr)

        # Drawdown
        snapshot.max_drawdown, snapshot.current_drawdown, snapshot.drawdown_duration_days = (
            _compute_drawdown(equity_curve)
        )

        # Per-strategy metrics
        snapshot.strategy_win_rates, snapshot.strategy_profit_factors = (
            self._compute_strategy_metrics(trades)
        )

        # Alpha/Beta vs SPY
        spy_returns = self._load_spy_returns(len(returns_arr))
        if len(spy_returns) > 0:
            snapshot.alpha_vs_spy, snapshot.beta_vs_spy = _compute_alpha_beta(returns_arr, spy_returns)
            snapshot.information_ratio = _compute_information_ratio(returns_arr, spy_returns)

        # Hit rate by confidence bucket
        snapshot.hit_rate_by_confidence = self._compute_hit_rates_by_confidence()

        return snapshot

    def _load_trades(self) -> list[dict]:
        """Load closed trades from database. Never raises."""
        try:
            import database
            conn = database._get_conn()
            rows = conn.execute(
                "SELECT symbol, strategy, side, entry_price, exit_price, "
                "qty, entry_time, exit_time, pnl, pnl_pct "
                "FROM trades ORDER BY exit_time ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"MetricsPipeline: Failed to load trades: {e}")
            return []

    def _build_daily_returns(self, trades: list[dict]) -> list[float]:
        """Aggregate trade PnL into daily returns."""
        from collections import defaultdict
        daily_pnl: dict[str, float] = defaultdict(float)
        daily_capital: dict[str, float] = defaultdict(float)

        for t in trades:
            exit_time = t.get("exit_time", "")
            if not exit_time:
                continue
            day = str(exit_time)[:10]
            pnl = t.get("pnl", 0.0) or 0.0
            capital = abs((t.get("entry_price", 0.0) or 0.0) * (t.get("qty", 0.0) or 0.0))
            daily_pnl[day] += pnl
            daily_capital[day] += max(capital, 1.0)

        # Convert to return series sorted by date
        returns = []
        for day in sorted(daily_pnl.keys()):
            cap = daily_capital[day]
            if cap > 0:
                returns.append(daily_pnl[day] / cap)
            else:
                returns.append(0.0)
        return returns

    def _compute_strategy_metrics(self, trades: list[dict]) -> tuple[dict, dict]:
        """Compute per-strategy win rate and profit factor."""
        from collections import defaultdict
        wins: dict[str, int] = defaultdict(int)
        losses: dict[str, int] = defaultdict(int)
        gross_profit: dict[str, float] = defaultdict(float)
        gross_loss: dict[str, float] = defaultdict(float)

        for t in trades:
            strat = t.get("strategy", "UNKNOWN")
            pnl = t.get("pnl", 0.0) or 0.0
            if pnl > 0:
                wins[strat] += 1
                gross_profit[strat] += pnl
            else:
                losses[strat] += 1
                gross_loss[strat] += abs(pnl)

        all_strats = set(wins.keys()) | set(losses.keys())
        win_rates = {}
        profit_factors = {}

        for s in all_strats:
            total = wins[s] + losses[s]
            win_rates[s] = wins[s] / total if total > 0 else 0.0
            profit_factors[s] = (
                gross_profit[s] / gross_loss[s]
                if gross_loss[s] > 0 else 10.0
            )

        return win_rates, profit_factors

    def _load_spy_returns(self, n_days: int) -> np.ndarray:
        """Load recent SPY daily returns for alpha/beta calculation.

        Falls back to empty array if data unavailable.
        """
        try:
            import database
            conn = database._get_conn()
            # Check if we have a market_data or spy_returns table
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "spy_daily_returns" in tables:
                rows = conn.execute(
                    "SELECT daily_return FROM spy_daily_returns "
                    "ORDER BY date DESC LIMIT ?", (n_days,)
                ).fetchall()
                if rows:
                    return np.array([r[0] for r in reversed(rows)], dtype=float)

            return np.array([], dtype=float)
        except Exception as e:
            logger.debug(f"MetricsPipeline: Could not load SPY returns: {e}")
            return np.array([], dtype=float)

    def _compute_hit_rates_by_confidence(self) -> dict:
        """Compute win rate bucketed by signal confidence.

        Buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        """
        try:
            import database
            conn = database._get_conn()

            # Check if signals table has confidence column
            cols = [r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()]
            if "confidence" not in cols:
                return {}

            rows = conn.execute(
                "SELECT s.confidence, "
                "CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END AS win "
                "FROM signals s "
                "JOIN trades t ON s.symbol = t.symbol AND s.strategy = t.strategy "
                "AND s.acted_on = 1 "
                "WHERE s.confidence IS NOT NULL "
                "ORDER BY s.timestamp DESC "
                "LIMIT 1000"
            ).fetchall()

            if not rows:
                return {}

            buckets = {
                "0.0-0.2": {"wins": 0, "total": 0},
                "0.2-0.4": {"wins": 0, "total": 0},
                "0.4-0.6": {"wins": 0, "total": 0},
                "0.6-0.8": {"wins": 0, "total": 0},
                "0.8-1.0": {"wins": 0, "total": 0},
            }

            for r in rows:
                conf = float(r[0])
                win = int(r[1])
                if conf < 0.2:
                    b = "0.0-0.2"
                elif conf < 0.4:
                    b = "0.2-0.4"
                elif conf < 0.6:
                    b = "0.4-0.6"
                elif conf < 0.8:
                    b = "0.6-0.8"
                else:
                    b = "0.8-1.0"
                buckets[b]["total"] += 1
                buckets[b]["wins"] += win

            return {
                k: v["wins"] / v["total"] if v["total"] > 0 else 0.0
                for k, v in buckets.items()
                if v["total"] > 0
            }
        except Exception as e:
            logger.debug(f"MetricsPipeline: Could not compute hit rates: {e}")
            return {}
