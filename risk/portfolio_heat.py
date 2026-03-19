"""V8: Portfolio heat tracking with correlation clustering.

Tracks total portfolio risk exposure as a single 'heat' metric and
clusters correlated positions to enforce concentration limits.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class PortfolioHeatTracker:
    """Track portfolio heat based on position correlation clustering."""

    def __init__(self):
        self._correlation_matrix: pd.DataFrame | None = None
        self._last_update: datetime | None = None
        self._current_heat: float = 0.0
        self._clusters: dict[int, list[str]] = {}

    def update_correlations(self, symbols: list[str]):
        """Update correlation matrix using recent returns.

        Uses yfinance for 20-day rolling returns correlation.
        Called periodically (not every scan).
        """
        if not symbols or not config.PORTFOLIO_HEAT_ENABLED:
            return

        try:
            import yfinance as yf
            from datetime import timedelta

            end = datetime.now()
            start = end - timedelta(days=config.HEAT_CORRELATION_LOOKBACK + 10)

            data = yf.download(symbols, start=start, end=end, progress=False)
            if data.empty:
                return

            closes = data["Close"] if "Close" in data.columns else data
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()

            returns = closes.pct_change().dropna()
            if len(returns) < 10:
                return

            self._correlation_matrix = returns.corr()
            self._last_update = datetime.now()
            self._update_clusters()

        except Exception as e:
            logger.warning(f"Portfolio heat correlation update failed: {e}")

    def _update_clusters(self):
        """Group symbols into clusters where pairwise correlation > threshold."""
        if self._correlation_matrix is None:
            return

        symbols = list(self._correlation_matrix.columns)
        visited = set()
        clusters = {}
        cluster_id = 0

        for sym in symbols:
            if sym in visited:
                continue

            cluster = [sym]
            visited.add(sym)

            for other in symbols:
                if other in visited:
                    continue
                try:
                    corr = self._correlation_matrix.loc[sym, other]
                    if abs(corr) > config.CLUSTER_CORRELATION_THRESHOLD:
                        cluster.append(other)
                        visited.add(other)
                except (KeyError, ValueError):
                    continue

            clusters[cluster_id] = cluster
            cluster_id += 1

        self._clusters = clusters

    def compute_heat(self, open_trades: dict, equity: float) -> float:
        """Compute current portfolio heat.

        Heat = sum(position_risk_pct * correlation_adjustment)
        Correlated positions amplify heat.
        """
        if not config.PORTFOLIO_HEAT_ENABLED or equity <= 0:
            return 0.0

        if not open_trades:
            self._current_heat = 0.0
            return 0.0

        total_heat = 0.0
        symbols_in_positions = set(open_trades.keys())

        for symbol, trade in open_trades.items():
            position_value = trade.entry_price * trade.qty
            position_risk_pct = position_value / equity

            # Find how many correlated positions exist
            corr_count = 1
            for cluster_symbols in self._clusters.values():
                if symbol in cluster_symbols:
                    corr_count = sum(1 for s in cluster_symbols if s in symbols_in_positions)
                    break

            # Amplify risk by sqrt of correlated position count
            correlation_adjustment = np.sqrt(corr_count)
            total_heat += position_risk_pct * correlation_adjustment

        self._current_heat = total_heat
        return total_heat

    def check_new_trade(self, symbol: str, position_value: float,
                        open_trades: dict, equity: float) -> tuple[bool, str]:
        """Check if a new trade would exceed heat limits.

        Returns (allowed, reason).
        """
        if not config.PORTFOLIO_HEAT_ENABLED:
            return True, ""

        current_heat = self.compute_heat(open_trades, equity)

        # Check total portfolio heat
        new_position_pct = position_value / equity if equity > 0 else 0
        projected_heat = current_heat + new_position_pct

        if projected_heat > config.PORTFOLIO_HEAT_MAX:
            return False, f"portfolio_heat={projected_heat:.2%}>{config.PORTFOLIO_HEAT_MAX:.0%}"

        # Check cluster heat
        for cluster_id, cluster_symbols in self._clusters.items():
            if symbol in cluster_symbols:
                cluster_heat = 0.0
                for s in cluster_symbols:
                    if s in open_trades:
                        t = open_trades[s]
                        cluster_heat += (t.entry_price * t.qty) / equity
                cluster_heat += new_position_pct

                if cluster_heat > config.CLUSTER_MAX_HEAT:
                    return False, f"cluster_heat={cluster_heat:.2%}>{config.CLUSTER_MAX_HEAT:.0%}"

        return True, ""

    @property
    def current_heat(self) -> float:
        return self._current_heat

    @property
    def clusters(self) -> dict[int, list[str]]:
        return self._clusters.copy()

    def reset_daily(self):
        """Reset daily state."""
        self._current_heat = 0.0
