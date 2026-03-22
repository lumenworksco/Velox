"""RISK-001: Multi-factor risk model for portfolio risk decomposition.

Decomposes portfolio risk into systematic (factor-driven) and idiosyncratic
components using rolling OLS regression against factor return proxies.

Factors:
    - Market (beta to SPY)
    - Size (SMB: IWM - SPY)
    - Value (HML: IWD - IWF)
    - Momentum (UMD: MTUM proxy)
    - Volatility (realized vol regime)
    - Sector (per-GICS via sector ETFs)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Factor return proxy tickers
FACTOR_PROXIES = {
    "market": "SPY",
    "size": ("IWM", "SPY"),       # IWM - SPY
    "value": ("IWD", "IWF"),      # IWD - IWF
    "momentum": "MTUM",
}

# Sector ETF tickers for sector factor exposures
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Default factor exposure limits (absolute value)
DEFAULT_FACTOR_LIMITS = {
    "market": 0.60,      # Max net market beta
    "size": 0.40,
    "value": 0.40,
    "momentum": 0.50,
    "volatility": 0.30,
}

# Per-sector max gross exposure as fraction of portfolio
DEFAULT_SECTOR_LIMIT = 0.35

ROLLING_WINDOW = 60  # Trading days for rolling regression


@dataclass
class FactorExposure:
    """Single factor exposure with metadata."""
    factor: str
    exposure: float
    t_stat: float = 0.0
    limit: float = 0.0
    breached: bool = False


@dataclass
class RiskDecomposition:
    """Systematic vs. idiosyncratic risk breakdown."""
    total_risk: float = 0.0           # Portfolio vol (annualized)
    systematic_risk: float = 0.0      # Factor-explained vol
    idiosyncratic_risk: float = 0.0   # Residual vol
    r_squared: float = 0.0           # Proportion explained by factors
    factor_contributions: dict = field(default_factory=dict)  # factor -> vol contribution


class FactorRiskModel:
    """Multi-factor risk model using rolling regression against factor proxies.

    Computes factor exposures (betas) for each position via 60-day rolling
    OLS, then aggregates to portfolio level for risk decomposition and
    limit checking.
    """

    def __init__(
        self,
        rolling_window: int = ROLLING_WINDOW,
        factor_limits: dict[str, float] | None = None,
        sector_limit: float = DEFAULT_SECTOR_LIMIT,
    ):
        self._rolling_window = rolling_window
        self._factor_limits = factor_limits or DEFAULT_FACTOR_LIMITS.copy()
        self._sector_limit = sector_limit

        # Cache: symbol -> {factor -> beta}
        self._exposure_cache: dict[str, dict[str, float]] = {}
        self._cache_timestamp: datetime | None = None
        self._last_decomposition: RiskDecomposition | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_factor_exposures(
        self,
        symbol: str,
        returns_data: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Compute factor exposures for a single symbol via OLS regression.

        V11.2 convenience wrapper: runs rolling OLS of symbol returns on factor
        return proxies and returns the betas.

        Args:
            symbol: Ticker symbol.
            returns_data: symbol -> daily return series (must include factor proxies).

        Returns:
            Dict of factor name -> beta (exposure). Returns zeros if data insufficient.
        """
        sym_returns = returns_data.get(symbol)
        if sym_returns is None or len(sym_returns) < self._rolling_window:
            return {f: 0.0 for f in self._factor_limits}

        factor_returns = self._build_factor_returns(returns_data)
        if factor_returns.empty:
            return {f: 0.0 for f in self._factor_limits}

        betas = self._rolling_regression(sym_returns, factor_returns)
        with self._lock:
            self._exposure_cache[symbol] = betas
        return betas

    def portfolio_factor_risk(
        self,
        positions: dict[str, float],
        returns_data: dict[str, pd.Series],
    ) -> dict:
        """Compute portfolio-level factor exposures and flag concentrations.

        V11.2: Sum of (weight x exposure) per factor with 0.40 concentration flag.

        Args:
            positions: symbol -> dollar exposure.
            returns_data: symbol -> daily return series.

        Returns:
            Dict with keys: exposures (dict), violations (list), concentrated (bool).
        """
        exposures = self.compute_factor_exposures(positions, returns_data)
        violations = []
        for factor, exp in exposures.items():
            if factor.startswith("sector_"):
                continue
            if abs(exp) > 0.40:
                violations.append(
                    f"{factor} concentration={exp:+.3f} exceeds 0.40 limit"
                )
        return {
            "exposures": exposures,
            "violations": violations,
            "concentrated": len(violations) > 0,
        }

    def compute_factor_exposures(
        self,
        positions: dict[str, float],
        returns_data: dict[str, pd.Series],
    ) -> dict[str, float]:
        """Compute portfolio-level factor exposures (weighted sum of position betas).

        Args:
            positions: symbol -> dollar exposure (positive = long, negative = short)
            returns_data: symbol -> daily return series (including factor proxy tickers)

        Returns:
            Dict of factor name -> portfolio-level exposure (beta)
        """
        if not positions:
            return {f: 0.0 for f in self._factor_limits}

        total_abs_exposure = sum(abs(v) for v in positions.values())
        if total_abs_exposure < 1e-8:
            return {f: 0.0 for f in self._factor_limits}

        # Build factor return series
        factor_returns = self._build_factor_returns(returns_data)
        if factor_returns.empty:
            logger.warning("Could not build factor returns -- insufficient data")
            return {f: 0.0 for f in self._factor_limits}

        # Compute per-position betas
        portfolio_exposures: dict[str, float] = {f: 0.0 for f in factor_returns.columns}

        for symbol, dollar_exposure in positions.items():
            weight = dollar_exposure / total_abs_exposure
            sym_returns = returns_data.get(symbol)
            if sym_returns is None or len(sym_returns) < self._rolling_window:
                continue

            betas = self._rolling_regression(sym_returns, factor_returns)
            with self._lock:
                self._exposure_cache[symbol] = betas

            for factor, beta in betas.items():
                portfolio_exposures[factor] = portfolio_exposures.get(factor, 0.0) + weight * beta

        # Add sector exposures
        sector_exposures = self._compute_sector_exposures(positions)
        portfolio_exposures.update(sector_exposures)

        with self._lock:
            self._cache_timestamp = datetime.now(config.ET)

        logger.info(
            f"Factor exposures: "
            + " ".join(f"{k}={v:+.3f}" for k, v in portfolio_exposures.items()
                       if not k.startswith("sector_"))
        )
        return portfolio_exposures

    def decompose_risk(
        self,
        positions: dict[str, float],
        returns_data: dict[str, pd.Series] | None = None,
        portfolio_returns: pd.Series | None = None,
    ) -> tuple[float, float]:
        """Decompose portfolio risk into systematic and idiosyncratic components.

        Args:
            positions: symbol -> dollar exposure
            returns_data: symbol -> daily return series (including factor proxies)
            portfolio_returns: Pre-computed portfolio return series (optional)

        Returns:
            (systematic_risk, idiosyncratic_risk) as annualized volatilities
        """
        if not positions:
            return 0.0, 0.0

        # Compute portfolio returns if not provided
        if portfolio_returns is None and returns_data is not None:
            portfolio_returns = self._compute_portfolio_returns(positions, returns_data)

        if portfolio_returns is None or len(portfolio_returns) < self._rolling_window:
            logger.warning("Insufficient data for risk decomposition")
            return 0.0, 0.0

        # Build factor returns
        factor_returns = self._build_factor_returns(
            {k: v for k, v in (returns_data or {}).items()}
        )
        if factor_returns.empty:
            total_vol = float(portfolio_returns.std() * np.sqrt(252))
            return 0.0, total_vol

        # Align dates
        common_idx = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_idx) < self._rolling_window:
            total_vol = float(portfolio_returns.std() * np.sqrt(252))
            return 0.0, total_vol

        y = portfolio_returns.loc[common_idx].values[-self._rolling_window:]
        X = factor_returns.loc[common_idx].values[-self._rolling_window:]

        # OLS with intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            total_vol = float(np.std(y) * np.sqrt(252))
            return 0.0, total_vol

        y_hat = X_with_const @ betas
        resid = y - y_hat

        total_var = float(np.var(y))
        systematic_var = float(np.var(y_hat))
        idiosyncratic_var = float(np.var(resid))

        # Annualize
        systematic_risk = float(np.sqrt(max(systematic_var, 0) * 252))
        idiosyncratic_risk = float(np.sqrt(max(idiosyncratic_var, 0) * 252))
        total_risk = float(np.sqrt(max(total_var, 0) * 252))

        r_squared = systematic_var / total_var if total_var > 1e-10 else 0.0

        # Factor contributions
        factor_names = list(factor_returns.columns)
        factor_contributions = {}
        factor_betas = betas[1:]  # Exclude intercept
        factor_cov = np.cov(X.T) if X.shape[1] > 1 else np.array([[np.var(X)]])

        for i, name in enumerate(factor_names):
            # Marginal contribution = beta_i * sum_j(beta_j * cov(f_i, f_j))
            if i < len(factor_betas) and i < factor_cov.shape[0]:
                marginal = factor_betas[i] * float(np.dot(factor_betas, factor_cov[i]))
                factor_contributions[name] = float(np.sqrt(max(abs(marginal) * 252, 0)))

        decomp = RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            r_squared=float(r_squared),
            factor_contributions=factor_contributions,
        )

        with self._lock:
            self._last_decomposition = decomp

        logger.info(
            f"Risk decomposition: total={total_risk:.2%} "
            f"systematic={systematic_risk:.2%} idio={idiosyncratic_risk:.2%} "
            f"R2={r_squared:.2%}"
        )
        return systematic_risk, idiosyncratic_risk

    def check_factor_limits(
        self,
        positions: dict[str, float],
        returns_data: dict[str, pd.Series] | None = None,
        exposures: dict[str, float] | None = None,
    ) -> list[str]:
        """Check if current factor exposures exceed configured limits.

        Args:
            positions: symbol -> dollar exposure
            returns_data: symbol -> daily return series (needed if exposures not cached)
            exposures: Pre-computed exposures (optional, avoids recomputation)

        Returns:
            List of violation description strings (empty = all OK)
        """
        if exposures is None:
            if returns_data is not None:
                exposures = self.compute_factor_exposures(positions, returns_data)
            else:
                # Use cached exposures if available
                with self._lock:
                    if not self._exposure_cache:
                        return []
                    # Rebuild from cache
                    total_abs = sum(abs(v) for v in positions.values())
                    if total_abs < 1e-8:
                        return []
                    exposures = {}
                    for sym, dollar_exp in positions.items():
                        weight = dollar_exp / total_abs
                        cached = self._exposure_cache.get(sym, {})
                        for factor, beta in cached.items():
                            exposures[factor] = exposures.get(factor, 0.0) + weight * beta

        violations = []

        # Check standard factor limits
        for factor, limit in self._factor_limits.items():
            exposure = abs(exposures.get(factor, 0.0))
            if exposure > limit:
                msg = (
                    f"Factor limit breach: {factor} exposure={exposure:+.3f} "
                    f"exceeds limit={limit:.3f}"
                )
                violations.append(msg)
                logger.warning(msg)

        # Check sector concentration limits
        for key, value in exposures.items():
            if key.startswith("sector_") and abs(value) > self._sector_limit:
                msg = (
                    f"Sector limit breach: {key} weight={value:.1%} "
                    f"exceeds limit={self._sector_limit:.1%}"
                )
                violations.append(msg)
                logger.warning(msg)

        return violations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_factor_returns(
        self, returns_data: dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Construct factor return series from proxy tickers."""
        factors = {}

        # Market factor: SPY returns
        spy = returns_data.get("SPY")
        if spy is not None and len(spy) >= self._rolling_window:
            factors["market"] = spy

        # Size factor: IWM - SPY
        iwm = returns_data.get("IWM")
        if iwm is not None and spy is not None:
            common = iwm.index.intersection(spy.index)
            if len(common) >= self._rolling_window:
                factors["size"] = iwm.loc[common] - spy.loc[common]

        # Value factor: IWD - IWF
        iwd = returns_data.get("IWD")
        iwf = returns_data.get("IWF")
        if iwd is not None and iwf is not None:
            common = iwd.index.intersection(iwf.index)
            if len(common) >= self._rolling_window:
                factors["value"] = iwd.loc[common] - iwf.loc[common]

        # Momentum factor: MTUM excess return over SPY
        mtum = returns_data.get("MTUM")
        if mtum is not None and spy is not None:
            common = mtum.index.intersection(spy.index)
            if len(common) >= self._rolling_window:
                factors["momentum"] = mtum.loc[common] - spy.loc[common]

        if not factors:
            return pd.DataFrame()

        df = pd.DataFrame(factors)
        df.dropna(inplace=True)
        return df

    def _rolling_regression(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Run OLS regression of asset returns on factor returns over rolling window.

        Returns dict of factor name -> beta coefficient.
        """
        common_idx = asset_returns.index.intersection(factor_returns.index)
        if len(common_idx) < self._rolling_window:
            return {col: 0.0 for col in factor_returns.columns}

        # Use most recent window
        common_idx = common_idx.sort_values()[-self._rolling_window:]
        y = asset_returns.loc[common_idx].values
        X = factor_returns.loc[common_idx].values

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            betas, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            return {col: 0.0 for col in factor_returns.columns}

        result = {}
        for i, col in enumerate(factor_returns.columns):
            result[col] = float(betas[i + 1])  # Skip intercept
        return result

    def _compute_sector_exposures(
        self, positions: dict[str, float]
    ) -> dict[str, float]:
        """Compute sector-level exposure as fraction of total gross exposure."""
        total_abs = sum(abs(v) for v in positions.values())
        if total_abs < 1e-8:
            return {}

        sector_exposure: dict[str, float] = {}
        for symbol, dollar_exp in positions.items():
            sector_etf = config.SECTOR_MAP.get(symbol, "unknown")
            sector_name = SECTOR_ETFS.get(sector_etf, sector_etf)
            key = f"sector_{sector_name}"
            sector_exposure[key] = sector_exposure.get(key, 0.0) + abs(dollar_exp) / total_abs

        return sector_exposure

    def _compute_portfolio_returns(
        self,
        positions: dict[str, float],
        returns_data: dict[str, pd.Series],
    ) -> pd.Series | None:
        """Compute a weighted portfolio return series from position weights."""
        total_abs = sum(abs(v) for v in positions.values())
        if total_abs < 1e-8:
            return None

        weighted_returns = None
        for symbol, dollar_exp in positions.items():
            weight = dollar_exp / total_abs
            sym_returns = returns_data.get(symbol)
            if sym_returns is None:
                continue
            if weighted_returns is None:
                weighted_returns = sym_returns * weight
            else:
                common = weighted_returns.index.intersection(sym_returns.index)
                weighted_returns = weighted_returns.loc[common] + sym_returns.loc[common] * weight

        return weighted_returns

    # ------------------------------------------------------------------
    # Status & diagnostics
    # ------------------------------------------------------------------

    @property
    def last_decomposition(self) -> RiskDecomposition | None:
        with self._lock:
            return self._last_decomposition

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "cached_symbols": len(self._exposure_cache),
                "cache_timestamp": (
                    self._cache_timestamp.isoformat() if self._cache_timestamp else None
                ),
                "rolling_window": self._rolling_window,
                "factor_limits": dict(self._factor_limits),
                "sector_limit": self._sector_limit,
                "last_r_squared": (
                    self._last_decomposition.r_squared
                    if self._last_decomposition else None
                ),
            }
