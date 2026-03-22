"""T7-005: Portfolio-Level Black-Litterman Optimization.

Combines market equilibrium returns with strategy signal views to produce
optimal portfolio weights. Designed to run at the start of each trading
session for session-level rebalancing.

The Black-Litterman model:
    1. Market equilibrium returns:  Pi = delta * Sigma * w_mkt
    2. Strategy views matrix:       P * E[R] = q + epsilon
    3. Combined returns:            E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                                           * [(tau*Sigma)^-1 * Pi + P'*Omega^-1 * q]
    4. Optimize weights subject to constraints.

Gate: BLACK_LITTERMAN_ENABLED config flag.
Numpy-only implementation; scipy.optimize used if available for constrained opt.

Reference: Black & Litterman (1992), "Global Portfolio Optimization"
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

import config

logger = logging.getLogger(__name__)

# Conditional scipy import
_SCIPY_AVAILABLE = False
try:
    from scipy.optimize import minimize as scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    scipy_minimize = None  # type: ignore
    logger.debug("T7-005: scipy not available — using projection-based optimization")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ViewSpec:
    """A single investor view for the Black-Litterman model.

    Absolute view: "Asset i will return q%"
        -> P row has 1.0 at position i, q is the expected return
    Relative view: "Asset i will outperform Asset j by q%"
        -> P row has +1.0 at i, -1.0 at j, q is the spread
    """
    assets: list[int]           # Asset indices involved
    weights: list[float]        # View weights (sum to 0 for relative views)
    expected_return: float      # q: expected return (decimal, e.g. 0.02 = 2%)
    confidence: float = 1.0     # View confidence [0, 1]; lower = more uncertain


@dataclass
class BLResult:
    """Result of Black-Litterman optimization."""
    weights: np.ndarray                 # Optimal portfolio weights
    expected_returns: np.ndarray        # BL combined expected returns
    covariance: np.ndarray              # Posterior covariance
    equilibrium_returns: np.ndarray     # Market-implied returns (Pi)
    symbols: list[str]                  # Asset symbols (same order as weights)
    risk_aversion: float                # Delta parameter used
    tau: float                          # Uncertainty scaling parameter
    n_views: int                        # Number of views incorporated
    timestamp: datetime | None = None
    optimization_method: str = ""       # "scipy" or "projection"


@dataclass
class SectorConstraint:
    """Constraint on sector-level weight."""
    sector: str
    max_weight: float = 0.30    # Max weight in this sector
    min_weight: float = 0.0     # Min weight


# ---------------------------------------------------------------------------
# Black-Litterman Optimizer
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer:
    """Portfolio-level Black-Litterman optimization engine.

    Combines market equilibrium returns with strategy-derived views to
    produce optimal weights subject to constraints:
    - Long-only (no short positions)
    - Sector limits
    - Factor exposure limits
    - Maximum single-asset weight

    Usage:
        bl = BlackLittermanOptimizer()
        bl.set_universe(["AAPL", "MSFT", "GOOG", ...])
        bl.set_market_data(covariance=cov_matrix, market_caps=caps)
        bl.add_view(ViewSpec(assets=[0], weights=[1.0], expected_return=0.02))
        result = bl.optimize()
        # result.weights -> optimal allocation
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        max_single_weight: float = 0.15,
        sector_constraints: list[SectorConstraint] | None = None,
    ):
        self._enabled = getattr(config, "BLACK_LITTERMAN_ENABLED", False)
        self._delta = risk_aversion       # Risk aversion parameter
        self._tau = tau                    # Uncertainty scaling (typically 0.01-0.10)
        self._max_single = max_single_weight
        self._sector_constraints = sector_constraints or []

        # Universe
        self._symbols: list[str] = []
        self._n_assets: int = 0

        # Market data
        self._cov: np.ndarray | None = None           # Sigma (n x n)
        self._market_weights: np.ndarray | None = None # w_mkt (n,)
        self._market_caps: np.ndarray | None = None

        # Views
        self._views: list[ViewSpec] = []

        # Results
        self._last_result: BLResult | None = None
        self._lock = threading.Lock()

        if not self._enabled:
            logger.info("T7-005: Black-Litterman disabled (BLACK_LITTERMAN_ENABLED=False)")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_universe(self, symbols: list[str]):
        """Set the asset universe."""
        self._symbols = list(symbols)
        self._n_assets = len(symbols)
        self._views.clear()
        logger.debug(f"T7-005: Universe set to {self._n_assets} assets")

    def set_market_data(
        self,
        covariance: np.ndarray,
        market_caps: np.ndarray | None = None,
        market_weights: np.ndarray | None = None,
    ):
        """Set market data for equilibrium return computation.

        Args:
            covariance: (n, n) covariance matrix of asset returns.
            market_caps: (n,) market capitalizations (used to derive weights).
            market_weights: (n,) market-cap weights (alternative to caps).
        """
        n = self._n_assets
        if covariance.shape != (n, n):
            raise ValueError(f"Covariance shape {covariance.shape} != ({n}, {n})")

        self._cov = covariance.copy()
        self._market_caps = market_caps

        if market_weights is not None:
            self._market_weights = market_weights / market_weights.sum()
        elif market_caps is not None:
            total_cap = market_caps.sum()
            if total_cap > 0:
                self._market_weights = market_caps / total_cap
            else:
                self._market_weights = np.ones(n) / n
        else:
            # Equal weight as default
            self._market_weights = np.ones(n) / n

    def add_view(self, view: ViewSpec):
        """Add a strategy view."""
        self._views.append(view)

    def clear_views(self):
        """Clear all views for a fresh session."""
        self._views.clear()

    def add_signal_views(
        self,
        signals: dict[str, float],
        base_confidence: float = 0.5,
    ):
        """Convert strategy signals to Black-Litterman views.

        Each signal is an expected return (positive = bullish, negative = bearish).
        Translates to absolute views on individual assets.

        Args:
            signals: Dict of symbol -> expected return (decimal).
            base_confidence: Base confidence level for views [0, 1].
        """
        for symbol, expected_ret in signals.items():
            if symbol not in self._symbols:
                continue
            idx = self._symbols.index(symbol)
            # Confidence scales with signal magnitude
            confidence = min(1.0, base_confidence * (1.0 + abs(expected_ret) * 10.0))
            self._views.append(ViewSpec(
                assets=[idx],
                weights=[1.0],
                expected_return=expected_ret,
                confidence=confidence,
            ))

    # ------------------------------------------------------------------
    # Core: Black-Litterman computation
    # ------------------------------------------------------------------

    def compute_equilibrium_returns(self) -> np.ndarray:
        """Compute market-implied equilibrium returns: Pi = delta * Sigma * w_mkt."""
        if self._cov is None or self._market_weights is None:
            raise ValueError("Market data not set. Call set_market_data first.")
        return self._delta * self._cov @ self._market_weights

    def compute_bl_returns(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute Black-Litterman combined returns and posterior covariance.

        Returns:
            (expected_returns, posterior_covariance)
        """
        if self._cov is None:
            raise ValueError("Covariance not set")

        pi = self.compute_equilibrium_returns()
        n = self._n_assets
        tau_sigma = self._tau * self._cov

        if not self._views:
            # No views: return equilibrium
            return pi, tau_sigma

        # Build P matrix (k views x n assets) and q vector (k,)
        k = len(self._views)
        P = np.zeros((k, n))
        q = np.zeros(k)
        omega_diag = np.zeros(k)  # Diagonal uncertainty matrix

        for i, view in enumerate(self._views):
            for asset_idx, weight in zip(view.assets, view.weights):
                if 0 <= asset_idx < n:
                    P[i, asset_idx] = weight
            q[i] = view.expected_return

            # Omega: view uncertainty proportional to (1 - confidence)
            # Higher confidence = lower uncertainty
            view_var = P[i:i+1] @ tau_sigma @ P[i:i+1].T
            omega_diag[i] = float(view_var) * max(0.01, (1.0 / max(0.01, view.confidence) - 1.0))

        Omega = np.diag(omega_diag)

        # Black-Litterman formula:
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*q]
        try:
            tau_sigma_inv = np.linalg.inv(tau_sigma)
        except np.linalg.LinAlgError:
            tau_sigma_inv = np.linalg.pinv(tau_sigma)

        try:
            omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            omega_inv = np.diag(1.0 / np.maximum(omega_diag, 1e-10))

        # Posterior precision
        precision = tau_sigma_inv + P.T @ omega_inv @ P

        try:
            posterior_cov = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            posterior_cov = np.linalg.pinv(precision)

        # Posterior mean
        bl_returns = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ q)

        return bl_returns, posterior_cov

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self) -> BLResult:
        """Run full Black-Litterman optimization.

        Computes BL returns, then optimizes weights subject to:
        - Long-only constraint (weights >= 0)
        - Weights sum to 1
        - Max single-asset weight
        - Sector constraints (if configured)

        Returns:
            BLResult with optimal weights and diagnostics.
        """
        if not self._enabled:
            # Return equal weights when disabled
            n = max(1, self._n_assets)
            return BLResult(
                weights=np.ones(n) / n,
                expected_returns=np.zeros(n),
                covariance=np.eye(n),
                equilibrium_returns=np.zeros(n),
                symbols=self._symbols,
                risk_aversion=self._delta,
                tau=self._tau,
                n_views=0,
                timestamp=datetime.now(config.ET),
                optimization_method="disabled",
            )

        if self._cov is None:
            raise ValueError("Cannot optimize: market data not set")

        # Compute BL returns
        bl_returns, posterior_cov = self.compute_bl_returns()
        pi = self.compute_equilibrium_returns()

        # Optimize
        if _SCIPY_AVAILABLE:
            weights = self._optimize_scipy(bl_returns, self._cov)
            method = "scipy"
        else:
            weights = self._optimize_projection(bl_returns, self._cov)
            method = "projection"

        result = BLResult(
            weights=weights,
            expected_returns=bl_returns,
            covariance=posterior_cov,
            equilibrium_returns=pi,
            symbols=list(self._symbols),
            risk_aversion=self._delta,
            tau=self._tau,
            n_views=len(self._views),
            timestamp=datetime.now(config.ET),
            optimization_method=method,
        )

        with self._lock:
            self._last_result = result

        # Log summary
        nonzero = np.sum(weights > 0.001)
        logger.info(
            f"T7-005: BL optimization complete ({method}): "
            f"{nonzero}/{self._n_assets} assets allocated, "
            f"{len(self._views)} views, "
            f"max weight={np.max(weights):.2%}, "
            f"expected return={weights @ bl_returns:.4%}"
        )

        return result

    def _optimize_scipy(self, expected_returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Optimize using scipy.optimize.minimize (SLSQP)."""
        n = len(expected_returns)

        def neg_utility(w):
            # Mean-variance utility: E[R] - (delta/2) * w' Sigma w
            port_return = w @ expected_returns
            port_risk = w @ cov @ w
            return -(port_return - 0.5 * self._delta * port_risk)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]

        # Sector constraints
        sector_map = getattr(config, "SECTOR_MAP", {})
        for sc in self._sector_constraints:
            indices = [i for i, s in enumerate(self._symbols) if sector_map.get(s) == sc.sector]
            if indices:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, idx=indices, mx=sc.max_weight: mx - sum(w[j] for j in idx),
                })
                if sc.min_weight > 0:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, idx=indices, mn=sc.min_weight: sum(w[j] for j in idx) - mn,
                    })

        # Bounds: long-only, max single weight
        bounds = [(0.0, self._max_single) for _ in range(n)]

        # Initial guess: market weights or equal
        w0 = self._market_weights if self._market_weights is not None else np.ones(n) / n

        try:
            result = scipy_minimize(
                neg_utility,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.success:
                weights = result.x
                # Clean up tiny negatives from numerical issues
                weights = np.maximum(weights, 0.0)
                weights /= weights.sum() if weights.sum() > 0 else 1.0
                return weights
            else:
                logger.warning(f"T7-005: scipy optimization did not converge: {result.message}")
        except Exception as e:
            logger.warning(f"T7-005: scipy optimization failed: {e}")

        # Fallback to projection
        return self._optimize_projection(expected_returns, cov)

    def _optimize_projection(self, expected_returns: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Optimize using iterative projection (no scipy required).

        Uses the analytical mean-variance solution followed by projection
        onto the constraint set (long-only, sum-to-1, max weight).
        """
        n = len(expected_returns)

        try:
            # Unconstrained mean-variance optimal: w* = (1/delta) * Sigma^-1 * mu
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        w_raw = (1.0 / self._delta) * cov_inv @ expected_returns

        # Project onto constraints iteratively (Dykstra's algorithm simplified)
        w = w_raw.copy()
        for _ in range(100):  # Max iterations
            # 1. Long-only: clamp negatives to zero
            w = np.maximum(w, 0.0)

            # 2. Max single weight
            w = np.minimum(w, self._max_single)

            # 3. Sum to 1
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(n) / n
                break

            # Check convergence
            if np.all(w >= 0) and np.all(w <= self._max_single) and abs(w.sum() - 1.0) < 1e-8:
                break

        return w

    # ------------------------------------------------------------------
    # Session rebalancing
    # ------------------------------------------------------------------

    def compute_rebalance_trades(
        self,
        current_weights: dict[str, float],
        portfolio_value: float,
        min_trade_value: float = 100.0,
    ) -> dict[str, float]:
        """Compute trades needed to rebalance to BL-optimal weights.

        Args:
            current_weights: Dict of symbol -> current weight.
            portfolio_value: Total portfolio value in dollars.
            min_trade_value: Minimum trade size to execute.

        Returns:
            Dict of symbol -> dollar amount to trade (positive = buy, negative = sell).
        """
        if self._last_result is None:
            logger.warning("T7-005: No BL result available for rebalancing")
            return {}

        trades = {}
        result = self._last_result

        for i, symbol in enumerate(result.symbols):
            target_weight = result.weights[i]
            current_weight = current_weights.get(symbol, 0.0)
            delta_weight = target_weight - current_weight
            trade_value = delta_weight * portfolio_value

            if abs(trade_value) >= min_trade_value:
                trades[symbol] = round(trade_value, 2)

        if trades:
            total_buy = sum(v for v in trades.values() if v > 0)
            total_sell = sum(v for v in trades.values() if v < 0)
            logger.info(
                f"T7-005: Rebalance plan: {len(trades)} trades, "
                f"buy ${total_buy:+,.0f}, sell ${total_sell:+,.0f}"
            )

        return trades

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> BLResult | None:
        with self._lock:
            return self._last_result

    @property
    def status(self) -> dict:
        with self._lock:
            result = self._last_result
            return {
                "enabled": self._enabled,
                "scipy_available": _SCIPY_AVAILABLE,
                "universe_size": self._n_assets,
                "n_views": len(self._views),
                "last_optimized": result.timestamp.isoformat() if result and result.timestamp else None,
                "optimization_method": result.optimization_method if result else None,
                "n_allocated": int(np.sum(result.weights > 0.001)) if result else 0,
                "max_weight": f"{np.max(result.weights):.2%}" if result else None,
            }
