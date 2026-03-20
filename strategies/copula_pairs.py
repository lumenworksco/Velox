"""Copula-Based Pairs Trading strategy (STRAT-005).

Fits marginal distributions to individual asset returns, transforms to
uniform via probability integral transform, then fits a copula (Clayton,
Gumbel, or Frank) to model the joint dependence structure.

Signals are generated when the conditional copula probability is extreme
(>95th or <5th percentile), indicating a divergence from the historical
dependence pattern.

Copula families:
    Clayton  -- lower-tail dependence (crash co-movement)
    Gumbel   -- upper-tail dependence (rally co-movement)
    Frank    -- symmetric dependence (no tail preference)

The best-fitting copula is selected by AIC on the training window.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from strategies.base import Signal

logger = logging.getLogger(__name__)

# Pair selection parameters
MIN_CORRELATION = 0.75
MIN_TRAINING_DAYS = 120
LOOKBACK_DAYS = 252

# Signal thresholds (copula CDF percentiles)
UPPER_THRESHOLD = 0.95  # Asset 1 rich relative to Asset 2
LOWER_THRESHOLD = 0.05  # Asset 1 cheap relative to Asset 2

# Position parameters
DEFAULT_TP_PCT = 0.02
DEFAULT_SL_PCT = 0.015
REFIT_INTERVAL_DAYS = 21  # Refit copula monthly

# Default pairs (can be overridden)
DEFAULT_PAIRS = [
    ("AAPL", "MSFT"),
    ("GOOGL", "META"),
    ("JPM", "GS"),
    ("XOM", "CVX"),
    ("V", "MA"),
    ("AMD", "NVDA"),
    ("UBER", "LYFT"),
]


@dataclass
class CopulaFit:
    """Fitted copula model for a pair."""
    symbol_a: str
    symbol_b: str
    family: str                # "clayton", "gumbel", "frank"
    theta: float               # Copula parameter
    marginal_a: str            # Distribution family for asset A
    marginal_b: str            # Distribution family for asset B
    marginal_params_a: Tuple   # scipy distribution parameters
    marginal_params_b: Tuple
    aic: float = 0.0
    fit_date: Optional[datetime] = None
    n_obs: int = 0


@dataclass
class CopulaSignal:
    """Intermediate signal from copula analysis."""
    symbol_a: str
    symbol_b: str
    cond_prob_a_given_b: float  # P(U_a | U_b)
    cond_prob_b_given_a: float  # P(U_b | U_a)
    u_a: float                  # Marginal CDF value for A
    u_b: float                  # Marginal CDF value for B
    copula_family: str
    signal_type: str            # "long_a_short_b" or "short_a_long_b"


class CopulaPairsStrategy:
    """Copula-Based Pairs Trading strategy.

    Workflow:
    1. Fit marginal distributions to each asset's returns
    2. Transform returns to uniform [0,1] via probability integral transform
    3. Fit copula to the joint uniform distribution
    4. Monitor conditional probabilities for extreme values
    5. Generate mean-reversion signals when dependence structure breaks down

    The strategy exploits temporary deviations from the historical
    dependence structure, expecting pairs to revert to their copula-implied
    relationship.
    """

    def __init__(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        upper_threshold: float = UPPER_THRESHOLD,
        lower_threshold: float = LOWER_THRESHOLD,
    ):
        self.pairs = pairs or list(DEFAULT_PAIRS)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self._copula_fits: Dict[Tuple[str, str], CopulaFit] = {}
        self._last_fit_date: Dict[Tuple[str, str], datetime] = {}
        self._active_signals: Dict[Tuple[str, str], str] = {}

    def reset_daily(self):
        """Clear per-day state. Preserve copula fits (expensive to compute)."""
        self._active_signals.clear()

    def generate_signals(
        self,
        bars: Dict[str, pd.DataFrame],
        now: Optional[datetime] = None,
    ) -> List[Signal]:
        """Generate copula-based pair trading signals.

        Args:
            bars: Dict mapping symbol -> DataFrame of daily OHLCV bars.
            now: Current datetime (defaults to datetime.now()).

        Returns:
            List of Signal objects for pair trades.
        """
        if now is None:
            now = datetime.now()

        signals: List[Signal] = []

        for sym_a, sym_b in self.pairs:
            try:
                pair_key = (sym_a, sym_b)
                bars_a = bars.get(sym_a)
                bars_b = bars.get(sym_b)

                if bars_a is None or bars_b is None:
                    continue
                if len(bars_a) < MIN_TRAINING_DAYS or len(bars_b) < MIN_TRAINING_DAYS:
                    continue

                # Refit copula if needed
                if self._needs_refit(pair_key, now):
                    fit = self._fit_copula(sym_a, sym_b, bars_a, bars_b, now)
                    if fit is None:
                        continue
                    self._copula_fits[pair_key] = fit
                    self._last_fit_date[pair_key] = now

                fit = self._copula_fits.get(pair_key)
                if fit is None:
                    continue

                # Compute current copula signal
                copula_signal = self._compute_signal(fit, bars_a, bars_b)
                if copula_signal is None:
                    continue

                # Generate trading signals from copula extremes
                pair_signals = self._create_trade_signals(
                    copula_signal, bars_a, bars_b, now
                )
                signals.extend(pair_signals)

            except Exception as e:
                logger.debug(f"Copula pairs error for {sym_a}/{sym_b}: {e}")
                continue

        if signals:
            logger.info(f"Copula pairs: generated {len(signals)} signals")
        return signals

    def _needs_refit(self, pair_key: Tuple[str, str], now: datetime) -> bool:
        """Check if a pair's copula model needs refitting."""
        if pair_key not in self._copula_fits:
            return True
        last_fit = self._last_fit_date.get(pair_key)
        if last_fit is None:
            return True
        days_since = (now - last_fit).days
        return days_since >= REFIT_INTERVAL_DAYS

    def _fit_copula(
        self,
        sym_a: str,
        sym_b: str,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        now: datetime,
    ) -> Optional[CopulaFit]:
        """Fit marginals and copula to a pair's return data.

        Steps:
        1. Compute log returns
        2. Align series by date
        3. Fit marginal distributions (best of normal, t, skewnorm)
        4. Transform to uniform via PIT
        5. Fit Clayton, Gumbel, Frank copulas
        6. Select best by AIC
        """
        try:
            returns_a = np.log(bars_a["close"] / bars_a["close"].shift(1)).dropna()
            returns_b = np.log(bars_b["close"] / bars_b["close"].shift(1)).dropna()

            # Align by index
            aligned = pd.concat(
                [returns_a.rename("a"), returns_b.rename("b")], axis=1
            ).dropna()

            if len(aligned) < MIN_TRAINING_DAYS:
                return None

            ra = aligned["a"].values
            rb = aligned["b"].values

            # Check correlation
            corr = np.corrcoef(ra, rb)[0, 1]
            if abs(corr) < MIN_CORRELATION:
                logger.debug(
                    f"Copula: {sym_a}/{sym_b} correlation {corr:.3f} "
                    f"below threshold {MIN_CORRELATION}"
                )
                return None

            # Fit marginals
            marg_name_a, marg_params_a = self._fit_marginal(ra)
            marg_name_b, marg_params_b = self._fit_marginal(rb)

            # Transform to uniform [0,1] via PIT
            dist_a = self._get_distribution(marg_name_a)
            dist_b = self._get_distribution(marg_name_b)
            u_a = dist_a.cdf(ra, *marg_params_a)
            u_b = dist_b.cdf(rb, *marg_params_b)

            # Clip to avoid boundary issues
            u_a = np.clip(u_a, 1e-6, 1 - 1e-6)
            u_b = np.clip(u_b, 1e-6, 1 - 1e-6)

            # Fit copulas and select best by AIC
            best_fit = self._select_best_copula(u_a, u_b)
            if best_fit is None:
                return None

            family, theta, aic = best_fit

            return CopulaFit(
                symbol_a=sym_a,
                symbol_b=sym_b,
                family=family,
                theta=theta,
                marginal_a=marg_name_a,
                marginal_b=marg_name_b,
                marginal_params_a=marg_params_a,
                marginal_params_b=marg_params_b,
                aic=aic,
                fit_date=now,
                n_obs=len(aligned),
            )

        except Exception as e:
            logger.debug(f"Copula fit failed for {sym_a}/{sym_b}: {e}")
            return None

    def _fit_marginal(self, returns: np.ndarray) -> Tuple[str, Tuple]:
        """Fit the best marginal distribution to return data.

        Tests normal, Student-t, and skew-normal. Selects by AIC.
        """
        candidates = {}

        # Normal
        try:
            params = stats.norm.fit(returns)
            ll = np.sum(stats.norm.logpdf(returns, *params))
            aic = 2 * len(params) - 2 * ll
            candidates["norm"] = (params, aic)
        except Exception:
            pass

        # Student-t
        try:
            params = stats.t.fit(returns)
            ll = np.sum(stats.t.logpdf(returns, *params))
            aic = 2 * len(params) - 2 * ll
            candidates["t"] = (params, aic)
        except Exception:
            pass

        # Skew-normal
        try:
            params = stats.skewnorm.fit(returns)
            ll = np.sum(stats.skewnorm.logpdf(returns, *params))
            aic = 2 * len(params) - 2 * ll
            candidates["skewnorm"] = (params, aic)
        except Exception:
            pass

        if not candidates:
            # Fallback to normal
            params = stats.norm.fit(returns)
            return "norm", params

        best_name = min(candidates, key=lambda k: candidates[k][1])
        return best_name, candidates[best_name][0]

    @staticmethod
    def _get_distribution(name: str):
        """Get a scipy distribution object by name."""
        return getattr(stats, name)

    def _select_best_copula(
        self, u: np.ndarray, v: np.ndarray
    ) -> Optional[Tuple[str, float, float]]:
        """Fit Clayton, Gumbel, and Frank copulas; return best by AIC.

        Returns:
            Tuple of (family_name, theta, aic) or None if all fits fail.
        """
        results = []

        # Clayton copula: theta > 0 (for positive dependence)
        theta_c = self._fit_clayton(u, v)
        if theta_c is not None and theta_c > 0:
            ll = self._clayton_loglik(theta_c, u, v)
            aic = 2 - 2 * ll
            results.append(("clayton", theta_c, aic))

        # Gumbel copula: theta >= 1
        theta_g = self._fit_gumbel(u, v)
        if theta_g is not None and theta_g >= 1:
            ll = self._gumbel_loglik(theta_g, u, v)
            aic = 2 - 2 * ll
            results.append(("gumbel", theta_g, aic))

        # Frank copula: theta != 0
        theta_f = self._fit_frank(u, v)
        if theta_f is not None and abs(theta_f) > 0.01:
            ll = self._frank_loglik(theta_f, u, v)
            aic = 2 - 2 * ll
            results.append(("frank", theta_f, aic))

        if not results:
            return None

        return min(results, key=lambda x: x[2])

    # ------------------------------------------------------------------
    # Clayton copula: C(u,v) = (u^(-theta) + v^(-theta) - 1)^(-1/theta)
    # ------------------------------------------------------------------

    @staticmethod
    def _clayton_density(theta: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Clayton copula density c(u,v)."""
        if theta <= 0:
            return np.ones_like(u)
        a = u ** (-theta) + v ** (-theta) - 1
        a = np.clip(a, 1e-12, None)
        density = (1 + theta) * (u * v) ** (-(1 + theta)) * a ** (-(2 + 1 / theta))
        return np.clip(density, 1e-12, None)

    def _clayton_loglik(self, theta: float, u: np.ndarray, v: np.ndarray) -> float:
        density = self._clayton_density(theta, u, v)
        return float(np.sum(np.log(density)))

    def _fit_clayton(self, u: np.ndarray, v: np.ndarray) -> Optional[float]:
        """Estimate Clayton theta by maximum likelihood."""
        try:
            def neg_ll(params):
                theta = params[0]
                if theta <= 0.01:
                    return 1e10
                return -self._clayton_loglik(theta, u, v)

            result = optimize.minimize(
                neg_ll, x0=[2.0], bounds=[(0.01, 50.0)], method="L-BFGS-B"
            )
            return float(result.x[0]) if result.success else None
        except Exception:
            return None

    @staticmethod
    def _clayton_conditional(theta: float, u: float, v: float) -> float:
        """P(U <= u | V = v) for Clayton copula."""
        if theta <= 0:
            return u
        a = u ** (-theta) + v ** (-theta) - 1
        if a <= 0:
            return 0.5
        return float(np.clip(v ** (-(1 + theta)) * a ** (-(1 + 1 / theta)), 0, 1))

    # ------------------------------------------------------------------
    # Gumbel copula: C(u,v) = exp(-( (-ln u)^theta + (-ln v)^theta )^(1/theta))
    # ------------------------------------------------------------------

    @staticmethod
    def _gumbel_density(theta: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gumbel copula density."""
        if theta < 1:
            return np.ones_like(u)
        lu = -np.log(np.clip(u, 1e-12, 1 - 1e-12))
        lv = -np.log(np.clip(v, 1e-12, 1 - 1e-12))
        a = lu ** theta + lv ** theta
        a = np.clip(a, 1e-12, None)
        C = np.exp(-a ** (1 / theta))
        term1 = C / (u * v)
        term2 = (lu * lv) ** (theta - 1)
        term3 = a ** (2 / theta - 2)
        term4 = a ** (1 / theta) + theta - 1
        density = term1 * term2 * term3 * term4
        return np.clip(density, 1e-12, None)

    def _gumbel_loglik(self, theta: float, u: np.ndarray, v: np.ndarray) -> float:
        density = self._gumbel_density(theta, u, v)
        return float(np.sum(np.log(density)))

    def _fit_gumbel(self, u: np.ndarray, v: np.ndarray) -> Optional[float]:
        """Estimate Gumbel theta by maximum likelihood."""
        try:
            def neg_ll(params):
                theta = params[0]
                if theta < 1.0:
                    return 1e10
                return -self._gumbel_loglik(theta, u, v)

            result = optimize.minimize(
                neg_ll, x0=[2.0], bounds=[(1.0, 50.0)], method="L-BFGS-B"
            )
            return float(result.x[0]) if result.success else None
        except Exception:
            return None

    @staticmethod
    def _gumbel_conditional(theta: float, u: float, v: float) -> float:
        """P(U <= u | V = v) for Gumbel copula."""
        if theta < 1:
            return u
        lu = -np.log(max(u, 1e-12))
        lv = -np.log(max(v, 1e-12))
        a = lu ** theta + lv ** theta
        if a <= 0:
            return 0.5
        C = np.exp(-a ** (1 / theta))
        h = C * (1 / v) * lv ** (theta - 1) * a ** (1 / theta - 1)
        return float(np.clip(h, 0, 1))

    # ------------------------------------------------------------------
    # Frank copula: C(u,v) = -1/theta * ln(1 + (e^(-theta*u)-1)(e^(-theta*v)-1)/(e^(-theta)-1))
    # ------------------------------------------------------------------

    @staticmethod
    def _frank_density(theta: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Frank copula density."""
        if abs(theta) < 0.01:
            return np.ones_like(u)
        et = np.exp(-theta)
        etu = np.exp(-theta * u)
        etv = np.exp(-theta * v)
        num = -theta * (et - 1) * np.exp(-theta * (u + v))
        den = ((et - 1) + (etu - 1) * (etv - 1)) ** 2
        den = np.clip(den, 1e-12, None)
        density = num / den
        return np.clip(np.abs(density), 1e-12, None)

    def _frank_loglik(self, theta: float, u: np.ndarray, v: np.ndarray) -> float:
        density = self._frank_density(theta, u, v)
        return float(np.sum(np.log(density)))

    def _fit_frank(self, u: np.ndarray, v: np.ndarray) -> Optional[float]:
        """Estimate Frank theta by maximum likelihood."""
        try:
            def neg_ll(params):
                theta = params[0]
                if abs(theta) < 0.01:
                    return 1e10
                return -self._frank_loglik(theta, u, v)

            result = optimize.minimize(
                neg_ll, x0=[5.0], bounds=[(-50.0, 50.0)], method="L-BFGS-B"
            )
            return float(result.x[0]) if result.success else None
        except Exception:
            return None

    @staticmethod
    def _frank_conditional(theta: float, u: float, v: float) -> float:
        """P(U <= u | V = v) for Frank copula."""
        if abs(theta) < 0.01:
            return u
        etu = np.exp(-theta * u)
        etv = np.exp(-theta * v)
        et = np.exp(-theta)
        num = (etu - 1) * etv
        den = (et - 1) + (etu - 1) * (etv - 1)
        if abs(den) < 1e-12:
            return 0.5
        return float(np.clip(num / den, 0, 1))

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _compute_signal(
        self,
        fit: CopulaFit,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
    ) -> Optional[CopulaSignal]:
        """Compute copula conditional probabilities for the latest observation.

        Returns a CopulaSignal if probabilities are extreme enough.
        """
        try:
            # Latest returns
            ret_a = np.log(bars_a["close"].iloc[-1] / bars_a["close"].iloc[-2])
            ret_b = np.log(bars_b["close"].iloc[-1] / bars_b["close"].iloc[-2])

            # Transform to uniform via marginal CDF
            dist_a = self._get_distribution(fit.marginal_a)
            dist_b = self._get_distribution(fit.marginal_b)
            u_a = float(np.clip(dist_a.cdf(ret_a, *fit.marginal_params_a), 1e-6, 1 - 1e-6))
            u_b = float(np.clip(dist_b.cdf(ret_b, *fit.marginal_params_b), 1e-6, 1 - 1e-6))

            # Conditional probabilities
            if fit.family == "clayton":
                cond_a_given_b = self._clayton_conditional(fit.theta, u_a, u_b)
                cond_b_given_a = self._clayton_conditional(fit.theta, u_b, u_a)
            elif fit.family == "gumbel":
                cond_a_given_b = self._gumbel_conditional(fit.theta, u_a, u_b)
                cond_b_given_a = self._gumbel_conditional(fit.theta, u_b, u_a)
            elif fit.family == "frank":
                cond_a_given_b = self._frank_conditional(fit.theta, u_a, u_b)
                cond_b_given_a = self._frank_conditional(fit.theta, u_b, u_a)
            else:
                return None

            # Check for extreme conditional probabilities
            signal_type = None
            if cond_a_given_b > self.upper_threshold:
                # A is "too high" given B -> short A, long B
                signal_type = "short_a_long_b"
            elif cond_a_given_b < self.lower_threshold:
                # A is "too low" given B -> long A, short B
                signal_type = "long_a_short_b"
            elif cond_b_given_a > self.upper_threshold:
                # B is "too high" given A -> long A, short B
                signal_type = "long_a_short_b"
            elif cond_b_given_a < self.lower_threshold:
                # B is "too low" given A -> short A, long B
                signal_type = "short_a_long_b"

            if signal_type is None:
                return None

            return CopulaSignal(
                symbol_a=fit.symbol_a,
                symbol_b=fit.symbol_b,
                cond_prob_a_given_b=cond_a_given_b,
                cond_prob_b_given_a=cond_b_given_a,
                u_a=u_a,
                u_b=u_b,
                copula_family=fit.family,
                signal_type=signal_type,
            )

        except Exception as e:
            logger.debug(f"Copula signal computation failed: {e}")
            return None

    def _create_trade_signals(
        self,
        csig: CopulaSignal,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        now: datetime,
    ) -> List[Signal]:
        """Create paired trading signals from a copula signal."""
        signals: List[Signal] = []
        pair_id = f"COPULA_{csig.symbol_a}_{csig.symbol_b}_{now.strftime('%Y%m%d')}"

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])

        if price_a <= 0 or price_b <= 0:
            return signals

        # Confidence based on extremity of conditional probability
        max_cond = max(csig.cond_prob_a_given_b, 1 - csig.cond_prob_a_given_b,
                       csig.cond_prob_b_given_a, 1 - csig.cond_prob_b_given_a)
        confidence = min(0.5 + (max_cond - 0.5) * 1.5, 0.95)

        common_metadata = {
            "copula_family": csig.copula_family,
            "cond_prob_a_given_b": round(csig.cond_prob_a_given_b, 4),
            "cond_prob_b_given_a": round(csig.cond_prob_b_given_a, 4),
            "u_a": round(csig.u_a, 4),
            "u_b": round(csig.u_b, 4),
        }

        if csig.signal_type == "long_a_short_b":
            # Long A
            signals.append(Signal(
                symbol=csig.symbol_a,
                strategy="COPULA_PAIRS",
                side="buy",
                entry_price=round(price_a, 2),
                take_profit=round(price_a * (1 + DEFAULT_TP_PCT), 2),
                stop_loss=round(price_a * (1 - DEFAULT_SL_PCT), 2),
                reason=(
                    f"Copula pairs LONG {csig.symbol_a} vs {csig.symbol_b} "
                    f"cond_p={csig.cond_prob_a_given_b:.3f} "
                    f"copula={csig.copula_family}"
                ),
                hold_type="swing",
                pair_id=pair_id,
                confidence=round(confidence, 4),
                metadata=common_metadata,
                timestamp=now,
                pair_symbol=csig.symbol_b,
            ))
            # Short B
            signals.append(Signal(
                symbol=csig.symbol_b,
                strategy="COPULA_PAIRS",
                side="sell",
                entry_price=round(price_b, 2),
                take_profit=round(price_b * (1 - DEFAULT_TP_PCT), 2),
                stop_loss=round(price_b * (1 + DEFAULT_SL_PCT), 2),
                reason=(
                    f"Copula pairs SHORT {csig.symbol_b} vs {csig.symbol_a} "
                    f"cond_p={csig.cond_prob_b_given_a:.3f} "
                    f"copula={csig.copula_family}"
                ),
                hold_type="swing",
                pair_id=pair_id,
                confidence=round(confidence, 4),
                metadata=common_metadata,
                timestamp=now,
                pair_symbol=csig.symbol_a,
            ))

        elif csig.signal_type == "short_a_long_b":
            # Short A
            signals.append(Signal(
                symbol=csig.symbol_a,
                strategy="COPULA_PAIRS",
                side="sell",
                entry_price=round(price_a, 2),
                take_profit=round(price_a * (1 - DEFAULT_TP_PCT), 2),
                stop_loss=round(price_a * (1 + DEFAULT_SL_PCT), 2),
                reason=(
                    f"Copula pairs SHORT {csig.symbol_a} vs {csig.symbol_b} "
                    f"cond_p={csig.cond_prob_a_given_b:.3f} "
                    f"copula={csig.copula_family}"
                ),
                hold_type="swing",
                pair_id=pair_id,
                confidence=round(confidence, 4),
                metadata=common_metadata,
                timestamp=now,
                pair_symbol=csig.symbol_b,
            ))
            # Long B
            signals.append(Signal(
                symbol=csig.symbol_b,
                strategy="COPULA_PAIRS",
                side="buy",
                entry_price=round(price_b, 2),
                take_profit=round(price_b * (1 + DEFAULT_TP_PCT), 2),
                stop_loss=round(price_b * (1 - DEFAULT_SL_PCT), 2),
                reason=(
                    f"Copula pairs LONG {csig.symbol_b} vs {csig.symbol_a} "
                    f"cond_p={csig.cond_prob_b_given_a:.3f} "
                    f"copula={csig.copula_family}"
                ),
                hold_type="swing",
                pair_id=pair_id,
                confidence=round(confidence, 4),
                metadata=common_metadata,
                timestamp=now,
                pair_symbol=csig.symbol_a,
            ))

        return signals

    def get_fit_summary(self) -> List[Dict]:
        """Return summary of fitted copulas for dashboard/logging."""
        summaries = []
        for key, fit in self._copula_fits.items():
            summaries.append({
                "pair": f"{fit.symbol_a}/{fit.symbol_b}",
                "family": fit.family,
                "theta": round(fit.theta, 4),
                "marginal_a": fit.marginal_a,
                "marginal_b": fit.marginal_b,
                "aic": round(fit.aic, 2),
                "n_obs": fit.n_obs,
                "fit_date": fit.fit_date.isoformat() if fit.fit_date else None,
            })
        return summaries
