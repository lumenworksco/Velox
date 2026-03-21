"""LPRADO-004: Entropy Features — market predictability measurement.

Implements entropy-based features from Lopez de Prado's framework for
quantifying the predictability (or randomness) of market dynamics.

Entropy interpretation for trading:
    Low entropy  -> Ordered / trending   -> Momentum strategies favoured
    High entropy -> Disordered / chaotic -> Mean reversion or sit out

Four entropy measures, each capturing different aspects:
    1. Shannon entropy: distribution shape of returns (rolling window)
    2. Approximate entropy (ApEn): regularity of patterns in time series
    3. Sample entropy (SampEn): improved ApEn (less self-match bias)
    4. Permutation entropy: ordinal pattern complexity

Usage:
    ef = EntropyFeatures()
    shannon = ef.compute_shannon_entropy(returns, window=20)
    sampen = ef.compute_sample_entropy(price_series, m=2, r=0.2)
    perm_en = ef.compute_permutation_entropy(price_series, order=3, delay=1)
    features = ef.compute_all(returns, prices)
"""

import logging
from math import factorial, log
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EntropyFeatures:
    """Compute entropy-based features for market regime classification.

    All methods accept either numpy arrays or pandas Series and return
    scalar values or Series (for rolling computations).
    """

    # ------------------------------------------------------------------
    # Shannon Entropy (rolling)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_shannon_entropy(
        returns: pd.Series | np.ndarray,
        window: int = 20,
        n_bins: int = 10,
    ) -> pd.Series:
        """Compute rolling Shannon entropy of the return distribution.

        Discretizes returns into bins and computes H = -sum(p * log2(p))
        over a rolling window. Higher values indicate more uniform (random)
        distributions; lower values indicate concentration (trending).

        Args:
            returns: Return series (simple or log returns).
            window: Rolling window size. Default 20 (one trading month).
            n_bins: Number of histogram bins for discretization. Default 10.

        Returns:
            Series of rolling Shannon entropy values. First (window-1)
            values are NaN.
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        n = len(returns)
        result = np.full(n, np.nan, dtype=np.float64)

        values = returns.values.astype(np.float64)

        for i in range(window - 1, n):
            window_data = values[i - window + 1: i + 1]

            # Skip if all NaN or constant
            valid = window_data[~np.isnan(window_data)]
            if len(valid) < 3:
                continue

            # Histogram-based probability estimation
            counts, _ = np.histogram(valid, bins=n_bins)
            probs = counts / counts.sum()

            # Shannon entropy: H = -sum(p * log2(p)) for p > 0
            nonzero = probs[probs > 0]
            entropy = -np.sum(nonzero * np.log2(nonzero))

            result[i] = entropy

        return pd.Series(result, index=returns.index, name="shannon_entropy")

    # ------------------------------------------------------------------
    # Approximate Entropy (ApEn)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_approximate_entropy(
        series: pd.Series | np.ndarray,
        m: int = 2,
        r: float = 0.2,
    ) -> float:
        """Compute Approximate Entropy (ApEn) of a time series.

        ApEn(m, r) measures the logarithmic likelihood that patterns of
        length m that are similar within tolerance r remain similar at
        length m+1. Lower ApEn indicates more regularity (predictability).

        Note: ApEn has known bias from self-matches. Prefer SampEn for
        most applications.

        Args:
            series: Input time series (prices, returns, or any signal).
            m: Embedding dimension (pattern length). Default 2.
            r: Tolerance as a fraction of the series std. Default 0.2.
                Absolute tolerance = r * std(series).

        Returns:
            Scalar ApEn value. Typical range [0, 2] for financial data.
            Returns 0.0 if series is too short or constant.
        """
        data = np.asarray(series, dtype=np.float64)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < m + 2:
            return 0.0

        std = np.std(data)
        if std < 1e-10:
            return 0.0

        tolerance = r * std

        def _phi(template_len: int) -> float:
            """Count similar template matches for embedding dimension template_len."""
            templates = np.array([
                data[i: i + template_len]
                for i in range(n - template_len + 1)
            ])
            n_templates = len(templates)
            counts = np.zeros(n_templates)

            for i in range(n_templates):
                # Chebyshev distance (max absolute difference)
                diffs = np.abs(templates - templates[i])
                max_diffs = diffs.max(axis=1)
                counts[i] = np.sum(max_diffs <= tolerance)

            # Include self-match (ApEn convention)
            counts = counts / n_templates
            return np.sum(np.log(counts[counts > 0])) / n_templates

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        return abs(phi_m - phi_m1)

    # ------------------------------------------------------------------
    # Sample Entropy (SampEn)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sample_entropy(
        series: pd.Series | np.ndarray,
        m: int = 2,
        r: float = 0.2,
    ) -> float:
        """Compute Sample Entropy (SampEn) of a time series.

        SampEn is a refinement of ApEn that excludes self-matches, giving
        a less biased estimate of regularity. It is the negative natural
        logarithm of the conditional probability that two sequences similar
        for m points remain similar at m+1 points.

        Args:
            series: Input time series.
            m: Embedding dimension. Default 2.
            r: Tolerance as fraction of series std. Default 0.2.

        Returns:
            Scalar SampEn value. Higher = more complex/random.
            Returns 0.0 if insufficient data; np.inf if no matches at m+1.
        """
        data = np.asarray(series, dtype=np.float64)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < m + 2:
            return 0.0

        std = np.std(data)
        if std < 1e-10:
            return 0.0

        tolerance = r * std

        def _count_matches(template_len: int) -> int:
            """Count template matches excluding self-matches."""
            templates = np.array([
                data[i: i + template_len]
                for i in range(n - template_len)
            ])
            n_templates = len(templates)
            total = 0

            for i in range(n_templates):
                for j in range(i + 1, n_templates):
                    if np.max(np.abs(templates[i] - templates[j])) <= tolerance:
                        total += 1

            return total

        count_m = _count_matches(m)
        count_m1 = _count_matches(m + 1)

        if count_m == 0:
            return 0.0
        if count_m1 == 0:
            return np.inf

        return -np.log(count_m1 / count_m)

    # ------------------------------------------------------------------
    # Permutation Entropy
    # ------------------------------------------------------------------

    @staticmethod
    def compute_permutation_entropy(
        series: pd.Series | np.ndarray,
        order: int = 3,
        delay: int = 1,
        normalize: bool = True,
    ) -> float:
        """Compute Permutation Entropy of a time series.

        Maps the time series to a sequence of ordinal patterns (permutations)
        and computes the Shannon entropy of the pattern distribution. Fast,
        robust to noise, and invariant to monotonic transformations.

        Args:
            series: Input time series.
            order: Permutation order (pattern length). Default 3.
                   Higher orders capture more complex patterns but need
                   more data (at least order! * 10 observations).
            delay: Embedding delay (time lag between pattern elements).
                   Default 1 (consecutive points).
            normalize: If True, normalize by log2(order!) so result is in [0, 1].
                       Default True.

        Returns:
            Permutation entropy value.
            If normalized: 0 = perfectly regular, 1 = maximally random.
        """
        data = np.asarray(series, dtype=np.float64)
        data = data[~np.isnan(data)]
        n = len(data)

        n_patterns = n - (order - 1) * delay
        if n_patterns < 1:
            return 0.0

        # Extract ordinal patterns
        pattern_counts: dict[tuple, int] = {}

        for i in range(n_patterns):
            # Extract the pattern: values at indices i, i+delay, i+2*delay, ...
            indices = list(range(i, i + order * delay, delay))
            pattern_values = data[indices]

            # Convert to ordinal pattern (rank order)
            # argsort of argsort gives rank
            ordinal = tuple(np.argsort(np.argsort(pattern_values)))

            pattern_counts[ordinal] = pattern_counts.get(ordinal, 0) + 1

        # Compute Shannon entropy of pattern distribution
        total = sum(pattern_counts.values())
        probs = np.array([c / total for c in pattern_counts.values()])
        # CRIT-005: Filter zero probabilities to avoid 0 * log2(0) = NaN
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        if normalize:
            max_entropy = log(factorial(order)) / log(2)
            if max_entropy > 0:
                entropy /= max_entropy

        return entropy

    # ------------------------------------------------------------------
    # Rolling variants for SampEn and PermEn
    # ------------------------------------------------------------------

    def compute_rolling_sample_entropy(
        self,
        series: pd.Series,
        window: int = 50,
        m: int = 2,
        r: float = 0.2,
    ) -> pd.Series:
        """Compute rolling Sample Entropy over a sliding window.

        Args:
            series: Input time series.
            window: Rolling window size. Default 50.
            m: Embedding dimension. Default 2.
            r: Tolerance factor. Default 0.2.

        Returns:
            Series of rolling SampEn values.
        """
        n = len(series)
        result = np.full(n, np.nan, dtype=np.float64)
        values = np.asarray(series, dtype=np.float64)

        for i in range(window - 1, n):
            window_data = values[i - window + 1: i + 1]
            result[i] = self.compute_sample_entropy(window_data, m=m, r=r)

        return pd.Series(
            result,
            index=series.index if isinstance(series, pd.Series) else None,
            name="rolling_sampen",
        )

    def compute_rolling_permutation_entropy(
        self,
        series: pd.Series,
        window: int = 50,
        order: int = 3,
        delay: int = 1,
    ) -> pd.Series:
        """Compute rolling Permutation Entropy over a sliding window.

        Args:
            series: Input time series.
            window: Rolling window size. Default 50.
            order: Permutation order. Default 3.
            delay: Embedding delay. Default 1.

        Returns:
            Series of rolling permutation entropy values (normalized [0, 1]).
        """
        n = len(series)
        result = np.full(n, np.nan, dtype=np.float64)
        values = np.asarray(series, dtype=np.float64)

        for i in range(window - 1, n):
            window_data = values[i - window + 1: i + 1]
            result[i] = self.compute_permutation_entropy(
                window_data, order=order, delay=delay, normalize=True,
            )

        return pd.Series(
            result,
            index=series.index if isinstance(series, pd.Series) else None,
            name="rolling_permen",
        )

    # ------------------------------------------------------------------
    # Combined feature extraction
    # ------------------------------------------------------------------

    def compute_all(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        window: int = 20,
    ) -> pd.DataFrame:
        """Compute all entropy features and return as a DataFrame.

        Args:
            returns: Return series for Shannon entropy.
            prices: Price series for SampEn and PermEn. If None, uses returns.
            window: Rolling window for Shannon entropy. Default 20.

        Returns:
            DataFrame with columns: shannon_entropy, rolling_sampen,
            rolling_permen.
        """
        target = prices if prices is not None else returns

        result = pd.DataFrame(index=returns.index)
        result["shannon_entropy"] = self.compute_shannon_entropy(returns, window=window)
        result["rolling_sampen"] = self.compute_rolling_sample_entropy(
            target, window=max(window, 30),
        )
        result["rolling_permen"] = self.compute_rolling_permutation_entropy(
            target, window=max(window, 30),
        )

        return result

    # ------------------------------------------------------------------
    # Regime classification helper
    # ------------------------------------------------------------------

    @staticmethod
    def classify_entropy_regime(
        entropy_value: float,
        low_threshold: float = 0.35,
        high_threshold: float = 0.65,
    ) -> str:
        """Classify entropy value into a trading regime.

        Args:
            entropy_value: Normalized entropy value in [0, 1].
            low_threshold: Below this = trending. Default 0.35.
            high_threshold: Above this = chaotic. Default 0.65.

        Returns:
            One of "trending", "transitional", or "chaotic".
        """
        if entropy_value < low_threshold:
            return "trending"
        elif entropy_value > high_threshold:
            return "chaotic"
        else:
            return "transitional"
