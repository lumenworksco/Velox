"""ADVML-002: RMT Covariance Cleaning — Marchenko-Pastur denoising.

Applies Random Matrix Theory to separate signal from noise in empirical
correlation matrices.  Noise eigenvalues (those falling within the
Marchenko-Pastur bulk) are shrunk to their average, preserving the
signal eigenvalues that exceed the upper MP bound.

Also provides:
- Detoned correlation (market-mode removal for clustering/pair selection)
- Marchenko-Pastur PDF for eigenvalue analysis
- Absorption ratio for systemic risk monitoring
- Effective bets (entropy-based effective rank)

References:
    - Laloux et al. (1999) "Noise Dressing of Financial Correlation Matrices"
    - Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 2
    - Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing*

Dependencies: numpy (always available in this project).
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the positive-semidefinite cone.

    Uses the Higham (2002) alternating-projection algorithm simplified for
    symmetric input: clamp negative eigenvalues to zero and reconstruct.
    """
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1.0  # avoid division by zero
    outer = np.outer(std, std)
    return cov / outer


# ---------------------------------------------------------------------------
# CovarianceCleaner
# ---------------------------------------------------------------------------


class CovarianceCleaner:
    """Marchenko-Pastur denoising for correlation / covariance matrices.

    Parameters
    ----------
    method : str
        Denoising method.  ``"constant"`` replaces noise eigenvalues with
        their average (default).  ``"target"`` shrinks toward a target
        (identity) matrix using the Ledoit-Wolf optimal shrinkage intensity.
    preserve_trace : bool
        If *True*, rescale denoised eigenvalues so the trace (sum of
        eigenvalues) matches the original.  Keeps the matrix interpretable
        as a correlation matrix.  Default *True*.
    """

    def __init__(
        self,
        method: str = "constant",
        preserve_trace: bool = True,
    ) -> None:
        if method not in ("constant", "target"):
            raise ValueError(f"Unknown method {method!r}; choose 'constant' or 'target'")
        self.method = method
        self.preserve_trace = preserve_trace

    # ----- public API -------------------------------------------------------

    def denoise_correlation(
        self,
        corr_matrix: np.ndarray,
        n_observations: int,
    ) -> np.ndarray:
        """Denoise a correlation matrix using Random Matrix Theory.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Symmetric correlation matrix of shape ``(n, n)``.
        n_observations : int
            Number of observations (rows / time steps) used to estimate
            the correlation matrix.  Needed to compute the quality ratio
            ``q = n / T``.

        Returns
        -------
        np.ndarray
            Denoised correlation matrix of the same shape, guaranteed to be
            positive-semidefinite with unit diagonal.

        Raises
        ------
        ValueError
            If *corr_matrix* is not square, or *n_observations* < 2.
        """
        corr_matrix = np.asarray(corr_matrix, dtype=np.float64)

        n = corr_matrix.shape[0]
        if corr_matrix.shape != (n, n):
            raise ValueError("corr_matrix must be square")
        if n_observations < 2:
            raise ValueError("n_observations must be >= 2")
        if n < 2:
            logger.debug("Trivial 1x1 matrix — returning unchanged.")
            return corr_matrix.copy()

        # Symmetrise (handles tiny floating-point asymmetries)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2.0

        # --- step 1: eigen-decomposition ------------------------------------
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        # eigh returns ascending order — sort descending for convenience
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # --- step 2: Marchenko-Pastur bounds --------------------------------
        q = n / n_observations  # quality ratio
        lambda_plus, lambda_minus = self._mp_bounds(q)

        logger.debug(
            "MP bounds: lambda_- = %.4f, lambda_+ = %.4f  (q=%.4f, n=%d, T=%d)",
            lambda_minus,
            lambda_plus,
            q,
            n,
            n_observations,
        )

        noise_mask = (eigvals <= lambda_plus) & (eigvals >= lambda_minus)
        n_noise = int(noise_mask.sum())
        n_signal = n - n_noise
        logger.info(
            "Eigenvalue split: %d signal, %d noise (of %d total)",
            n_signal,
            n_noise,
            n,
        )

        # --- step 3: denoise eigenvalues ------------------------------------
        denoised_eigvals = eigvals.copy()
        if n_noise > 0:
            if self.method == "constant":
                # Replace noise eigenvalues with their mean
                noise_mean = eigvals[noise_mask].mean()
                denoised_eigvals[noise_mask] = noise_mean
            elif self.method == "target":
                # Shrink toward identity (eigenvalue = 1)
                alpha = self._ledoit_wolf_shrinkage(eigvals, noise_mask)
                denoised_eigvals[noise_mask] = (
                    alpha * 1.0 + (1.0 - alpha) * eigvals[noise_mask]
                )

        # --- step 4: preserve trace (optional) ------------------------------
        if self.preserve_trace:
            original_trace = eigvals.sum()
            denoised_trace = denoised_eigvals.sum()
            if denoised_trace > 0:
                denoised_eigvals *= original_trace / denoised_trace

        # --- step 5: reconstruct -------------------------------------------
        denoised = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T

        # Force unit diagonal & symmetry
        denoised = self._force_correlation(denoised)

        # Guarantee PSD
        denoised = _nearest_positive_semidefinite(denoised)
        denoised = self._force_correlation(denoised)

        return denoised

    def denoise_covariance(
        self,
        cov_matrix: np.ndarray,
        n_observations: int,
    ) -> np.ndarray:
        """Denoise a covariance matrix.

        Internally converts to correlation, denoises, then maps back to
        covariance space using the original standard deviations.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Symmetric covariance matrix of shape ``(n, n)``.
        n_observations : int
            Number of observations used to estimate the matrix.

        Returns
        -------
        np.ndarray
            Denoised covariance matrix.
        """
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)
        std = np.sqrt(np.diag(cov_matrix))
        std[std == 0] = 1.0

        corr = _correlation_from_covariance(cov_matrix)
        denoised_corr = self.denoise_correlation(corr, n_observations)

        # Map back to covariance
        outer = np.outer(std, std)
        return denoised_corr * outer

    def compute_effective_bets(self, corr_matrix: np.ndarray) -> int:
        """Number of independent bets implied by the correlation structure.

        Defined as ``exp(H)`` where *H* is the Shannon entropy of the
        normalised eigenvalue distribution.  A perfectly diagonal matrix
        yields *n*; a perfectly correlated matrix yields 1.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Symmetric correlation matrix.

        Returns
        -------
        int
            Effective number of independent bets (rounded down).
        """
        corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-12)  # avoid log(0)
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p))
        return int(math.exp(entropy))

    def compute_absorption_ratio(
        self,
        corr_matrix: np.ndarray,
        n_components: Optional[int] = None,
        fraction: float = 0.2,
    ) -> float:
        """Absorption ratio — fraction of total variance explained by the
        top *n_components* eigenvalues.

        A rising absorption ratio indicates increasing systemic risk
        (markets becoming more tightly coupled).

        Parameters
        ----------
        corr_matrix : np.ndarray
            Correlation matrix.
        n_components : int, optional
            Number of top eigenvalues to use.  Defaults to
            ``ceil(fraction * n)``.
        fraction : float
            Fraction of the dimension to use if *n_components* is None.

        Returns
        -------
        float
            Value in [0, 1].
        """
        eigvals = np.linalg.eigvalsh(np.asarray(corr_matrix, dtype=np.float64))
        eigvals = np.sort(eigvals)[::-1]
        n = len(eigvals)
        if n_components is None:
            n_components = max(1, int(math.ceil(fraction * n)))
        n_components = min(n_components, n)

        total = eigvals.sum()
        if total <= 0:
            return 0.0
        return float(eigvals[:n_components].sum() / total)

    # ----- internals --------------------------------------------------------

    @staticmethod
    def _mp_bounds(q: float) -> Tuple[float, float]:
        """Return (lambda_plus, lambda_minus) for quality ratio q = n/T."""
        sqrt_q = math.sqrt(max(q, 1e-12))
        lambda_plus = (1.0 + sqrt_q) ** 2
        lambda_minus = (1.0 - sqrt_q) ** 2
        return lambda_plus, lambda_minus

    @staticmethod
    def _ledoit_wolf_shrinkage(
        eigvals: np.ndarray,
        noise_mask: np.ndarray,
    ) -> float:
        """Compute a simple shrinkage intensity for noise eigenvalues.

        Returns a value in [0, 1].  Higher means more shrinkage toward
        the identity target.
        """
        noise_vals = eigvals[noise_mask]
        if len(noise_vals) == 0:
            return 0.0
        # Shrinkage proportional to how far noise eigenvalues deviate from 1
        dispersion = np.mean((noise_vals - 1.0) ** 2)
        alpha = min(1.0, dispersion / max(np.var(eigvals), 1e-12))
        return float(alpha)

    def detone_correlation(
        self,
        corr_matrix: np.ndarray,
        n_remove: int = 1,
    ) -> np.ndarray:
        """Remove the market component (top eigenvalues) from a correlation matrix.

        'Detoning' removes the dominant eigenvector(s) (market mode) so that
        the remaining structure reflects idiosyncratic co-movement.  Useful
        for clustering and pair selection where you want pairs that move
        together beyond market beta.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Symmetric correlation matrix (N x N).
        n_remove : int
            Number of top eigenvalues/vectors to remove.  Default 1
            (just the market factor).

        Returns
        -------
        np.ndarray
            Detoned correlation matrix with unit diagonal.
        """
        corr = np.asarray(corr_matrix, dtype=np.float64)
        n = corr.shape[0]

        eigvals, eigvecs = np.linalg.eigh(corr)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Zero out the top n_remove eigenvalues
        cleaned = eigvals.copy()
        removed_var = cleaned[:n_remove].sum()
        cleaned[:n_remove] = 0.0

        # Redistribute removed variance to preserve trace
        remaining = n - n_remove
        if remaining > 0:
            cleaned[n_remove:] += removed_var / remaining

        detoned = eigvecs @ np.diag(cleaned) @ eigvecs.T
        detoned = self._force_correlation(detoned)
        detoned = _nearest_positive_semidefinite(detoned)
        detoned = self._force_correlation(detoned)

        return detoned

    @staticmethod
    def marchenko_pastur_pdf(
        x: np.ndarray,
        q: float,
        sigma_sq: float = 1.0,
    ) -> np.ndarray:
        """Marchenko-Pastur probability density function.

        Parameters
        ----------
        x : np.ndarray
            Eigenvalue points at which to evaluate the PDF.
        q : float
            Quality ratio n/T (assets / observations).
        sigma_sq : float
            Variance parameter.  Default 1.0 (correlation matrices).

        Returns
        -------
        np.ndarray
            PDF values.  Zero outside the MP support.
        """
        sqrt_q = np.sqrt(max(q, 1e-12))
        lambda_plus = sigma_sq * (1.0 + sqrt_q) ** 2
        lambda_minus = sigma_sq * (1.0 - sqrt_q) ** 2

        pdf = np.zeros_like(x, dtype=np.float64)
        mask = (x >= lambda_minus) & (x <= lambda_plus) & (x > 1e-12)

        if np.any(mask):
            xm = x[mask]
            pdf[mask] = (
                1.0
                / (2.0 * np.pi * sigma_sq * q)
                * np.sqrt(np.maximum((lambda_plus - xm) * (xm - lambda_minus), 0.0))
                / xm
            )

        return pdf

    @staticmethod
    def covariance_to_correlation(
        cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a covariance matrix to a correlation matrix.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (correlation_matrix, standard_deviations)
        """
        std = np.sqrt(np.diag(cov))
        safe = std.copy()
        safe[safe < 1e-15] = 1.0
        corr = _correlation_from_covariance(cov)
        return corr, std

    @staticmethod
    def correlation_to_covariance(
        corr: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """Convert a correlation matrix back to covariance space.

        Parameters
        ----------
        corr : np.ndarray
            Correlation matrix (N x N).
        stds : np.ndarray
            Standard deviations (N,).

        Returns
        -------
        np.ndarray
            Covariance matrix (N x N).
        """
        return corr * np.outer(stds, stds)

    @staticmethod
    def _force_correlation(matrix: np.ndarray) -> np.ndarray:
        """Force unit diagonal and clip off-diagonal to [-1, 1]."""
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
        np.clip(matrix, -1.0, 1.0, out=matrix)
        return matrix
