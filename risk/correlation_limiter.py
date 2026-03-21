"""V10 Risk — Correlation-based portfolio concentration limiter.

Prevents the portfolio from accumulating too much correlated risk by:
1. Tracking pairwise correlations between open positions
2. Computing portfolio-level concentration (effective number of bets)
3. Blocking new positions that would increase concentration beyond threshold

This goes beyond the simple `is_too_correlated()` check (which only checks
individual pairs) by looking at portfolio-level correlation structure.
"""

import logging
import threading
import time as _time
from dataclasses import dataclass

import numpy as np

import config

logger = logging.getLogger(__name__)

# MED-037: Cache TTL for correlation matrix computation (seconds)
_CORR_MATRIX_CACHE_TTL = 300  # 5 minutes


@dataclass
class ConcentrationResult:
    """Portfolio concentration analysis."""
    effective_bets: float = 0.0         # Effective number of independent bets
    max_pairwise_corr: float = 0.0      # Highest pairwise correlation
    avg_pairwise_corr: float = 0.0      # Average pairwise correlation
    sector_concentration: float = 0.0   # Herfindahl index of sector exposure
    too_concentrated: bool = False       # Whether we should block new entries
    reason: str = ""


class CorrelationLimiter:
    """Portfolio-level correlation and concentration limiter.

    Uses the correlation matrix to compute the effective number of
    independent bets, and blocks new entries when the portfolio is
    too concentrated in correlated positions.
    """

    def __init__(
        self,
        max_pairwise_corr: float = None,
        min_effective_bets: float = None,
        max_sector_weight: float = None,
    ):
        self.max_pairwise_corr = max_pairwise_corr or getattr(config, "MAX_PAIRWISE_CORRELATION", 0.70)
        self.min_effective_bets = min_effective_bets or getattr(config, "MIN_EFFECTIVE_BETS", 2.0)
        self.max_sector_weight = max_sector_weight or getattr(config, "MAX_SECTOR_WEIGHT", 0.50)

        self._correlation_cache: dict[tuple[str, str], float] = {}
        self._sector_map: dict[str, str] = dict(getattr(config, 'SECTOR_MAP', {}))
        self._lock = threading.Lock()  # MED-008: protect _sector_map and _correlation_cache

        # MED-037: Cache the computed correlation matrix to avoid recomputing every call
        self._matrix_cache_key: tuple | None = None  # frozenset of symbols
        self._matrix_cache_result: np.ndarray | None = None
        self._matrix_cache_time: float = 0.0

    def set_sector_map(self, sector_map: dict[str, str]):
        """Set the symbol-to-sector mapping (thread-safe)."""
        with self._lock:
            self._sector_map = dict(sector_map)

    def update_correlation(self, sym1: str, sym2: str, corr: float):
        """Update cached pairwise correlation (thread-safe)."""
        key = tuple(sorted([sym1, sym2]))
        with self._lock:
            self._correlation_cache[key] = corr

    def check_new_position(
        self,
        new_symbol: str,
        open_symbols: list[str],
        correlations: dict[tuple[str, str], float] | None = None,
    ) -> ConcentrationResult:
        """Check if adding a new position would make the portfolio too concentrated.

        Args:
            new_symbol: Symbol to potentially add
            open_symbols: Currently open position symbols
            correlations: Optional correlation matrix override

        Returns:
            ConcentrationResult with analysis
        """
        if not open_symbols:
            return ConcentrationResult(effective_bets=1.0)

        # MED-008: snapshot mutable state under lock
        with self._lock:
            corr_source = correlations or dict(self._correlation_cache)
            sector_map_snapshot = dict(self._sector_map)
        all_symbols = open_symbols + [new_symbol]
        n = len(all_symbols)

        # Build correlation matrix with Ledoit-Wolf shrinkage (HIGH-030)
        # MED-037: Use cached matrix if symbols haven't changed and TTL hasn't expired
        cache_key = frozenset(all_symbols)
        now = _time.time()
        if (self._matrix_cache_key == cache_key
                and self._matrix_cache_result is not None
                and (now - self._matrix_cache_time) < _CORR_MATRIX_CACHE_TTL):
            corr_matrix = self._matrix_cache_result
        else:
            corr_matrix = self._build_corr_matrix(all_symbols, corr_source, n)
            self._matrix_cache_key = cache_key
            self._matrix_cache_result = corr_matrix
            self._matrix_cache_time = now

        # 1. Max pairwise correlation check
        # Only check correlations involving the NEW symbol
        new_idx = n - 1
        new_corrs = [abs(corr_matrix[new_idx, i]) for i in range(n - 1)]
        max_new_corr = max(new_corrs) if new_corrs else 0.0

        if max_new_corr > self.max_pairwise_corr:
            most_correlated = open_symbols[int(np.argmax(new_corrs))]
            return ConcentrationResult(
                max_pairwise_corr=max_new_corr,
                too_concentrated=True,
                reason=f"high_corr_with_{most_correlated}_{max_new_corr:.2f}",
            )

        # 2. Effective number of bets (eigenvalue-based)
        # ENB = (sum(eigenvalues))^2 / sum(eigenvalues^2)
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Clamp negative eigenvalues
            sum_eig = float(np.sum(eigenvalues))
            sum_eig_sq = float(np.sum(eigenvalues ** 2))
            effective_bets = (sum_eig ** 2 / sum_eig_sq) if sum_eig_sq > 0 else n
        except Exception:
            effective_bets = n  # Fallback: assume independent

        if effective_bets < self.min_effective_bets and n > 2:
            return ConcentrationResult(
                effective_bets=effective_bets,
                max_pairwise_corr=max_new_corr,
                too_concentrated=True,
                reason=f"low_effective_bets_{effective_bets:.1f}",
            )

        # 3. Sector concentration (Herfindahl index)
        # HIGH-005: Default unknown sectors to "OTHER" and exclude from concentration calc
        sector_weights = {}
        for sym in all_symbols:
            sector = sector_map_snapshot.get(sym, "OTHER")
            sector_weights[sector] = sector_weights.get(sector, 0) + 1.0 / n

        # Exclude "OTHER" from concentration checks — unknown sectors shouldn't
        # trigger false concentration alerts
        known_sector_weights = {s: w for s, w in sector_weights.items() if s != "OTHER"}
        hhi = sum(w ** 2 for w in known_sector_weights.values()) if known_sector_weights else 0.0
        max_sector = max(known_sector_weights.values()) if known_sector_weights else 0.0

        if max_sector > self.max_sector_weight:
            top_sector = max(sector_weights, key=sector_weights.get)
            return ConcentrationResult(
                effective_bets=effective_bets,
                max_pairwise_corr=max_new_corr,
                sector_concentration=hhi,
                too_concentrated=True,
                reason=f"sector_{top_sector}_overweight_{max_sector:.0%}",
            )

        # All checks passed
        pairwise_corrs = [
            abs(corr_matrix[i, j])
            for i in range(n) for j in range(i + 1, n)
        ]
        avg_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else 0.0

        return ConcentrationResult(
            effective_bets=effective_bets,
            max_pairwise_corr=max_new_corr,
            avg_pairwise_corr=avg_corr,
            sector_concentration=hhi,
            too_concentrated=False,
        )

    @staticmethod
    def _build_corr_matrix(
        all_symbols: list[str],
        corr_source: dict[tuple[str, str], float],
        n: int,
    ) -> np.ndarray:
        """HIGH-030: Build correlation matrix with Ledoit-Wolf shrinkage.

        Applies shrinkage toward the identity matrix to improve estimation
        stability. Falls back to identity matrix on singular matrix errors.
        """
        raw = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                key = tuple(sorted([all_symbols[i], all_symbols[j]]))
                c = corr_source.get(key, 0.0)
                raw[i, j] = c
                raw[j, i] = c

        # MED-016: Replace NaN values with 0.0 (assume uncorrelated when unknown)
        nan_count = int(np.isnan(raw).sum())
        if nan_count > 0:
            logger.warning("Correlation matrix has %d NaN values — replacing with 0.0", nan_count)
            np.nan_to_num(raw, copy=False, nan=0.0)

        # Ledoit-Wolf shrinkage toward identity
        try:
            # Shrinkage intensity: use a simple analytical formula
            # alpha = optimal shrinkage intensity (between 0 and 1)
            # Shrunk = alpha * Identity + (1 - alpha) * Sample
            target = np.eye(n)
            # Frobenius norm-based shrinkage estimate
            off_diag = raw - target
            # Simple shrinkage: proportional to how noisy the off-diag is
            # relative to the number of observations (use n as proxy)
            frobenius_sq = float(np.sum(off_diag ** 2))
            # Shrinkage intensity: higher for smaller portfolios (less data)
            alpha = min(1.0, max(0.0, 1.0 / (n + 1)))
            shrunk = alpha * target + (1.0 - alpha) * raw

            # Verify positive semi-definiteness
            eigenvalues = np.linalg.eigvalsh(shrunk)
            if np.any(eigenvalues < -1e-10):
                # Clamp negative eigenvalues (shouldn't happen with shrinkage, but be safe)
                logger.debug("Clamping negative eigenvalues after shrinkage")
                eigvals, eigvecs = np.linalg.eigh(shrunk)
                eigvals = np.maximum(eigvals, 0)
                shrunk = eigvecs @ np.diag(eigvals) @ eigvecs.T
                # Re-normalize diagonal to 1.0
                d = np.sqrt(np.diag(shrunk))
                d[d == 0] = 1.0
                shrunk = shrunk / np.outer(d, d)

            return shrunk

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(
                "Correlation matrix shrinkage failed (%s), falling back to identity", e
            )
            return np.eye(n)

    @property
    def status(self) -> dict:
        return {
            "cached_correlations": len(self._correlation_cache),
            "tracked_sectors": len(set(self._sector_map.values())),
            "max_pairwise_corr_limit": self.max_pairwise_corr,
            "min_effective_bets": self.min_effective_bets,
            "max_sector_weight": self.max_sector_weight,
        }
