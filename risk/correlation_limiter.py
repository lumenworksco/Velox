"""V10 Risk — Correlation-based portfolio concentration limiter.

Prevents the portfolio from accumulating too much correlated risk by:
1. Tracking pairwise correlations between open positions
2. Computing portfolio-level concentration (effective number of bets)
3. Blocking new positions that would increase concentration beyond threshold

This goes beyond the simple `is_too_correlated()` check (which only checks
individual pairs) by looking at portfolio-level correlation structure.
"""

import logging
from dataclasses import dataclass

import numpy as np

import config

logger = logging.getLogger(__name__)


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
        self._sector_map: dict[str, str] = {}  # symbol -> sector

    def set_sector_map(self, sector_map: dict[str, str]):
        """Set the symbol-to-sector mapping."""
        self._sector_map = sector_map

    def update_correlation(self, sym1: str, sym2: str, corr: float):
        """Update cached pairwise correlation."""
        key = tuple(sorted([sym1, sym2]))
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

        corr_source = correlations or self._correlation_cache
        all_symbols = open_symbols + [new_symbol]
        n = len(all_symbols)

        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                key = tuple(sorted([all_symbols[i], all_symbols[j]]))
                c = corr_source.get(key, 0.0)
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

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
        sector_weights = {}
        for sym in all_symbols:
            sector = self._sector_map.get(sym, "unknown")
            sector_weights[sector] = sector_weights.get(sector, 0) + 1.0 / n

        hhi = sum(w ** 2 for w in sector_weights.values())
        max_sector = max(sector_weights.values())

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

    @property
    def status(self) -> dict:
        return {
            "cached_correlations": len(self._correlation_cache),
            "tracked_sectors": len(set(self._sector_map.values())),
            "max_pairwise_corr_limit": self.max_pairwise_corr,
            "min_effective_bets": self.min_effective_bets,
            "max_sector_weight": self.max_sector_weight,
        }
