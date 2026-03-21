"""PROD-015: Options strategy foundations.

Provides framework classes for options-based strategies:
- CoveredCallWriter: Sells covered calls against long equity positions.
- ProtectivePutBuyer: Buys protective puts for downside hedging.
- GammaScalper: Delta-neutral gamma scalping via options + underlying.

These are framework stubs — full execution requires broker API support
for options order types (which Alpaca does not yet fully support).
"""

from strategies.options.covered_calls import CoveredCallWriter
from strategies.options.protective_puts import ProtectivePutBuyer
from strategies.options.gamma_scalp import GammaScalper

__all__ = [
    "CoveredCallWriter",
    "ProtectivePutBuyer",
    "GammaScalper",
]
