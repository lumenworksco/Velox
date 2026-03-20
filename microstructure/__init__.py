"""Market microstructure analysis (Phase 7).

Provides real-time microstructure metrics for trade filtering,
position sizing, and regime detection:

- VPIN: Volume-Synchronized Probability of Informed Trading
- Order Book Imbalance: Top-of-book bid/ask imbalance
- Trade Classifier: Institutional vs. retail flow detection
- Spread Analysis: Effective spread, realized spread, adverse selection
"""

from microstructure.vpin import VPIN
from microstructure.order_book import OrderBookAnalyzer
from microstructure.trade_classifier import TradeClassifier, TradeType
from microstructure.spread_analysis import SpreadAnalyzer

__all__ = [
    "VPIN",
    "OrderBookAnalyzer",
    "TradeClassifier",
    "TradeType",
    "SpreadAnalyzer",
]
