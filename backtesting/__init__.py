"""Advanced backtesting infrastructure (Phase 8).

Provides event-driven backtesting, Monte Carlo robustness testing,
performance attribution, and alpha decay analysis:

- EventDrivenBacktester: Bar-by-bar simulation with realistic fills
- MonteCarloTester: Trade shuffling, skipping, parameter perturbation
- PerformanceAttribution: Strategy, factor, timing, and execution analysis
- AlphaDecayAnalyzer: Signal decay curve and strategy health monitoring
"""

from backtesting.event_backtester import EventDrivenBacktester, BacktestResult
from backtesting.monte_carlo import MonteCarloTester, MonteCarloResult
from backtesting.attribution import PerformanceAttribution, AttributionReport, BrinsonFachlerResult
from backtesting.alpha_decay import AlphaDecayAnalyzer, DecayCurve

__all__ = [
    "EventDrivenBacktester",
    "BacktestResult",
    "MonteCarloTester",
    "MonteCarloResult",
    "PerformanceAttribution",
    "AttributionReport",
    "BrinsonFachlerResult",
    "AlphaDecayAnalyzer",
    "DecayCurve",
]
