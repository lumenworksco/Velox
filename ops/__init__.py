"""Operations package.

Provides operational support modules for tax optimization, drawdown risk
management, and disaster recovery.

Modules:
    tax_harvesting   — Tax-loss harvesting with wash sale detection
    drawdown_risk    — Drawdown metrics and dynamic exposure management
    disaster_recovery — State recovery, backup, and heartbeat monitoring
"""

from ops.tax_harvesting import TaxLossHarvester
from ops.drawdown_risk import DrawdownRiskManager
from ops.disaster_recovery import DisasterRecovery

__all__ = [
    "TaxLossHarvester",
    "DrawdownRiskManager",
    "DisasterRecovery",
]
