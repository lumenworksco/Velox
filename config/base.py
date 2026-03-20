"""ARCH-006: Immutable base constants — market hours, exchange info, system limits.

These values NEVER change at runtime and are not loaded from YAML.
They represent fundamental market structure and exchange rules.
"""

from datetime import time
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# US equity market hours (Eastern Time)
# ---------------------------------------------------------------------------

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

# Trading windows used by the bot
TRADING_START = time(10, 0)       # Avoid first 30 min volatility
ORB_EXIT_TIME = time(15, 45)     # Close ORB positions before close
EOD_SUMMARY_TIME = time(16, 15)  # End-of-day reporting
MR_UNIVERSE_PREP_TIME = time(9, 0)  # Pre-market universe selection

# ---------------------------------------------------------------------------
# Exchange & settlement
# ---------------------------------------------------------------------------

SETTLEMENT_DAYS = 1              # T+1 settlement for US equities
MIN_TICK_SIZE = 0.01             # Minimum price increment
LOT_SIZE = 1                    # US equities trade in single shares

# ---------------------------------------------------------------------------
# Alpaca-specific
# ---------------------------------------------------------------------------

ALPACA_BASE_URL_LIVE = "https://api.alpaca.markets"
ALPACA_BASE_URL_PAPER = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# ---------------------------------------------------------------------------
# System limits (safety rails)
# ---------------------------------------------------------------------------

MAX_SYMBOLS_TOTAL = 200          # Hard cap on universe size
MAX_OPEN_ORDERS = 50             # Hard cap on concurrent open orders
MIN_ORDER_VALUE_USD = 1.0        # Alpaca minimum
MAX_ORDER_VALUE_USD = 1_000_000  # Sanity cap

# ---------------------------------------------------------------------------
# PDT rule
# ---------------------------------------------------------------------------

PDT_EQUITY_THRESHOLD = 25_000.0  # Below this, 3 day-trades per 5 business days
PDT_MAX_DAY_TRADES = 3           # Per rolling 5-day window

# ---------------------------------------------------------------------------
# Holidays (NYSE observed) — update annually
# ---------------------------------------------------------------------------

NYSE_EARLY_CLOSE_TIME = time(13, 0)  # 1 PM ET on half-days
