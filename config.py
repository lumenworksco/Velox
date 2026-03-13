"""Configuration — all settings and environment variable handling."""

import os
from datetime import time
from zoneinfo import ZoneInfo

# --- Timezone ---
ET = ZoneInfo("America/New_York")

# --- Alpaca API ---
PAPER_MODE = os.getenv("ALPACA_LIVE", "false") != "true"
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

BASE_URL = (
    "https://paper-api.alpaca.markets"
    if PAPER_MODE
    else "https://api.alpaca.markets"
)
DATA_URL = "https://data.alpaca.markets"

# --- Trading ---
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false") == "true"
MAX_POSITIONS = 10
MAX_PORTFOLIO_DEPLOY = 0.40
TRADE_SIZE_PCT = 0.03               # Legacy flat sizing (kept for reference)
DAILY_LOSS_HALT = -0.04
SCAN_INTERVAL_SEC = 60

# --- V2: Position Sizing (ATR-based) ---
RISK_PER_TRADE_PCT = 0.01           # Risk 1% of portfolio per trade
MAX_POSITION_PCT = 0.08             # Hard cap: max 8% of portfolio per position
MIN_POSITION_VALUE = 100             # Min $100 per trade

# --- Strategy Filters ---
MAX_GAP_PCT = 0.04                  # Skip ORB if gap > 4%
MAX_ORB_RANGE_PCT = 0.035           # Skip ORB if range > 3.5% of price
MAX_INTRADAY_MOVE_PCT = 0.04        # Skip VWAP if stock moved > 4% today

# --- ORB Settings ---
ORB_VOLUME_MULTIPLIER = 1.2
ORB_TAKE_PROFIT_MULT = 1.5
ORB_STOP_LOSS_MULT = 0.5
ORB_TOP_N_SYMBOLS = 15
ORB_ENTRY_SLIPPAGE = 0.0005
ORB_RSI_MIN = 50                         # V5: min RSI for ORB entry
ORB_RSI_MAX = 78                         # V5: max RSI for ORB entry

# --- VWAP Settings ---
VWAP_BAND_STD = 1.2
VWAP_RSI_OVERSOLD = 45
VWAP_RSI_OVERBOUGHT = 55
VWAP_STOP_EXTENSION = 0.5
VWAP_TIME_STOP_MINUTES = 60

# --- V5: Minimum Stop/TP Distances ---
ORB_MIN_STOP_PCT = 0.003                 # 0.3% minimum stop distance
ORB_MIN_TP_PCT = 0.006                   # 0.6% minimum TP distance
VWAP_MIN_STOP_PCT = 0.0025              # 0.25% minimum stop distance
VWAP_MIN_TP_PCT = 0.004                  # 0.4% minimum TP distance

# --- V5: Entry Improvements ---
ORB_PULLBACK_ENTRY = True                # Wait for pullback after breakout
ORB_PULLBACK_TOLERANCE = 0.002           # Allow 0.2% overshoot on pullback
ORB_PULLBACK_TIMEOUT = 2                 # Enter at market after 2 scans
VWAP_CONFIRMATION_BARS = 2               # Require 2-bar confirmation

# --- V2: Momentum Settings ---
ALLOW_MOMENTUM = os.getenv("ALLOW_MOMENTUM", "true") == "true"
MAX_MOMENTUM_POSITIONS = 1          # Only 1 swing position at once
MOMENTUM_MIN_MOVE_PCT = 0.04        # Yesterday must move > +4%
MOMENTUM_VOL_MULTIPLIER = 2.0       # Volume > 2x 30-day average
MOMENTUM_CONSOLIDATION_PCT = 0.015  # Today within 1.5% of yesterday's close
MOMENTUM_MAX_STOP_PCT = 0.02        # Max -2% stop from entry
MOMENTUM_TP1_PCT = 0.03             # Sell 50% at +3%
MOMENTUM_TP2_PCT = 0.06             # Sell remaining at +6%
MOMENTUM_TRAILING_STOP_PCT = 0.015  # 1.5% trailing stop
MOMENTUM_MAX_HOLD_DAYS = 5          # Exit on day 5 regardless
MOMENTUM_SCAN_TIME = time(10, 30)   # Scan once at 10:30 AM

# --- Market Regime ---
REGIME_CHECK_INTERVAL_MIN = 30
REGIME_EMA_PERIOD = 20
BEARISH_SIZE_CUT = 0.40

# --- V2: Filters ---
EARNINGS_FILTER_DAYS = 2            # Skip symbols with earnings within 2 days
CORRELATION_THRESHOLD = 0.88        # Skip if correlated > 88% with open position

# --- Market Hours (ET) ---
MARKET_OPEN = time(9, 30)
ORB_END = time(10, 0)
TRADING_START = time(10, 0)
ORB_EXIT_TIME = time(15, 45)
MARKET_CLOSE = time(16, 0)
EOD_SUMMARY_TIME = time(16, 15)

# --- Core Symbol Universe (original 50) ---
CORE_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "NFLX", "AMD", "INTC", "SOFI", "PLTR", "COIN", "PYPL",
    "SNAP", "BABA", "JD", "SHOP", "JPM", "BAC", "GS", "V", "MA",
    "UNH", "CVX", "XOM", "LLY", "PFE", "SQ", "UBER", "LYFT", "ABNB",
    "RBLX", "AFRM", "HOOD", "CRWD", "PANW", "ZS", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "ARKK", "GLD", "SLV", "TLT",
]

# --- V2: Extended Universe (+100) ---
LARGE_CAP_GROWTH = [
    "ADBE", "CRM", "NOW", "SNOW", "DDOG", "NET", "FTNT", "OKTA",
    "ZM", "DOCU", "TWLO", "MDB", "ESTC", "CFLT", "GTLB", "BILL",
    "HUBS", "VEEV", "WDAY", "ANSS", "TTD", "ROKU", "PINS", "ETSY",
    "W", "CHWY", "DASH", "APP",
]

SECTOR_ETFS = [
    "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "XLY", "SOXX",
    "SMH", "IBB", "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ",
    "TNA", "FAS", "FAZ", "LABU", "LABD",
]

HIGH_MOMENTUM_MIDCAPS = [
    "CELH", "SMCI", "AXON", "PODD", "ENPH", "FSLR", "RUN", "BLNK",
    "CHPT", "BE", "IONQ", "RGTI", "QUBT", "ARQQ", "BBAI", "SOUN",
    "ASTS", "RDW", "RKLB", "LUNR", "DUOL", "CAVA", "BROS", "SHAK",
    "WING", "TXRH", "CMG", "DPZ", "DNUT", "JACK",
]

# Full universe
SYMBOLS = CORE_SYMBOLS + LARGE_CAP_GROWTH + SECTOR_ETFS + HIGH_MOMENTUM_MIDCAPS

# Leveraged ETFs — ONLY use VWAP on these, never ORB or Momentum
LEVERAGED_ETFS = {
    "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ", "TNA", "FAS", "FAZ", "LABU", "LABD",
}

# Non-leveraged symbols for ORB and Momentum
STANDARD_SYMBOLS = [s for s in SYMBOLS if s not in LEVERAGED_ETFS]

# --- State & Persistence ---
STATE_FILE = "state.json"
DB_FILE = "bot.db"
LOG_FILE = "bot.log"
STATE_SAVE_INTERVAL_SEC = 60

# --- Backtest ---
BACKTEST_SLIPPAGE = 0.0005          # 0.05% slippage per trade
BACKTEST_COMMISSION = 0.0035        # $0.0035 per share
BACKTEST_RISK_FREE_RATE = 0.045     # 4.5% annual
BACKTEST_TOP_N = 20                 # Run on top 20 most liquid symbols

# =============================================================================
# V3 ADDITIONS
# =============================================================================

# --- V3: ML Signal Filter ---
USE_ML_FILTER = os.getenv("USE_ML_FILTER", "true") == "true"
ML_MIN_TRADES = 200                  # Min labeled trades before ML filter activates
ML_PROBABILITY_THRESHOLD = 0.55      # Min probability to take a trade
ML_MIN_PRECISION = 0.58              # Model must achieve > 58% precision to be used

# --- V3: Short Selling ---
SHORT_SIZE_MULTIPLIER = 0.75         # Short positions = 75% of equivalent long size
SHORT_HARD_STOP_PCT = 0.04           # Close short if goes against you > 4%
NO_SHORT_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA"}  # Never short broad market ETFs

# --- V3: Dynamic Capital Allocation ---
DYNAMIC_ALLOCATION = os.getenv("DYNAMIC_ALLOCATION", "true") == "true"
ALLOCATION_LOOKBACK_DAYS = 20        # Rolling window for Sharpe-based allocation
ALLOCATION_MIN_WEIGHT = 0.10         # Min 10% per strategy
ALLOCATION_RECALC_TIME = time(9, 0)  # Recalculate at 9:00 AM ET

# --- V3: WebSocket Monitoring ---
WEBSOCKET_MONITORING = os.getenv("WEBSOCKET_MONITORING", "true") == "true"
WEBSOCKET_RECONNECT_SEC = 60         # Reconnect interval on disconnect

# --- V3: Gap & Go Strategy ---
GAP_GO_ENABLED = os.getenv("GAP_GO_ENABLED", "true") == "true"
GAP_MIN_PCT = 0.03                   # Minimum gap size (3%)
GAP_MAX_PCT = 0.08                   # Maximum gap size (8%)
GAP_MAX_POSITIONS = 2                # Max 2 Gap & Go positions
GAP_PREMARKET_VOL_MULT = 2.0         # Pre-market vol must be 2x average
GAP_MIN_PRICE = 5.0                  # Minimum stock price $5
GAP_PREMARKET_SCAN_TIME = time(9, 0) # Pre-market scan at 9:00 AM
GAP_ENTRY_TIME = time(9, 45)         # Enter after 9:45 AM
GAP_EXIT_TIME = time(11, 30)         # Time stop at 11:30 AM
GAP_FIRST_CANDLE_MINUTES = 15        # First candle period

# --- V3: Relative Strength ---
USE_RS_FILTER = os.getenv("USE_RS_FILTER", "true") == "true"
RS_LONG_THRESHOLD = 0.2              # Long only if RS > 0.2
RS_SHORT_THRESHOLD = -0.2            # Short only if RS < -0.2

# --- V3: WhatsApp Notifications (via Meta Cloud API) ---
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false") == "true"
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_RECIPIENT_NUMBER = os.getenv("WHATSAPP_RECIPIENT_NUMBER", "")

# --- V3: Web Dashboard ---
WEB_DASHBOARD_ENABLED = os.getenv("WEB_DASHBOARD_ENABLED", "true") == "true"
WEB_DASHBOARD_PORT = int(os.getenv("WEB_DASHBOARD_PORT", "8080"))

# --- V3: Weekly Auto-Optimization ---
AUTO_OPTIMIZE = os.getenv("AUTO_OPTIMIZE", "true") == "true"
OPTIMIZE_MIN_IMPROVEMENT = 0.10      # Only update params if Sharpe improves > 10%
OPTIMIZE_LOOKBACK_WEEKS = 8          # Use last 8 weeks of data
OPTIMIZE_TIMEOUT_SEC = 7200          # Max 2 hours for optimization run

# --- V3: Sector ETF Mapping (for relative strength) ---
SECTOR_MAP = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
    "CRM": "XLK", "ADBE": "XLK", "NOW": "XLK", "PLTR": "XLK", "CRWD": "XLK",
    "PANW": "XLK", "ZS": "XLK", "SNOW": "XLK", "DDOG": "XLK", "NET": "XLK",
    "FTNT": "XLK", "OKTA": "XLK", "TWLO": "XLK", "MDB": "XLK", "ESTC": "XLK",
    "CFLT": "XLK", "GTLB": "XLK", "BILL": "XLK", "HUBS": "XLK", "VEEV": "XLK",
    "WDAY": "XLK", "ANSS": "XLK", "SMCI": "XLK", "IONQ": "XLK", "RGTI": "XLK",
    "QUBT": "XLK", "ARQQ": "XLK", "BBAI": "XLK", "SOUN": "XLK", "APP": "XLK",
    # Semiconductors (more specific)
    "SOXX": "SMH", "SMH": "SMH", "SOXL": "SMH",
    # Communication Services
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "SNAP": "XLC", "TTD": "XLC",
    "ROKU": "XLC", "PINS": "XLC", "ZM": "XLC", "DOCU": "XLC", "DUOL": "XLC",
    # Consumer Discretionary
    "TSLA": "XLY", "AMZN": "XLY", "SHOP": "XLY", "BABA": "XLY", "JD": "XLY",
    "UBER": "XLY", "LYFT": "XLY", "ABNB": "XLY", "RBLX": "XLY", "ETSY": "XLY",
    "W": "XLY", "CHWY": "XLY", "DASH": "XLY", "CAVA": "XLY", "BROS": "XLY",
    "SHAK": "XLY", "WING": "XLY", "TXRH": "XLY", "CMG": "XLY", "DPZ": "XLY",
    "DNUT": "XLY", "JACK": "XLY",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "V": "XLF", "MA": "XLF",
    "PYPL": "XLF", "SQ": "XLF", "SOFI": "XLF", "COIN": "XLF", "AFRM": "XLF",
    "HOOD": "XLF",
    # Healthcare
    "UNH": "XLV", "LLY": "XLV", "PFE": "XLV", "PODD": "XLV",
    # Energy
    "CVX": "XLE", "XOM": "XLE",
    # Clean Energy / EV
    "ENPH": "XLE", "FSLR": "XLE", "RUN": "XLE", "BLNK": "XLE",
    "CHPT": "XLE", "BE": "XLE",
    # Biotech
    "IBB": "IBB", "LABU": "IBB", "LABD": "IBB",
    # Aerospace / Space
    "AXON": "XLI", "ASTS": "XLI", "RDW": "XLI", "RKLB": "XLI", "LUNR": "XLI",
    # Consumer / Other
    "CELH": "XLP",
}

# =============================================================================
# V4 ADDITIONS
# =============================================================================

# --- V4: Multi-Timeframe Signal Confirmation ---
MTF_CONFIRMATION_ENABLED = os.getenv("MTF_CONFIRMATION_ENABLED", "true") == "true"
MTF_CACHE_SECONDS = 300              # Cache higher-TF result for 5 minutes

# --- V4: VIX-Based Dynamic Risk Scaling ---
VIX_RISK_SCALING_ENABLED = os.getenv("VIX_RISK_SCALING_ENABLED", "true") == "true"
VIX_HALT_THRESHOLD = 40              # VIX > 40 = halt all new positions
VIX_CACHE_SECONDS = 900              # Refresh VIX every 15 minutes

# --- V4: News Sentiment Filter ---
NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "true") == "true"
NEWS_LOOKBACK_HOURS = 24             # Check news from last 24 hours
NEWS_CACHE_CLEAR_TIME = time(9, 25)  # Clear stale news at 9:25 AM

# --- V4: Sector Rotation Strategy ---
SECTOR_ROTATION_ENABLED = os.getenv("SECTOR_ROTATION_ENABLED", "false") == "true"
SECTOR_ROTATION_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
]
MAX_SECTOR_POSITIONS = 4             # Max 2 long + 2 short
SECTOR_POSITION_SIZE_PCT = 0.05      # 5% of portfolio per sector position
SECTOR_STOP_PCT = 0.035              # -3.5% stop loss
SECTOR_MAX_HOLD_DAYS = 10            # Max 10 trading days
SECTOR_SCAN_TIME = time(10, 30)      # Daily scan at 10:30 AM
SECTOR_MIN_SCORE = 0.005             # Min momentum score to enter

# --- V4: Pairs Trading Strategy ---
PAIRS_TRADING_ENABLED = os.getenv("PAIRS_TRADING_ENABLED", "false") == "true"
MAX_PAIRS_POSITIONS = 3              # Max 3 active pairs
PAIRS_ZSCORE_ENTRY = 2.0             # Enter when z-score > 2.0
PAIRS_ZSCORE_EXIT = 0.5              # Exit when z-score < 0.5
PAIRS_ZSCORE_STOP = 3.5              # Stop loss: z-score > 3.5 (diverging)
PAIRS_MAX_HOLD_DAYS = 15             # Max 15 trading days
PAIRS_MIN_CORRELATION = 0.85         # Min correlation for pair selection
PAIRS_REVALIDATION_THRESHOLD = 0.75  # Drop pair if correlation < 0.75
PAIRS_DISCOVERY_SYMBOLS = 80         # Top N most liquid for pair discovery
PAIRS_COINT_PVALUE = 0.05            # Cointegration significance threshold

# --- V4: Advanced Exit Mechanics ---
ADVANCED_EXITS_ENABLED = os.getenv("ADVANCED_EXITS_ENABLED", "true") == "true"
SCALED_TP_ENABLED = True             # Scaled take profit (33%/50%/rest)
BREAKEVEN_STOP_ENABLED = True        # Move stop to breakeven after first partial
RSI_EXIT_THRESHOLD = 80              # Exit if RSI > 80 and profitable
ATR_EXPANSION_MULT = 2.0             # Exit if ATR > 2x entry ATR and losing
TRAILING_STOP_PCT = 0.015            # 1.5% trailing stop for swing positions

# --- V4: Async Mode ---
ASYNC_MODE = os.getenv("ASYNC_MODE", "false") == "true"

# =============================================================================
# V5 ADDITIONS
# =============================================================================

# --- V5: Multi-Timeframe Per-Strategy Toggle ---
MTF_ENABLED_FOR = {
    "ORB": True,
    "MOMENTUM": True,
    "SECTOR_ROTATION": True,
    "VWAP": False,           # Mean reversion is counter-trend by nature
    "GAP_GO": False,         # Gap trades in first hour, trend not established
    "PAIRS": False,          # Market neutral — trend irrelevant
    "EMA_SCALP": False,      # Ribbon defines its own trend
}

# --- V5: Focus List (25 highest-ATR, highest-volume symbols) ---
FOCUS_LIST = [
    "NVDA", "TSLA", "AMD", "META", "AMZN", "GOOGL", "MSFT", "AAPL",
    "SPY", "QQQ", "COIN", "PLTR", "SOFI", "SMCI", "CRWD",
    "SOXL", "TQQQ", "ARKK", "IWM", "XLK",
    "AXON", "DDOG", "NET", "SNOW", "PANW",
]

# --- V5: Two-Tier Scan Frequency ---
TIER1_SCAN_INTERVAL_SEC = 30             # FOCUS_LIST scan interval
TIER2_SCAN_INTERVAL_SEC = 90             # Full universe scan interval

# --- V5: Shadow Mode ---
STRATEGY_MODES = {
    "ORB": "live",
    "VWAP": "live",
    "MOMENTUM": "live",
    "GAP_GO": "live",
    "SECTOR_ROTATION": "shadow",
    "PAIRS": "shadow",
    "EMA_SCALP": "shadow",
}
SHADOW_PROMOTE_SHARPE = 0.8              # Promote to live if shadow Sharpe > 0.8

# --- V5: EMA Ribbon Scalper ---
EMA_SCALP_ENABLED = os.getenv("EMA_SCALP_ENABLED", "true") == "true"
EMA_SCALP_PERIODS = [5, 8, 13, 21]
EMA_SCALP_VOLUME_MULT = 1.3
EMA_SCALP_MAX_POSITIONS = 3
EMA_SCALP_START_TIME = time(10, 0)
EMA_SCALP_END_TIME = time(15, 30)
EMA_SCALP_TIME_STOP_MINUTES = 30
EMA_SCALP_STOP_PCT = 0.004              # 0.4% stop
EMA_SCALP_TP_MULT = 2.0                 # 2:1 R/R (TP = 2x stop distance)
EMA_SCALP_COOLDOWN_SEC = 600            # 10-min cooldown per symbol

# --- V5: Morning Strategy Health Check ---
MORNING_HEALTH_CHECK_ENABLED = os.getenv("MORNING_HEALTH_CHECK_ENABLED", "true") == "true"
MORNING_HEALTH_CHECK_TIME = time(9, 0)
HEALTH_CHECK_LOOKBACK_DAYS = 30
HEALTH_CHECK_MIN_TRADES = 10
HEALTH_CHECK_MIN_SHARPE = 0.0

# --- V3: Runtime-mutable strategy parameters (can be updated by optimizer) ---
_runtime_params: dict = {}

def get_param(key: str, default=None):
    """Get a runtime parameter (optimizer-modified or config default)."""
    return _runtime_params.get(key, default)

def set_param(key: str, value):
    """Set a runtime parameter (used by optimizer)."""
    _runtime_params[key] = value
