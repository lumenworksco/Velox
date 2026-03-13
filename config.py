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
MAX_POSITIONS = 12
MAX_PORTFOLIO_DEPLOY = 0.55
DAILY_LOSS_HALT = -0.04

# --- V2: Position Sizing (ATR-based) ---
RISK_PER_TRADE_PCT = 0.01           # Risk 1% of portfolio per trade (overridden by V6 below)
MAX_POSITION_PCT = 0.08             # Hard cap: max 8% of portfolio per position
MIN_POSITION_VALUE = 100             # Min $100 per trade

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

# --- V3: Short Selling ---
SHORT_SIZE_MULTIPLIER = 0.75         # Short positions = 75% of equivalent long size
SHORT_HARD_STOP_PCT = 0.04           # Close short if goes against you > 4%
MOMENTUM_TRAILING_STOP_PCT = 0.02    # Legacy: trailing stop for position monitor
NO_SHORT_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA"}  # Never short broad market ETFs

# --- V3: Dynamic Capital Allocation ---
DYNAMIC_ALLOCATION = os.getenv("DYNAMIC_ALLOCATION", "true") == "true"
ALLOCATION_LOOKBACK_DAYS = 20        # Rolling window for Sharpe-based allocation
ALLOCATION_MIN_WEIGHT = 0.10         # Min 10% per strategy
ALLOCATION_RECALC_TIME = time(9, 0)  # Recalculate at 9:00 AM ET

# --- V3: WhatsApp Notifications (via Meta Cloud API) ---
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false") == "true"
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_RECIPIENT_NUMBER = os.getenv("WHATSAPP_RECIPIENT_NUMBER", "")

# --- V3: Web Dashboard ---
WEB_DASHBOARD_ENABLED = os.getenv("WEB_DASHBOARD_ENABLED", "true") == "true"
WEB_DASHBOARD_PORT = int(os.getenv("WEB_DASHBOARD_PORT", "8080"))

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

# --- V4: Async Mode ---
ASYNC_MODE = os.getenv("ASYNC_MODE", "false") == "true"

# =============================================================================
# V6 ADDITIONS — Consistent Returns Rebuild
# =============================================================================

# --- V6: Strategy Allocations ---
STRATEGY_ALLOCATIONS = {
    'STAT_MR': 0.60,
    'KALMAN_PAIRS': 0.25,
    'MICRO_MOM': 0.15,
}

# --- V6: Statistical Mean Reversion ---
MR_ZSCORE_ENTRY = 1.5
MR_ZSCORE_EXIT_FULL = 0.2
MR_ZSCORE_EXIT_PARTIAL = 0.5
MR_ZSCORE_STOP = 2.5
MR_RSI_PERIOD = 7
MR_RSI_OVERSOLD = 40
MR_RSI_OVERBOUGHT = 60
MR_HURST_MAX = 0.52
MR_ADF_PVALUE_MAX = 0.05
MR_HALFLIFE_MIN_HOURS = 1
MR_HALFLIFE_MAX_HOURS = 48
MR_UNIVERSE_SIZE = 40
MR_UNIVERSE_PREP_TIME = time(9, 0)
MR_SCAN_INTERVAL_SEC = 120
MR_MIN_GAIN_PCT = 0.002
MR_MIN_RR_RATIO = 1.5

# --- V6: Kalman Pairs Trading ---
PAIRS_ZSCORE_ENTRY = 2.0
PAIRS_ZSCORE_EXIT = 0.2
PAIRS_ZSCORE_STOP = 3.0
PAIRS_MAX_HOLD_DAYS = 10
PAIRS_MAX_ACTIVE = 15
PAIRS_CAPITAL_PCT = 0.05
PAIRS_MIN_CORRELATION = 0.80
PAIRS_COINT_PVALUE = 0.05
PAIRS_SCAN_INTERVAL_SEC = 120
KALMAN_DELTA = 1e-4
KALMAN_OBS_NOISE = 0.001

# --- V6: Sector Groups for Pair Selection ---
SECTOR_GROUPS = {
    'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
    'semis': ['NVDA', 'AMD', 'INTC'],
    'banks': ['JPM', 'BAC', 'GS', 'V', 'MA'],
    'energy': ['XOM', 'CVX'],
    'fintech': ['PYPL', 'SQ', 'SOFI', 'COIN', 'AFRM', 'HOOD'],
    'cloud_saas': ['CRM', 'NOW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'PANW', 'ZS'],
    'consumer': ['UBER', 'LYFT', 'ABNB', 'DASH'],
    'etf_pairs': [('SPY', 'QQQ'), ('IWM', 'QQQ'), ('XLK', 'QQQ'), ('GLD', 'SLV')],
}

# --- V6: Intraday Micro-Momentum ---
MICRO_SPY_VOL_SPIKE_MULT = 3.0
MICRO_SPY_MIN_MOVE_PCT = 0.0015
MICRO_MAX_HOLD_MINUTES = 8
MICRO_STOP_PCT = 0.003
MICRO_TARGET_PCT = 0.006
MICRO_MAX_TRADES_PER_EVENT = 3
MICRO_MAX_DAILY_GAIN_DISABLE = 0.015
MICRO_TOP_BETA_STOCKS = 5
MICRO_POSITION_SIZE_PCT = 0.02

# --- V6: Volatility Targeting ---
VOL_TARGET_DAILY = 0.01
VOL_TARGET_MAX = 0.015
VOL_SCALAR_MIN = 0.3
VOL_SCALAR_MAX = 1.5
RISK_PER_TRADE_PCT = 0.008

# --- V6: Daily P&L Lock ---
PNL_GAIN_LOCK_PCT = 0.015
PNL_LOSS_HALT_PCT = -0.010
PNL_GAIN_LOCK_SIZE_MULT = 0.30

# --- V6: Beta Neutralization ---
BETA_MAX_ABS = 0.3
BETA_CHECK_INTERVAL_MIN = 15
BETA_SKIP_FIRST_MINUTES = 15

# --- V6: Scan Configuration ---
SCAN_INTERVAL_SEC = 120

# --- WebSocket Position Monitoring ---
WEBSOCKET_MONITORING = True
WEBSOCKET_RECONNECT_SEC = 5

# --- Runtime-mutable strategy parameters (can be updated by optimizer) ---
_runtime_params: dict = {}

def get_param(key: str, default=None):
    """Get a runtime parameter (optimizer-modified or config default)."""
    return _runtime_params.get(key, default)

def set_param(key: str, value):
    """Set a runtime parameter (used by optimizer)."""
    _runtime_params[key] = value
