# Velox V11: Institutional-Grade Quant System — Complete Upgrade Roadmap

**Document Purpose:** This document contains every change, addition, removal, and bug fix needed to transform Velox V10 into **Velox V11** — an institutional-grade quantitative trading system capable of competing with mid-tier quant hedge funds and outperforming 99% of algorithmic trading systems in production.

**How to use this document:** Each section is self-contained. Work top-down within each phase. Items marked 🔴 are critical bugs that should be fixed immediately. Items marked 🟡 are high-priority enhancements. Items marked 🟢 are forward-looking upgrades. After implementing everything in this document, the system should be renamed to **Velox V11**.

**Estimated total effort:** 20-28 developer-weeks across a full team.

**Target outcome:** A system with 200+ features, 6+ strategy families, ML-enhanced alpha, institutional risk management, and robust execution — capable of delivering consistent risk-adjusted returns (target Sharpe > 2.0) across market regimes.

---

## Table of Contents

1. [Phase 0: Critical Bug Fixes (Fix Immediately)](#phase-0-critical-bug-fixes) — 14 critical bugs
2. [Phase 1: Architecture & Infrastructure Overhaul](#phase-1-architecture--infrastructure-overhaul) — 9 items
3. [Phase 2: Data Infrastructure Upgrade](#phase-2-data-infrastructure-upgrade) — 6 items
4. [Phase 3: Advanced Alpha Generation](#phase-3-advanced-alpha-generation) — 8 items
5. [Phase 4: Portfolio Construction & Risk Management](#phase-4-portfolio-construction--risk-management) — 8 items
6. [Phase 5: Execution Engine Upgrade](#phase-5-execution-engine-upgrade) — 6 items
7. [Phase 6: Machine Learning Pipeline](#phase-6-machine-learning-pipeline) — 5 items
8. [Phase 7: Market Microstructure Analysis](#phase-7-market-microstructure-analysis) — 4 items
9. [Phase 8: Backtesting & Research Framework](#phase-8-backtesting--research-framework) — 6 items
10. [Phase 9: Monitoring, Observability & Alerting](#phase-9-monitoring-observability--alerting) — 5 items
11. [Phase 10: Compliance & Audit Infrastructure](#phase-10-compliance--audit-infrastructure) — 4 items
12. [Phase 11: Testing & Quality Assurance](#phase-11-testing--quality-assurance) — 4 items
13. [Phase 12: Advanced Data Science (López de Prado Framework)](#phase-12-advanced-data-science-techniques-lópez-de-prado-framework) — 5 items
14. [Phase 13: Advanced ML & AI Techniques](#phase-13-advanced-ml--ai-techniques) — 6 items
15. [Phase 14: Additional Strategy Modules](#phase-14-additional-strategy-modules) — 3 items
16. [Phase 15: Operational Excellence](#phase-15-operational-excellence) — 5 items
17. [Phase 16: Additional Bug Fixes & Code Quality](#phase-16-additional-bug-fixes--code-quality) — 12 items
18. [What to Remove or Replace](#what-to-remove-or-replace) — 7 items
19. [Appendix: Competition Benchmark](#appendix-competition-benchmark)

**Total items: 115+ actionable tasks across 18 phases**

---

## Phase 0: Critical Bug Fixes

These are production bugs that can lose money or cause system failures. Fix before anything else.

### 🔴 BUG-001: Circuit Breaker Tier Never De-escalates
**File:** `risk/circuit_breaker.py` (Lines 60-104)
**Problem:** Tiers only escalate based on P&L thresholds. If daily P&L recovers from -3% to -1%, the circuit breaker stays at RED instead of dropping back to YELLOW. This means once a bad morning triggers RED, the bot is effectively dead for the rest of the day even if the market recovers.
**Fix:** Add de-escalation logic that checks current P&L against each tier threshold and sets the appropriate tier. The tier should always reflect the current P&L level, not the worst P&L seen. Add a hysteresis buffer (e.g., 0.2%) to prevent rapid oscillation between tiers.

### 🔴 BUG-002: Circuit Breaker Tier Sort Order Incorrect
**File:** `risk/circuit_breaker.py` (Line 85)
**Problem:** Iterates `for tier in sorted(self.tiers.keys(), reverse=True)` but `CircuitTier.NORMAL` (value 0) comes first in reverse sorted order when it should come last. The loop logic may skip the NORMAL tier, causing the bot to stay at a higher tier than appropriate when P&L is positive.
**Fix:** Rewrite the tier determination as a simple descending threshold check: iterate from BLACK → RED → ORANGE → YELLOW → NORMAL, returning the first tier whose threshold is exceeded.

### 🔴 BUG-003: Partial Exit Counter Never Incremented
**File:** `exit_manager.py` (Lines 113-142)
**Problem:** Scaled take-profit logic gates on `trade.partial_exits == 0` and `trade.partial_exits == 1` to determine which partial exit level to apply, but `partial_exits` is never incremented after a partial exit executes. This means the second partial exit level (z < 0.2) will never trigger, and the bot will keep attempting the first partial exit repeatedly.
**Fix:** After each successful partial exit order submission, increment `trade.partial_exits += 1`. Add a guard to skip partial exit logic if `trade.partial_exits >= max_partial_levels`.

### 🔴 BUG-004: Undefined Private Attribute on TradeRecord
**File:** `exit_manager.py` (Line 128)
**Problem:** Sets `trade._partial_closed_qty` as a private attribute via string concatenation, but `TradeRecord` never initializes `_partial_closed_qty`. Subsequent reads of this attribute will raise `AttributeError`, or if caught silently, partial quantity tracking is broken.
**Fix:** Add `partial_closed_qty: int = 0` to the `TradeRecord` dataclass definition. Use proper integer arithmetic instead of string concatenation.

### 🔴 BUG-005: Division by Zero in Multiple Locations
**Files and locations:**
- `risk/vol_targeting.py` (Lines 115-120): No guard against `entry_price <= 0` before `shares = risk_dollars / risk_per_share`
- `backtester.py` (Lines 95-96): `range_pct = (orb_high - orb_low) / ((orb_high + orb_low) / 2)` — midpoint can be zero
- `strategies/orb_v2.py` (Lines 81-82): `range_pct = (orb_high - orb_low) / orb_low` — `orb_low` can be zero
- `risk/kelly.py` (Lines 46-47): Guards against `avg_loss < 1e-8` by setting to `1e-6`, but `win_loss_ratio` calculation still uses the original unguarded `avg_loss` if it's between 1e-8 and 1e-6

**Fix:** Add explicit zero-guards before every division operation. Create a utility function `safe_divide(numerator, denominator, default=0.0)` and use it project-wide. For Kelly, ensure the guarded value is used consistently in all downstream calculations.

### 🔴 BUG-006: Race Condition in Exit Manager
**File:** `exit_manager.py` (Lines 69-72)
**Problem:** `_evaluate_trade()` modifies `trade.highest_price_seen` and `trade.lowest_price_seen` without synchronization. The main scan loop and WebSocket handler can race on these fields, causing corrupted trailing stop calculations.
**Fix:** Either use a lock per trade object, or make TradeRecord fields thread-safe using `threading.Lock` for mutable state. Alternatively, copy trade state into a local snapshot before evaluation.

### 🔴 BUG-007: Race Condition in Risk Manager
**File:** `risk/risk_manager.py`
**Problem:** `_strategy_weights` dict is accessed without lock in `update_equity()` and other methods, but modified elsewhere under `_lock`. Inconsistent locking allows concurrent modification and potential `RuntimeError: dictionary changed size during iteration`.
**Fix:** Ensure ALL accesses to `_strategy_weights` (reads and writes) are protected by the same lock. Consider using `threading.RLock` to allow recursive locking within the same thread.

### 🔴 BUG-008: Thread-Unsafe Singleton Pattern
**File:** `engine/signal_processor.py` (Lines 30-37)
**Problem:** `_seasonality_instance` singleton is created without a lock. Two threads calling `_get_seasonality()` simultaneously could create two instances, leading to duplicated state and incorrect seasonality scoring.
**Fix:** Use a `threading.Lock` around the singleton creation, or use Python's `functools.lru_cache` which is thread-safe for creation (though not for the cached object's methods).

### 🔴 BUG-009: Timezone Bugs Throughout Codebase
**Files:**
- `database.py` (Lines 14-23): `_to_iso()` adds timezone only if `dt.tzinfo is None`, but doesn't validate the timezone is Eastern. Datetimes already aware in UTC or another zone won't be converted.
- `data_quality.py` (Lines 61-69): Compares timezone-aware `now` with potentially timezone-naive `last_bar_time`, causing `TypeError`.
- `oms/order.py` (Lines 70-88): Uses `datetime.now()` which returns naive datetime. Should be `datetime.now(config.ET)`.

**Fix:** Establish a project-wide timezone policy: ALL datetimes must be timezone-aware in US/Eastern. Create a utility `now_et() -> datetime` and `ensure_et(dt) -> datetime` that converts any datetime to Eastern. Replace every `datetime.now()` and `datetime.utcnow()` call with `now_et()`. Add a linting rule to flag naive datetime usage.

### 🔴 BUG-010: Data Cache Race Condition
**File:** `data_cache.py` (Lines 98-100)
**Problem:** LRU eviction happens under lock, but `move_to_end()` is called after checking TTL without holding the lock. Another thread could evict between the check and the move.
**Fix:** Hold the lock for the entire get-check-move operation. Use `with self._lock:` to wrap the full cache lookup including TTL check and LRU update.

### 🔴 BUG-011: TWAP Orders Not Tracked in OMS
**File:** `execution.py` (Lines 122-160)
**Problem:** Submits multiple TWAP slices but doesn't register them individually in the OMS. If one slice fails, the bot doesn't know the partial fill percentage. This can lead to position size mismatches between the bot's internal state and the broker.
**Fix:** Register each TWAP slice as a child order in the OMS, linked to a parent order. Track aggregate fill percentage. If any slice fails, cancel remaining slices and calculate actual position from filled slices.

### 🔴 BUG-012: Idempotency Key Can Be Empty
**File:** `oms/order_manager.py` (Lines 30-62)
**Problem:** Accepts empty `idempotency_key`. If multiple callers submit with empty key, all are treated as unique, completely defeating the idempotency protection that prevents duplicate orders.
**Fix:** If no idempotency key is provided, auto-generate one from `f"{symbol}_{side}_{qty}_{strategy}_{timestamp_bucket}"` where timestamp_bucket rounds to the nearest 5 seconds. This ensures near-simultaneous duplicate submissions are caught.

### 🔴 BUG-013: close_all_positions() Return Value Ambiguity
**File:** `execution.py` (Line 215)
**Problem:** Returns 0 on success but -1 on error. Callers can't distinguish "closed 0 positions successfully" from "failed." The circuit breaker's RED/BLACK tier handlers depend on this function working correctly.
**Fix:** Return a tuple `(success: bool, closed_count: int, failed_symbols: List[str])` or raise an exception on failure.

### 🔴 BUG-014: Kalman Filter Dimension Mismatch
**File:** `strategies/kalman_pairs.py` (Lines 94-99)
**Problem:** Initializes Kalman state with `theta: np.array([hedge_ratio, 0.0])` (2-element), but there's no validation that hedge_ratio is scalar. If hedge_ratio is itself an array (from OLS), the theta vector will be wrong dimensions, causing silent linear algebra errors.
**Fix:** Explicitly cast `hedge_ratio = float(hedge_ratio)` before initialization. Add a dimension assertion after Kalman update.

---

## Phase 1: Architecture & Infrastructure Overhaul

### 🟡 ARCH-001: Decompose main.py God Module
**File:** `main.py` (800+ lines)
**Problem:** Contains inline logic for strategy management, order submission, exit handling, scheduling, and lifecycle management all in one file.
**Action:** Split into:
- `engine/strategy_manager.py` — Strategy registration, lifecycle, scan orchestration
- `engine/scheduler.py` — Time-based task scheduling (universe prep, EOD close, weekly tasks)
- `engine/lifecycle.py` — Startup, shutdown, health check coordination
- `main.py` — Entry point only (argument parsing, component wiring, start)

### 🟡 ARCH-002: Implement Dependency Injection
**Problem:** All modules use global clients (`_trading_client`, `_data_client` in `data.py`). Makes testing impossible without monkeypatch. Makes it impossible to run multiple bot instances with different configurations.
**Action:**
- Create a `Container` class that holds all shared dependencies (broker client, data client, database, risk manager, etc.)
- Pass the container to each component's constructor
- Replace all global module-level clients with injected dependencies
- Use a lightweight DI framework like `dependency-injector` or roll your own

### 🟡 ARCH-003: Event-Driven Architecture
**Problem:** Current scan loop polls every 120 seconds regardless of market activity. Misses fast-moving opportunities and wastes resources during quiet periods.
**Action:**
- Implement an event bus using `asyncio` or a message queue (e.g., `aioredis` pub/sub)
- Events: `BarUpdate`, `QuoteUpdate`, `TradeUpdate`, `OrderFill`, `RiskAlert`, `RegimeChange`
- Strategies subscribe to relevant events instead of polling
- Execution engine reacts to fill events immediately
- Risk manager subscribes to all position-changing events
- This is how firms like Jane Street and Jump Trading architect their systems

### 🟡 ARCH-004: Async I/O for Network Operations
**Problem:** All Alpaca API calls are synchronous, blocking the main thread. Scanning 130+ symbols takes 30-60 seconds per cycle.
**Action:**
- Convert data fetching to async using `aiohttp` or `httpx`
- Use `asyncio.gather()` to parallelize bar fetches across symbols
- Use `asyncio.TaskGroup` for strategy signal generation
- Expected speedup: 5-10x on scan cycle time
- Keep risk checks synchronous (they must complete before order submission)

### 🟡 ARCH-005: Strategy Interface Abstraction
**Problem:** Strategies have inconsistent interfaces. Some return signal dicts, others return tuples. No formal contract.
**Action:**
- Define an abstract `Strategy` base class with methods:
  - `prepare_universe(date) -> List[str]`
  - `generate_signals(bars: Dict[str, DataFrame]) -> List[Signal]`
  - `get_exit_params(trade: Trade) -> ExitParams`
  - `reset_daily()`
  - `get_metadata() -> StrategyMetadata`
- Create a `Signal` dataclass: `(symbol, side, strategy, confidence, entry_price, stop_loss, take_profit, metadata)`
- All strategies must implement this interface
- Makes adding new strategies trivial and testable

### 🟡 ARCH-006: Configuration Management Overhaul
**File:** `config.py`
**Problem:** 500+ line config file mixing constants, runtime parameters, API keys, and strategy parameters. Hardcoded values make optimization impossible.
**Action:**
- Split into:
  - `config/base.py` — Immutable constants (market hours, exchange info)
  - `config/strategies.yaml` — Strategy parameters (loadable, versionable, optimizable)
  - `config/risk.yaml` — Risk parameters
  - `config/universe.yaml` — Symbol universe definitions
  - `config/secrets.env` — API keys (loaded via `python-dotenv`, never committed)
- Implement a `ConfigManager` class that validates and provides typed access
- Support environment-based overrides (dev/staging/production)
- Enable hot-reloading of strategy parameters without restart

### 🟡 ARCH-007: Database Layer Upgrade
**File:** `database.py`
**Problem:** Raw SQLite with hand-written queries. No migrations, no connection pooling, no query optimization. Concurrent access from watchdog thread causes "database is locked" errors.
**Action:**
- Replace with SQLAlchemy ORM with proper models
- Add Alembic for schema migrations
- Use connection pooling (`sqlalchemy.pool.QueuePool`)
- For production: migrate to PostgreSQL (supports concurrent writes, JSON columns, window functions)
- Add TimescaleDB extension for time-series data (bars, trades, P&L)
- Define proper indices on frequently-queried columns (symbol, timestamp, strategy)

### 🟡 ARCH-008: Secrets Management
**Files:** `config.py` (Lines 14-15, 455)
**Problem:** API keys stored in environment variables with no encryption. Anthropic API key in config passed in plaintext.
**Action:**
- Implement a secrets manager abstraction supporting:
  - Local: `python-dotenv` with `.env` file (dev)
  - AWS: AWS Secrets Manager (production)
  - Generic: HashiCorp Vault
- Never log or print secrets
- Rotate API keys on a schedule
- Add a pre-commit hook that scans for hardcoded secrets

### 🟢 ARCH-009: Microservices Consideration
**Long-term:** For true institutional scale, consider splitting into separate services:
- **Data Service** — Bar fetching, caching, alternative data ingestion
- **Signal Service** — Strategy execution, signal generation
- **Risk Service** — Portfolio risk, position limits, circuit breakers
- **Execution Service** — Order management, smart routing, fill tracking
- **Analytics Service** — Performance attribution, reporting, dashboards
- Communication via gRPC or Apache Kafka
- Enables independent scaling (e.g., more GPU for ML signal service)

---

## Phase 2: Data Infrastructure Upgrade

### 🟡 DATA-001: Tick-Level Data Pipeline
**Problem:** Bot operates on 1-min and 2-min bars. Misses intrabar dynamics that institutional systems exploit.
**Action:**
- Subscribe to Alpaca's WebSocket for real-time trades and quotes
- Build an in-memory order book from quote updates (top-of-book at minimum)
- Aggregate ticks into custom bars (volume bars, dollar bars, tick bars) — these have better statistical properties than time bars per Marcos López de Prado's research
- Store raw ticks in a time-series database (TimescaleDB, InfluxDB, or QuestDB)
- Build a configurable bar aggregation engine that strategies can query

### 🟡 DATA-002: Feature Store
**Problem:** Each strategy independently computes features from raw bars. Duplicated computation, no feature reuse, no feature versioning.
**Action:**
- Build a centralized feature store:
  - **Online store:** Redis or in-memory dict for real-time feature serving (<1ms latency)
  - **Offline store:** PostgreSQL/Parquet files for historical features (backtesting)
  - **Feature registry:** Catalog of all computed features with metadata (name, version, computation logic, staleness threshold)
- Features should be computed once and served to all strategies
- Examples: RSI(7), RSI(14), VWAP deviation, ATR(14), OU z-score, Hurst exponent, volume ratio, bid-ask spread, order flow imbalance
- Use a DAG-based computation engine (like `dagster` or custom) to manage feature dependencies

### 🟡 DATA-003: Alternative Data Integration
**Problem:** Bot only uses price/volume data and Alpaca news headlines. Institutional systems use 10+ alternative data sources.
**Action — implement in priority order:**

1. **SEC Filing Analysis (EDGAR)**
   - Fetch 10-K, 10-Q, 8-K filings via SEC EDGAR API
   - NLP extraction: revenue guidance changes, risk factor changes, insider transactions
   - Compare filing language to prior quarter (semantic diff)
   - Signal: Negative language increase → bearish, positive → bullish

2. **Options Flow Data**
   - Source: CBOE DataShop, Unusual Whales API, or Alpaca options data
   - Track unusual options activity: large OTM call/put sweeps, put/call ratio spikes
   - Signal: Unusual call volume → bullish institutional positioning

3. **Short Interest Data**
   - Source: FINRA short interest reports (bi-monthly), Ortex for real-time estimates
   - Track days-to-cover, short interest % of float
   - Signal: Rising short interest + rising price → potential squeeze

4. **Institutional Holdings (13F)**
   - Source: SEC EDGAR 13F filings (quarterly)
   - Track position changes by top funds
   - Signal: Smart money accumulation/distribution patterns

5. **Social Sentiment (Advanced)**
   - Source: Twitter/X API, Reddit API (r/wallstreetbets, r/stocks), StockTwits
   - NLP sentiment scoring using FinBERT (finance-tuned BERT)
   - Track sentiment velocity (rate of change, not just level)
   - Signal: Sentiment divergence from price → potential reversal

6. **Macroeconomic Data**
   - Source: FRED API (Federal Reserve Economic Data)
   - Track: CPI, PPI, NFP, unemployment claims, ISM, housing starts
   - Pre-compute economic surprise index (actual vs. consensus)
   - Use for regime detection enhancement

7. **Satellite & Transaction Data (Long-term)**
   - Satellite imagery for retail foot traffic (Orbital Insight, Planet Labs)
   - Credit card transaction data (Second Measure, Earnest Research)
   - These provide 2-4 week lead time on earnings

### 🟡 DATA-004: Dynamic Universe Selection
**File:** `config.py` (Lines 52-70)
**Problem:** `LARGE_CAP_GROWTH`, `SECTOR_ETFS`, `HIGH_MOMENTUM_MIDCAPS` are hardcoded lists. Dead companies stay in, new high-performing stocks are missed.
**Action:**
- Build a daily universe refresh pipeline:
  - Pull all US equities from Alpaca's asset list
  - Filter by: market cap > $1B, average daily volume > $10M, price > $5, listed > 90 days
  - Rank by: 20-day ADV, volatility, sector diversity
  - Output: Top N symbols per strategy requirement
- Add a weekly "universe health check" that removes:
  - Delisted symbols
  - Symbols with pending mergers (M&A event risk)
  - Symbols with upcoming earnings (for non-PEAD strategies)
  - Symbols with regulatory actions (SEC halt, etc.)
- Store universe snapshots with timestamps for backtest reproducibility

### 🟡 DATA-005: Data Quality Framework
**File:** `data_quality.py`
**Problem:** Basic staleness and zero-volume checks. Misses many data quality issues.
**Action:**
- Implement comprehensive data quality checks:
  - **Staleness:** Reject bars > 5 minutes old (configurable per strategy)
  - **Volume anomalies:** Flag if volume is < 10% or > 1000% of 20-day average
  - **Price anomalies:** Flag if price moved > 15% in a single bar (likely data error or halt)
  - **Gap detection:** Flag if bar timestamps have gaps > 2x expected interval
  - **OHLC consistency:** Verify high >= open, close, low and low <= open, close, high
  - **Bid-ask sanity:** Verify bid < ask, spread > 0, spread < 5% of mid
  - **Corporate action detection:** Flag if overnight price change > 20% (likely split/dividend)
  - **Cross-validation:** Compare Alpaca prices to a secondary source (IEX, polygon.io) for critical signals
- Create a `DataQualityScore` (0-1) for each bar
- Strategies should weight signals by data quality score

### 🟡 DATA-006: Caching Layer Overhaul
**File:** `data_cache.py`
**Problem:** Cache key mismatch between `TimeFrame` objects and string representations. TTL too aggressive for some data, too lenient for others.
**Action:**
- Normalize all cache keys to canonical string format
- Implement tiered caching:
  - L1: In-memory (sub-millisecond, for hot data like current positions, latest bars)
  - L2: Redis (millisecond, for computed features, recent bars)
  - L3: Database (for historical data, trade history)
- Per-data-type TTLs:
  - Real-time quotes: 0s (always fresh from WebSocket)
  - Intraday bars: 30s
  - Daily bars: 5 min
  - VIX: 30s (currently 5 min — too stale in crisis)
  - News: 5 min
  - Earnings calendar: 1 hour
  - Sector correlations: 1 hour

---

## Phase 3: Advanced Alpha Generation

### 🟡 ALPHA-001: Transformer-Based Price Prediction
**Problem:** Current signals are purely statistical (z-scores, RSI). No learned representations of price dynamics.
**Action:**
- Implement a Temporal Fusion Transformer (TFT) for multi-horizon price prediction
- Input features: OHLCV, technical indicators, regime state, cross-asset features, sentiment
- Output: Predicted return distribution (mean + uncertainty) at 5min, 15min, 1hr horizons
- Train on 3+ years of intraday data per symbol
- Use uncertainty estimates to modulate signal confidence
- Framework: PyTorch with `pytorch-forecasting` library
- Train weekly on GPU, serve inference on CPU (or GPU if latency-critical)
- This is what Renaissance Technologies pioneered and firms like Two Sigma now use

### 🟡 ALPHA-002: Reinforcement Learning for Entry/Exit Optimization
**Problem:** Entry and exit rules are hardcoded (z-score > 1.5, take profit at z < 0.2). No adaptation to changing market dynamics.
**Action:**
- Implement a PPO (Proximal Policy Optimization) agent for entry/exit timing
- State space: Current price, position, unrealized P&L, time held, volatility, regime, spread
- Action space: Enter long, enter short, exit, hold, scale up, scale down
- Reward function: Risk-adjusted P&L (Sharpe contribution) minus transaction costs
- Train in a market simulator built from historical data
- Start with one strategy (StatMR) and expand if successful
- Use Safe RL with constraint satisfaction to prevent the agent from violating risk limits
- Framework: `stable-baselines3` or `RLlib`

### 🟡 ALPHA-003: Cross-Asset Signal Generation
**Problem:** Current cross-asset monitoring (VIX, TLT, HYG, UUP, GLD) is used only for regime bias. Not used to generate alpha directly.
**Action:**
- Implement cross-asset lead-lag signals:
  - Bond yield curve slope → equity sector rotation
  - Credit spread changes (HYG-IEF) → risk appetite signal
  - Dollar strength (UUP) → export-heavy stock signal
  - Gold momentum → inflation/deflation signal
  - VIX term structure (VIX-VIX3M) → volatility regime change signal
- Compute rolling lead-lag correlations to find predictive relationships
- Use Granger causality tests to validate signal direction
- Add as features to ML models and as standalone signals

### 🟡 ALPHA-004: Order Flow Imbalance Signal
**Problem:** No order flow analysis. Institutional systems exploit order flow for short-term alpha.
**Action:**
- Subscribe to Alpaca WebSocket trades/quotes
- Compute real-time order flow metrics:
  - **Order Flow Imbalance (OFI):** Net buy vs. sell volume at bid/ask
  - **VPIN (Volume-Synchronized Probability of Informed Trading):** Classifies volume as buyer/seller-initiated
  - **Trade Flow Toxicity:** Fraction of aggressive orders (market orders hitting the book)
  - **Book pressure:** Bid size vs. ask size at top N levels
- Signal: Strong buy-side OFI + oversold condition → high-confidence long
- Signal: Elevated VPIN → reduce position sizes (informed traders active)

### 🟡 ALPHA-005: Earnings Sentiment Deep Dive (PEAD Enhancement)
**Problem:** PEAD strategy uses basic earnings surprise (actual vs. estimate). Misses qualitative signals.
**Action:**
- Add earnings call transcript analysis:
  - Source: Alpha Vantage, Financial Modeling Prep, or Seeking Alpha API
  - NLP analysis using FinBERT for sentence-level sentiment
  - Track management tone changes (more confident vs. more hedging language)
  - Extract forward guidance keywords
- Add analyst revision momentum:
  - Track consensus estimate changes over 30/60/90 days pre-earnings
  - Rising estimates → stronger drift signal post-surprise
- Add institutional positioning pre-earnings:
  - Unusual options activity → smart money positioning
  - Short interest changes → potential squeeze or confirmation

### 🟡 ALPHA-006: Intraday Seasonality Enhancement
**Problem:** Current seasonality is learned from own trade history (60 days). Too small a sample, biased by own execution.
**Action:**
- Build comprehensive intraday seasonality model from market data:
  - Compute average returns by 15-minute bucket across 2+ years of data
  - Segment by day-of-week, month, and VIX regime
  - Known patterns: Monday morning weakness, Friday afternoon selling, month-end rebalancing, options expiration effects
- Add calendar effects:
  - FOMC meeting days (reduced sizing pre-announcement, momentum post)
  - Options expiration (monthly, quarterly — increased pinning behavior)
  - Index rebalance days (Russell reconstitution, S&P additions)
  - Tax-loss selling season (December, January effect)
- Weight signals by seasonality favorability

### 🟢 ALPHA-007: Pairs Trading via Copula Models
**Problem:** Current pairs trading uses linear cointegration (Engle-Granger). Misses non-linear dependencies.
**Action:**
- Implement copula-based pairs trading:
  - Fit marginal distributions per asset (kernel density estimation or parametric)
  - Fit copula (Clayton, Gumbel, Frank, Student-t) to joint distribution
  - Detect dependency regime changes (copula parameter drift)
  - Generate signals when copula-implied probabilities are extreme
- This captures non-linear tail dependencies that linear cointegration misses
- Particularly valuable during market stress when correlations change

### 🟢 ALPHA-008: Sector Momentum Rotation
**Problem:** No sector-level alpha generation. Bot trades individual stocks without sector context.
**Action:**
- Implement sector momentum rotation strategy:
  - Rank sectors by 1-month, 3-month, 6-month momentum
  - Go long top 3 sectors, short bottom 3 (via sector ETFs)
  - Rebalance weekly
  - Use regime filter: only in LOW_VOL_BULL and MEAN_REVERTING regimes
- Add sector-stock relative strength:
  - If a stock is strong AND its sector is strong → higher conviction long
  - If a stock is strong BUT sector is weak → lower conviction (divergence risk)

---

## Phase 4: Portfolio Construction & Risk Management

### 🟡 RISK-001: Factor Risk Model
**Problem:** Risk is measured at position level (individual VaR, correlation). No understanding of factor exposures driving portfolio risk.
**Action:**
- Implement a multi-factor risk model:
  - **Market factor:** Beta to SPY
  - **Size factor:** Market cap exposure (SMB)
  - **Value factor:** Book-to-market exposure (HML)
  - **Momentum factor:** 12-1 month return exposure (UMD)
  - **Volatility factor:** Low-vol vs. high-vol exposure
  - **Sector factors:** Per-GICS sector exposure
- Compute daily factor exposures for the portfolio
- Decompose portfolio risk into: systematic (factor) + idiosyncratic (stock-specific)
- Set limits on factor exposures (e.g., market beta between 0.8 and 1.2 for market-neutral targets)
- Alert when factor concentrations exceed thresholds
- Use Barra-style risk model or build a PCA-based custom model from returns data

### 🟡 RISK-002: Hierarchical Risk Parity (HRP) for Allocation
**Problem:** Strategy allocation is static (40/20/20/5/5/10). Doesn't adapt to changing risk contributions.
**Action:**
- Implement HRP (Marcos López de Prado, 2016):
  - Compute correlation matrix across strategies' return streams
  - Apply hierarchical clustering (single-linkage)
  - Allocate capital using inverse-variance within clusters
  - Rebalance weekly
- Benefits over current static allocation:
  - No need to estimate expected returns (major source of error)
  - No matrix inversion (stable with small samples)
  - Automatically reduces allocation to correlated strategies
  - Naturally diversifies across strategy clusters
- Keep minimum/maximum allocation bounds per strategy (e.g., 5%-50%)

### 🟡 RISK-003: Stress Testing Framework
**Problem:** No stress testing. VaR assumes normal conditions. No modeling of extreme scenarios.
**Action:**
- Implement scenario-based stress tests:
  - **Flash Crash:** SPY drops 5% in 10 minutes, VIX spikes to 80
  - **Liquidity Crisis:** Bid-ask spreads widen 10x, volume drops 90%
  - **Correlation Breakdown:** All correlations go to 1.0 (flight to safety)
  - **Gap Opening:** All positions gap -3% overnight
  - **Fed Surprise:** Rates move 50bps unexpectedly
  - **Sector Shock:** One sector drops 10% while others flat
  - **Technology Failure:** No order execution for 5 minutes during crisis
- Run stress tests daily before market open
- Output: Portfolio P&L under each scenario, margin impact, liquidation risk
- Block new positions if worst-case scenario exceeds -5% portfolio loss

### 🟡 RISK-004: Dynamic Hedging
**Problem:** No portfolio-level hedging. When markets turn, all strategies lose simultaneously.
**Action:**
- Implement automatic portfolio hedging:
  - **Delta hedge:** When net portfolio beta exceeds 1.5, buy SPY puts or short SPY futures
  - **Tail hedge:** Always maintain a small VIX call position (2-3% of portfolio) as tail insurance
  - **Sector hedge:** When sector concentration exceeds 40%, hedge with sector inverse ETF
- Implement dynamic hedge ratios:
  - In LOW_VOL regimes: minimal hedging (0.5-1% of portfolio)
  - In HIGH_VOL regimes: increase hedging (2-5% of portfolio)
  - In CRISIS: maximum hedging (5-10% of portfolio)
- Track hedge P&L separately from alpha P&L for performance attribution

### 🟡 RISK-005: Intraday Risk Controls Enhancement
**File:** `risk/daily_pnl_lock.py`
**Problem:** Only tracks daily P&L. No intraday velocity controls. Account could lose 2% in 30 minutes within the daily limit.
**Action:**
- Add rolling window P&L controls:
  - **5-minute P&L limit:** -0.3% → halt new entries for 10 minutes
  - **30-minute P&L limit:** -0.5% → reduce sizing 50% for 30 minutes
  - **1-hour P&L limit:** -0.8% → halt all new entries for 1 hour
- Add velocity controls:
  - If 3+ stop losses hit within 15 minutes → pause strategy for 30 minutes
  - If losing trades outnumber winners 4:1 in last hour → reduce sizing 50%
- Make circuit breaker check real-time via WebSocket (not every 120 seconds)

### 🟡 RISK-006: Overnight & Gap Risk Management
**Problem:** No modeling of overnight gap risk. Heat calculation assumes continuous trading.
**Action:**
- Compute overnight gap risk for each position:
  - Historical overnight gap distribution (mean, std, max)
  - Earnings-related gap risk (much larger)
  - Weekend gap risk (larger than weeknight)
- Reduce position sizes for overnight holds based on gap risk
- Set maximum overnight exposure (e.g., 30% of portfolio)
- Close all non-swing positions by 3:55 PM to avoid overnight risk
- For swing positions (PEAD), explicitly budget for gap risk in position sizing

### 🟡 RISK-007: Liquidation Scenario Handling
**Problem:** No handling of margin calls or forced liquidation scenarios.
**Action:**
- Monitor margin utilization in real-time
- At 70% margin used: alert
- At 80% margin used: halt new short positions
- At 90% margin used: begin orderly unwinding of most-margined positions
- Implement a liquidation priority queue: close most-volatile positions first, keep most-profitable
- Test this logic with paper trading margin scenarios

### 🟡 RISK-008: Correlation Matrix Improvement
**Problem:** Correlation computed from 20-day rolling returns. Stale in fast-moving markets. Doesn't capture intraday correlation changes.
**Action:**
- Use exponentially-weighted correlation (half-life = 10 days for daily, 2 hours for intraday)
- Implement DCC-GARCH (Dynamic Conditional Correlation) for time-varying correlation estimation
- Shrink the correlation matrix toward a structured target (Ledoit-Wolf shrinkage) to reduce estimation noise
- Update intraday correlations every 30 minutes using high-frequency returns
- Use the shrunk correlation matrix for all portfolio construction decisions

---

## Phase 5: Execution Engine Upgrade

### 🟡 EXEC-001: Smart Order Router
**Problem:** Basic spread-based order type selection (limit if spread < 0.15%, else market). No venue optimization.
**Action:**
- Implement an adaptive smart order router:
  - **Venue selection:** Route to exchange with best price + rebate combination
  - **Order type selection:** Limit, market, midpoint peg, or iceberg based on:
    - Spread width
    - Order size relative to displayed liquidity
    - Urgency (signal decay rate)
    - Volatility
  - **Adaptive aggressiveness:** Start passive (limit at mid), escalate to aggressive (market) based on fill probability decay
  - **Child order management:** Split large orders into children, track fill rates per venue
- Note: Alpaca's API abstracts away venue selection, but order type optimization is still valuable

### 🟡 EXEC-002: Almgren-Chriss Optimal Execution
**Problem:** TWAP splits orders into equal time-weighted slices. Doesn't account for market impact or urgency.
**Action:**
- Implement Almgren-Chriss framework:
  - Model temporary and permanent market impact
  - Optimize execution trajectory to minimize total cost (impact + risk)
  - Parameter `lambda` controls urgency (high = execute fast, accept more impact; low = execute slowly, minimize impact)
  - Calibrate impact parameters from historical fill data
- For each order, compute the optimal execution schedule:
  - Front-load if signal is decaying (momentum strategies)
  - Back-load if signal is persistent (mean reversion)
  - Uniform if no urgency preference

### 🟡 EXEC-003: Slippage Model
**Problem:** Transaction cost filter uses a static slippage estimate. Real slippage varies by time, volatility, and order size.
**Action:**
- Build an empirical slippage model from historical fills:
  - Features: spread at order time, volatility, order size / ADV, time of day, VIX level
  - Target: (fill price - mid price at order time) / mid price
  - Model: Linear regression or gradient boosting
  - Update weekly with new fill data
- Use the slippage model in:
  - Pre-trade cost estimation (signal filtering)
  - Position sizing (expected cost reduces expected profit)
  - Backtesting (realistic fill simulation)
  - Performance attribution (execution alpha vs. slippage drag)

### 🟡 EXEC-004: Fill Quality Analytics
**Problem:** No measurement of execution quality. Can't tell if execution is improving or degrading.
**Action:**
- Track per-fill metrics:
  - **Implementation Shortfall:** (decision price - fill price) / decision price
  - **VWAP benchmark:** Fill price vs. VWAP during execution window
  - **Arrival price benchmark:** Fill price vs. mid at order submission
  - **Spread capture:** Did we fill inside the spread?
  - **Latency:** Time from signal to order submission to fill
- Aggregate daily/weekly:
  - Average implementation shortfall by strategy
  - Execution cost as % of alpha
  - Improvement trends over time

### 🟡 EXEC-005: Order Retry with Exponential Backoff
**File:** `execution.py` (Lines 174-195)
**Problem:** Retries only once after 2 seconds. Could double-submit if Alpaca is slow.
**Action:**
- Implement exponential backoff with jitter:
  - Attempt 1: Immediate
  - Attempt 2: 1-2 seconds (random jitter)
  - Attempt 3: 3-6 seconds
  - Attempt 4: 8-15 seconds
  - Max attempts: 4
- Before each retry, check if the original order was actually filled (query order status)
- Use the idempotency key to prevent duplicate fills
- Log each retry attempt with reason

### 🟡 EXEC-006: Pre-Trade Validation Enhancement
**Problem:** Basic validation before order submission. Missing several critical checks.
**Action:**
- Add pre-trade checks:
  - **Buying power:** Verify sufficient buying power before submission (not just position count)
  - **Margin requirement:** Calculate margin impact of new position
  - **Short locate:** For short sales, verify availability before signal processing (not after)
  - **Price reasonableness:** Reject limit orders > 2% away from current mid (likely stale price)
  - **Symbol validity:** Verify symbol is still tradeable (not halted, delisted, or pending corporate action)
  - **Market hours:** Verify market is actually open (handle early closes, holidays)
  - **Min notional:** Alpaca has minimum order values — check before submission

---

## Phase 6: Machine Learning Pipeline

### 🟡 ML-001: Feature Engineering Framework
**Problem:** Features are computed ad-hoc within each strategy. No systematic feature engineering.
**Action:**
- Build a feature computation engine with 200+ features across categories:

  **Price-based (30+ features):**
  - Returns at multiple horizons (1-bar, 5-bar, 20-bar, 60-bar)
  - Volatility at multiple horizons (realized vol, Parkinson, Garman-Klass)
  - Price relative to moving averages (5, 10, 20, 50, 200)
  - Bollinger Band width and %B
  - ATR at multiple periods
  - Price acceleration (return of returns)

  **Volume-based (15+ features):**
  - Volume ratio (current / 20-bar average)
  - On-Balance Volume (OBV) and OBV divergence
  - Volume-weighted return
  - Accumulation/Distribution line
  - Money Flow Index
  - Volume at price levels (volume profile)

  **Microstructure (20+ features):**
  - Bid-ask spread (absolute and relative)
  - Order flow imbalance
  - VPIN
  - Trade size distribution (entropy, skewness)
  - Quote-to-trade ratio
  - Effective spread vs. quoted spread

  **Cross-asset (15+ features):**
  - SPY return, VIX level, VIX change
  - Sector ETF return vs. SPY (relative strength)
  - Bond-equity correlation (rolling)
  - Credit spread level and change
  - Dollar index level and change

  **Sentiment (10+ features):**
  - News sentiment score (rolling average, velocity)
  - Social media sentiment (if available)
  - Analyst estimate revisions
  - Short interest level and change

  **Calendar (10+ features):**
  - Day of week, month, quarter
  - Days to/from earnings
  - Days to/from FOMC
  - Options expiration proximity
  - End-of-month/quarter indicator

### 🟡 ML-002: Model Training Pipeline
**Problem:** No ML training infrastructure. LLM scorer is the only ML component and it's optional/external.
**Action:**
- Build a reproducible ML training pipeline:
  - **Data:** Pull features from feature store, split into train/validation/test with temporal ordering (no future leakage)
  - **Purging:** Implement purged K-fold cross-validation (remove overlapping samples between train and test)
  - **Embargo:** Add embargo period between train and test splits (prevent leakage from label overlap)
  - **Models:** Train ensemble of:
    - LightGBM (fast, handles missing values, good for tabular data)
    - XGBoost (more regularization options)
    - Random Forest (variance reduction baseline)
    - LSTM (temporal patterns)
    - Transformer (attention-based patterns)
  - **Stacking:** Use first-level model predictions as features for a meta-learner
  - **Hyperparameter optimization:** Optuna with purged cross-validation as objective
  - **Model registry:** Version models, track performance, enable rollback
- Training schedule: Weekly full retrain, daily incremental update
- Use MLflow or Weights & Biases for experiment tracking

### 🟡 ML-003: Online Learning & Model Adaptation
**Problem:** Static models degrade as market regimes change.
**Action:**
- Implement online learning for fast adaptation:
  - **Exponential decay weighting:** Recent observations weighted more heavily
  - **Concept drift detection:** Monitor model prediction error over time; retrain if error exceeds threshold
  - **Shadow models:** Train new models alongside production; swap when shadow outperforms
  - **Regime-conditional models:** Train separate models per regime, select based on HMM state
- Monitor model performance in real-time:
  - Rolling accuracy, precision, recall by 1-hour windows
  - Feature importance drift (which features matter changes over time)
  - Prediction calibration (are 60% confidence predictions actually right 60% of the time?)

### 🟡 ML-004: Overfitting Prevention
**Problem:** Strategy parameters are hardcoded with suspiciously specific values (MR_ZSCORE_ENTRY=1.5, ORB_TP_MULT=1.5). Parameter optimizer exists but is disabled.
**Action:**
- Implement anti-overfitting measures:
  - **Combinatorial Purged Cross-Validation (CPCV):** Superior to walk-forward for detecting overfitting
  - **Deflated Sharpe Ratio (DSR):** Adjusts Sharpe for multiple testing bias
  - **Probability of Backtest Overfitting (PBO):** Quantifies likelihood that a strategy's backtest performance is due to overfitting
  - **Minimum Backtest Length (MinBTL):** Calculate minimum data needed for statistically significant results
  - **Parameter sensitivity analysis:** Vary each parameter ±20% and verify performance doesn't collapse
  - **Out-of-sample degradation ratio:** Acceptable if OOS Sharpe > 0.5 × IS Sharpe
- Any strategy failing these tests should be demoted or removed

### 🟢 ML-005: Reinforcement Learning for Portfolio Management
**Problem:** Portfolio-level decisions (which signals to trade, how much capital to allocate) are rule-based.
**Action:**
- Implement a portfolio-level RL agent:
  - State: Current positions, unrealized P&L, available signals, regime, risk metrics
  - Action: For each signal, decide: trade/skip, and position size (continuous)
  - Reward: Risk-adjusted portfolio return (Sharpe) minus turnover costs
  - Constraints: Must satisfy all risk limits (max positions, sector limits, correlation limits)
- This replaces the 13-filter pipeline with a learned decision function
- The RL agent can discover non-obvious interactions between positions
- Long-term project — keep rule-based system as fallback

---

## Phase 7: Market Microstructure Analysis

### 🟡 MICRO-001: VPIN Implementation
**Problem:** No order flow toxicity measurement. Bot doesn't know when informed traders are active.
**Action:**
- Implement Volume-Synchronized Probability of Informed Trading (VPIN):
  - Bucket trades into volume-equal bins (not time-equal)
  - Classify each trade as buyer or seller-initiated (tick rule or Lee-Ready algorithm)
  - Compute VPIN = |buy volume - sell volume| / total volume, averaged over N buckets
  - High VPIN (> 0.7) → informed traders active → widen stops, reduce sizing
  - VPIN spike → potential upcoming volatility → defensive posture
- Use VPIN as:
  - A filter (block entries when VPIN > threshold)
  - A sizing modifier (reduce size as VPIN increases)
  - A regime indicator (supplement HMM with microstructure signal)

### 🟡 MICRO-002: Order Book Imbalance
**Problem:** No order book analysis. Missing near-term directional signals.
**Action:**
- Track top-of-book imbalance from WebSocket quotes:
  - `imbalance = (bid_size - ask_size) / (bid_size + ask_size)`
  - Positive imbalance → near-term buy pressure
  - Negative imbalance → near-term sell pressure
- Compute weighted imbalance across multiple levels (if available)
- Use as a short-term directional signal (1-5 minute horizon)
- Particularly useful for:
  - Micro-Momentum strategy entry timing
  - ORB strategy breakout confirmation
  - Exit timing (imbalance turning against position)

### 🟡 MICRO-003: Trade Size Analysis
**Problem:** No analysis of trade sizes. Institutional trades have different characteristics than retail.
**Action:**
- Classify trades by size:
  - Small (< 100 shares): Retail flow
  - Medium (100-1000 shares): Mixed
  - Large (> 1000 shares or > $50k): Institutional
- Track institutional flow direction:
  - Net large-trade volume (buy vs. sell)
  - Large trade arrival rate
  - Block trade detection
- Signal: Institutional buying + retail selling → bullish (smart money accumulating)
- Signal: Institutional selling + retail buying → bearish (distribution)

### 🟢 MICRO-004: Effective Spread & Information Content
**Action:**
- Compute realized measures:
  - **Effective spread:** 2 × |trade price - midpoint| (actual cost of trading)
  - **Realized spread:** Effective spread minus price impact (market maker's profit)
  - **Price impact:** Trade price minus midpoint 5 minutes later (information content)
- Use effective spread in transaction cost model
- Use price impact to estimate how much of a stock's spread is due to adverse selection (informed trading)
- High adverse selection → reduce position sizes

---

## Phase 8: Backtesting & Research Framework

### 🟡 BACKTEST-001: Event-Driven Backtester
**File:** `backtester.py`
**Problem:** Current backtester uses vectorized operations with look-ahead bias. Groups by trading day, then looks for breakouts using full-day bars.
**Action:**
- Build a proper event-driven backtester:
  - Process bars one-by-one in chronological order
  - Maintain simulated order book with realistic fill models
  - Track positions, P&L, margin in real-time
  - Apply all filters in the same order as live trading
  - Support multiple strategies running simultaneously
  - Use the same code for live and backtest (strategy logic shouldn't know if it's live or historical)
- Key features:
  - **No look-ahead bias:** State at time T only uses data available at time T
  - **Transaction cost modeling:** Use the empirical slippage model (EXEC-003)
  - **Market impact:** For large orders, model temporary and permanent impact
  - **Partial fills:** Simulate partial fills based on available volume
  - **Latency simulation:** Add configurable delay between signal and order

### 🟡 BACKTEST-002: Combinatorial Purged Cross-Validation (CPCV)
**Problem:** Walk-forward validation with single split is unreliable. High variance, regime-dependent.
**Action:**
- Implement CPCV (López de Prado, 2018):
  - Divide data into N groups (e.g., N=10)
  - Generate all possible combinations of K test groups (e.g., K=2)
  - For each combination: train on non-test groups, test on test groups
  - Purge: remove samples from training set that overlap with test labels
  - Embargo: remove E samples after each purge gap
  - Result: Distribution of Sharpe ratios across many train/test splits
- Compute:
  - **PBO (Probability of Backtest Overfitting):** P(OOS Sharpe < 0 | IS Sharpe > 0)
  - **DSR (Deflated Sharpe Ratio):** Adjusts for multiple testing
  - Reject strategies with PBO > 0.5 or DSR p-value > 0.05

### 🟡 BACKTEST-003: Monte Carlo Stress Testing
**Problem:** Single backtest path gives false confidence. No understanding of result distribution.
**Action:**
- Implement Monte Carlo robustness testing:
  - **Trade shuffling:** Randomize trade order to test sequence dependence
  - **Trade skipping:** Randomly skip 10-30% of trades to test robustness
  - **Parameter perturbation:** Add noise to strategy parameters (±10-20%)
  - **Data bootstrapping:** Resample returns with replacement
  - **Regime randomization:** Randomly reassign regime labels
- Run 1000+ simulations per strategy
- Report: Median Sharpe, 5th percentile Sharpe, max drawdown distribution, probability of ruin
- Reject strategies where 5th percentile Sharpe < 0

### 🟡 BACKTEST-004: Performance Attribution
**Problem:** Limited understanding of where P&L comes from. Can't distinguish skill from luck.
**Action:**
- Implement multi-level attribution:
  - **Strategy-level:** P&L per strategy, per day/week/month
  - **Factor-level:** How much P&L came from market beta, sector, momentum, mean reversion, etc.
  - **Timing-level:** P&L from entry timing, exit timing, position sizing
  - **Execution-level:** P&L lost to slippage, spread, market impact
  - **Risk-level:** P&L contribution from hedges, circuit breaker actions
- Brinson-Fachler attribution for sector allocation vs. stock selection
- This tells the dev team where to focus optimization efforts

### 🟡 BACKTEST-005: Alpha Decay Analysis
**Problem:** Walk-forward validation runs weekly but only checks Sharpe threshold. Doesn't model how fast alpha decays.
**Action:**
- For each strategy, measure alpha decay curve:
  - Signal t=0: Expected return at signal generation
  - Signal t+1min, t+5min, t+15min, t+1hr: How much expected return remains?
  - Plot decay curve → informs optimal execution urgency
- Track alpha decay over time:
  - Is mean reversion alpha decaying faster than 6 months ago?
  - Is momentum alpha increasing or decreasing?
  - Are new strategies needed to replace decaying ones?
- Auto-demote strategies with alpha decay rate exceeding threshold

### 🟢 BACKTEST-006: Market Simulation Environment
**Action:**
- Build a realistic market simulator for strategy development and RL training:
  - Replay historical order books with synthetic liquidity
  - Model market impact of the bot's own orders on prices
  - Simulate other market participants (noise traders, market makers, informed traders)
  - Support "what-if" scenarios (what if VIX doubles? what if correlations spike?)
- Use for:
  - RL agent training (ALPHA-002)
  - Execution algorithm testing (EXEC-001, EXEC-002)
  - Stress testing (RISK-003)
  - New strategy development before live deployment

---

## Phase 9: Monitoring, Observability & Alerting

### 🟡 MON-001: Real-Time Dashboard
**File:** `dashboard.py`
**Problem:** Incomplete dashboard. No real-time P&L, no position visualization, no risk metrics display.
**Action:**
- Build a comprehensive real-time dashboard (Streamlit, Grafana, or custom React):
  - **P&L panel:** Real-time daily P&L curve, cumulative P&L, per-strategy P&L
  - **Positions panel:** Current positions with unrealized P&L, entry price, stop/target, time held
  - **Risk panel:** Circuit breaker status, VaR utilization, sector concentration, factor exposures
  - **Execution panel:** Recent orders, fill rates, slippage metrics
  - **Regime panel:** Current HMM state, VIX level, cross-asset indicators
  - **Signal panel:** Active signals awaiting execution, filtered signals with rejection reasons
  - **Health panel:** System health, API latency, data freshness, error counts

### 🟡 MON-002: Performance Metrics Pipeline
**Problem:** Metrics computed on-demand. No time-series tracking of performance metrics.
**Action:**
- Compute and store metrics every hour:
  - Rolling Sharpe (1-day, 1-week, 1-month, 3-month, inception)
  - Rolling Sortino
  - Rolling max drawdown and drawdown duration
  - Rolling win rate and profit factor per strategy
  - Rolling alpha and beta vs. SPY
  - Rolling information ratio
  - Hit rate by signal confidence bucket
- Store in time-series database for trend analysis
- Alert when metrics degrade below thresholds

### 🟡 MON-003: Alerting System
**File:** `notifications.py`
**Problem:** No rate limiting on Telegram notifications. No tiered severity. No escalation.
**Action:**
- Implement tiered alerting:
  - **INFO:** Trade executed, daily summary → Telegram (rate-limited to 1/minute)
  - **WARNING:** Strategy underperforming, high VaR, data quality issue → Telegram + email
  - **CRITICAL:** Circuit breaker triggered, position sync mismatch, system error → Telegram + email + SMS
  - **EMERGENCY:** Kill switch activated, margin call, system crash → Telegram + email + SMS + phone call (via Twilio)
- Add rate limiting per alert type (e.g., max 1 CRITICAL alert per 5 minutes for same issue)
- Add alert suppression rules (maintenance windows, known issues)
- Log all alerts with timestamps for audit

### 🟡 MON-004: Latency & Performance Monitoring
**Problem:** No tracking of system performance metrics.
**Action:**
- Instrument all critical paths with timing:
  - Bar fetch latency (per symbol, per source)
  - Signal computation time (per strategy)
  - Filter pipeline time (per filter)
  - Order submission latency (submission to acknowledgment)
  - Fill latency (order to fill)
  - Full cycle time (signal generation to order fill)
- Use Prometheus metrics + Grafana dashboards (or DataDog for SaaS)
- Alert if:
  - Bar fetch latency > 5 seconds (normally < 500ms)
  - Scan cycle time > 90 seconds (normally < 60s)
  - Order submission latency > 2 seconds

### 🟡 MON-005: Broker Position Reconciliation
**Problem:** No automated reconciliation between bot's internal state and broker's state.
**Action:**
- Run reconciliation every 30 minutes and at market open/close:
  - Fetch all positions from Alpaca
  - Compare to bot's internal position tracking
  - Flag discrepancies: phantom positions (bot thinks it has, broker doesn't), ghost positions (broker has, bot doesn't know), quantity mismatches, price discrepancies
  - Auto-heal: If broker has a position the bot doesn't track, create a tracking record
  - Alert on any discrepancy (could indicate a bug, a manually-placed trade, or a sync failure)
- At market open: Force full sync before first trade

---

## Phase 10: Compliance & Audit Infrastructure

### 🟡 COMPLY-001: Complete Audit Trail
**Problem:** Limited audit logging. Can't reconstruct why a specific trade was made.
**Action:**
- Log every decision point with full context:
  - Signal generation: Which strategy, what inputs, what output, confidence score
  - Filter pipeline: Which filters passed/failed, with reasons
  - Position sizing: All multipliers applied, final size, rationale
  - Order submission: Full order details, timestamp, idempotency key
  - Fill: Fill price, slippage, execution venue
  - Exit: Exit reason, P&L, hold time
- Store in append-only log (never delete or modify)
- Format: Structured JSON per event, queryable via SQL or Elasticsearch
- Retention: Minimum 7 years (SEC requirement for broker-dealers, good practice for all)

### 🟡 COMPLY-002: Market Manipulation Detection
**Problem:** No self-surveillance. Bot could inadvertently create patterns that look like manipulation.
**Action:**
- Monitor for:
  - **Wash trading:** Buying and selling the same symbol within seconds (could happen with multiple strategies)
  - **Spoofing patterns:** Large limit orders cancelled quickly (shouldn't happen, but verify)
  - **Marking the close:** Trades in last 5 minutes that move closing price (reduce EOD trading)
  - **Layering:** Multiple orders at different prices that are cancelled (verify OMS doesn't create this)
  - **Front-running:** Trading ahead of known large orders (shouldn't be possible, but verify)
- Add a self-surveillance module that runs daily
- Log all flagged patterns for review

### 🟡 COMPLY-003: Best Execution Documentation
**Problem:** No documentation of execution quality for regulatory compliance.
**Action:**
- Generate daily best execution report:
  - For each trade: decision price, order type, venue, fill price, benchmark comparison (VWAP, TWAP, arrival)
  - Aggregate: Average slippage, percentage of trades with positive/negative execution alpha
  - Smart order routing decisions: Why was this order type chosen? Why this venue?
- Store reports for regulatory review
- SEC and FINRA increasingly require best execution documentation

### 🟡 COMPLY-004: PDT Rule Enhancement
**Problem:** Basic PDT check exists but may not handle edge cases around the 4:00 PM cutoff or pattern detection.
**Action:**
- Implement robust PDT compliance:
  - Track rolling 5-business-day window of day trades
  - A day trade = open and close same position same calendar day
  - If equity < $25,000: max 3 day trades in 5-business-day window
  - If equity drops below $25,000 mid-day: immediately block new day trades
  - If day trade count = 3: block all intraday strategies until window rolls over
  - Log all PDT calculations for audit

---

## Phase 11: Testing & Quality Assurance

### 🟡 TEST-001: Unit Test Coverage
**Problem:** Missing unit tests for critical edge cases.
**Action:**
- Write unit tests for every module, targeting 80%+ coverage:

  **Risk module tests:**
  - Kelly calculation: 100% win rate, 0% win rate, 50/50, negative expectancy
  - Circuit breaker: All tier transitions, de-escalation, reset at EOD
  - VaR: Zero returns, extreme returns, insufficient history
  - Vol targeting: VIX = 0, VIX = 100, zero volatility, infinite volatility
  - Position sizing: Zero equity, max equity, zero risk-per-share

  **Strategy tests:**
  - OU parameter fitting: Degenerate data (constant price, all zeros, NaN values)
  - Mean reversion: Entry at exact z-score threshold, data with gaps
  - Pairs: Cointegration test with perfectly correlated series, uncorrelated series
  - ORB: Zero range, gap > max, overlapping ranges

  **Execution tests:**
  - Order rejection handling (insufficient funds, halted symbol, bad price)
  - Partial fills (fill 50%, then cancel remainder)
  - Order cancellation race conditions
  - TWAP with varying slice counts

  **Data tests:**
  - Timezone boundaries (market close vs. 4:00 PM ET vs. extended hours)
  - Missing bars, duplicate bars, out-of-order bars
  - Corporate action handling (2:1 split, reverse split, special dividend)
  - Weekend/holiday gap handling

### 🟡 TEST-002: Integration Tests
**Action:**
- Build integration tests that test full signal-to-execution pipeline:
  - Strategy generates signal → filters pass → sizing computed → order submitted → fill received → position tracked → exit evaluated → exit executed
  - Test with paper trading account on Alpaca
  - Run nightly against next day's universe
- Test broker reconnection:
  - Simulate WebSocket disconnect and reconnect
  - Verify position sync after reconnection
  - Verify pending orders are still tracked

### 🟡 TEST-003: Regression Tests
**Action:**
- After every bug fix, add a regression test that reproduces the bug
- Maintain a "golden set" of known scenarios with expected outputs:
  - Given these bars, this strategy should generate this signal
  - Given this portfolio state, this risk check should pass/fail
  - Given this order flow, execution should choose this order type
- Run regression tests on every code change (CI/CD pipeline)

### 🟡 TEST-004: Shadow Trading / Paper Trading Validation
**Action:**
- Before any strategy goes live, require 30 days of paper trading:
  - Run live alongside production, generate signals, but don't execute
  - Compare shadow signals to live signals (should be identical)
  - Track shadow P&L as if orders were filled at market price
  - Require shadow Sharpe > 0.5 and shadow max drawdown < 5% to approve live
- After major code changes, run 1 week of shadow trading before deploying to production

---

## Phase 12: Advanced Data Science Techniques (López de Prado Framework)

### 🟡 LPRADO-001: Fractional Differentiation of Time Series
**Problem:** ML models need stationary inputs, but differencing (returns) destroys memory. Price levels have memory but aren't stationary. This is a fundamental tension in financial ML.
**Action:**
- Implement fractional differentiation (d between 0 and 1) on all price-based features:
  - Use the `fracdiff` or `tsfracdiff` library
  - For each feature, find the minimum d that achieves stationarity (ADF test p < 0.05)
  - Typical values: d ≈ 0.3-0.5 for most price series
  - This preserves long memory while achieving stationarity
- Apply to: Price, volume, VWAP, OBV, cumulative returns
- Use fractionally-differentiated features as ML inputs instead of raw returns
- This is the single most impactful ML improvement per López de Prado's research
- Reference: *Advances in Financial Machine Learning*, Chapter 5

### 🟡 LPRADO-002: Triple-Barrier Method & Meta-Labeling
**Problem:** Current labels are binary (price went up/down). This creates noisy labels that degrade ML models. No confidence scoring on labels.
**Action:**
- **Triple-Barrier Method** for label generation:
  - Define three barriers per trade: upper (take-profit), lower (stop-loss), vertical (time expiry)
  - Label = which barrier was hit first: +1 (upper), -1 (lower), 0 (time expiry)
  - Barrier widths based on daily volatility (e.g., 2x daily vol for TP/SL)
  - This produces cleaner labels than simple return-based labeling
- **Meta-Labeling** (second-stage model):
  - Primary model: Generate directional signal (buy/sell)
  - Meta-model: Predict probability that the primary model's signal will be profitable
  - Input to meta-model: Primary model's signal + market features
  - Output: Bet size (0 to 1)
  - Benefits: Reduces false positives, enables dynamic position sizing based on ML confidence
  - Train meta-model using purged cross-validation
- **Label concurrency handling:**
  - Track overlapping labels (multiple active trades at same time)
  - Use uniqueness weighting to reduce impact of concurrent labels on training
  - Average uniqueness = 1 / (average number of concurrent labels)
- Reference: *Advances in Financial Machine Learning*, Chapters 3-4

### 🟡 LPRADO-003: Information-Driven Bars
**Problem:** Time bars (1-min, 2-min) sample at uniform time intervals regardless of market activity. This means quiet periods oversample noise, while active periods undersample information.
**Action:**
- Implement three types of information-driven bars:
  - **Tick Imbalance Bars (TIB):** Sample when the running tick imbalance (buy ticks minus sell ticks) exceeds an EWMA-based threshold. Bars are generated when there's a significant imbalance in buy vs. sell pressure.
  - **Volume Imbalance Bars (VIB):** Same concept but using volume instead of tick count. More sensitive to institutional flow.
  - **Dollar Imbalance Bars (DIB):** Same concept using dollar volume. Best for cross-sectional comparisons.
- Benefits:
  - Bars are sampled in proportion to information arrival
  - Returns are closer to IID (better for ML)
  - Fewer bars during quiet periods (less noise)
  - More bars during active periods (more signal captured)
- Use as default bar type for all ML-based strategies
- Keep time bars for strategies that explicitly need them (ORB needs 30-min time windows)
- Reference: *Advances in Financial Machine Learning*, Chapter 2

### 🟡 LPRADO-004: Entropy-Based Features
**Problem:** No information-theoretic measures. Missing a class of features that captures market complexity.
**Action:**
- Implement entropy features:
  - **Shannon entropy** of return distribution (rolling window): Measures market randomness
  - **Approximate entropy (ApEn):** Measures regularity/predictability of price series
  - **Sample entropy (SampEn):** Improved version of ApEn (less bias for short series)
  - **Transfer entropy:** Directional information flow between assets (A → B vs. B → A)
  - **Permutation entropy:** Order-based complexity measure, robust to noise
- Applications:
  - Low entropy → Market is trending (predictable) → Momentum strategies favored
  - High entropy → Market is chaotic → Mean reversion or sit out
  - Transfer entropy spike → Information flowing from one asset to another → Lead-lag signal
- Add as features to ML models and as regime indicators

### 🟢 LPRADO-005: Sample Weights by Uniqueness
**Problem:** ML models treat all training samples equally, but concurrent labels mean some samples carry redundant information.
**Action:**
- Compute sample uniqueness: For each label, calculate average number of concurrent labels
- Weight = 1 / average_concurrency
- Apply weights in model training (most tree-based and neural network frameworks support sample weights)
- This dramatically reduces overfitting in time-series ML
- Combine with time-decay weighting (recent samples weighted more)

---

## Phase 13: Advanced ML & AI Techniques

### 🟡 ADVML-001: Graph Neural Networks for Stock Relationships
**Problem:** Current correlation analysis is pairwise and linear. Misses complex network effects (sector cascades, supply chain relationships).
**Action:**
- Build a stock relationship graph:
  - **Nodes:** Individual stocks
  - **Edges:** Weighted by correlation, supply chain links, sector co-membership, co-mention in news
  - Update edge weights daily from rolling correlation
- Implement a GNN (Graph Attention Network or Spatio-Temporal GNN):
  - Input: Node features (per-stock technical/fundamental features) + graph structure
  - Output: Per-stock predicted return or risk score
  - The GNN can learn how information propagates through the stock network
- Applications:
  - Predict which stocks will be affected when a sector leader moves
  - Identify stocks that are mis-priced relative to their network position
  - Improve correlation-based risk filters (replace pairwise with graph-based)
- Framework: PyTorch Geometric or DGL (Deep Graph Library)

### 🟡 ADVML-002: Random Matrix Theory for Covariance Cleaning
**Problem:** Empirical covariance matrices from financial data are extremely noisy. With 130 stocks and 60 days of data, approximately 94% of the correlation matrix is noise.
**Action:**
- Implement Marchenko-Pastur denoising:
  - Compute eigenvalues of the correlation matrix
  - Apply the Marchenko-Pastur distribution to identify noise eigenvalues
  - Replace noise eigenvalues with their average (shrink toward identity)
  - Reconstruct the denoised correlation matrix
- Apply to:
  - Portfolio concentration analysis (eigenvalue-based effective bets)
  - Pairs trading correlation screening
  - HRP portfolio construction
  - Sector correlation monitoring
- This is standard at institutional quant firms and significantly improves all correlation-dependent decisions
- Reference: Bun, Bouchaud, Potters (2017) — *Cleaning large correlation matrices*

### 🟡 ADVML-003: Bayesian Online Change-Point Detection (BOCPD)
**Problem:** Current HMM regime detection is trained weekly on 3 years of data. Slow to detect regime changes. Can't detect intraday regime shifts.
**Action:**
- Implement BOCPD for real-time regime change detection:
  - Maintains a posterior distribution over the time since the last change point
  - Updates in O(1) per observation (real-time capable)
  - Detects structural breaks as they happen, not with a weekly lag
- Apply to:
  - Volatility regime changes (VIX spike detection within minutes, not 15 minutes)
  - Correlation regime changes (breakdown of pairs relationships)
  - Strategy performance regime changes (alpha decay detection in real-time)
- Use alongside HMM: BOCPD for fast detection, HMM for classification
- Score-driven BOCPD with time-varying parameters performs best on financial data

### 🟡 ADVML-004: Synthetic Data Generation
**Problem:** Limited historical data for training ML models (especially for rare events like crashes). Can't stress-test models against unseen scenarios.
**Action:**
- Implement financial data generation models:
  - **TimeGAN:** Generates realistic time-series preserving temporal dynamics
  - **CGAN (Conditional GAN):** Generate data conditioned on regime (e.g., "generate a crash scenario")
  - Validate generated data preserves stylized facts: fat tails, volatility clustering, leverage effect, autocorrelation of squared returns
- Use cases:
  - **Training augmentation:** 10x more training data for rare events (crashes, squeezes)
  - **Stress testing:** Generate 1000 synthetic crash scenarios with different characteristics
  - **Model robustness:** Train on synthetic + real data, test on real only
  - **RL training:** Generate diverse market environments for RL agent training
- Framework: `ydata-synthetic` or custom PyTorch implementations

### 🟡 ADVML-005: Explainable AI (XAI) Layer
**Problem:** ML model predictions are black boxes. Can't explain why a signal was generated. Regulatory risk and debugging difficulty.
**Action:**
- Implement multi-level explainability:
  - **Global feature importance:** SHAP values across all predictions (which features matter most overall?)
  - **Local feature importance:** Per-prediction SHAP values (why THIS specific signal?)
  - **Attention visualization:** For transformer models, visualize which time steps the model attends to
  - **Counterfactual explanations:** "This signal would not have been generated if RSI were above 45 instead of 38"
- Integration:
  - Store SHAP values with each ML-generated signal in the audit trail
  - Dashboard panel showing feature attribution for active signals
  - Weekly report: Top 10 features driving P&L
- This satisfies regulatory requirements (best execution justification) and helps the dev team debug model behavior

### 🟢 ADVML-006: Attention-Based Market Regime Detection
**Action:**
- Replace or supplement HMM with a transformer-based regime classifier:
  - Input: 60-day window of multi-asset features (SPY, VIX, TLT, HYG, sector ETFs)
  - Output: Probability distribution over regime states
  - Self-attention learns which historical patterns are most relevant to current conditions
  - Can capture non-Markovian dynamics that HMM misses (HMM assumes memoryless transitions)
- Train on labeled historical regimes (backtested regime assignments)
- Use attention weights for interpretability: "Model classified this as HIGH_VOL_BEAR because of features X, Y, Z"

---

## Phase 14: Additional Strategy Modules

### 🟡 STRAT-001: Options-Enhanced Equity Strategies
**Problem:** Bot is equity-only. Options provide leverage, hedging, and volatility exposure that pure equity strategies can't achieve.
**Action:**
- **Covered Call Writing:**
  - For long equity positions with unrealized gains, sell OTM calls to capture premium
  - Enhances yield during low-volatility periods
  - Selection: Strike at 1.5-2σ above current price, 30-45 DTE
  - Exit: Roll or let expire
- **Protective Put Overlay:**
  - For portfolio-level tail risk protection
  - Buy 5-10% OTM puts on SPY, rolling monthly
  - Cost: ~0.5-1% of portfolio annually (insurance premium)
  - This replaces the conceptual tail hedge in RISK-004 with an actual implementation
- **Volatility Arbitrage / Gamma Scalping (advanced):**
  - Compare implied volatility (from options prices) to realized volatility
  - When IV > RV: Sell straddles, delta-hedge continuously → profit from volatility overpricing
  - When IV < RV: Buy straddles, delta-hedge → profit from volatility underpricing
  - Requires: Real-time Greeks computation, continuous delta rebalancing (every 5 minutes)
  - Sharpe enhancement: +0.3-0.5 based on 2025 research
  - Transaction cost sensitivity: High — only profitable with tight spreads

### 🟡 STRAT-002: Multi-Timeframe Strategy Integration
**Problem:** Each strategy operates on a single timeframe. No multi-timeframe confirmation.
**Action:**
- For each signal, check alignment across timeframes:
  - **Short-term (2-min bars):** Signal direction (the primary signal)
  - **Medium-term (15-min bars):** Trend alignment (does the medium-term trend agree?)
  - **Long-term (daily bars):** Major trend (are we trading with or against the daily trend?)
- Scoring:
  - All three aligned: Full confidence (1.0x sizing)
  - Two aligned: Moderate confidence (0.7x sizing)
  - Only one: Low confidence (0.4x sizing or skip)
- This is a classic institutional technique that dramatically reduces false signals

### 🟢 STRAT-003: Cross-Sectional Momentum
**Problem:** No cross-sectional alpha. Bot treats each stock independently.
**Action:**
- Implement long-short cross-sectional momentum:
  - Rank stocks by 1-month return (skip most recent week to avoid reversal)
  - Go long top decile, short bottom decile
  - Rebalance weekly
  - Market-neutral by construction
- Enhancements:
  - Residual momentum (momentum after removing factor exposures) is more persistent
  - Industry-neutral momentum (rank within sectors) reduces sector concentration
  - Combine with mean reversion: Momentum for entry direction, mean reversion for timing

---

## Phase 15: Operational Excellence

### 🟡 OPS-001: Tax-Loss Harvesting Engine
**Problem:** No tax optimization. Real after-tax returns are what matter to capital allocators.
**Action:**
- Implement automated tax-loss harvesting:
  - Track cost basis for every lot (FIFO, LIFO, HIFO, SpecID methods)
  - Daily scan: Identify positions with unrealized losses > $X threshold
  - Sell losing positions, immediately buy correlated substitute (to maintain exposure)
  - Respect wash sale rules: No repurchase of "substantially identical" security within 30 days
  - Track wash sale disallowances and adjust cost basis accordingly
- Estimated tax alpha: 0.5-1.5% annually for taxable accounts
- Implementation: Run after market close, before next day's trading

### 🟡 OPS-002: Drawdown-Based Risk Measures
**Problem:** Risk measured primarily by VaR and volatility. Investors care more about drawdowns.
**Action:**
- Add drawdown-centric risk metrics:
  - **Calmar Ratio:** Annualized return / max drawdown (target: > 2.0)
  - **Conditional Drawdown at Risk (CDaR):** Average of worst 10% drawdowns
  - **Ulcer Index:** Root-mean-square of drawdown from peak
  - **Drawdown duration:** Average and max time to recover from drawdowns
- Add drawdown constraints to portfolio optimization:
  - Max acceptable drawdown: 8%
  - If current drawdown > 5%: Reduce gross exposure by 30%
  - If current drawdown > 8%: Halt all new positions until recovery to -5%
- This is what institutional allocators actually evaluate when selecting managers

### 🟡 OPS-003: Multi-Account Management
**Problem:** Single account only. Can't manage multiple accounts with different risk profiles.
**Action:**
- Abstract account management to support:
  - Multiple Alpaca accounts (paper + live, different strategies per account)
  - Account-level risk limits (different max drawdown per account)
  - Aggregate risk across accounts (total exposure monitoring)
  - Account-level performance reporting
- Use case: Run aggressive strategies in one account, conservative in another

### 🟡 OPS-004: Disaster Recovery & Business Continuity
**Problem:** No disaster recovery plan. If the server crashes mid-day, recovery is manual.
**Action:**
- Implement automated recovery:
  - On startup: Full position reconciliation with broker
  - On startup: Reconstruct pending order state from broker's order history
  - On startup: Verify circuit breaker state from today's P&L
  - Heartbeat monitoring: External watchdog that restarts bot if it stops responding
  - Database backup: Every hour to a separate location
  - Config backup: Version-controlled, encrypted
- Runbook documentation:
  - Step-by-step recovery for every failure mode
  - Contact information for broker support
  - Manual kill procedures if automation fails

### 🟡 OPS-005: CI/CD Pipeline
**Problem:** No continuous integration. Code changes go directly to production.
**Action:**
- Implement a deployment pipeline:
  - **Lint:** flake8/ruff for code quality, mypy for type checking
  - **Unit tests:** pytest with 80%+ coverage requirement
  - **Integration tests:** Paper trading simulation
  - **Shadow trading:** 1-day parallel run comparing new vs. old signals
  - **Canary deployment:** Route 10% of capital to new version, monitor for 1 week
  - **Rollback:** One-command rollback to previous version
- Git workflow: Feature branches → PR with code review → merge to staging → shadow test → production

---

## Phase 16: Additional Bug Fixes & Code Quality

### 🔴 BUG-015: Password in URL Query Parameter
**File:** `web_dashboard.py` (Line 141)
**Problem:** Login password passed as URL query parameter. Query parameters are logged in server access logs, browser history, and potentially in proxy logs. This is a critical security vulnerability.
**Fix:** Move authentication to POST request body. Use HTTPS. Implement proper session-based authentication with JWT tokens (the `auth/jwt_auth.py` module exists but may not be fully integrated).

### 🔴 BUG-016: 76 Silent Exception Handlers
**Files:** Throughout codebase (76 instances of bare `except Exception:` clauses)
**Problem:** Exceptions are swallowed without logging. Real errors are hidden, making debugging nearly impossible. Critical failures in strategies, execution, and risk management may go undetected.
**Fix:**
- Audit every `except Exception` clause. For each one:
  - Add `logger.error(f"...: {e}", exc_info=True)` at minimum
  - Determine if the exception should be re-raised or handled
  - For critical paths (order submission, risk checks): re-raise or trigger alert
  - For non-critical paths (notifications, dashboard): log and continue
- Create a coding standard: No bare `except Exception: pass` without a comment explaining why

### 🟡 BUG-017: IB Broker Skeleton in Production Code
**File:** `broker/ib_broker.py`
**Problem:** File is explicitly marked as skeleton/incomplete implementation. If accidentally selected as broker, it could cause undefined behavior.
**Fix:** Either complete the IB implementation or add a hard guard: `raise NotImplementedError("IB broker not yet implemented for production use")` at the top of every method. Remove from production code path until ready.

### 🟡 BUG-018: Database Migration Placeholder
**File:** `db/migrations/versions/651e01fa3df3_v10_initial_schema.py`
**Problem:** Migration file contains only `pass` statement. If Alembic runs this migration, it does nothing, potentially leaving the database in an inconsistent state.
**Fix:** Either implement the full schema migration or remove the placeholder and create proper migrations when the schema is finalized.

### 🟡 BUG-019: Cointegration Testing Ignores Corporate Actions
**File:** `strategies/kalman_pairs.py` (Lines 87-89)
**Problem:** Cointegration testing on 60 days of daily data doesn't check for stock splits, dividends, or mergers. A 2:1 split will break the cointegration relationship and produce false signals.
**Fix:** Use adjusted prices for all cointegration testing. Detect corporate actions (large overnight price changes > 20%) and refit the model when detected. Alternatively, use returns-based cointegration which is naturally adjusted.

### 🟡 BUG-020: Look-Ahead Bias in OU Half-Life
**File:** `strategies/stat_mean_reversion.py` (Lines 103-109)
**Problem:** OU half-life calculated from the same bars used to generate signals. The model "knows" future bar values when computing the half-life at the current time.
**Fix:** Compute OU parameters using only bars up to T-1 (excluding the current bar). Or better: use the half-life from the most recent universe preparation (which uses a separate lookback window) rather than recomputing on signal bars.

### 🟡 BUG-021: Walk-Forward Survivorship Bias
**File:** `walk_forward.py` (Lines 71-73)
**Problem:** Uses `trades_30d` without checking if failing strategies were already removed. If a strategy was demoted and its trades excluded, the walk-forward results are upward-biased.
**Fix:** Include all strategies in walk-forward analysis, including demoted ones. Track demotion separately from validation results. The walk-forward should evaluate ALL strategies to detect recovery, not just active ones.

### 🟡 BUG-022: Daily Tasks Error Handling
**File:** `engine/daily_tasks.py`
**Problem:** EOD close and weekly rebalance tasks don't catch exceptions from individual strategies' `reset_daily()`. One failing strategy halts the entire daily reset, leaving other strategies in an inconsistent state.
**Fix:** Wrap each strategy's `reset_daily()` in a try/except. Log errors but continue resetting remaining strategies. Report all failures in a summary alert.

### 🟡 BUG-023: Stale VIX in Crisis Scenarios
**File:** `config.py` (Line 279)
**Problem:** VIX cached for 5 minutes. During a crisis (exactly when VIX matters most), the cached value could be 5+ points stale. This means risk scaling lags reality by minutes during the most dangerous periods.
**Fix:** Reduce VIX cache to 30 seconds. Better: Subscribe to VIX via WebSocket or compute VIX proxy from SPY options data in real-time.

### 🟡 BUG-024: Missing Halted Symbol Detection
**Problem:** No detection of trading halts (LULD halts, T1/T2 regulatory halts, news-pending halts). Bot could submit orders for halted symbols, causing rejections and wasting API calls.
**Fix:**
- Check for zero volume over 5+ consecutive bars (potential halt indicator)
- Use Alpaca's asset status endpoint to check for halts before order submission
- Subscribe to halt/resume notifications if available via WebSocket
- Add halted symbols to a temporary blocklist until trading resumes

### 🟡 BUG-025: No Corporate Action Handling
**Problem:** Stock splits, reverse splits, and special dividends are not handled. After a 2:1 split: position quantities are wrong, historical indicators are broken, stop losses trigger incorrectly, pairs trading hedge ratios are wrong.
**Fix:**
- Detect corporate actions from overnight price changes > 20% with corresponding volume spike
- On split detection:
  - Adjust position quantities and cost basis
  - Invalidate all cached technical indicators for that symbol
  - Refit OU/cointegration models for affected pairs
  - Adjust stop loss and take profit levels
- Query Alpaca's corporate actions API daily before market open

### 🟡 BUG-026: Transaction Cost Win Rate Not Strategy-Specific
**File:** `oms/transaction_cost.py` (Lines 71-120)
**Problem:** Uses a default 55% win rate for transaction cost calculations. This is not realistic — mean reversion strategies have 65-75% win rate while momentum has 40-50%. Using 55% for all strategies either overestimates or underestimates expected profit.
**Fix:** Fetch per-strategy win rate from the trade database. If insufficient history (< 30 trades), use the strategy's theoretical win rate from config as default.

---

## What to Remove or Replace

### REMOVE-001: Hardcoded Beta Table
**File:** `config.py` (Lines 21-33)
**Problem:** Static beta values (TSLA=2.0, AAPL=1.05, etc.) that are never updated.
**Replace with:** Dynamic beta computation from 60-day rolling regression against SPY. Update daily at market open.

### REMOVE-002: Hardcoded Symbol Universe
**File:** `config.py` (Lines 52-70)
**Problem:** Static lists that become stale as companies are delisted, acquired, or lose liquidity.
**Replace with:** Dynamic universe selection pipeline (DATA-004).

### REMOVE-003: Static Strategy Allocation
**File:** `config.py` (allocation weights)
**Problem:** 40/20/20/5/5/10 allocation never adapts.
**Replace with:** Hierarchical Risk Parity (RISK-002) with weekly rebalancing.

### REMOVE-004: Simple News Sentiment Scoring
**File:** `engine/signal_processor.py` (news filter)
**Problem:** Keyword-based sentiment (counting "beat"/"miss" in headlines) is crude and easily fooled ("Company beats expectations" vs. "Company fails to beat expectations").
**Replace with:** FinBERT-based sentiment scoring that understands context and negation.

### REMOVE-005: Vectorized Backtester
**File:** `backtester.py`
**Problem:** Look-ahead bias, unrealistic fills, no transaction costs.
**Replace with:** Event-driven backtester (BACKTEST-001).

### REMOVE-006: Manual ETF Earnings Exemption List
**File:** `config.py` (120+ hardcoded ETF symbols)
**Problem:** Maintained manually. New ETFs not included. Non-ETFs might be incorrectly included.
**Replace with:** Dynamic check using Alpaca's asset classification API. ETFs don't have earnings — query asset type at runtime.

### REMOVE-007: Global Module-Level Clients
**Files:** `data.py`, `execution.py` (module-level _trading_client, _data_client)
**Problem:** Prevents testing, prevents multiple instances, creates import-time side effects.
**Replace with:** Dependency injection (ARCH-002).

---

## Appendix: Competition Benchmark

### What Top Firms Have That Velox V10 Doesn't

| Capability | Renaissance | Two Sigma | Citadel | DE Shaw | Velox V10 Current |
|---|---|---|---|---|---|
| Deep learning alpha | ✅ Transformers, RL | ✅ ML at scale | ✅ Multi-model | ✅ Hybrid | ❌ None |
| Alternative data | ✅ 100+ sources | ✅ Satellite, NLP | ✅ Transaction data | ✅ Patent, NLP | ❌ News headlines only |
| Feature count | 1000+ | 1000+ | 500+ | 500+ | ~30 |
| Factor risk model | ✅ Custom | ✅ Custom | ✅ Barra + custom | ✅ Custom | ❌ None |
| Portfolio optimization | ✅ Advanced | ✅ Black-Litterman | ✅ HRP + custom | ✅ Custom | ❌ Static allocation |
| Execution optimization | ✅ Sub-ms | ✅ Adaptive SOR | ✅ Custom algos | ✅ Custom | ❌ Basic TWAP |
| Backtesting rigor | ✅ CPCV + MC | ✅ Custom | ✅ Custom | ✅ Custom | ❌ Vectorized + look-ahead |
| Market microstructure | ✅ Full | ✅ VPIN + LOB | ✅ Full | ✅ Full | ❌ None |
| Infrastructure | ✅ FPGA + GPU | ✅ Distributed | ✅ Global colocation | ✅ Custom | ❌ Single-threaded Python |
| Stress testing | ✅ Daily | ✅ Continuous | ✅ Real-time | ✅ Daily | ❌ None |
| Test coverage | ✅ >90% | ✅ >90% | ✅ >90% | ✅ >90% | ❌ Minimal |

### Realistic Competitive Positioning After Full Implementation

After implementing everything in this roadmap, Velox V10 would be competitive with **mid-tier quantitative hedge funds** (AUM $50M-$500M). To compete with Renaissance or Citadel would additionally require:
- Proprietary data sources (not available via APIs)
- Colocation at exchanges (sub-millisecond latency)
- Teams of 50+ PhDs with decades of experience
- $100M+ in infrastructure investment
- Access to prime brokerage services (multi-venue, dark pools)

However, the system would be significantly more sophisticated than 99% of retail algorithmic trading systems and most small prop trading firms.

### Key Academic References

The dev team should read these foundational texts:

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning* — Feature engineering, CPCV, meta-labeling, fractional differentiation
2. **López de Prado, M. (2020).** *Machine Learning for Asset Managers* — Portfolio construction with ML, hierarchical risk parity
3. **Avellaneda, M. & Lee, J. (2010).** *Statistical Arbitrage in the US Equities Market* — OU process for mean reversion, eigenportfolios
4. **Almgren, R. & Chriss, N. (2001).** *Optimal Execution of Portfolio Transactions* — Execution cost minimization framework
5. **Cartea, Á., Jaimungal, S., & Penalva, J. (2015).** *Algorithmic and High-Frequency Trading* — Market microstructure, optimal execution
6. **Chan, E. (2013).** *Algorithmic Trading: Winning Strategies and Their Rationale* — Practical mean reversion and momentum strategies
7. **Black, F. & Litterman, R. (1992).** *Global Portfolio Optimization* — Black-Litterman model
8. **Easley, D., López de Prado, M., & O'Hara, M. (2012).** *Flow Toxicity and Liquidity in a High-Frequency World* — VPIN methodology

---

---

## Appendix B: V11 Feature Checklist (Quick Reference)

Use this checklist to track implementation progress:

**Data Layer:**
- [ ] Tick-level data pipeline with WebSocket
- [ ] Information-driven bars (TIB/VIB/DIB)
- [ ] Centralized feature store (200+ features)
- [ ] Alternative data integration (SEC, options flow, short interest, social, macro)
- [ ] Dynamic universe selection (daily refresh)
- [ ] Comprehensive data quality framework
- [ ] Fractionally-differentiated features
- [ ] Entropy-based features

**Alpha Generation:**
- [ ] Transformer-based price prediction
- [ ] RL for entry/exit optimization
- [ ] Cross-asset lead-lag signals
- [ ] Order flow imbalance signals
- [ ] Enhanced PEAD with NLP
- [ ] Copula-based pairs trading
- [ ] Sector momentum rotation
- [ ] Cross-sectional momentum
- [ ] Multi-timeframe confirmation
- [ ] Meta-labeling for signal confidence
- [ ] GNN for stock relationships

**Portfolio & Risk:**
- [ ] Factor risk model (multi-factor)
- [ ] Hierarchical Risk Parity allocation
- [ ] Stress testing framework (6+ scenarios)
- [ ] Dynamic hedging (delta, tail, sector)
- [ ] Rolling window P&L controls
- [ ] Gap risk management
- [ ] Liquidation scenario handling
- [ ] DCC-GARCH correlation
- [ ] RMT covariance cleaning
- [ ] Drawdown-based risk measures (CDaR, Calmar)
- [ ] BOCPD regime change detection

**Execution:**
- [ ] Adaptive smart order router
- [ ] Almgren-Chriss optimal execution
- [ ] Empirical slippage model
- [ ] Fill quality analytics
- [ ] Exponential backoff with jitter
- [ ] Enhanced pre-trade validation

**ML Pipeline:**
- [ ] Feature engineering (200+ features)
- [ ] Model training (LightGBM, XGBoost, LSTM, Transformer stacking)
- [ ] Online learning with concept drift detection
- [ ] CPCV overfitting prevention
- [ ] XAI (SHAP, attention visualization)
- [ ] Synthetic data generation

**Market Microstructure:**
- [ ] VPIN implementation
- [ ] Order book imbalance
- [ ] Trade size analysis (institutional flow)
- [ ] Effective spread measurement

**Backtesting:**
- [ ] Event-driven backtester (no look-ahead bias)
- [ ] CPCV with PBO/DSR
- [ ] Monte Carlo stress testing (1000+ paths)
- [ ] Performance attribution (factor, timing, execution)
- [ ] Alpha decay analysis

**Infrastructure:**
- [ ] Event-driven architecture
- [ ] Async I/O for network
- [ ] Dependency injection
- [ ] Configuration management (YAML + hot reload)
- [ ] PostgreSQL + TimescaleDB
- [ ] Secrets management
- [ ] CI/CD pipeline
- [ ] Disaster recovery

**Monitoring:**
- [ ] Real-time dashboard
- [ ] Performance metrics pipeline
- [ ] Tiered alerting system
- [ ] Latency monitoring
- [ ] Broker position reconciliation

**Compliance:**
- [ ] Complete audit trail
- [ ] Self-surveillance for manipulation
- [ ] Best execution documentation
- [ ] Robust PDT compliance

**Testing:**
- [ ] 80%+ unit test coverage
- [ ] Integration test suite
- [ ] Regression test suite
- [ ] Shadow trading validation

**Bug Fixes:**
- [ ] All 26 identified bugs resolved
- [ ] 76 silent exception handlers audited
- [ ] All timezone issues resolved
- [ ] All race conditions fixed

---

*This document was generated from a comprehensive audit of the Velox V10 codebase and benchmarking against state-of-the-art institutional quantitative trading systems as of March 2026. It serves as the complete blueprint for the Velox V11 upgrade.*
