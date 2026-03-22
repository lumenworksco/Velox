"""COMP-001: EDGAR 10-K/10-Q NLP analysis — detect meaningful filing changes.

Fetches SEC EDGAR filings via the public EDGAR API (no API key needed).
Extracts key sections (risk factors, MD&A) and computes text similarity
to previous filings to detect meaningful changes. Uses TF-IDF for
similarity scoring and simple keyword extraction for topic detection.

Fail-open: if EDGAR is unreachable or parsing fails, returns neutral
results and never blocks trading.

Usage::

    analyzer = SECFilingAnalyzer(user_agent="MyBot admin@example.com")
    analyzer.fetch_recent_filings("AAPL", filing_type="10-K", count=2)
    result = analyzer.analyze_changes("AAPL")
    # result.similarity_score, result.changed_sections, result.key_topics
"""

import logging
import re
import time
import time as _time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC EDGAR API constants
# ---------------------------------------------------------------------------
EDGAR_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"
EDGAR_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{query}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=companyName%3A%22{company}%22"

# Key sections to extract from filings
KEY_SECTIONS = {
    "risk_factors": r"(?:Item\s+1A[\.\s]*Risk\s+Factors)",
    "mda": r"(?:Item\s+7[\.\s]*Management.s\s+Discussion)",
    "business": r"(?:Item\s+1[\.\s]*Business)",
    "legal": r"(?:Item\s+3[\.\s]*Legal\s+Proceedings)",
}

# Minimum similarity threshold — below this we flag as "significant change"
SIMILARITY_THRESHOLD = 0.75

# Cache TTL in seconds (filings don't change, so cache aggressively)
CACHE_TTL = 86400  # 24 hours

# Rate-limit: SEC asks for max 10 requests per second
REQUEST_DELAY = 0.12  # seconds between requests


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilingSection:
    """A single extracted section from an SEC filing."""
    name: str
    text: str
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class FilingRecord:
    """Metadata and extracted content from one filing."""
    cik: str
    accession: str
    filing_type: str  # "10-K" or "10-Q"
    filed_date: str
    sections: Dict[str, FilingSection] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class FilingChangeResult:
    """Result of comparing two consecutive filings."""
    symbol: str
    filing_type: str
    current_date: str
    previous_date: str
    similarity_score: float  # 0.0 (totally different) to 1.0 (identical)
    changed_sections: List[str] = field(default_factory=list)
    section_similarities: Dict[str, float] = field(default_factory=dict)
    key_topics: List[str] = field(default_factory=list)
    significant_change: bool = False


# Neutral result returned on failure
_NEUTRAL_RESULT = FilingChangeResult(
    symbol="",
    filing_type="",
    current_date="",
    previous_date="",
    similarity_score=1.0,
    significant_change=False,
)


# ---------------------------------------------------------------------------
# TF-IDF helpers (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase tokenization with stop-word removal."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "not",
        "this", "that", "these", "those", "it", "its", "we", "our", "us",
        "they", "their", "them", "he", "she", "his", "her", "you", "your",
        "i", "me", "my", "which", "who", "whom", "what", "where", "when",
        "how", "if", "then", "than", "so", "no", "nor", "each", "every",
        "all", "any", "both", "such", "into", "over", "after", "before",
        "between", "under", "above", "up", "down", "out", "about", "through",
    }
    tokens = re.findall(r"[a-z][a-z']{2,}", text.lower())
    return [t for t in tokens if t not in stop_words]


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (normalized by document length)."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count / total for word, count in counts.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency across a set of documents."""
    n_docs = len(documents)
    if n_docs == 0:
        return {}
    doc_freq: Counter = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1
    return {
        term: math.log((n_docs + 1) / (df + 1)) + 1
        for term, df in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF vector for a single document."""
    tf = _compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_key_topics(tokens: List[str], top_n: int = 10) -> List[str]:
    """Extract most frequent meaningful terms as key topics."""
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SECFilingAnalyzer:
    """Fetch and analyze SEC EDGAR filings for textual changes.

    Uses the SEC EDGAR public API (HTTPS, no API key). Requires a valid
    User-Agent string per SEC guidelines.
    """

    def __init__(self, user_agent: str = "VeloxBot admin@velox.dev"):
        self._user_agent = user_agent
        self._filings_cache: Dict[str, List[FilingRecord]] = {}  # symbol -> filings
        self._cik_cache: Dict[str, str] = {}  # symbol -> CIK
        self._last_request: float = 0.0

    # ------------------------------------------------------------------
    # Rate-limited HTTP helper
    # ------------------------------------------------------------------

    def _get(self, url: str) -> Optional[Any]:
        """Rate-limited GET request returning parsed JSON or None."""
        try:
            import urllib.request
            import json

            # Respect SEC rate limits
            elapsed = time.time() - self._last_request
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)

            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request = time.time()
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("EDGAR request failed for %s: %s", url, exc)
            return None

    def _get_text(self, url: str) -> Optional[str]:
        """Rate-limited GET request returning raw text or None."""
        try:
            import urllib.request

            elapsed = time.time() - self._last_request
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)

            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=30) as resp:
                self._last_request = time.time()
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("EDGAR text request failed for %s: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # CIK lookup
    # ------------------------------------------------------------------

    def _resolve_cik(self, symbol: str) -> Optional[str]:
        """Resolve a ticker symbol to a CIK number via EDGAR."""
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]

        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{symbol}%22&forms=10-K"
        try:
            # Alternative: use the company tickers JSON
            import urllib.request
            import json

            ticker_url = "https://www.sec.gov/files/company_tickers.json"
            req = urllib.request.Request(
                ticker_url, headers={"User-Agent": self._user_agent}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for entry in data.values():
                if entry.get("ticker", "").upper() == symbol.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self._cik_cache[symbol] = cik
                    return cik
        except Exception as exc:
            logger.debug("CIK lookup failed for %s: %s", symbol, exc)

        return None

    # ------------------------------------------------------------------
    # Filing fetching
    # ------------------------------------------------------------------

    def fetch_recent_filings(
        self,
        symbol: str,
        filing_type: str = "10-K",
        count: int = 2,
    ) -> List[FilingRecord]:
        """Fetch recent filings from EDGAR for a given symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            filing_type: "10-K" or "10-Q".
            count: Number of most recent filings to fetch.

        Returns:
            List of FilingRecord objects (most recent first).
        """
        cache_key = f"{symbol}:{filing_type}"
        if cache_key in self._filings_cache:
            return self._filings_cache[cache_key][:count]

        cik = self._resolve_cik(symbol)
        if not cik:
            logger.info("Could not resolve CIK for %s", symbol)
            return []

        submissions_url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        data = self._get(submissions_url)
        if not data:
            return []

        filings: List[FilingRecord] = []
        try:
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])

            found = 0
            for i, form in enumerate(forms):
                if found >= count:
                    break
                if form != filing_type:
                    continue

                accession = accessions[i].replace("-", "")
                filed_date = dates[i]
                filename = primary_docs[i]

                # Fetch the actual filing text
                doc_url = EDGAR_FILING_URL.format(
                    cik=cik.lstrip("0"), accession=accession, filename=filename
                )
                raw_text = self._get_text(doc_url)
                if not raw_text:
                    continue

                record = FilingRecord(
                    cik=cik,
                    accession=accessions[i],
                    filing_type=filing_type,
                    filed_date=filed_date,
                    raw_text=raw_text,
                )
                record.sections = self._extract_sections(raw_text)
                filings.append(record)
                found += 1

        except Exception as exc:
            logger.warning("Failed to parse EDGAR submissions for %s: %s", symbol, exc)

        self._filings_cache[cache_key] = filings
        return filings

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> Dict[str, FilingSection]:
        """Extract key sections from a filing's raw text.

        Uses regex patterns to locate section headers and extract text
        between consecutive headers. Strips HTML tags if present.
        """
        # Strip HTML tags for cleaner text
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)

        sections: Dict[str, FilingSection] = {}

        for name, pattern in KEY_SECTIONS.items():
            try:
                match = re.search(pattern, clean, re.IGNORECASE)
                if not match:
                    continue

                start = match.end()
                # Find next "Item" header to delimit this section
                next_item = re.search(r"Item\s+\d", clean[start:], re.IGNORECASE)
                end = start + next_item.start() if next_item else min(start + 50000, len(clean))

                section_text = clean[start:end].strip()
                # Limit section to ~10000 words for performance
                words = section_text.split()
                if len(words) > 10000:
                    section_text = " ".join(words[:10000])

                sections[name] = FilingSection(name=name, text=section_text)
            except Exception as exc:
                logger.debug("Section extraction failed for %s: %s", name, exc)

        return sections

    # ------------------------------------------------------------------
    # Change analysis
    # ------------------------------------------------------------------

    def analyze_changes(self, symbol: str, filing_type: str = "10-K") -> FilingChangeResult:
        """Compare the two most recent filings and quantify changes.

        Returns a FilingChangeResult with similarity scores per section
        and a list of significantly changed sections.
        """
        filings = self.fetch_recent_filings(symbol, filing_type, count=2)
        if len(filings) < 2:
            logger.info("Not enough filings for %s to compare", symbol)
            result = FilingChangeResult(
                symbol=symbol,
                filing_type=filing_type,
                current_date=filings[0].filed_date if filings else "",
                previous_date="",
                similarity_score=1.0,
                significant_change=False,
            )
            return result

        current, previous = filings[0], filings[1]

        # Tokenize full documents for IDF computation
        current_tokens = _tokenize(current.raw_text[:200000])
        previous_tokens = _tokenize(previous.raw_text[:200000])
        all_docs = [current_tokens, previous_tokens]
        idf = _compute_idf(all_docs)

        # Overall similarity
        vec_curr = _tfidf_vector(current_tokens, idf)
        vec_prev = _tfidf_vector(previous_tokens, idf)
        overall_sim = _cosine_similarity(vec_curr, vec_prev)

        # Per-section similarities
        section_sims: Dict[str, float] = {}
        changed_sections: List[str] = []

        all_section_names = set(current.sections.keys()) | set(previous.sections.keys())
        for section_name in all_section_names:
            curr_sec = current.sections.get(section_name)
            prev_sec = previous.sections.get(section_name)

            if curr_sec and prev_sec:
                sec_curr_tokens = _tokenize(curr_sec.text)
                sec_prev_tokens = _tokenize(prev_sec.text)
                sec_idf = _compute_idf([sec_curr_tokens, sec_prev_tokens])
                sec_vec_curr = _tfidf_vector(sec_curr_tokens, sec_idf)
                sec_vec_prev = _tfidf_vector(sec_prev_tokens, sec_idf)
                sim = _cosine_similarity(sec_vec_curr, sec_vec_prev)
            elif curr_sec and not prev_sec:
                sim = 0.0  # New section
            elif prev_sec and not curr_sec:
                sim = 0.0  # Removed section
            else:
                sim = 1.0

            section_sims[section_name] = sim
            if sim < SIMILARITY_THRESHOLD:
                changed_sections.append(section_name)

        # Key topics from the current filing
        key_topics = _extract_key_topics(current_tokens, top_n=15)

        return FilingChangeResult(
            symbol=symbol,
            filing_type=filing_type,
            current_date=current.filed_date,
            previous_date=previous.filed_date,
            similarity_score=overall_sim,
            changed_sections=changed_sections,
            section_similarities=section_sims,
            key_topics=key_topics,
            significant_change=overall_sim < SIMILARITY_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Batch screening
    # ------------------------------------------------------------------

    def screen_symbols(
        self,
        symbols: List[str],
        filing_type: str = "10-K",
    ) -> List[FilingChangeResult]:
        """Screen multiple symbols for significant filing changes.

        Returns only symbols with significant changes (similarity below
        threshold), sorted by most changed first.
        """
        results: List[FilingChangeResult] = []
        for symbol in symbols:
            try:
                result = self.analyze_changes(symbol, filing_type)
                if result.significant_change:
                    results.append(result)
            except Exception as exc:
                logger.debug("Screening failed for %s: %s", symbol, exc)

        results.sort(key=lambda r: r.similarity_score)
        return results

    def clear_cache(self) -> None:
        """Clear all cached filings and CIK lookups."""
        self._filings_cache.clear()
        self._cik_cache.clear()


# ======================================================================
# T7-003: SEC EDGAR Real-Time 8-K Parser
# ======================================================================

# 8-K Item type classification
_8K_ITEM_CLASSIFICATION: Dict[str, Dict[str, Any]] = {
    "1.01": {"label": "Entry into Material Agreement (M&A)", "strength": "strong", "bias_mult": 1.5},
    "1.02": {"label": "Termination of Material Agreement", "strength": "strong", "bias_mult": -1.0},
    "1.03": {"label": "Bankruptcy/Receivership", "strength": "strong", "bias_mult": -2.0},
    "2.01": {"label": "Completion of Acquisition/Disposition", "strength": "strong", "bias_mult": 1.2},
    "2.02": {"label": "Results of Operations (Earnings)", "strength": "covered_by_pead", "bias_mult": 0.0},
    "2.03": {"label": "Creation of Financial Obligation", "strength": "moderate", "bias_mult": -0.3},
    "2.04": {"label": "Triggering Events", "strength": "moderate", "bias_mult": -0.5},
    "2.05": {"label": "Costs of Exit/Disposal", "strength": "moderate", "bias_mult": -0.6},
    "2.06": {"label": "Material Impairment", "strength": "strong", "bias_mult": -1.2},
    "3.01": {"label": "Delisting Notice", "strength": "strong", "bias_mult": -1.5},
    "4.01": {"label": "Auditor Change", "strength": "moderate", "bias_mult": -0.4},
    "4.02": {"label": "Non-Reliance on Financial Statements", "strength": "strong", "bias_mult": -1.8},
    "5.01": {"label": "Changes in Control", "strength": "strong", "bias_mult": 0.8},
    "5.02": {"label": "Director/Officer Changes", "strength": "moderate", "bias_mult": -0.2},
    "5.03": {"label": "Amendments to Articles", "strength": "weak", "bias_mult": 0.0},
    "5.07": {"label": "Submission of Matters to Vote", "strength": "weak", "bias_mult": 0.0},
    "7.01": {"label": "Regulation FD Disclosure", "strength": "moderate", "bias_mult": 0.3},
    "8.01": {"label": "Other Events", "strength": "weak", "bias_mult": 0.1},
    "9.01": {"label": "Financial Statements and Exhibits", "strength": "weak", "bias_mult": 0.0},
}

# Sentiment keyword dictionaries for 8-K text scoring
_POSITIVE_KEYWORDS = {
    "growth", "exceeded", "strong", "positive", "record", "increased",
    "expansion", "partnership", "agreement", "awarded", "approved",
    "upgrade", "innovation", "synergy", "profitable", "outperform",
    "momentum", "breakthrough", "favorable", "exceeded expectations",
}

_NEGATIVE_KEYWORDS = {
    "loss", "decline", "negative", "impairment", "bankruptcy", "default",
    "investigation", "lawsuit", "termination", "restructuring", "layoff",
    "downgrade", "concern", "weakness", "deterioration", "shortfall",
    "restatement", "fraud", "violation", "penalty", "recall", "warning",
}


@dataclass
class Filing8KResult:
    """Result from parsing a single 8-K filing."""
    symbol: str
    cik: str
    filed_date: str
    accession: str
    item_types: List[str]                  # e.g., ["1.01", "9.01"]
    item_labels: List[str]                 # Human-readable labels
    strength: str                          # "strong", "moderate", "weak", "covered_by_pead"
    sentiment_score: float                 # [-1, 1] from text analysis
    signal_bias: float                     # Combined directional bias for signal_processor
    summary_text: str = ""                 # First ~500 chars of filing
    url: str = ""


@dataclass
class EdgarMonitorStatus:
    """Status of the EDGAR monitor."""
    enabled: bool = False
    last_poll: Optional[str] = None
    filings_found_today: int = 0
    signals_emitted: int = 0
    rate_limit_remaining: int = 60
    errors_today: int = 0


class EdgarMonitor:
    """T7-003: Real-time SEC EDGAR 8-K filing monitor.

    Polls the EDGAR full-text search API every 5 minutes during market hours,
    filters for universe symbols, classifies 8-K items, and produces signal
    biases for the signal processor.

    Rate limiting: max 1 API call per minute (SEC guideline: 10 req/sec,
    but we are conservative to avoid blocks).

    Gate: EDGAR_MONITOR_ENABLED config flag.

    Usage:
        monitor = EdgarMonitor(universe=["AAPL", "MSFT", "GOOG"])
        await monitor.poll()  # or monitor.poll_sync()
        for result in monitor.get_recent_filings():
            signal_processor.apply_soft_bias(result.symbol, result.signal_bias)
    """

    # EDGAR EFTS (full-text search) endpoint
    EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(
        self,
        universe: List[str] | None = None,
        user_agent: str = "VeloxBot admin@velox.dev",
        poll_interval_sec: int = 300,      # 5 minutes
        min_request_interval: float = 60.0, # 1 minute between API calls
    ):
        self._enabled = getattr(config, "EDGAR_MONITOR_ENABLED", False)
        self._universe = set(s.upper() for s in (universe or []))
        self._user_agent = user_agent
        self._poll_interval = poll_interval_sec
        self._min_request_interval = min_request_interval

        # CIK <-> Ticker mapping
        self._cik_to_ticker: Dict[str, str] = {}
        self._ticker_to_cik: Dict[str, str] = {}

        # State
        self._last_poll: Optional[datetime] = None
        self._last_request: float = 0.0
        self._recent_filings: List[Filing8KResult] = []
        self._seen_accessions: set = set()
        self._lock = threading.Lock()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        # Daily counters
        self._filings_today = 0
        self._signals_today = 0
        self._errors_today = 0

        if not self._enabled:
            logger.info("T7-003: EDGAR monitor disabled (EDGAR_MONITOR_ENABLED=False)")

    def start(self):
        """Start background polling thread."""
        if not self._enabled:
            return
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="edgar-monitor"
        )
        self._poll_thread.start()
        logger.info("T7-003: EDGAR monitor started (polling every %ds)", self._poll_interval)

    def stop(self):
        """Stop background polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=10)
        logger.info("T7-003: EDGAR monitor stopped")

    def poll_sync(self) -> List[Filing8KResult]:
        """Synchronous poll for new 8-K filings. Safe to call manually."""
        if not self._enabled:
            return []

        try:
            # Ensure CIK mapping is loaded
            if not self._cik_to_ticker:
                self._load_cik_mapping()

            filings = self._fetch_recent_8k()
            new_filings = []

            for filing in filings:
                if filing.accession in self._seen_accessions:
                    continue
                self._seen_accessions.add(filing.accession)
                new_filings.append(filing)
                self._filings_today += 1

            with self._lock:
                self._recent_filings.extend(new_filings)
                # Keep last 500 filings
                if len(self._recent_filings) > 500:
                    self._recent_filings = self._recent_filings[-500:]
                self._last_poll = datetime.now(config.ET)

            if new_filings:
                logger.info(
                    f"T7-003: Found {len(new_filings)} new 8-K filings: "
                    f"{', '.join(f.symbol for f in new_filings[:5])}"
                    f"{'...' if len(new_filings) > 5 else ''}"
                )

            return new_filings

        except Exception as e:
            self._errors_today += 1
            logger.warning(f"T7-003: Poll failed: {e}")
            return []

    def get_recent_filings(self, symbol: str | None = None) -> List[Filing8KResult]:
        """Get recent 8-K filings, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                return [f for f in self._recent_filings if f.symbol == symbol.upper()]
            return list(self._recent_filings)

    def get_signal_bias(self, symbol: str) -> float:
        """Get the aggregate signal bias for a symbol from recent 8-K filings.

        Returns a value in [-1, 1] where:
        - Positive = bullish signal (e.g., M&A, positive earnings surprise)
        - Negative = bearish signal (e.g., bankruptcy, restatement)
        - 0.0 = neutral (no recent filings or covered by PEAD)
        """
        filings = self.get_recent_filings(symbol)
        if not filings:
            return 0.0

        # Weight more recent filings higher
        total_bias = 0.0
        total_weight = 0.0
        now = datetime.now(config.ET)

        for f in filings:
            try:
                filed = datetime.fromisoformat(f.filed_date)
                age_hours = max(1, (now - filed).total_seconds() / 3600)
            except (ValueError, TypeError):
                age_hours = 24.0

            # Decay: full weight at 0h, half at 24h, quarter at 48h
            weight = 1.0 / (1.0 + age_hours / 24.0)
            total_bias += f.signal_bias * weight
            total_weight += weight

        if total_weight > 0:
            return max(-1.0, min(1.0, total_bias / total_weight))
        return 0.0

    @property
    def status(self) -> dict:
        """Return monitor status."""
        return {
            "enabled": self._enabled,
            "running": self._running,
            "last_poll": self._last_poll.isoformat() if self._last_poll else None,
            "universe_size": len(self._universe),
            "filings_today": self._filings_today,
            "signals_today": self._signals_today,
            "errors_today": self._errors_today,
            "cached_filings": len(self._recent_filings),
        }

    # ------------------------------------------------------------------
    # Internal: Background polling
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                # Only poll during market hours (roughly 9:00-16:30 ET)
                now = datetime.now(config.ET)
                hour = now.hour
                if 9 <= hour <= 16 and now.weekday() < 5:
                    self.poll_sync()
            except Exception as e:
                self._errors_today += 1
                logger.warning(f"T7-003: Poll loop error: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(self._poll_interval):
                if not self._running:
                    return
                _time.sleep(1)

    # ------------------------------------------------------------------
    # Internal: CIK mapping
    # ------------------------------------------------------------------

    def _load_cik_mapping(self):
        """Load CIK <-> ticker mapping from SEC."""
        try:
            import urllib.request
            import json

            elapsed = _time.time() - self._last_request
            if elapsed < self._min_request_interval:
                _time.sleep(self._min_request_interval - elapsed)

            req = urllib.request.Request(
                self.COMPANY_TICKERS_URL,
                headers={"User-Agent": self._user_agent},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request = _time.time()
                data = json.loads(resp.read().decode("utf-8"))

            for entry in data.values():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if ticker and cik:
                    self._cik_to_ticker[cik] = ticker
                    self._ticker_to_cik[ticker] = cik

            logger.info(f"T7-003: Loaded {len(self._cik_to_ticker)} CIK->ticker mappings")

        except Exception as e:
            logger.warning(f"T7-003: Failed to load CIK mapping: {e}")

    # ------------------------------------------------------------------
    # Internal: Fetch and parse 8-K filings
    # ------------------------------------------------------------------

    def _fetch_recent_8k(self) -> List[Filing8KResult]:
        """Fetch recent 8-K filings from EDGAR EFTS."""
        import urllib.request
        import json

        # Rate limit
        elapsed = _time.time() - self._last_request
        if elapsed < self._min_request_interval:
            _time.sleep(self._min_request_interval - elapsed)

        # Search for recent 8-K filings
        now = datetime.now(config.ET)
        start_date = (now - timedelta(hours=6)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        url = (
            f"{self.EFTS_SEARCH_URL}?q=*&forms=8-K"
            f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
            f"&from=0&size=50"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
            with urllib.request.urlopen(req, timeout=15) as resp:
                self._last_request = _time.time()
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug(f"T7-003: EFTS search failed: {e}")
            return []

        filings: List[Filing8KResult] = []
        hits = data.get("hits", {}).get("hits", [])

        for hit in hits:
            try:
                source = hit.get("_source", {})
                cik = str(source.get("entity_id", "")).zfill(10)
                accession = source.get("file_num", "") or hit.get("_id", "")
                filed_date = source.get("file_date", "")

                # Filter by universe
                ticker = self._cik_to_ticker.get(cik, "")
                if not ticker:
                    continue
                if self._universe and ticker not in self._universe:
                    continue

                # Extract and classify 8-K items from the filing text
                text = source.get("text", "") or source.get("display_names", [""])[0]
                items = self._extract_8k_items(text)
                if not items:
                    items = ["8.01"]  # Default: Other Events

                # Classify
                classifications = [_8K_ITEM_CLASSIFICATION.get(item, {}) for item in items]
                labels = [c.get("label", f"Item {item}") for c, item in zip(classifications, items)]

                # Determine overall strength (strongest wins)
                strength_order = {"strong": 3, "moderate": 2, "covered_by_pead": 1, "weak": 0}
                strengths = [c.get("strength", "weak") for c in classifications]
                overall_strength = max(strengths, key=lambda s: strength_order.get(s, 0))

                # Sentiment scoring
                sentiment = self._score_sentiment(text)

                # Combined signal bias
                bias_mults = [c.get("bias_mult", 0.0) for c in classifications]
                avg_bias_mult = sum(bias_mults) / len(bias_mults) if bias_mults else 0.0

                # If covered by PEAD, set bias to 0 (let PEAD handle it)
                if overall_strength == "covered_by_pead":
                    signal_bias = 0.0
                else:
                    signal_bias = avg_bias_mult * (0.5 + 0.5 * sentiment)
                    signal_bias = max(-1.0, min(1.0, signal_bias))

                filings.append(Filing8KResult(
                    symbol=ticker,
                    cik=cik,
                    filed_date=filed_date,
                    accession=accession,
                    item_types=items,
                    item_labels=labels,
                    strength=overall_strength,
                    sentiment_score=sentiment,
                    signal_bias=signal_bias,
                    summary_text=text[:500] if text else "",
                ))

            except Exception as e:
                logger.debug(f"T7-003: Failed to parse 8-K hit: {e}")
                continue

        return filings

    @staticmethod
    def _extract_8k_items(text: str) -> List[str]:
        """Extract 8-K item numbers from filing text."""
        # Match patterns like "Item 1.01", "Item 2.02", etc.
        pattern = r"Item\s+(\d+\.\d+)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Deduplicate while preserving order
        seen = set()
        items = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                items.append(m)
        return items

    @staticmethod
    def _score_sentiment(text: str) -> float:
        """Simple keyword-based sentiment score for 8-K text.

        Returns:
            Float in [-1, 1]. Positive = bullish, negative = bearish.
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        words = set(re.findall(r"[a-z]+", text_lower))

        pos_count = len(words & _POSITIVE_KEYWORDS)
        neg_count = len(words & _NEGATIVE_KEYWORDS)
        total = pos_count + neg_count

        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total
