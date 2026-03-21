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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math

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
