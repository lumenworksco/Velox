"""COMP-004: 13F quarterly position tracking — institutional "smart money" signals.

Parses SEC EDGAR 13F filings to track institutional holdings. Detects
large position changes by major institutions and computes aggregate
"smart money" positioning per symbol.

Key features:
- Fetch 13F filings from EDGAR for tracked institutions
- Detect quarter-over-quarter position changes
- Compute aggregate institutional sentiment
- Track concentration and conviction metrics

Fail-open: returns neutral results if EDGAR is unreachable.

Usage::

    tracker = InstitutionalHoldingsTracker(user_agent="MyBot admin@example.com")
    tracker.update_institution("Berkshire Hathaway", cik="0001067983")
    changes = tracker.detect_position_changes("Berkshire Hathaway")
    sentiment = tracker.get_smart_money_sentiment("AAPL")
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

# Well-known institutions to track (name -> CIK)
NOTABLE_INSTITUTIONS = {
    "Berkshire Hathaway": "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors": "0001423053",
    "Two Sigma Investments": "0001179392",
    "DE Shaw": "0001009207",
    "AQR Capital": "0001167557",
    "Tiger Global": "0001167483",
    "Soros Fund Management": "0001029160",
    "Appaloosa Management": "0001656456",
}

# Thresholds for detecting significant changes
POSITION_CHANGE_SIGNIFICANT = 0.25  # 25% change in shares
NEW_POSITION_MIN_VALUE = 1_000_000  # $1M minimum for new position alert
EXIT_MIN_VALUE = 500_000            # $500K minimum for exit alert

# Rate-limit for SEC EDGAR
REQUEST_DELAY = 0.12

# Cache TTL (13F filed quarterly, check weekly)
CACHE_TTL = 604800  # 7 days

# 13F XML namespace
NS_13F = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Holding:
    """A single position from a 13F filing."""
    symbol: str                         # CUSIP-resolved ticker (best-effort)
    cusip: str                          # CUSIP identifier
    name_of_issuer: str
    title_of_class: str
    value: int                          # Market value in thousands ($000s)
    shares: int                         # Number of shares
    share_type: str = "SH"             # SH (shares) or PRN (principal)
    investment_discretion: str = "SOLE"
    voting_authority_sole: int = 0
    voting_authority_shared: int = 0
    voting_authority_none: int = 0


@dataclass
class InstitutionReport:
    """A complete 13F filing for one institution."""
    institution_name: str
    cik: str
    filing_date: str                    # ISO date
    report_date: str                    # Quarter-end date
    holdings: List[Holding] = field(default_factory=list)
    total_value: int = 0                # Sum of all position values ($000s)
    position_count: int = 0


@dataclass
class PositionChange:
    """A detected change in an institution's position."""
    institution: str
    symbol: str
    cusip: str
    change_type: str                    # "new", "increased", "decreased", "exited"
    current_shares: int
    previous_shares: int
    current_value: int                  # $000s
    previous_value: int                 # $000s
    change_pct: float                   # Percent change in shares
    value_change: int                   # Change in value ($000s)

    @property
    def summary(self) -> str:
        if self.change_type == "new":
            return f"{self.institution} NEW position in {self.symbol}: {self.current_shares:,} shares (${self.current_value:,}K)"
        elif self.change_type == "exited":
            return f"{self.institution} EXITED {self.symbol}: was {self.previous_shares:,} shares"
        else:
            return (
                f"{self.institution} {self.change_type.upper()} {self.symbol}: "
                f"{self.previous_shares:,} -> {self.current_shares:,} shares ({self.change_pct:+.1%})"
            )


@dataclass
class SmartMoneySentiment:
    """Aggregate institutional sentiment for a symbol."""
    symbol: str
    num_holders: int                    # Institutions holding this symbol
    total_shares: int                   # Aggregate shares held
    total_value: int                    # Aggregate value ($000s)
    net_buyers: int                     # Institutions that increased
    net_sellers: int                    # Institutions that decreased
    new_positions: int                  # Institutions that initiated
    exits: int                          # Institutions that exited
    sentiment_score: float              # -1.0 (bearish) to +1.0 (bullish)
    conviction: float                   # 0.0 to 1.0 (strength of consensus)
    top_holders: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CUSIP-to-ticker mapping (common large-caps)
# ---------------------------------------------------------------------------

# Partial mapping for common CUSIPs. In production, use a full CUSIP database.
_CUSIP_TICKER_MAP = {
    "037833100": "AAPL",
    "594918104": "MSFT",
    "02079K107": "GOOG",
    "02079K305": "GOOGL",
    "023135106": "AMZN",
    "67066G104": "NVDA",
    "30303M102": "META",
    "88160R101": "TSLA",
    "084670702": "BRK.B",
    "46625H100": "JPM",
    "92826C839": "V",
    "478160104": "JNJ",
    "742718109": "PG",
    "91324P102": "UNH",
    "459200101": "IBM",
}


def _cusip_to_ticker(cusip: str) -> str:
    """Best-effort CUSIP to ticker resolution."""
    return _CUSIP_TICKER_MAP.get(cusip, cusip)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InstitutionalHoldingsTracker:
    """Track institutional 13F filings and detect position changes."""

    def __init__(self, user_agent: str = "VeloxBot admin@velox.dev"):
        self._user_agent = user_agent
        self._reports: Dict[str, List[InstitutionReport]] = {}  # inst_name -> reports
        self._last_request: float = 0.0
        self._last_update: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get(self, url: str) -> Optional[Any]:
        """Rate-limited GET returning parsed JSON or None."""
        try:
            import urllib.request
            import json

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
        """Rate-limited GET returning raw text or None."""
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
    # Filing fetching
    # ------------------------------------------------------------------

    def update_institution(
        self,
        name: str,
        cik: Optional[str] = None,
        count: int = 2,
    ) -> List[InstitutionReport]:
        """Fetch recent 13F filings for an institution.

        Args:
            name: Institution name (used as cache key).
            cik: CIK number. If None, looks up from NOTABLE_INSTITUTIONS.
            count: Number of recent filings to fetch.

        Returns:
            List of InstitutionReport objects (most recent first).
        """
        if cik is None:
            cik = NOTABLE_INSTITUTIONS.get(name)
        if not cik:
            logger.info("No CIK found for institution: %s", name)
            return []

        elapsed = time.time() - self._last_update.get(name, 0)
        if elapsed < CACHE_TTL and name in self._reports:
            return self._reports[name][:count]

        submissions_url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        data = self._get(submissions_url)
        if not data:
            return self._reports.get(name, [])[:count]

        reports: List[InstitutionReport] = []
        try:
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])

            found = 0
            for i, form_type in enumerate(forms):
                if found >= count:
                    break
                if form_type not in ("13F-HR", "13F-HR/A"):
                    continue

                accession = accessions[i].replace("-", "")
                filing_date = dates[i]

                # Try to find the information table XML
                info_table = self._fetch_info_table(cik.lstrip("0"), accession)
                holdings = self._parse_info_table(info_table) if info_table else []

                total_value = sum(h.value for h in holdings)
                report = InstitutionReport(
                    institution_name=name,
                    cik=cik,
                    filing_date=filing_date,
                    report_date=filing_date,  # Approximation
                    holdings=holdings,
                    total_value=total_value,
                    position_count=len(holdings),
                )
                reports.append(report)
                found += 1

        except Exception as exc:
            logger.warning("Failed to parse 13F filings for %s: %s", name, exc)

        self._reports[name] = reports
        self._last_update[name] = time.time()
        return reports

    def _fetch_info_table(self, cik: str, accession: str) -> Optional[str]:
        """Try to find and fetch the 13F information table XML."""
        # The info table is usually named *infotable.xml or similar
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
        index_data = self._get(index_url)
        if not index_data:
            return None

        try:
            for item in index_data.get("directory", {}).get("item", []):
                fname = item.get("name", "")
                if "infotable" in fname.lower() or fname.endswith(".xml"):
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{fname}"
                    return self._get_text(doc_url)
        except Exception as exc:
            logger.debug("Info table fetch failed: %s", exc)

        return None

    def _parse_info_table(self, xml_text: str) -> List[Holding]:
        """Parse 13F information table XML into Holding objects."""
        holdings: List[Holding] = []
        try:
            root = ET.fromstring(xml_text)

            # Try with and without namespace
            for entry in root.iter():
                if "infoTable" in entry.tag:
                    holding = self._parse_info_entry(entry)
                    if holding:
                        holdings.append(holding)
        except ET.ParseError:
            # Fall back to regex-based parsing for malformed XML
            holdings = self._parse_info_table_regex(xml_text)
        except Exception as exc:
            logger.debug("13F XML parsing failed: %s", exc)

        return holdings

    def _parse_info_entry(self, entry) -> Optional[Holding]:
        """Parse a single infoTable entry from XML."""
        try:
            def _find(tag: str) -> str:
                for elem in entry.iter():
                    if tag.lower() in elem.tag.lower():
                        return (elem.text or "").strip()
                return ""

            cusip = _find("cusip")
            name_of_issuer = _find("nameOfIssuer")
            title = _find("titleOfClass")
            value_str = _find("value")
            shares_str = _find("sshPrnamt") or _find("shares")
            share_type = _find("sshPrnamtType") or "SH"

            value = int(value_str) if value_str else 0
            shares = int(shares_str) if shares_str else 0

            if not cusip:
                return None

            return Holding(
                symbol=_cusip_to_ticker(cusip),
                cusip=cusip,
                name_of_issuer=name_of_issuer,
                title_of_class=title,
                value=value,
                shares=shares,
                share_type=share_type,
            )
        except Exception:
            return None

    def _parse_info_table_regex(self, text: str) -> List[Holding]:
        """Fallback regex parser for 13F XML that may be malformed."""
        holdings = []
        # Simple regex to extract CUSIP and value pairs
        cusip_pattern = re.findall(r"<cusip>([\w]+)</cusip>", text, re.IGNORECASE)
        value_pattern = re.findall(r"<value>(\d+)</value>", text, re.IGNORECASE)
        shares_pattern = re.findall(r"<sshPrnamt>(\d+)</sshPrnamt>", text, re.IGNORECASE)
        name_pattern = re.findall(r"<nameOfIssuer>([^<]+)</nameOfIssuer>", text, re.IGNORECASE)

        for i, cusip in enumerate(cusip_pattern):
            value = int(value_pattern[i]) if i < len(value_pattern) else 0
            shares = int(shares_pattern[i]) if i < len(shares_pattern) else 0
            name = name_pattern[i].strip() if i < len(name_pattern) else ""

            holdings.append(Holding(
                symbol=_cusip_to_ticker(cusip),
                cusip=cusip,
                name_of_issuer=name,
                title_of_class="",
                value=value,
                shares=shares,
            ))
        return holdings

    # ------------------------------------------------------------------
    # Position change detection
    # ------------------------------------------------------------------

    def detect_position_changes(self, institution_name: str) -> List[PositionChange]:
        """Compare the two most recent 13F filings to detect changes.

        Returns a list of PositionChange objects for significant moves.
        """
        reports = self._reports.get(institution_name, [])
        if len(reports) < 2:
            logger.info("Need at least 2 reports for %s to detect changes", institution_name)
            return []

        current_report, previous_report = reports[0], reports[1]
        current_map = {h.cusip: h for h in current_report.holdings}
        previous_map = {h.cusip: h for h in previous_report.holdings}

        changes: List[PositionChange] = []
        all_cusips = set(current_map.keys()) | set(previous_map.keys())

        for cusip in all_cusips:
            curr = current_map.get(cusip)
            prev = previous_map.get(cusip)

            if curr and not prev:
                # New position
                if curr.value >= NEW_POSITION_MIN_VALUE / 1000:
                    changes.append(PositionChange(
                        institution=institution_name,
                        symbol=curr.symbol,
                        cusip=cusip,
                        change_type="new",
                        current_shares=curr.shares,
                        previous_shares=0,
                        current_value=curr.value,
                        previous_value=0,
                        change_pct=1.0,
                        value_change=curr.value,
                    ))
            elif prev and not curr:
                # Exited position
                if prev.value >= EXIT_MIN_VALUE / 1000:
                    changes.append(PositionChange(
                        institution=institution_name,
                        symbol=prev.symbol,
                        cusip=cusip,
                        change_type="exited",
                        current_shares=0,
                        previous_shares=prev.shares,
                        current_value=0,
                        previous_value=prev.value,
                        change_pct=-1.0,
                        value_change=-prev.value,
                    ))
            elif curr and prev and prev.shares > 0:
                # Existing position — check for significant change
                change_pct = (curr.shares - prev.shares) / prev.shares
                if abs(change_pct) >= POSITION_CHANGE_SIGNIFICANT:
                    change_type = "increased" if change_pct > 0 else "decreased"
                    changes.append(PositionChange(
                        institution=institution_name,
                        symbol=curr.symbol,
                        cusip=cusip,
                        change_type=change_type,
                        current_shares=curr.shares,
                        previous_shares=prev.shares,
                        current_value=curr.value,
                        previous_value=prev.value,
                        change_pct=change_pct,
                        value_change=curr.value - prev.value,
                    ))

        # Sort by absolute value change
        changes.sort(key=lambda c: abs(c.value_change), reverse=True)
        return changes

    # ------------------------------------------------------------------
    # Smart money aggregation
    # ------------------------------------------------------------------

    def get_smart_money_sentiment(self, symbol: str) -> SmartMoneySentiment:
        """Aggregate institutional sentiment for a symbol across all tracked institutions.

        Computes a sentiment score based on net buying/selling activity
        and a conviction score based on the strength of consensus.
        """
        holders: List[str] = []
        total_shares = 0
        total_value = 0
        net_buyers = 0
        net_sellers = 0
        new_positions = 0
        exits = 0

        for inst_name, reports in self._reports.items():
            if not reports:
                continue

            current = reports[0]
            current_holding = None
            for h in current.holdings:
                if h.symbol == symbol or h.cusip == symbol:
                    current_holding = h
                    break

            if current_holding:
                holders.append(inst_name)
                total_shares += current_holding.shares
                total_value += current_holding.value

            # Check changes if we have history
            if len(reports) >= 2:
                changes = self.detect_position_changes(inst_name)
                for change in changes:
                    if change.symbol != symbol:
                        continue
                    if change.change_type == "new":
                        new_positions += 1
                        net_buyers += 1
                    elif change.change_type == "exited":
                        exits += 1
                        net_sellers += 1
                    elif change.change_type == "increased":
                        net_buyers += 1
                    elif change.change_type == "decreased":
                        net_sellers += 1

        # Sentiment score
        total_actors = net_buyers + net_sellers
        if total_actors > 0:
            sentiment_score = (net_buyers - net_sellers) / total_actors
        else:
            sentiment_score = 0.0

        # Conviction: how strongly do institutions agree?
        if total_actors > 0:
            conviction = abs(net_buyers - net_sellers) / total_actors
        else:
            conviction = 0.0

        return SmartMoneySentiment(
            symbol=symbol,
            num_holders=len(holders),
            total_shares=total_shares,
            total_value=total_value,
            net_buyers=net_buyers,
            net_sellers=net_sellers,
            new_positions=new_positions,
            exits=exits,
            sentiment_score=sentiment_score,
            conviction=conviction,
            top_holders=holders[:10],
        )

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def update_all_notable(self, count: int = 2) -> int:
        """Fetch 13F filings for all notable institutions.

        Returns the number of institutions successfully updated.
        """
        updated = 0
        for name, cik in NOTABLE_INSTITUTIONS.items():
            try:
                reports = self.update_institution(name, cik, count)
                if reports:
                    updated += 1
            except Exception as exc:
                logger.debug("Failed to update %s: %s", name, exc)
        return updated

    def screen_smart_money(
        self,
        symbols: List[str],
        min_holders: int = 2,
    ) -> List[SmartMoneySentiment]:
        """Screen symbols for strong institutional sentiment.

        Returns symbols where multiple institutions are actively
        buying or selling, sorted by absolute sentiment score.
        """
        results: List[SmartMoneySentiment] = []
        for symbol in symbols:
            try:
                sentiment = self.get_smart_money_sentiment(symbol)
                if sentiment.num_holders >= min_holders and abs(sentiment.sentiment_score) > 0.2:
                    results.append(sentiment)
            except Exception as exc:
                logger.debug("Smart money screening failed for %s: %s", symbol, exc)

        results.sort(key=lambda s: abs(s.sentiment_score), reverse=True)
        return results

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._reports.clear()
        self._last_update.clear()
