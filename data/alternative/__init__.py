"""Alternative data package — competitive edge data sources (COMP-001 through COMP-006).

Provides modules for:
    - SEC EDGAR filing analysis (sec_filings)
    - Unusual options activity detection (options_flow)
    - Short interest and squeeze detection (short_interest)
    - Institutional 13F holdings tracking (institutional_holdings)
    - Social media sentiment scoring (social_sentiment)
    - Macro economic surprise index (macro_surprise)

All modules follow fail-open design: if data is unavailable, they return
neutral defaults and never block trading.
"""

from data.alternative.sec_filings import SECFilingAnalyzer, EdgarMonitor
from data.alternative.options_flow import OptionsFlowDetector
from data.alternative.short_interest import ShortInterestTracker
from data.alternative.institutional_holdings import InstitutionalHoldingsTracker
from data.alternative.social_sentiment import SocialSentimentAnalyzer
from data.alternative.macro_surprise import MacroSurpriseIndex

__all__ = [
    "SECFilingAnalyzer",
    "EdgarMonitor",
    "OptionsFlowDetector",
    "ShortInterestTracker",
    "InstitutionalHoldingsTracker",
    "SocialSentimentAnalyzer",
    "MacroSurpriseIndex",
]
