"""
Client classes for live data access in TradingAgents.
"""

# Re-export existing clients from dataflows
from tradingagents.dataflows.reddit_utils import RedditClient

from .base import BaseClient
from .finnhub_client import FinnhubClient
from .google_news_client import GoogleNewsClient
from .yfinance_client import YFinanceClient

__all__ = [
    "BaseClient",
    "YFinanceClient",
    "GoogleNewsClient",
    "FinnhubClient",
    "RedditClient",
]
