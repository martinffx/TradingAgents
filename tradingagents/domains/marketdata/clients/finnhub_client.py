"""
Finnhub client for financial data access.
"""

import logging
from datetime import date

import finnhub

from ..models import (
    CompanyProfile,
    InsiderSentimentResponse,
    InsiderTransactionsResponse,
    ReportedFinancialsResponse,
)

logger = logging.getLogger(__name__)


class FinnhubClient:
    """
    Finnhub API client for accessing financial data including fundamental data.

    Provides access to:
    - Company news
    - Insider transactions
    - Insider sentiment
    - Real-time quotes
    - Company profiles
    - Financial statements (balance sheet, income statement, cash flow)
    """

    def __init__(self, api_key: str):
        """
        Initialize Finnhub client with official SDK.

        Args:
            api_key: Finnhub API key
        """
        self.client = finnhub.Client(api_key=api_key)

    def get_reported_financials(
        self, symbol: str, frequency: str
    ) -> ReportedFinancialsResponse:
        """
        Get reported financials data from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            Reported financials data from Finnhub API
        """
        try:
            # Finnhub SDK expects frequency as 'quarterly' or 'annual'
            freq = "quarterly" if frequency.lower() in ["quarterly", "q"] else "annual"
            response = self.client.financials_reported(symbol=symbol.upper(), freq=freq)
            return ReportedFinancialsResponse.from_dict(response)
        except Exception as e:
            logger.error(f"Error fetching reported financials for {symbol}: {e}")
            raise

    def get_insider_transactions(
        self, symbol: str, start_date: date, end_date: date
    ) -> InsiderTransactionsResponse:
        """
        Get insider transactions for a company.

        Args:
            symbol: Stock symbol
            start_date: Start date as date object
            end_date: End date as date object

        Returns:
            Insider transaction data
        """
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        try:
            response = self.client.stock_insider_transactions(
                symbol.upper(), _from=start_str, to=end_str
            )
            if isinstance(response, dict):
                return InsiderTransactionsResponse.from_dict(response)
            else:
                # Return empty response if API returns unexpected format
                return InsiderTransactionsResponse(data=[], symbol=symbol.upper())
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            raise

    def get_insider_sentiment(
        self, symbol: str, start_date: date, end_date: date
    ) -> InsiderSentimentResponse:
        """
        Get insider sentiment data for a company.

        Args:
            symbol: Stock symbol
            start_date: Start date as date object
            end_date: End date as date object

        Returns:
            Insider sentiment data
        """
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        try:
            response = self.client.stock_insider_sentiment(
                symbol.upper(), _from=start_str, to=end_str
            )
            if isinstance(response, dict):
                return InsiderSentimentResponse.from_dict(response)
            else:
                # Return empty response if API returns unexpected format
                return InsiderSentimentResponse(data=[], symbol=symbol.upper())
        except Exception as e:
            logger.error(f"Error fetching insider sentiment for {symbol}: {e}")
            raise

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """
        Get company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Company profile data
        """
        try:
            response = self.client.company_profile2(symbol=symbol.upper())
            return CompanyProfile.from_dict(response)
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            raise
