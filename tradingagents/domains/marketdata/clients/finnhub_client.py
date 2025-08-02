"""
Finnhub client for financial data access.
"""

import logging
from datetime import date
from typing import Any

import finnhub

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
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)

    def test_connection(self) -> bool:
        """Test if the Finnhub API connection is working."""
        try:
            # Test with a simple quote request
            response = self.client.quote("AAPL")
            return "c" in response  # 'c' is current price field
        except Exception as e:
            logger.error(f"Finnhub connection test failed: {e}")
            return False

    def get_balance_sheet(
        self, symbol: str, frequency: str, report_date: date
    ) -> dict[str, Any]:
        """
        Get balance sheet data from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            frequency: Reporting frequency ('quarterly' or 'annual')
            report_date: Report date as date object

        Returns:
            Balance sheet data from Finnhub API
        """
        try:
            # Finnhub SDK expects frequency as 'quarterly' or 'annual'
            freq = "quarterly" if frequency.lower() in ["quarterly", "q"] else "annual"
            response = self.client.financials_reported(symbol=symbol.upper(), freq=freq)
            return response if isinstance(response, dict) else {"data": []}
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return {"data": []}

    def get_income_statement(
        self, symbol: str, frequency: str, report_date: date
    ) -> dict[str, Any]:
        """
        Get income statement data from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            frequency: Reporting frequency ('quarterly' or 'annual')
            report_date: Report date as date object

        Returns:
            Income statement data from Finnhub API
        """
        try:
            freq = "quarterly" if frequency.lower() in ["quarterly", "q"] else "annual"
            response = self.client.financials_reported(symbol=symbol.upper(), freq=freq)
            return response if isinstance(response, dict) else {"data": []}
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return {"data": []}

    def get_cash_flow(
        self, symbol: str, frequency: str, report_date: date
    ) -> dict[str, Any]:
        """
        Get cash flow statement data from Finnhub.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            frequency: Reporting frequency ('quarterly' or 'annual')
            report_date: Report date as date object

        Returns:
            Cash flow statement data from Finnhub API
        """
        try:
            freq = "quarterly" if frequency.lower() in ["quarterly", "q"] else "annual"
            response = self.client.financials_reported(symbol=symbol.upper(), freq=freq)
            return response if isinstance(response, dict) else {"data": []}
        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return {"data": []}

    def get_company_news(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """
        Get company news for a specific symbol and date range.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date as date object
            end_date: End date as date object

        Returns:
            List of news articles
        """
        # Convert date objects to strings for API
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        try:
            response = self.client.company_news(
                symbol.upper(), _from=start_str, to=end_str
            )
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def get_insider_transactions(
        self, symbol: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
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
            return response if isinstance(response, dict) else {"data": []}
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            return {"data": []}

    def get_insider_sentiment(
        self, symbol: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
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
            return response if isinstance(response, dict) else {"data": []}
        except Exception as e:
            logger.error(f"Error fetching insider sentiment for {symbol}: {e}")
            return {"data": []}

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data with current price, change, etc.
        """
        try:
            response = self.client.quote(symbol.upper())
            return response if isinstance(response, dict) else {}
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}

    def get_company_profile(self, symbol: str) -> dict[str, Any]:
        """
        Get company profile information.

        Args:
            symbol: Stock symbol

        Returns:
            Company profile data
        """
        try:
            response = self.client.company_profile2(symbol=symbol.upper())
            return response if isinstance(response, dict) else {}
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {}

    def get_client_info(self) -> dict[str, Any]:
        """
        Get information about this client.

        Returns:
            Dict[str, Any]: Client metadata
        """
        return {
            "client_type": self.__class__.__name__,
            "api_key_set": bool(self.api_key),
            "sdk_version": getattr(finnhub, "__version__", "unknown"),
        }
