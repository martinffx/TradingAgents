"""
Fundamental Data Service for aggregating and analyzing financial statement data.
"""

import logging
from datetime import date

from tradingagents.config import TradingAgentsConfig

from .clients.finnhub_client import FinnhubClient
from .models import (
    BalanceSheetContext,
    CashFlowContext,
    DataQuality,
    FundamentalContext,
    IncomeStatementContext,
)
from .repos.fundamental_data_repository import FundamentalDataRepository

logger = logging.getLogger(__name__)


class FundamentalDataService:
    """Service for fundamental financial data aggregation and analysis."""

    def __init__(
        self,
        finnhub_client: FinnhubClient,
        repository: FundamentalDataRepository,
    ):
        """Initialize Fundamental Data Service.

        Args:
            finnhub_client: Client for FinnHub financial API access
            repository: Repository for cached fundamental data
            online_mode: Whether to fetch live data or use cached data
        """
        self.finnhub_client = finnhub_client
        self.repository = repository

    @staticmethod
    def build(_config: TradingAgentsConfig):
        client = FinnhubClient("")
        repo = FundamentalDataRepository("")
        return FundamentalDataService(client, repo)

    def get_fundamental_context(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> FundamentalContext:
        """Get fundamental analysis context for a company.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            FundamentalContext with financial statements and key ratios
        """
        # TODO: implement
        return FundamentalContext(
            symbol=symbol, start_date=start_date, end_date=end_date
        )

    def get_balance_sheet_context(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> BalanceSheetContext:
        """Get balance sheet context for a company.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            BalanceSheetContext with balance sheet data and ratios
        """
        # TODO: implement

        # Return empty context if no data
        return BalanceSheetContext(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            balance_sheet_data=[],
            key_ratios=[],
            data_quality=DataQuality.LOW,
            source="none",
            metadata={"error": "No balance sheet data available"},
        )

    def get_income_statement_context(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> IncomeStatementContext:
        """Get income statement context for a company.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            IncomeStatementContext with income statement data and ratios
        """
        # TODO: implement

        # Return empty context if no data
        return IncomeStatementContext(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            income_statement_data=[],
            key_ratios=[],
            data_quality=DataQuality.LOW,
            source="none",
            metadata={"error": "No income statement data available"},
        )

    def get_cashflow_context(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> CashFlowContext:
        """Get cash flow context for a company.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            CashFlowContext with cash flow data and ratios
        """
        # TODOL implement

        # Return empty context if no data
        return CashFlowContext(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            cash_flow_data=[],
            key_ratios=[],
            data_quality=DataQuality.LOW,
            source="none",
            metadata={"error": "No cash flow data available"},
        )

    def update_fundamental_data(
        self,
        symbol: str,
        date: date,
        frequency: str = "quarterly",
    ) -> bool:
        """Update fundamental data by fetching from FinnHub and storing in repository.

        Args:
            symbol: Stock ticker symbol
            date: Date for the financial data
            frequency: Reporting frequency ('quarterly' or 'annual')

        Returns:
            bool: True if update was successful
        """
        try:
            # Fetch reported financials data from FinnHub using the unified method
            reported_financials = self.finnhub_client.get_reported_financials(
                symbol, frequency
            )

            # Store the reported financials data in repository
            return self.repository.store_reported_financials(
                symbol=symbol, date=date, reported_financials=reported_financials
            )

        except Exception as e:
            logger.error(f"Error updating fundamental data for {symbol}: {e}")
            return False
