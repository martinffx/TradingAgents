"""
Fundamental Data Service for aggregating and analyzing financial statement data.
"""

import logging
from datetime import date, datetime
from typing import Any

from tradingagents.clients import FinnhubClient
from tradingagents.models.context import (
    DataQuality,
    FinancialStatement,
    FundamentalContext,
)
from tradingagents.repositories.fundamental_repository import FundamentalDataRepository
from tradingagents.services.base import BaseService

logger = logging.getLogger(__name__)


class FundamentalDataService(BaseService):
    """Service for fundamental financial data aggregation and analysis."""

    def __init__(
        self,
        finnhub_client: FinnhubClient,
        repository: FundamentalDataRepository,
        data_dir: str = "data",
        **kwargs,
    ):
        """Initialize Fundamental Data Service.

        Args:
            finnhub_client: Client for Finnhub/financial API access
            repository: Repository for cached fundamental data
            data_dir: Directory for data storage
        """
        super().__init__(online_mode=True, data_dir=data_dir, **kwargs)
        self.finnhub_client = finnhub_client
        self.repository = repository

    def get_fundamental_context(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "quarterly",
        force_refresh: bool = False,
        **kwargs,
    ) -> FundamentalContext:
        """Get fundamental analysis context for a company.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency ('quarterly' or 'annual')
            force_refresh: If True, skip local data and fetch fresh from APIs

        Returns:
            FundamentalContext with financial statements and key ratios
        """
        # Validate date strings first
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

        # Check date order
        if end_dt < start_dt:
            raise ValueError(f"End date {end_date} is before start date {start_date}")

        balance_sheet = None
        income_statement = None
        cash_flow = None
        error_info = {}
        errors = []
        data_source = "unknown"

        try:
            # Local-first data strategy with force refresh option
            if force_refresh:
                # Skip local data, fetch fresh from APIs
                balance_sheet, income_statement, cash_flow, data_source = (
                    self._fetch_and_cache_fresh_fundamental_data(
                        symbol, start_date, end_date, frequency
                    )
                )
            else:
                # Check local data first, fetch missing if needed
                balance_sheet, income_statement, cash_flow, data_source = (
                    self._get_fundamental_data_local_first(
                        symbol, start_date, end_date, frequency
                    )
                )

        except Exception as e:
            logger.error(f"Error fetching fundamental data: {e}")
            errors.append(str(e))

        # Add error info if there were any errors
        if errors:
            error_info = {"error": "; ".join(errors)}

        # Calculate key financial ratios
        key_ratios = self._calculate_key_ratios(
            balance_sheet, income_statement, cash_flow
        )

        # Determine data quality based on data source
        data_quality = self._determine_data_quality(
            data_source=data_source,
            statement_count=sum(
                [
                    balance_sheet is not None,
                    income_statement is not None,
                    cash_flow is not None,
                ]
            ),
            has_errors=bool(errors),
        )

        # Handle partial data scenarios gracefully
        context = self._handle_partial_statements(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow=cash_flow,
            key_ratios=key_ratios,
            data_quality=data_quality,
            data_source=data_source,
            force_refresh=force_refresh,
            error_info=error_info,
        )

        return context

    def get_context(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "quarterly",
        **kwargs,
    ) -> FundamentalContext:
        """Alias for get_fundamental_context for consistency with other services."""
        return self.get_fundamental_context(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            **kwargs,
        )

    def _get_balance_sheet(
        self, symbol: str, frequency: str, report_date: date
    ) -> FinancialStatement | None:
        """Get balance sheet data from client."""
        try:
            data = self.finnhub_client.get_balance_sheet(symbol, frequency, report_date)
            return self._convert_to_financial_statement(data)
        except Exception as e:
            logger.warning(f"Failed to get balance sheet for {symbol}: {e}")
            return None

    def _get_income_statement(
        self, symbol: str, frequency: str, report_date: date
    ) -> FinancialStatement | None:
        """Get income statement data from client."""
        try:
            data = self.finnhub_client.get_income_statement(
                symbol, frequency, report_date
            )
            return self._convert_to_financial_statement(data)
        except Exception as e:
            logger.warning(f"Failed to get income statement for {symbol}: {e}")
            return None

    def _get_cash_flow(
        self, symbol: str, frequency: str, report_date: date
    ) -> FinancialStatement | None:
        """Get cash flow statement data from client."""
        try:
            data = self.finnhub_client.get_cash_flow(symbol, frequency, report_date)
            return self._convert_to_financial_statement(data)
        except Exception as e:
            logger.warning(f"Failed to get cash flow for {symbol}: {e}")
            return None

    def _convert_to_financial_statement(
        self, data: dict[str, Any]
    ) -> FinancialStatement | None:
        """Convert raw financial data to FinancialStatement object."""
        if not data or "data" not in data or not data["data"]:
            return None

        try:
            return FinancialStatement(
                period=data.get("period", "Unknown"),
                report_date=data.get("report_date", ""),
                publish_date=data.get("publish_date", ""),
                currency=data.get("currency", "USD"),
                data=data["data"],
            )
        except Exception as e:
            logger.warning(f"Failed to convert financial statement: {e}")
            return None

    def _parse_cached_statements(self, cached_data: dict[str, Any]) -> tuple:
        """Parse cached repository data into financial statements."""
        balance_sheet = None
        income_statement = None
        cash_flow = None

        if cached_data and "financial_statements" in cached_data:
            statements = cached_data["financial_statements"]

            if "balance_sheet" in statements:
                balance_sheet = FinancialStatement(**statements["balance_sheet"])
            if "income_statement" in statements:
                income_statement = FinancialStatement(**statements["income_statement"])
            if "cash_flow" in statements:
                cash_flow = FinancialStatement(**statements["cash_flow"])

        return balance_sheet, income_statement, cash_flow

    def _get_fundamental_data_local_first(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> tuple[
        FinancialStatement | None,
        FinancialStatement | None,
        FinancialStatement | None,
        str,
    ]:
        """Get fundamental data using local-first strategy: check local data first, fetch missing if needed."""
        try:
            # Check if we have sufficient local data
            if self.repository.has_data_for_period(
                symbol, start_date, end_date, frequency=frequency
            ):
                logger.info(
                    f"Using local fundamental data for {symbol} ({start_date} to {end_date})"
                )
                cached_data = self.repository.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                )
                balance_sheet, income_statement, cash_flow = (
                    self._parse_cached_statements(cached_data)
                )
                return balance_sheet, income_statement, cash_flow, "local_cache"

            # We don't have sufficient local data - need to fetch from APIs
            logger.info(
                f"Local data insufficient, fetching from APIs for {symbol} ({start_date} to {end_date})"
            )
            balance_sheet, income_statement, cash_flow, _ = (
                self._fetch_fresh_fundamental_data(
                    symbol, start_date, end_date, frequency
                )
            )

            # Cache the fresh data
            if any([balance_sheet, income_statement, cash_flow]):
                try:
                    cache_data = {
                        "symbol": symbol,
                        "frequency": frequency,
                        "financial_statements": {},
                        "metadata": {"cached_at": datetime.utcnow().isoformat()},
                    }

                    if balance_sheet:
                        cache_data["financial_statements"]["balance_sheet"] = (
                            balance_sheet.model_dump()
                        )
                    if income_statement:
                        cache_data["financial_statements"]["income_statement"] = (
                            income_statement.model_dump()
                        )
                    if cash_flow:
                        cache_data["financial_statements"]["cash_flow"] = (
                            cash_flow.model_dump()
                        )

                    self.repository.store_data(symbol, cache_data, frequency=frequency)
                    logger.debug(f"Cached fresh fundamental data for {symbol}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cache fundamental data for {symbol}: {e}"
                    )

            return balance_sheet, income_statement, cash_flow, "live_api"

        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return None, None, None, "error"

    def _fetch_and_cache_fresh_fundamental_data(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> tuple[
        FinancialStatement | None,
        FinancialStatement | None,
        FinancialStatement | None,
        str,
    ]:
        """Force fetch fresh fundamental data from APIs and cache it, bypassing local data."""
        try:
            logger.info(
                f"Force refreshing fundamental data from APIs for {symbol} ({start_date} to {end_date})"
            )

            # Clear existing data
            try:
                self.repository.clear_data(
                    symbol, start_date, end_date, frequency=frequency
                )
                logger.debug(f"Cleared existing fundamental data for {symbol}")
            except Exception as e:
                logger.warning(
                    f"Failed to clear existing fundamental data for {symbol}: {e}"
                )

            # Fetch fresh data
            balance_sheet, income_statement, cash_flow, _ = (
                self._fetch_fresh_fundamental_data(
                    symbol, start_date, end_date, frequency
                )
            )

            # Cache the fresh data
            if any([balance_sheet, income_statement, cash_flow]):
                try:
                    cache_data = {
                        "symbol": symbol,
                        "frequency": frequency,
                        "financial_statements": {},
                        "metadata": {"refreshed_at": datetime.utcnow().isoformat()},
                    }

                    if balance_sheet:
                        cache_data["financial_statements"]["balance_sheet"] = (
                            balance_sheet.model_dump()
                        )
                    if income_statement:
                        cache_data["financial_statements"]["income_statement"] = (
                            income_statement.model_dump()
                        )
                    if cash_flow:
                        cache_data["financial_statements"]["cash_flow"] = (
                            cash_flow.model_dump()
                        )

                    self.repository.store_data(
                        symbol, cache_data, frequency=frequency, overwrite=True
                    )
                    logger.debug(f"Cached refreshed fundamental data for {symbol}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cache refreshed fundamental data for {symbol}: {e}"
                    )

            return balance_sheet, income_statement, cash_flow, "live_api_refresh"

        except Exception as e:
            logger.error(f"Error force refreshing fundamental data for {symbol}: {e}")
            return None, None, None, "refresh_error"

    def _fetch_fresh_fundamental_data(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> tuple[
        FinancialStatement | None,
        FinancialStatement | None,
        FinancialStatement | None,
        str,
    ]:
        """Fetch fresh fundamental data from APIs."""
        balance_sheet = None
        income_statement = None
        cash_flow = None

        if self.is_online() and self.finnhub_client:
            # Parse end_date string to date object for client calls
            try:
                end_date_obj = date.fromisoformat(end_date)
            except ValueError as e:
                logger.error(f"Invalid end_date format '{end_date}': {e}")
                return balance_sheet, income_statement, cash_flow, "date_error"

            # Get financial statements from Finnhub client
            balance_sheet = self._get_balance_sheet(symbol, frequency, end_date_obj)
            income_statement = self._get_income_statement(
                symbol, frequency, end_date_obj
            )
            cash_flow = self._get_cash_flow(symbol, frequency, end_date_obj)

        return balance_sheet, income_statement, cash_flow, "live_api"

    def _calculate_key_ratios(
        self,
        balance_sheet: FinancialStatement | None,
        income_statement: FinancialStatement | None,
        cash_flow: FinancialStatement | None,
    ) -> dict[str, float]:
        """Calculate key financial ratios from financial statements."""
        ratios = {}

        try:
            # Extract data from statements
            bs_data = balance_sheet.data if balance_sheet else {}
            is_data = income_statement.data if income_statement else {}

            # Liquidity Ratios
            if (
                "Total Current Assets" in bs_data
                and "Total Current Liabilities" in bs_data
            ):
                current_liabilities = bs_data["Total Current Liabilities"]
                if current_liabilities > 0:
                    ratios["current_ratio"] = (
                        bs_data["Total Current Assets"] / current_liabilities
                    )

            # Quick ratio (more conservative)
            if all(
                k in bs_data
                for k in [
                    "Cash and Cash Equivalents",
                    "Short-term Investments",
                    "Accounts Receivable",
                    "Total Current Liabilities",
                ]
            ):
                quick_assets = (
                    bs_data["Cash and Cash Equivalents"]
                    + bs_data.get("Short-term Investments", 0)
                    + bs_data["Accounts Receivable"]
                )
                current_liabilities = bs_data["Total Current Liabilities"]
                if current_liabilities > 0:
                    ratios["quick_ratio"] = quick_assets / current_liabilities

            # Cash ratio
            if (
                "Cash and Cash Equivalents" in bs_data
                and "Total Current Liabilities" in bs_data
            ):
                current_liabilities = bs_data["Total Current Liabilities"]
                if current_liabilities > 0:
                    cash_and_equivalents = bs_data[
                        "Cash and Cash Equivalents"
                    ] + bs_data.get("Short-term Investments", 0)
                    ratios["cash_ratio"] = cash_and_equivalents / current_liabilities

            # Leverage Ratios
            if "Long-term Debt" in bs_data and "Total Shareholders Equity" in bs_data:
                equity = bs_data["Total Shareholders Equity"]
                if equity > 0:
                    ratios["debt_to_equity"] = bs_data["Long-term Debt"] / equity

            if "Long-term Debt" in bs_data and "Total Assets" in bs_data:
                assets = bs_data["Total Assets"]
                if assets > 0:
                    ratios["debt_to_assets"] = bs_data["Long-term Debt"] / assets

            if "Total Assets" in bs_data and "Total Shareholders Equity" in bs_data:
                equity = bs_data["Total Shareholders Equity"]
                if equity > 0:
                    ratios["equity_multiplier"] = bs_data["Total Assets"] / equity

            # Profitability Ratios
            if "Total Revenue" in is_data and "Cost of Revenue" in is_data:
                revenue = is_data["Total Revenue"]
                if revenue > 0:
                    ratios["gross_margin"] = (
                        revenue - is_data["Cost of Revenue"]
                    ) / revenue

            if "Operating Income" in is_data and "Total Revenue" in is_data:
                revenue = is_data["Total Revenue"]
                if revenue > 0:
                    ratios["operating_margin"] = is_data["Operating Income"] / revenue

            if "Net Income" in is_data and "Total Revenue" in is_data:
                revenue = is_data["Total Revenue"]
                if revenue > 0:
                    ratios["net_margin"] = is_data["Net Income"] / revenue

            # Return on Equity (ROE)
            if "Net Income" in is_data and "Total Shareholders Equity" in bs_data:
                equity = bs_data["Total Shareholders Equity"]
                if equity > 0:
                    ratios["roe"] = is_data["Net Income"] / equity

            # Return on Assets (ROA)
            if "Net Income" in is_data and "Total Assets" in bs_data:
                assets = bs_data["Total Assets"]
                if assets > 0:
                    ratios["roa"] = is_data["Net Income"] / assets

            # Efficiency Ratios
            if "Total Revenue" in is_data and "Total Assets" in bs_data:
                assets = bs_data["Total Assets"]
                if assets > 0:
                    ratios["asset_turnover"] = is_data["Total Revenue"] / assets

            # Inventory turnover
            if "Cost of Revenue" in is_data and "Inventory" in bs_data:
                inventory = bs_data["Inventory"]
                if inventory > 0:
                    ratios["inventory_turnover"] = (
                        is_data["Cost of Revenue"] / inventory
                    )

            # Receivables turnover
            if "Total Revenue" in is_data and "Accounts Receivable" in bs_data:
                receivables = bs_data["Accounts Receivable"]
                if receivables > 0:
                    ratios["receivables_turnover"] = (
                        is_data["Total Revenue"] / receivables
                    )

        except Exception as e:
            logger.warning(f"Error calculating financial ratios: {e}")

        return ratios

    def _handle_partial_statements(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str,
        balance_sheet: FinancialStatement | None,
        income_statement: FinancialStatement | None,
        cash_flow: FinancialStatement | None,
        key_ratios: dict[str, float],
        data_quality: DataQuality,
        data_source: str,
        force_refresh: bool,
        error_info: dict[str, Any],
    ) -> FundamentalContext:
        """Create context even if some statements are missing.

        - If all statements fail: Raise exception
        - If some statements succeed: Return partial context
        - Mark missing statements in metadata
        """
        statement_count = sum(
            [
                balance_sheet is not None,
                income_statement is not None,
                cash_flow is not None,
            ]
        )

        # If all statements failed, raise exception
        if statement_count == 0 and data_source not in ["local_cache"]:
            error_msg = f"Failed to fetch any financial statements for {symbol}"
            if error_info:
                error_msg += f": {error_info.get('error', 'Unknown error')}"
            raise ValueError(error_msg)

        # Create metadata with partial data information
        metadata = {
            "data_quality": data_quality,
            "service": "fundamental_data",
            "online_mode": self.is_online(),
            "frequency": frequency,
            "data_source": data_source,
            "force_refresh": force_refresh,
            "has_balance_sheet": balance_sheet is not None,
            "has_income_statement": income_statement is not None,
            "has_cash_flow": cash_flow is not None,
            "partial_data": statement_count < 3,
            "statement_count": statement_count,
            **error_info,
        }

        return FundamentalContext(
            symbol=symbol,
            period={"start": start_date, "end": end_date},
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow=cash_flow,
            key_ratios=key_ratios,
            metadata=metadata,
        )

    def detect_fundamental_gaps(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> list[str]:
        """
        Returns list of report dates that need fetching.

        Example: If requesting quarterly from 2024-01-01 to 2024-12-31
        and cache has Q1 and Q3, returns ["2024-06-30", "2024-09-30", "2024-12-31"]

        For quarterly: Check for Q1 (Mar 31), Q2 (Jun 30), Q3 (Sep 30), Q4 (Dec 31)
        For annual: Check for fiscal year ends
        """
        try:
            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)
        except ValueError:
            logger.error(
                f"Invalid date format in gap detection: {start_date}, {end_date}"
            )
            return []

        # Get existing data from repository
        try:
            cached_data = self.repository.get_data(
                symbol, start_date, end_date, frequency
            )
            existing_dates = set()

            if cached_data and "financial_statements" in cached_data:
                for statement_type in [
                    "balance_sheet",
                    "income_statement",
                    "cash_flow",
                ]:
                    if statement_type in cached_data["financial_statements"]:
                        stmt = cached_data["financial_statements"][statement_type]
                        if "report_date" in stmt:
                            existing_dates.add(stmt["report_date"])
        except Exception as e:
            logger.warning(f"Error checking cached data for gap detection: {e}")
            existing_dates = set()

        # Calculate expected report dates based on frequency
        expected_dates = []
        current_year = start_dt.year
        end_year = end_dt.year

        if frequency == "quarterly":
            # Standard quarterly dates: Mar 31, Jun 30, Sep 30, Dec 31
            quarter_dates = [
                (3, 31),  # Q1
                (6, 30),  # Q2
                (9, 30),  # Q3
                (12, 31),  # Q4
            ]

            for year in range(current_year, end_year + 1):
                for month, day in quarter_dates:
                    report_date = date(year, month, day)
                    if start_dt <= report_date <= end_dt:
                        expected_dates.append(report_date.isoformat())

        elif frequency == "annual":
            # Standard fiscal year end: Dec 31
            for year in range(current_year, end_year + 1):
                report_date = date(year, 12, 31)
                if start_dt <= report_date <= end_dt:
                    expected_dates.append(report_date.isoformat())

        # Return dates that are expected but not in cache
        missing_dates = [d for d in expected_dates if d not in existing_dates]

        if missing_dates:
            logger.info(
                f"Gap detection for {symbol}: missing {len(missing_dates)} report periods"
            )

        return missing_dates

    def _determine_data_quality(
        self, data_source: str, statement_count: int, has_errors: bool = False
    ) -> DataQuality:
        """Determine data quality based on source, statement count, and errors."""
        if has_errors or statement_count == 0:
            return DataQuality.LOW

        if data_source in ["local_cache", "error", "refresh_error"]:
            return DataQuality.LOW
        elif data_source in ["live_api", "live_api_refresh"]:
            if statement_count == 3:
                return DataQuality.HIGH  # All three statements available
            elif statement_count == 2:
                return DataQuality.MEDIUM  # Two statements available
            else:
                return DataQuality.LOW  # One or no statements
        else:
            return DataQuality.MEDIUM
