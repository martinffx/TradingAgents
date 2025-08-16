"""
Yahoo Finance client for live market data.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Client for Yahoo Finance API using yfinance library."""

    def __init__(self, **kwargs):
        """
        Initialize Yahoo Finance client.

        Args:
            **kwargs: Configuration options
        """
        super().__init__(**kwargs)
        self.session = None

    def test_connection(self) -> bool:
        """Test Yahoo Finance connection by fetching a known ticker."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            return bool(info and "symbol" in info)
        except Exception as e:
            logger.error(f"Yahoo Finance connection test failed: {e}")
            return False

    def get_data(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Price data with metadata
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            # Add one day to end_date to make it inclusive
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_adjusted = end_date_obj + timedelta(days=1)
            end_date_str = end_date_adjusted.strftime("%Y-%m-%d")

            data = ticker.history(start=start_date, end=end_date_str)

            if data.empty:
                logger.warning(
                    f"No data found for {symbol} between {start_date} and {end_date}"
                )
                return {
                    "symbol": symbol,
                    "data": [],
                    "metadata": {
                        "source": "yahoo_finance",
                        "empty": True,
                        "reason": "no_data_available",
                    },
                }

            # Remove timezone info and format data
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            # Reset index to make Date a column
            data = data.reset_index()
            data["Date"] = data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Round numerical values
            numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].round(2)

            # Convert to list of dictionaries
            records = data.to_dict("records")

            return {
                "symbol": symbol,
                "period": {"start": start_date, "end": end_date},
                "data": records,
                "metadata": {
                    "source": "yahoo_finance",
                    "record_count": len(records),
                    "columns": list(data.columns),
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            raise

    def get_company_info(self, symbol: str) -> dict[str, Any]:
        """
        Get company information for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict[str, Any]: Company information
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            return {
                "symbol": symbol,
                "info": info,
                "metadata": {
                    "source": "yahoo_finance",
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {
                "symbol": symbol,
                "info": {},
                "metadata": {
                    "source": "yahoo_finance",
                    "error": str(e),
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            }

    def get_financials(
        self, symbol: str, statement_type: str = "income"
    ) -> dict[str, Any]:
        """
        Get financial statements for a symbol.

        Args:
            symbol: Stock ticker symbol
            statement_type: Type of statement ("income", "balance", "cashflow")

        Returns:
            Dict[str, Any]: Financial statement data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            if statement_type == "income":
                annual = ticker.financials
                quarterly = ticker.quarterly_financials
            elif statement_type == "balance":
                annual = ticker.balance_sheet
                quarterly = ticker.quarterly_balance_sheet
            elif statement_type == "cashflow":
                annual = ticker.cashflow
                quarterly = ticker.quarterly_cashflow
            else:
                raise ValueError(f"Unknown statement type: {statement_type}")

            result = {
                "symbol": symbol,
                "statement_type": statement_type,
                "annual": {},
                "quarterly": {},
                "metadata": {
                    "source": "yahoo_finance",
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            }

            # Process annual data
            if not annual.empty:
                annual_data = annual.copy()
                if isinstance(annual_data.columns, pd.DatetimeIndex):
                    annual_data.columns = annual_data.columns.strftime("%Y-%m-%d")
                result["annual"] = annual_data.to_dict()

            # Process quarterly data
            if not quarterly.empty:
                quarterly_data = quarterly.copy()
                if isinstance(quarterly_data.columns, pd.DatetimeIndex):
                    quarterly_data.columns = quarterly_data.columns.strftime("%Y-%m-%d")
                result["quarterly"] = quarterly_data.to_dict()

            return result

        except Exception as e:
            logger.error(
                f"Error fetching {statement_type} financials for {symbol}: {e}"
            )
            return {
                "symbol": symbol,
                "statement_type": statement_type,
                "annual": {},
                "quarterly": {},
                "metadata": {
                    "source": "yahoo_finance",
                    "error": str(e),
                    "retrieved_at": datetime.now(timezone.utc)
                    .replace(tzinfo=None)
                    .isoformat(),
                },
            }

    def get_available_symbols(self) -> list[str]:
        """
        Yahoo Finance doesn't provide a direct way to list all symbols.
        Return common major symbols as examples.
        """
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "AMD",
            "JPM",
            "JNJ",
            "V",
            "WMT",
            "PG",
            "UNH",
            "HD",
            "MA",
            "BAC",
            "DIS",
        ]
