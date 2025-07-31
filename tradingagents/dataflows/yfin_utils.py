# gets data/stats

from functools import lru_cache
from typing import Annotated, cast

import pandas as pd
import yfinance as yf
from pandas import DataFrame, Series

from .utils import SavePathType


# Module-level cache to avoid memory leaks with instance methods
@lru_cache(maxsize=100)
def _get_cached_ticker(symbol: str) -> yf.Ticker:
    """Get a cached yfinance Ticker instance."""
    return yf.Ticker(symbol)


class YFinanceUtils:
    """Clean YFinance utilities with ticker caching for better performance."""

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a cached yfinance Ticker instance."""
        return _get_cached_ticker(symbol)

    def get_stock_data(
        self,
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str, "start date for retrieving stock price data, YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "end date for retrieving stock price data, YYYY-mm-dd"
        ],
        save_path: SavePathType | None = None,
    ) -> DataFrame:
        """Retrieve stock price data for designated ticker symbol."""
        ticker = self._get_ticker(symbol)

        # Add one day to the end_date so that the data range is inclusive
        end_date_adjusted = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date_str = end_date_adjusted.strftime("%Y-%m-%d")

        stock_data = ticker.history(start=start_date, end=end_date_str)
        return stock_data

    def get_stock_info(self, symbol: Annotated[str, "ticker symbol"]) -> dict:
        """Fetches and returns latest stock information."""
        ticker = self._get_ticker(symbol)
        return ticker.info

    def get_company_info(
        self,
        symbol: Annotated[str, "ticker symbol"],
        save_path: str | None = None,
    ) -> DataFrame:
        """Fetches and returns company information as a DataFrame."""
        ticker = self._get_ticker(symbol)
        info = ticker.info

        company_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }

        company_info_df = DataFrame([company_info])

        if save_path:
            company_info_df.to_csv(save_path)
            print(f"Company info for {symbol} saved to {save_path}")

        return company_info_df

    def get_stock_dividends(
        self,
        symbol: Annotated[str, "ticker symbol"],
        save_path: str | None = None,
    ) -> Series:
        """Fetches and returns the latest dividends data as a DataFrame."""
        ticker = self._get_ticker(symbol)
        dividends = ticker.dividends

        if save_path:
            dividends.to_csv(save_path)
            print(f"Dividends for {symbol} saved to {save_path}")

        return dividends

    def get_income_stmt(self, symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest income statement of the company as a DataFrame."""
        ticker = self._get_ticker(symbol)
        return ticker.financials

    def get_balance_sheet(self, symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest balance sheet of the company as a DataFrame."""
        ticker = self._get_ticker(symbol)
        return ticker.balance_sheet

    def get_cash_flow(self, symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """Fetches and returns the latest cash flow statement of the company as a DataFrame."""
        ticker = self._get_ticker(symbol)
        return ticker.cashflow

    def get_analyst_recommendations(
        self, symbol: Annotated[str, "ticker symbol"]
    ) -> tuple[str | None, int]:
        """Fetches the latest analyst recommendations and returns the most common recommendation and its count."""
        ticker = self._get_ticker(symbol)
        recommendations = cast("DataFrame", ticker.recommendations)

        if recommendations is None or recommendations.empty:
            return None, 0  # No recommendations available

        # Get the most recent recommendation row (excluding 'period' column if it exists)
        try:
            row_0 = recommendations.iloc[0, 1:]  # Skip first column (likely 'period')

            # Find the maximum voting result
            max_votes = row_0.max()
            majority_voting_result = row_0[row_0 == max_votes].index.tolist()

            return majority_voting_result[0], int(max_votes)
        except (IndexError, KeyError):
            return None, 0

    def clear_cache(self) -> None:
        """Clear the ticker cache. Useful for testing or memory management."""
        _get_cached_ticker.cache_clear()

    def cache_info(self) -> dict:
        """Get information about the ticker cache."""
        info = _get_cached_ticker.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
        }
