"""
Repository for historical market data (CSV files).
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

# from .base import BaseRepository  # Not found, removing import

logger = logging.getLogger(__name__)


class MarketDataRepository:
    """Repository for accessing historical market data from CSV files."""

    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize market data repository.

        Args:
            data_dir: Base directory for market data storage
            **kwargs: Additional configuration
        """
        _ = kwargs  # Acknowledge unused parameter
        self.market_data_dir = Path(data_dir) / "market_data"
        self.market_data_dir.mkdir(parents=True, exist_ok=True)

    def get_market_data_df(
        self, symbol: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Get historical market data as DataFrame for a symbol and date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Market data filtered by date range
        """
        csv_path = self.market_data_dir / f"{symbol}.csv"

        if not csv_path.exists():
            logger.warning(f"No CSV file found for symbol {symbol} at {csv_path}")
            return pd.DataFrame()

        try:
            # Read CSV file
            df = pd.read_csv(csv_path)

            if df.empty:
                logger.warning(f"Empty CSV file for symbol {symbol}")
                return pd.DataFrame()

            # Convert Date column to date objects for filtering
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.date

                # Filter by date range
                mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
                filtered_df: pd.DataFrame = df.loc[mask].copy()

                logger.info(
                    f"Retrieved {len(filtered_df)} records for {symbol} from {start_date} to {end_date}"
                )
                return filtered_df
            else:
                logger.warning(f"No 'Date' column found in {csv_path}")
                return df

        except Exception as e:
            logger.error(f"Error reading CSV file {csv_path}: {e}")
            return pd.DataFrame()

    def store_marketdata(self, symbol: str, marketdata: pd.DataFrame) -> pd.DataFrame:
        """
        Store market data DataFrame to CSV file, appending or replacing existing data.

        Args:
            symbol: Stock ticker symbol
            marketdata: DataFrame with market data to store

        Returns:
            pd.DataFrame: The combined DataFrame that was stored
        """
        if marketdata.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}")
            return marketdata

        csv_path = self.market_data_dir / f"{symbol}.csv"

        try:
            if csv_path.exists():
                # Load existing data
                existing_df = pd.read_csv(csv_path)

                if not existing_df.empty and "Date" in existing_df.columns:
                    # Ensure Date columns are in same format for comparison
                    existing_df["Date"] = pd.to_datetime(
                        existing_df["Date"]
                    ).dt.strftime("%Y-%m-%d")
                    marketdata_copy = marketdata.copy()
                    marketdata_copy["Date"] = pd.to_datetime(
                        marketdata_copy["Date"]
                    ).dt.strftime("%Y-%m-%d")

                    # Combine and remove duplicates by Date, keeping newer data
                    combined_df = pd.concat(
                        [existing_df, marketdata_copy], ignore_index=True
                    )
                    combined_df = combined_df.drop_duplicates(
                        subset=["Date"], keep="last"
                    )
                    combined_df = combined_df.sort_values("Date").reset_index(drop=True)
                else:
                    # Existing file is empty or malformed, use new data
                    combined_df = marketdata.copy()
            else:
                # No existing file, use new data
                combined_df = marketdata.copy()

            # Save to CSV
            combined_df.to_csv(csv_path, index=False)
            logger.info(
                f"Stored {len(marketdata)} records for {symbol}, total records: {len(combined_df)}"
            )

            return combined_df

        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            raise
