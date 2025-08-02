"""
Repository for fundamental financial data (balance sheets, income statements, cash flow).
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .base import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatement:
    """Represents a financial statement with standardized structure matching the service."""

    period: str
    report_date: str
    publish_date: str
    currency: str
    data: dict[str, float]


@dataclass
class FinancialData:
    """Container for all financial statements for a symbol and date."""

    symbol: str
    date: date
    financial_statements: dict[
        str, FinancialStatement
    ]  # "balance_sheet", "income_statement", "cash_flow"


class FundamentalDataRepository(BaseRepository):
    """Repository for accessing cached fundamental financial data as a KV store."""

    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize fundamental data repository.

        Args:
            data_dir: Base directory for fundamental data storage
            **kwargs: Additional configuration
        """
        self.fundamental_data_dir = Path(data_dir) / "fundamental_data"
        self.fundamental_data_dir.mkdir(parents=True, exist_ok=True)

    def get_financial_data(
        self, symbol: str, start_date: date, end_date: date
    ) -> dict[date, FinancialData]:
        """
        Get cached fundamental data for a symbol and date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dict[date, FinancialData]: Financial data keyed by date
        """
        symbol_dir = self.fundamental_data_dir / symbol

        if not symbol_dir.exists():
            logger.warning(f"No data directory found for symbol {symbol}")
            return {}

        financial_data = {}

        # Scan for JSON files in the symbol directory
        for json_file in symbol_dir.glob("*.json"):
            try:
                # Parse date from filename (YYYY-MM-DD.json)
                date_str = json_file.stem
                file_date = date.fromisoformat(date_str)

                # Filter by date range
                if start_date <= file_date <= end_date:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Create FinancialStatement objects from JSON data
                    financial_statements = {}
                    for statement_type, statement_data in data.get(
                        "financial_statements", {}
                    ).items():
                        financial_statements[statement_type] = FinancialStatement(
                            **statement_data
                        )

                    # Create FinancialData container
                    financial_data[file_date] = FinancialData(
                        symbol=symbol,
                        date=file_date,
                        financial_statements=financial_statements,
                    )

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error reading financial data from {json_file}: {e}")
                continue

        logger.info(f"Retrieved {len(financial_data)} financial records for {symbol}")
        return financial_data

    def has_data_for_period(
        self, symbol: str, start_date: str, end_date: str, frequency: str = "quarterly"
    ) -> bool:
        """
        Check if we have sufficient data for the given period.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency (not used in this implementation)

        Returns:
            bool: True if we have any data in the period
        """
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
        financial_data = self.get_financial_data(symbol, start_dt, end_dt)
        return len(financial_data) > 0

    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "quarterly",
        **kwargs,
    ) -> dict[str, any]:
        """
        Get data in the format expected by the service.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency
            **kwargs: Additional parameters

        Returns:
            Dict with financial_statements structure expected by service
        """
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
        financial_data = self.get_financial_data(symbol, start_dt, end_dt)

        if not financial_data:
            return {}

        # Get the most recent data
        latest_date = max(financial_data.keys())
        latest_data = financial_data[latest_date]

        return {
            "financial_statements": {
                statement_type: asdict(statement)
                for statement_type, statement in latest_data.financial_statements.items()
            }
        }

    def store_data(
        self,
        symbol: str,
        cache_data: dict,
        frequency: str = "quarterly",
        overwrite: bool = False,
    ) -> bool:
        """
        Store data in the format expected by the service.

        Args:
            symbol: Stock ticker symbol
            cache_data: Data dictionary with financial_statements
            frequency: Reporting frequency (not used)
            overwrite: Whether to overwrite existing data

        Returns:
            bool: True if successful
        """
        try:
            # Extract financial statements from cache_data
            statements_data = cache_data.get("financial_statements", {})

            if not statements_data:
                logger.warning(
                    f"No financial statements found in cache_data for {symbol}"
                )
                return False

            # Use today's date as the storage date
            storage_date = date.today()

            # Convert to FinancialStatement objects
            financial_statements = {}
            for statement_type, statement_dict in statements_data.items():
                financial_statements[statement_type] = FinancialStatement(
                    **statement_dict
                )

            # Store the statements
            self.store_financial_statements(symbol, storage_date, financial_statements)
            return True

        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
            return False

    def clear_data(
        self, symbol: str, start_date: str, end_date: str, frequency: str = "quarterly"
    ) -> bool:
        """
        Clear data for a symbol and date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Reporting frequency (not used)

        Returns:
            bool: True if successful
        """
        try:
            symbol_dir = self.fundamental_data_dir / symbol

            if not symbol_dir.exists():
                return True  # Nothing to clear

            start_dt = date.fromisoformat(start_date)
            end_dt = date.fromisoformat(end_date)

            # Remove files in the date range
            for json_file in symbol_dir.glob("*.json"):
                try:
                    date_str = json_file.stem
                    file_date = date.fromisoformat(date_str)

                    if start_dt <= file_date <= end_dt:
                        json_file.unlink()
                        logger.debug(f"Removed {json_file}")

                except (ValueError, OSError) as e:
                    logger.warning(f"Error removing {json_file}: {e}")
                    continue

            return True

        except Exception as e:
            logger.error(f"Error clearing data for {symbol}: {e}")
            return False

    def store_financial_statements(
        self,
        symbol: str,
        date: date,
        statements: dict[str, FinancialStatement],
    ) -> tuple[date, dict[str, FinancialStatement]]:
        """
        Store financial statements for a symbol and date.

        Args:
            symbol: Stock ticker symbol
            date: Date of the financial statements
            statements: Dictionary of statements keyed by type ("balance_sheet", "income_statement", "cash_flow")

        Returns:
            Tuple[date, dict[str, FinancialStatement]]: The stored date and statements
        """
        # Create symbol directory
        symbol_dir = self.fundamental_data_dir / symbol
        self._ensure_path_exists(symbol_dir)

        # Create JSON file path
        file_path = symbol_dir / f"{date.isoformat()}.json"

        try:
            # Prepare data for JSON serialization
            statements_data = {}
            for statement_type, statement in statements.items():
                statements_data[statement_type] = asdict(statement)

            data = {
                "symbol": symbol,
                "date": date.isoformat(),
                "financial_statements": statements_data,
                "metadata": {
                    "stored_at": date.today().isoformat(),
                    "repository": "fundamental_data_repository",
                    "statement_count": len(statements),
                },
            }

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(
                f"Stored {len(statements)} financial statements for {symbol} on {date}"
            )
            return (date, statements)

        except Exception as e:
            logger.error(
                f"Error storing financial statements for {symbol} on {date}: {e}"
            )
            raise
