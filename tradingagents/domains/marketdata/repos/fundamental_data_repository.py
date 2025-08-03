"""
Repository for fundamental financial data (balance sheets, income statements, cash flow).
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ReportedFinancialsResponse

# Base repository functionality inline

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


class FundamentalDataRepository:
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

    def store_reported_financials(
        self,
        symbol: str,
        date: date,
        reported_financials: "ReportedFinancialsResponse",
    ) -> bool:
        """
        Store reported financials data from Finnhub API.

        Args:
            symbol: Stock ticker symbol
            date: Date for the financial data
            reported_financials: ReportedFinancialsResponse from Finnhub

        Returns:
            bool: True if storage was successful
        """
        # Create symbol directory
        symbol_dir = self.fundamental_data_dir / symbol
        self._ensure_path_exists(symbol_dir)

        # Create JSON file path
        file_path = symbol_dir / f"{date.isoformat()}_reported_financials.json"

        try:
            # Convert dataclass to dict for JSON serialization
            data = {
                "symbol": symbol,
                "date": date.isoformat(),
                "reported_financials": {
                    "start_date": reported_financials.start_date,
                    "end_date": reported_financials.end_date,
                    "year": reported_financials.year,
                    "quarter": reported_financials.quarter,
                    "access_number": reported_financials.access_number,
                    "data": {
                        "bs": [asdict(item) for item in reported_financials.data.bs],
                        "ic": [asdict(item) for item in reported_financials.data.ic],
                        "cf": [asdict(item) for item in reported_financials.data.cf],
                    },
                },
                "metadata": {
                    "stored_at": date.today().isoformat(),
                    "repository": "fundamental_data_repository",
                    "data_source": "finnhub_reported_financials",
                    "bs_items": len(reported_financials.data.bs),
                    "ic_items": len(reported_financials.data.ic),
                    "cf_items": len(reported_financials.data.cf),
                },
            }

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(
                f"Stored reported financials for {symbol} on {date} "
                f"(BS: {len(reported_financials.data.bs)}, "
                f"IC: {len(reported_financials.data.ic)}, "
                f"CF: {len(reported_financials.data.cf)} items)"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error storing reported financials for {symbol} on {date}: {e}"
            )
            return False

    def _ensure_path_exists(self, path: Path) -> None:
        """Ensure a directory path exists."""
        path.mkdir(parents=True, exist_ok=True)
