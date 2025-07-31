"""
Base client abstraction for live data access.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class BaseClient(ABC):
    """
    Base class for all data clients that access live APIs.

    Provides common interface for different data sources while allowing
    each client to implement its specific data fetching logic.
    """

    def __init__(self, **kwargs):
        """Initialize client with configuration."""
        self.config = kwargs

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the client can connect to its data source.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def get_data(self, *args, **kwargs) -> dict[str, Any]:
        """
        Get data from the client's data source.

        Args:
            *args: Positional arguments
            **kwargs: Client-specific parameters

        Returns:
            Dict[str, Any]: Raw data from the source
        """
        pass

    def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols/tickers from this data source.

        Returns:
            List[str]: Available symbols, empty list if not supported
        """
        return []

    def get_data_range(
        self, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """
        Get data for a specific date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional client-specific parameters

        Returns:
            Dict[str, Any]: Data for the specified range
        """
        return self.get_data(start_date=start_date, end_date=end_date, **kwargs)

    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate that the date range is acceptable.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            bool: True if date range is valid
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            return start_dt <= end_dt <= datetime.now()
        except ValueError:
            return False

    def get_client_info(self) -> dict[str, Any]:
        """
        Get information about this client.

        Returns:
            Dict[str, Any]: Client metadata
        """
        return {
            "client_type": self.__class__.__name__,
            "supports_symbols": len(self.get_available_symbols()) > 0,
            "config": self.config,
        }
