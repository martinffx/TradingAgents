"""
Base client interface for TradingAgents data sources.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseClient(ABC):
    """Abstract base class for all data clients."""

    @abstractmethod
    def get_data(self, **kwargs) -> dict[str, Any]:
        """
        Get data from the client source.

        Args:
            **kwargs: Client-specific parameters

        Returns:
            dict: Data dictionary with standardized structure
        """
        pass
