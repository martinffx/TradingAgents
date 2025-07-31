"""
Base service class for TradingAgents services.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseService:
    """Base service class with common functionality."""

    def __init__(self, online_mode: bool = True, data_dir: str = "data", **kwargs):
        """Initialize base service.

        Args:
            online_mode: Whether to use live APIs or cached data only
            data_dir: Directory for data storage
        """
        self.online_mode = online_mode
        self.data_dir = data_dir

    def is_online(self) -> bool:
        """Check if service is in online mode."""
        return self.online_mode

    def set_online_mode(self, online: bool) -> None:
        """Set online mode for the service."""
        self.online_mode = online
