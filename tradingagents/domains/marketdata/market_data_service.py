"""
Market data service that provides structured market context.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels for market data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TechnicalIndicatorData:
    """Technical indicator data point."""

    date: str
    value: float | dict[str, Any]
    indicator_type: str


@dataclass
class MarketDataContext:
    """Market data context for trading analysis."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    price_data: list[dict[str, Any]]
    technical_indicators: dict[str, list[TechnicalIndicatorData]]
    metadata: dict[str, Any]


@dataclass
class TAReportContext:
    """Technical Analysis Report context for specific indicators."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    indicator: str
    indicator_data: list[TechnicalIndicatorData]
    analysis_summary: str
    signal_strength: float  # -1.0 to 1.0
    recommendation: str  # "BUY", "SELL", "HOLD"
    metadata: dict[str, Any]


@dataclass
class PriceDataContext:
    """Price Data context for historical price information."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    price_data: list[dict[str, Any]]
    latest_price: float
    price_change: float
    price_change_percent: float
    volume_info: dict[str, Any]
    metadata: dict[str, Any]


class MarketDataService:
    """Service for market data and technical indicators."""

    def __init__(
        self,
        yfin_client: YFinClient,
        repo: MarketdataRepository,
    ):
        """
        Initialize market data service.

        Args:
            client: Client for live market data
            repository: Repository for historical market data
            online_mode: Whether to use live data
            **kwargs: Additional configuration
        """
        self.finnhub_client = finnhub_client
        self.yfin_client = yfin_client
        self.repo = repo

    def get_market_data_context(
        self, symbol: str, start_date: str, end_date: str
    ) -> PriceDataContext:
        """
        Get focused price data context with key metrics.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            PriceDataContext: Focused price data context
        """
        # return PriceDataContext(
        #     symbol=symbol,
        #     period={"start": start_date, "end": end_date},
        #     price_data=price_data.get("data", []),
        #     latest_price=latest_price,
        #     price_change=price_change,
        #     price_change_percent=price_change_percent,
        #     volume_info=volume_info,
        #     metadata=metadata,
        # )

        pass  # TODO: get data from repo

    def get_ta_report_context(
        self, symbol: str, indicator: str, start_date: str, end_date: str
    ) -> TAReportContext:
        """
        Get technical analysis report context for a specific indicator.

        Args:
            symbol: Stock ticker symbol
            indicator: Technical indicator name (e.g., 'rsi', 'macd', 'sma')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            TAReportContext: Focused technical analysis context
        """

        # return TAReportContext(
        #     symbol=symbol,
        #     period={"start": start_date, "end": end_date},
        #     indicator=indicator,
        #     indicator_data=indicator_data.get(indicator, []),
        #     analysis_summary=analysis_summary,
        #     signal_strength=signal_strength,
        #     recommendation=recommendation,
        #     metadata=metadata,
        # )

        pass  # TODO get data from repo and calculate indicator with TALib?

    def update_market_data(self, symbol: str, start_date: str, end_date: str):
        pass  # TODO: fetch market data and save
