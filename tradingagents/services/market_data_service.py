"""
Market data service that provides structured market context.
"""

import logging
from typing import Any

from tradingagents.clients.base import BaseClient
from tradingagents.dataflows.stockstats_utils import StockstatsUtils
from tradingagents.models.context import (
    MarketDataContext,
    TechnicalIndicatorData,
)
from tradingagents.repositories.base import BaseRepository

from .base import BaseService

logger = logging.getLogger(__name__)


class MarketDataService(BaseService):
    """Service for market data and technical indicators."""

    def __init__(
        self,
        client: BaseClient | None = None,
        repository: BaseRepository | None = None,
        online_mode: bool = True,
        **kwargs,
    ):
        """
        Initialize market data service.

        Args:
            client: Client for live market data
            repository: Repository for historical market data
            online_mode: Whether to use live data
            **kwargs: Additional configuration
        """
        super().__init__(online_mode, **kwargs)
        self.client = client
        self.repository = repository
        self.stockstats_utils = StockstatsUtils()

    def get_context(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        indicators: list[str] | None = None,
        force_refresh: bool = False,
        **kwargs,
    ) -> MarketDataContext:
        """
        Get market data context with price data and technical indicators.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            indicators: List of technical indicators to calculate
            force_refresh: If True, skip local data and fetch fresh from API
            **kwargs: Additional parameters

        Returns:
            MarketDataContext: Structured market data context
        """
        if indicators is None:
            indicators = ["rsi", "macd", "close_50_sma"]

        # Local-first data strategy with force refresh option
        if force_refresh:
            # Skip local data, fetch fresh from API
            price_data = self._fetch_and_cache_fresh_data(symbol, start_date, end_date)
            data_source = "live_api_refresh"
        else:
            # Check local data first, fetch missing if needed
            price_data = self._get_price_data_local_first(symbol, start_date, end_date)
            data_source = price_data.get("metadata", {}).get("source", "unknown")

        # Calculate technical indicators
        technical_indicators = self._calculate_indicators(
            symbol, start_date, end_date, indicators
        )

        # Determine data quality
        data_quality = self._determine_data_quality(
            data_source=data_source,
            record_count=len(price_data.get("data", [])),
            has_errors="error" in price_data.get("metadata", {}),
        )

        # Create metadata
        metadata = self._create_base_metadata(
            data_quality=data_quality,
            price_data_source=data_source,
            indicator_count=len(technical_indicators),
            symbol=symbol,
            force_refresh=force_refresh,
        )

        return MarketDataContext(
            symbol=symbol,
            period={"start": start_date, "end": end_date},
            price_data=price_data.get("data", []),
            technical_indicators=technical_indicators,
            metadata=metadata,
        )

    def get_price_context(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> MarketDataContext:
        """
        Get market data context with just price data (no indicators).

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            MarketDataContext: Market context with price data only
        """
        return self.get_context(symbol, start_date, end_date, indicators=[], **kwargs)

    def _get_price_data_local_first(
        self, symbol: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Get price data using local-first strategy: check local data first, fetch missing if needed."""
        try:
            # Check if we have sufficient local data
            if self.repository and self.repository.has_data_for_period(
                symbol, start_date, end_date
            ):
                logger.info(
                    f"Using local data for {symbol} ({start_date} to {end_date})"
                )
                local_data = self.repository.get_data(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )
                local_data["metadata"] = local_data.get("metadata", {})
                local_data["metadata"]["source"] = "local_cache"
                return local_data

            # We don't have sufficient local data - need to fetch from API
            if self.client:
                logger.info(
                    f"Local data insufficient, fetching from API for {symbol} ({start_date} to {end_date})"
                )
                fresh_data = self.client.get_data(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )

                # Cache the fresh data if we have a repository
                if fresh_data and self.repository:
                    try:
                        self.repository.store_data(symbol, fresh_data)
                        logger.debug(f"Cached fresh data for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to cache data for {symbol}: {e}")

                fresh_data["metadata"] = fresh_data.get("metadata", {})
                fresh_data["metadata"]["source"] = "live_api"
                return fresh_data

            # No client available, try repository as fallback
            elif self.repository:
                logger.warning(
                    f"No API client available, using partial local data for {symbol}"
                )
                local_data = self.repository.get_data(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )
                local_data["metadata"] = local_data.get("metadata", {})
                local_data["metadata"]["source"] = "local_partial"
                return local_data

            else:
                logger.warning(f"No data source available for {symbol}")
                return {
                    "symbol": symbol,
                    "data": [],
                    "metadata": {
                        "source": "none",
                        "error": "No client or repository configured",
                    },
                }

        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "data": [],
                "metadata": {"source": "error", "error": str(e)},
            }

    def _fetch_and_cache_fresh_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Force fetch fresh data from API and cache it, bypassing local data."""
        try:
            if not self.client:
                logger.warning(f"No API client available for force refresh of {symbol}")
                return {
                    "symbol": symbol,
                    "data": [],
                    "metadata": {
                        "source": "no_client",
                        "error": "No API client configured for force refresh",
                    },
                }

            logger.info(
                f"Force refreshing data from API for {symbol} ({start_date} to {end_date})"
            )

            # Clear existing data if we have a repository
            if self.repository:
                try:
                    self.repository.clear_data(symbol, start_date, end_date)
                    logger.debug(f"Cleared existing data for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to clear existing data for {symbol}: {e}")

            # Fetch fresh data
            fresh_data = self.client.get_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )

            # Cache the fresh data
            if fresh_data and self.repository:
                try:
                    self.repository.store_data(symbol, fresh_data, overwrite=True)
                    logger.debug(f"Cached refreshed data for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to cache refreshed data for {symbol}: {e}")

            fresh_data["metadata"] = fresh_data.get("metadata", {})
            fresh_data["metadata"]["source"] = "live_api_refresh"
            return fresh_data

        except Exception as e:
            logger.error(f"Error force refreshing data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "data": [],
                "metadata": {"source": "refresh_error", "error": str(e)},
            }

    def _calculate_indicators(
        self, symbol: str, start_date: str, end_date: str, indicators: list[str]
    ) -> dict[str, list[TechnicalIndicatorData]]:
        """Calculate technical indicators."""
        if not indicators:
            return {}

        technical_data = {}

        for indicator in indicators:
            try:
                logger.info(f"Calculating {indicator} for {symbol}")

                # Use existing stockstats utility
                indicator_data = self._get_indicator_data(
                    symbol, indicator, start_date, end_date
                )

                if indicator_data:
                    technical_data[indicator] = indicator_data
                else:
                    logger.warning(f"No data returned for indicator {indicator}")

            except Exception as e:
                logger.error(f"Error calculating {indicator} for {symbol}: {e}")
                continue

        return technical_data

    def _get_indicator_data(
        self, symbol: str, indicator: str, start_date: str, end_date: str
    ) -> list[TechnicalIndicatorData]:
        """Get indicator data using StockstatsUtils."""
        try:
            from datetime import datetime, timedelta

            # Get data for the date range
            current_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")

            indicator_points = []

            # Iterate through date range
            while current_date >= start_date_dt:
                date_str = current_date.strftime("%Y-%m-%d")

                try:
                    # Use stockstats utility to get indicator value
                    # This assumes the existing data directory structure
                    data_dir = self.config.get("data_dir", "data")
                    price_data_dir = f"{data_dir}/market_data/price_data"

                    indicator_value = StockstatsUtils.get_stock_stats(
                        symbol,
                        indicator,
                        date_str,
                        price_data_dir,
                        online=self.online_mode,
                    )

                    if indicator_value is not None and indicator_value != "":
                        # Handle different indicator value types
                        if isinstance(indicator_value, int | float):
                            value = float(indicator_value)
                        elif isinstance(indicator_value, str):
                            try:
                                value = float(indicator_value)
                            except ValueError:
                                logger.warning(
                                    f"Could not parse indicator value: {indicator_value}"
                                )
                                current_date -= timedelta(days=1)
                                continue
                        else:
                            # For complex indicators like MACD, this might be a dict
                            value = indicator_value

                        indicator_points.append(
                            TechnicalIndicatorData(
                                date=date_str, value=value, indicator_type=indicator
                            )
                        )

                except Exception as e:
                    logger.debug(
                        f"Could not get {indicator} for {symbol} on {date_str}: {e}"
                    )

                current_date -= timedelta(days=1)

            # Return in chronological order
            return list(reversed(indicator_points))

        except Exception as e:
            logger.error(f"Error getting indicator data for {indicator}: {e}")
            return []
