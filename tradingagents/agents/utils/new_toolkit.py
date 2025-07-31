"""
New Toolkit class using Service/Client/Repository architecture with JSON context.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool

from tradingagents.config import TradingAgentsConfig
from tradingagents.services.builders import build_toolkit_services

if TYPE_CHECKING:
    from tradingagents.services.market_data_service import MarketDataService
    from tradingagents.services.news_service import NewsService

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = TradingAgentsConfig()


def create_msg_delete():
    """Create message deletion function for agents."""

    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


class Toolkit:
    """
    New Toolkit class that uses services to provide JSON context to agents.

    This replaces the old interface.py approach with structured Pydantic models
    that agents can process more dynamically.
    """

    def __init__(
        self,
        config: TradingAgentsConfig | None = None,
        services: dict[str, Any] | None = None,
    ):
        """
        Initialize Toolkit with services.

        Args:
            config: TradingAgents configuration
            services: Pre-built services dict, or None to build from config
        """
        self.config = config or DEFAULT_CONFIG

        if services:
            self.services = services
        else:
            logger.info("Building services from config")
            self.services = build_toolkit_services(self.config)

        # Set up individual services
        self.market_service: MarketDataService | None = self.services.get("market_data")
        self.news_service: NewsService | None = self.services.get("news")

        logger.info(f"Toolkit initialized with {len(self.services)} services")

    # Market Data Tools
    def get_market_data(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve market data context for a given ticker symbol.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: JSON context containing market data with price data and metadata
        """
        if not self.market_service:
            return self._create_error_context("MarketDataService not available")

        try:
            context = self.market_service.get_price_context(
                symbol, start_date, end_date
            )
            return context.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return self._create_error_context(f"Error fetching market data: {str(e)}")

    @tool
    def get_market_data_with_indicators(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        indicators: Annotated[
            str, "Comma-separated list of indicators (e.g. 'rsi,macd,close_50_sma')"
        ] = "rsi,macd",
    ) -> str:
        """
        Retrieve market data context with technical indicators.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            indicators (str): Comma-separated indicators
        Returns:
            str: JSON context containing market data with technical indicators
        """
        if not self.market_service:
            return self._create_error_context("MarketDataService not available")

        try:
            indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]
            context = self.market_service.get_context(
                symbol, start_date, end_date, indicators=indicator_list
            )
            return context.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error getting market data with indicators for {symbol}: {e}")
            return self._create_error_context(
                f"Error fetching market data with indicators: {str(e)}"
            )

    # News Tools
    @tool
    def get_company_news(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve news context for a specific company.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: JSON context containing news articles, sentiment analysis, and metadata
        """
        if not self.news_service:
            return self._create_error_context("NewsService not available")

        try:
            context = self.news_service.get_company_news_context(
                symbol, start_date, end_date
            )
            return context.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error getting company news for {symbol}: {e}")
            return self._create_error_context(f"Error fetching company news: {str(e)}")

    @tool
    def get_global_news(
        self,
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        categories: Annotated[
            str, "Comma-separated news categories (e.g. 'economy,markets,finance')"
        ] = "economy,markets",
    ) -> str:
        """
        Retrieve global/macro news context.
        Args:
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            categories (str): Comma-separated news categories
        Returns:
            str: JSON context containing global news articles and sentiment analysis
        """
        if not self.news_service:
            return self._create_error_context("NewsService not available")

        try:
            category_list = [c.strip() for c in categories.split(",") if c.strip()]
            context = self.news_service.get_global_news_context(
                start_date, end_date, categories=category_list
            )
            return context.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error getting global news: {e}")
            return self._create_error_context(f"Error fetching global news: {str(e)}")

    @tool
    def get_news_by_query(
        self,
        query: Annotated[str, "Search query for news"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve news context for a specific query.
        Args:
            query (str): Search query for news
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: JSON context containing news articles and sentiment analysis
        """
        if not self.news_service:
            return self._create_error_context("NewsService not available")

        try:
            context = self.news_service.get_context(query, start_date, end_date)
            return context.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error getting news for query '{query}': {e}")
            return self._create_error_context(f"Error fetching news: {str(e)}")

    # Legacy compatibility methods (return JSON instead of markdown)
    @tool
    def get_YFin_data(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Legacy method: Retrieve market data (now returns JSON context).
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: JSON context containing market data
        """
        return self.get_market_data(symbol, start_date, end_date)

    @tool
    def get_finnhub_news(
        self,
        ticker: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Legacy method: Retrieve company news (now returns JSON context).
        Args:
            ticker (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: JSON context containing news data
        """
        return self.get_company_news(ticker, start_date, end_date)

    # Utility methods
    def _create_error_context(self, error_message: str) -> str:
        """Create a JSON error context."""
        error_context = {
            "error": True,
            "message": error_message,
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "source": "toolkit",
            },
        }
        import json

        return json.dumps(error_context, indent=2)

    def get_available_tools(self) -> list:
        """Get list of available tools based on configured services."""
        tools = []

        if self.market_service:
            tools.extend(
                [
                    "get_market_data",
                    "get_market_data_with_indicators",
                    "get_YFin_data",  # legacy
                ]
            )

        if self.news_service:
            tools.extend(
                [
                    "get_company_news",
                    "get_global_news",
                    "get_news_by_query",
                    "get_finnhub_news",  # legacy
                ]
            )

        return tools

    def get_toolkit_info(self) -> dict[str, Any]:
        """Get information about the toolkit configuration."""
        return {
            "toolkit_type": "service_based",
            "config": {
                "online_mode": self.config.online_tools,
                "data_dir": self.config.data_dir,
            },
            "services": list(self.services.keys()),
            "available_tools": self.get_available_tools(),
        }
