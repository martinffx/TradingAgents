"""
Updated Toolkit class using Service/Client/Repository architecture with JSON context.
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
    Toolkit class that uses services to provide JSON context to agents.

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

    # Create tool methods as static methods with service access via closure
    def _create_market_data_tool(self):
        """Create market data tool with service access."""
        market_service = self.market_service

        @tool
        def get_market_data(
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
            if not market_service:
                return _create_error_context("MarketDataService not available")

            try:
                context = market_service.get_price_context(symbol, start_date, end_date)
                return context.model_dump_json(indent=2)
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
                return _create_error_context(f"Error fetching market data: {str(e)}")

        return get_market_data

    def _create_market_indicators_tool(self):
        """Create market data with indicators tool."""
        market_service = self.market_service

        @tool
        def get_market_data_with_indicators(
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
            if not market_service:
                return _create_error_context("MarketDataService not available")

            try:
                indicator_list = [i.strip() for i in indicators.split(",") if i.strip()]
                context = market_service.get_context(
                    symbol, start_date, end_date, indicators=indicator_list
                )
                return context.model_dump_json(indent=2)
            except Exception as e:
                logger.error(
                    f"Error getting market data with indicators for {symbol}: {e}"
                )
                return _create_error_context(
                    f"Error fetching market data with indicators: {str(e)}"
                )

        return get_market_data_with_indicators

    def _create_company_news_tool(self):
        """Create company news tool."""
        news_service = self.news_service

        @tool
        def get_company_news(
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
            if not news_service:
                return _create_error_context("NewsService not available")

            try:
                context = news_service.get_company_news_context(
                    symbol, start_date, end_date
                )
                return context.model_dump_json(indent=2)
            except Exception as e:
                logger.error(f"Error getting company news for {symbol}: {e}")
                return _create_error_context(f"Error fetching company news: {str(e)}")

        return get_company_news

    def _create_global_news_tool(self):
        """Create global news tool."""
        news_service = self.news_service

        @tool
        def get_global_news(
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
            if not news_service:
                return _create_error_context("NewsService not available")

            try:
                category_list = [c.strip() for c in categories.split(",") if c.strip()]
                context = news_service.get_global_news_context(
                    start_date, end_date, categories=category_list
                )
                return context.model_dump_json(indent=2)
            except Exception as e:
                logger.error(f"Error getting global news: {e}")
                return _create_error_context(f"Error fetching global news: {str(e)}")

        return get_global_news

    def get_tools(self):
        """Get all available tools as LangChain tools."""
        tools = []

        if self.market_service:
            tools.append(self._create_market_data_tool())
            tools.append(self._create_market_indicators_tool())

        if self.news_service:
            tools.append(self._create_company_news_tool())
            tools.append(self._create_global_news_tool())

        return tools

    def get_available_tools(self) -> list:
        """Get list of available tool names based on configured services."""
        tools = []

        if self.market_service:
            tools.extend(["get_market_data", "get_market_data_with_indicators"])

        if self.news_service:
            tools.extend(["get_company_news", "get_global_news"])

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


def _create_error_context(error_message: str) -> str:
    """Create a JSON error context."""
    import json

    error_context = {
        "error": True,
        "message": error_message,
        "metadata": {"created_at": datetime.utcnow().isoformat(), "source": "toolkit"},
    }
    return json.dumps(error_context, indent=2)
