"""
Service-based toolkit for agents using the new JSON context services.
Replaces the old markdown-based interface.py with structured service calls.
"""

import json
from datetime import datetime, timedelta
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool

from tradingagents.config import TradingAgentsConfig
from tradingagents.services.fundamental_data_service import FundamentalDataService
from tradingagents.services.insider_data_service import InsiderDataService
from tradingagents.services.market_data_service import MarketDataService
from tradingagents.services.news_service import NewsService
from tradingagents.services.openai_data_service import OpenAIDataService
from tradingagents.services.social_media_service import SocialMediaService

DEFAULT_CONFIG = TradingAgentsConfig()


def create_msg_delete():
    """Create message deletion function for Anthropic compatibility."""

    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


class ServiceToolkit:
    """Service-based toolkit using the new JSON context services."""

    def __init__(self, config: TradingAgentsConfig | None = None):
        """
        Initialize the service toolkit.

        Args:
            config: Configuration object for services
        """
        self._config = config or DEFAULT_CONFIG

        # Services will be lazily initialized
        self._market_service = None
        self._news_service = None
        self._social_service = None
        self._fundamental_service = None
        self._insider_service = None
        self._openai_service = None

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def update_config(self, config: TradingAgentsConfig):
        """Update the configuration and reset services."""
        self._config = config
        # Reset services to force re-initialization with new config
        self._market_service = None
        self._news_service = None
        self._social_service = None
        self._fundamental_service = None
        self._insider_service = None
        self._openai_service = None

    def _get_market_service(self) -> MarketDataService:
        """Lazy initialization of market data service."""
        if self._market_service is None:
            # This would typically use a service factory/builder
            # For now, return a basic service
            from tradingagents.services.builders import create_market_data_service

            self._market_service = create_market_data_service(self._config)
        return self._market_service

    def _get_news_service(self) -> NewsService:
        """Lazy initialization of news service."""
        if self._news_service is None:
            from tradingagents.services.builders import create_news_service

            self._news_service = create_news_service(self._config)
        return self._news_service

    def _get_social_service(self) -> SocialMediaService:
        """Lazy initialization of social media service."""
        if self._social_service is None:
            from tradingagents.services.builders import create_social_media_service

            self._social_service = create_social_media_service(self._config)
        return self._social_service

    def _get_fundamental_service(self) -> FundamentalDataService:
        """Lazy initialization of fundamental data service."""
        if self._fundamental_service is None:
            from tradingagents.services.builders import create_fundamental_data_service

            self._fundamental_service = create_fundamental_data_service(self._config)
        return self._fundamental_service

    def _get_insider_service(self) -> InsiderDataService:
        """Lazy initialization of insider data service."""
        if self._insider_service is None:
            from tradingagents.services.builders import create_insider_data_service

            self._insider_service = create_insider_data_service(self._config)
        return self._insider_service

    def _get_openai_service(self) -> OpenAIDataService:
        """Lazy initialization of OpenAI data service."""
        if self._openai_service is None:
            from tradingagents.services.builders import create_openai_data_service

            self._openai_service = create_openai_data_service(self._config)
        return self._openai_service

    def _context_to_string(self, context: Any) -> str:
        """Convert a context object to a formatted string for agents."""
        # For now, convert to JSON string
        # In the future, we might want more sophisticated formatting
        return json.dumps(context.model_dump(), indent=2, default=str)

    def _calculate_date_range(
        self, curr_date: str, look_back_days: int
    ) -> tuple[str, str]:
        """Calculate start and end dates from current date and lookback days."""
        end_date = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=look_back_days)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    @tool
    def get_reddit_news(
        self,
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context containing the latest global news from Reddit.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 7)

        social_service = self._get_social_service()
        context = social_service.get_global_trends(
            start_date=start_date,
            end_date=end_date,
            subreddits=["news", "worldnews", "Economics"],
        )

        return self._context_to_string(context)

    @tool
    def get_finnhub_news(
        self,
        ticker: Annotated[str, "Search query of a company, e.g. 'AAPL, TSM, etc."],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Finnhub within a date range.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context containing news about the company.
        """
        news_service = self._get_news_service()
        context = news_service.get_company_news_context(
            symbol=ticker, start_date=start_date, end_date=end_date, sources=["finnhub"]
        )

        return self._context_to_string(context)

    @tool
    def get_reddit_stock_info(
        self,
        ticker: Annotated[str, "Ticker of a company. e.g. AAPL, TSM"],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> str:
        """
        Retrieve the latest news about a given stock from Reddit.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            str: A JSON-formatted context containing the latest news about the company.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 7)

        social_service = self._get_social_service()
        context = social_service.get_company_social_context(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            subreddits=["investing", "stocks", "wallstreetbets"],
        )

        return self._context_to_string(context)

    @tool
    def get_YFin_data(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context containing stock price data and technical indicators.
        """
        market_service = self._get_market_service()
        context = market_service.get_context(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            force_refresh=False,  # Use local-first strategy
        )

        return self._context_to_string(context)

    @tool
    def get_YFin_data_online(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve fresh stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context containing fresh stock price data.
        """
        market_service = self._get_market_service()
        context = market_service.get_context(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            force_refresh=True,  # Force fresh data
        )

        return self._context_to_string(context)

    @tool
    def get_stockstats_indicators_report(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A JSON-formatted context containing technical indicators.
        """
        start_date, end_date = self._calculate_date_range(curr_date, look_back_days)

        market_service = self._get_market_service()
        context = market_service.get_context(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            indicators=[indicator],
            force_refresh=False,
        )

        return self._context_to_string(context)

    @tool
    def get_stockstats_indicators_report_online(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
    ) -> str:
        """
        Retrieve fresh stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A JSON-formatted context containing fresh technical indicators.
        """
        start_date, end_date = self._calculate_date_range(curr_date, look_back_days)

        market_service = self._get_market_service()
        context = market_service.get_context(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            indicators=[indicator],
            force_refresh=True,
        )

        return self._context_to_string(context)

    @tool
    def get_finnhub_company_insider_sentiment(
        self,
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[str, "current date of you are trading at, yyyy-mm-dd"],
    ) -> str:
        """
        Retrieve insider sentiment information about a company for the past 30 days.
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: A JSON-formatted context with insider trading sentiment analysis.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 30)

        insider_service = self._get_insider_service()
        context = insider_service.get_insider_context(
            symbol=ticker, start_date=start_date, end_date=end_date
        )

        return self._context_to_string(context)

    @tool
    def get_finnhub_company_insider_transactions(
        self,
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> str:
        """
        Retrieve insider transaction information about a company for the past 30 days.
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: A JSON-formatted context with insider transaction details.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 30)

        insider_service = self._get_insider_service()
        context = insider_service.get_insider_context(
            symbol=ticker, start_date=start_date, end_date=end_date
        )

        return self._context_to_string(context)

    @tool
    def get_simfin_balance_sheet(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> str:
        """
        Retrieve the most recent balance sheet of a company.
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: A JSON-formatted context with the company's balance sheet.
        """
        # Use a reasonable date range for fundamental data
        start_date = (
            datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")

        fundamental_service = self._get_fundamental_service()
        context = fundamental_service.get_fundamental_context(
            symbol=ticker, start_date=start_date, end_date=curr_date, frequency=freq
        )

        return self._context_to_string(context)

    @tool
    def get_simfin_cashflow(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> str:
        """
        Retrieve the most recent cash flow statement of a company.
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: A JSON-formatted context with the company's cash flow statement.
        """
        start_date = (
            datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")

        fundamental_service = self._get_fundamental_service()
        context = fundamental_service.get_fundamental_context(
            symbol=ticker, start_date=start_date, end_date=curr_date, frequency=freq
        )

        return self._context_to_string(context)

    @tool
    def get_simfin_income_stmt(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> str:
        """
        Retrieve the most recent income statement of a company.
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            str: A JSON-formatted context with the company's income statement.
        """
        start_date = (
            datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")

        fundamental_service = self._get_fundamental_service()
        context = fundamental_service.get_fundamental_context(
            symbol=ticker, start_date=start_date, end_date=curr_date, frequency=freq
        )

        return self._context_to_string(context)

    @tool
    def get_google_news(
        self,
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context containing the latest news from Google News.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 7)

        news_service = self._get_news_service()
        context = news_service.get_context(
            query=query, start_date=start_date, end_date=end_date, sources=["google"]
        )

        return self._context_to_string(context)

    @tool
    def get_stock_news_openai(
        self,
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve AI-generated news analysis about a given stock.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context with AI news analysis.
        """
        # First get the news context
        start_date, end_date = self._calculate_date_range(curr_date, 7)
        news_service = self._get_news_service()
        news_context = news_service.get_company_news_context(
            symbol=ticker, start_date=start_date, end_date=end_date
        )

        # Then get AI analysis of the news
        openai_service = self._get_openai_service()
        context = openai_service.get_news_impact_analysis(
            symbol=ticker, news_context=self._context_to_string(news_context)
        )

        return self._context_to_string(context)

    @tool
    def get_global_news_openai(
        self,
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve AI-generated macroeconomic news analysis.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context with AI macroeconomic analysis.
        """
        start_date, end_date = self._calculate_date_range(curr_date, 7)

        # Get global news context
        news_service = self._get_news_service()
        news_context = news_service.get_global_news_context(
            start_date=start_date,
            end_date=end_date,
            categories=["economy", "markets", "finance"],
        )

        # Get AI analysis
        openai_service = self._get_openai_service()
        context = openai_service.get_news_impact_analysis(
            symbol="GLOBAL",  # Use a placeholder for global analysis
            news_context=self._context_to_string(news_context),
        )

        return self._context_to_string(context)

    @tool
    def get_fundamentals_openai(
        self,
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    ) -> str:
        """
        Retrieve AI-generated fundamental analysis about a given stock.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
        Returns:
            str: A JSON-formatted context with AI fundamental analysis.
        """
        # Get fundamental data context
        start_date = (
            datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=365)
        ).strftime("%Y-%m-%d")
        fundamental_service = self._get_fundamental_service()
        fundamental_context = fundamental_service.get_fundamental_context(
            symbol=ticker,
            start_date=start_date,
            end_date=curr_date,
            frequency="quarterly",
        )

        # Get market data for additional context
        market_start = (
            datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=30)
        ).strftime("%Y-%m-%d")
        market_service = self._get_market_service()
        market_context = market_service.get_context(
            symbol=ticker, start_date=market_start, end_date=curr_date
        )

        # Combine contexts for AI analysis
        combined_context = {
            "fundamental_data": fundamental_context.model_dump(),
            "market_data": market_context.model_dump(),
        }

        # Get AI analysis
        openai_service = self._get_openai_service()
        context = openai_service.get_market_sentiment_analysis(
            symbol=ticker, market_data_context=json.dumps(combined_context, default=str)
        )

        return self._context_to_string(context)


# Create a default instance for backward compatibility
default_toolkit = ServiceToolkit()

# Export individual tools for use in agents
get_reddit_news = default_toolkit.get_reddit_news
get_finnhub_news = default_toolkit.get_finnhub_news
get_reddit_stock_info = default_toolkit.get_reddit_stock_info
get_YFin_data = default_toolkit.get_YFin_data
get_YFin_data_online = default_toolkit.get_YFin_data_online
get_stockstats_indicators_report = default_toolkit.get_stockstats_indicators_report
get_stockstats_indicators_report_online = (
    default_toolkit.get_stockstats_indicators_report_online
)
get_finnhub_company_insider_sentiment = (
    default_toolkit.get_finnhub_company_insider_sentiment
)
get_finnhub_company_insider_transactions = (
    default_toolkit.get_finnhub_company_insider_transactions
)
get_simfin_balance_sheet = default_toolkit.get_simfin_balance_sheet
get_simfin_cashflow = default_toolkit.get_simfin_cashflow
get_simfin_income_stmt = default_toolkit.get_simfin_income_stmt
get_google_news = default_toolkit.get_google_news
get_stock_news_openai = default_toolkit.get_stock_news_openai
get_global_news_openai = default_toolkit.get_global_news_openai
get_fundamentals_openai = default_toolkit.get_fundamentals_openai
