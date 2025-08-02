import logging
import re
from datetime import datetime, timedelta
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.config import DEFAULT_CONFIG, TradingAgentsConfig
from tradingagents.domains.marketdata.fundamental_data_service import (
    BalanceSheetContext,
    CashFlowContext,
    FundamentalDataService,
    IncomeStatementContext,
)
from tradingagents.domains.marketdata.insider_data_service import (
    InsiderDataService,
    InsiderSentimentContext,
    InsiderTransactionContext,
)
from tradingagents.domains.marketdata.market_data_service import (
    MarketDataService,
    PriceDataContext,
    TAReportContext,
)

# Import context models
from tradingagents.domains.news.news_service import (
    GlobalNewsContext,
    NewsContext,
    NewsService,
)
from tradingagents.domains.socialmedia.social_media_service import (
    SocialMediaService,
    StockSocialContext,
)

logger = logging.getLogger(__name__)


class AgentToolkit:
    def __init__(
        self,
        news_service: NewsService,
        marketdata_service: MarketDataService,
        fundamentaldata_service: FundamentalDataService,
        socialmedia_service: SocialMediaService,
        insiderdata_service: InsiderDataService,
        config: TradingAgentsConfig = DEFAULT_CONFIG,
    ):
        self._news_service = news_service
        self._marketdata_service = marketdata_service
        self._fundamentaldata_service = fundamentaldata_service
        self._socialmedia_service = socialmedia_service
        self._insiderdata_service = insiderdata_service
        self._config = config

    @tool
    def get_global_news(
        self,
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
    ) -> GlobalNewsContext:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
        Returns:
            GlobalNewsContext: Structured global news context with articles and sentiment analysis.
        """
        # Calculate date range (current date only)
        start_date = curr_date
        end_date = curr_date

        # Call specialized service method
        return self._news_service.get_global_news_context(
            start_date=start_date,
            end_date=end_date,
            categories=["general", "business", "politics"],
        )

    @tool
    def get_news(
        self,
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> NewsContext:
        """
        Retrieve the latest news about a given stock from Finnhub within a date range
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            NewsContext: Structured news context with articles and sentiment analysis for the company.
        """
        try:
            ticker = self._validate_ticker(ticker)
            # Validate date formats
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")

            return self._news_service.get_context(
                query=ticker, start_date=start_date, end_date=end_date, symbol=ticker
            )
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            raise

    @tool
    def get_socialmedia_stock_info(
        self,
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
    ) -> StockSocialContext:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
        Returns:
            StockSocialContext: Structured social media context with posts and sentiment analysis for the stock.
        """
        try:
            ticker = self._validate_ticker(ticker)
            # Validate date format
            datetime.strptime(curr_date, "%Y-%m-%d")

            return self._socialmedia_service.get_stock_social_context(
                symbol=ticker, start_date=curr_date, end_date=curr_date
            )
        except Exception as e:
            logger.error(f"Error getting social media info for {ticker}: {e}")
            raise

    @tool
    def get_market_data(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    ) -> PriceDataContext:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            PriceDataContext: Structured price data context with historical prices and key metrics.
        """
        try:
            symbol = self._validate_ticker(symbol)
            # Validate date formats
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")

            return self._marketdata_service.get_market_data_context(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            raise

    @tool
    def get_ta_report(
        self,
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = None,
    ) -> TAReportContext:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, uses config default if None
        Returns:
            TAReportContext: Structured technical analysis context with indicator data and signals.
        """
        try:
            symbol = self._validate_ticker(symbol)
            if look_back_days is None:
                look_back_days = self._config.default_ta_lookback_days
            start_date, end_date = self._calculate_date_range(curr_date, look_back_days)

            return self._marketdata_service.get_ta_report_context(
                symbol=symbol,
                indicator=indicator,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Error getting TA report for {symbol}: {e}")
            raise

    @tool
    def get_insider_sentiment(
        self,
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
    ) -> InsiderSentimentContext:
        """
        Retrieve insider sentiment information about a company (retrieved from public SEC information) for the configured lookback period
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            InsiderSentimentContext: Structured insider sentiment analysis with transaction data and sentiment scores.
        """
        try:
            ticker = self._validate_ticker(ticker)
            start_date, end_date = self._calculate_date_range(curr_date)

            return self._insiderdata_service.get_insider_sentiment_context(
                symbol=ticker, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            logger.error(f"Error getting insider sentiment for {ticker}: {e}")
            raise

    @tool
    def get_insider_transactions(
        self,
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
    ) -> InsiderTransactionContext:
        """
        Retrieve insider transaction information about a company (retrieved from public SEC information) for the configured lookback period
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            InsiderTransactionContext: Structured insider transaction analysis with detailed transaction data.
        """
        try:
            ticker = self._validate_ticker(ticker)
            start_date, end_date = self._calculate_date_range(curr_date)

            return self._insiderdata_service.get_insider_transaction_context(
                symbol=ticker, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            logger.error(f"Error getting insider transactions for {ticker}: {e}")
            raise

    @tool
    def get_balance_sheet(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> BalanceSheetContext:
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            BalanceSheetContext: Structured balance sheet analysis with key liquidity and debt metrics.
        """
        return self._fundamentaldata_service.get_balance_sheet_context(
            symbol=ticker,
            start_date=curr_date,
            end_date=curr_date,
            frequency=freq.lower(),
        )

    @tool
    def get_cashflow(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> CashFlowContext:
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            CashFlowContext: Structured cash flow analysis with operating cash flow metrics.
        """
        return self._fundamentaldata_service.get_cashflow_context(
            symbol=ticker,
            start_date=curr_date,
            end_date=curr_date,
            frequency=freq.lower(),
        )

    @tool
    def get_income_stmt(
        self,
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    ) -> IncomeStatementContext:
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency of the company's financial history: annual / quarterly
            curr_date (str): current date you are trading at, yyyy-mm-dd
        Returns:
            IncomeStatementContext: Structured income statement analysis with profitability metrics.
        """
        return self._fundamentaldata_service.get_income_statement_context(
            symbol=ticker,
            start_date=curr_date,
            end_date=curr_date,
            frequency=freq.lower(),
        )

    def _calculate_date_range(
        self, curr_date: str, lookback_days: int | None = None
    ) -> tuple[str, str]:
        """
        Calculate start and end dates based on current date and lookback period.

        Args:
            curr_date: Current date in YYYY-MM-DD format
            lookback_days: Number of days to look back (uses config default if None)

        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format

        Raises:
            ValueError: If date format is invalid
        """
        try:
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format '{curr_date}': {e}")
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {curr_date}")

        if lookback_days is None:
            lookback_days = self._config.default_lookback_days

        start_date_obj = curr_date_obj - timedelta(days=lookback_days)
        return start_date_obj.strftime("%Y-%m-%d"), curr_date

    def _validate_ticker(self, ticker: str) -> str:
        """
        Validate and sanitize ticker symbol.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            Sanitized ticker symbol

        Raises:
            ValueError: If ticker is invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")

        # Remove whitespace and convert to uppercase
        ticker = ticker.strip().upper()

        # Basic validation: only letters, numbers, and common symbols
        if not re.match(r"^[A-Z0-9.-]{1,10}$", ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")

        return ticker
