"""
Simple builder functions for dependency injection in TradingAgents services.
"""

import logging

from tradingagents.clients import (
    FinnhubClient,
    GoogleNewsClient,
    RedditClient,
    YFinanceClient,
)
from tradingagents.config import TradingAgentsConfig
from tradingagents.repositories import (
    FundamentalDataRepository,
    InsiderDataRepository,
    MarketDataRepository,
    NewsRepository,
    OpenAIRepository,
    SocialRepository,
)

from .fundamental_data_service import FundamentalDataService
from .insider_data_service import InsiderDataService
from .market_data_service import MarketDataService
from .news_service import NewsService
from .openai_data_service import OpenAIDataService
from .social_media_service import SocialMediaService

logger = logging.getLogger(__name__)


def build_market_data_service(config: TradingAgentsConfig) -> MarketDataService:
    """
    Build MarketDataService with appropriate client and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        MarketDataService: Configured service
    """
    client = None
    repository = None

    # Create client for online mode
    if config.online_tools:
        try:
            client = YFinanceClient()
            logger.info("Created YFinanceClient for live data")
        except Exception as e:
            logger.warning(f"Failed to create YFinanceClient: {e}")

    # Always create repository for fallback/offline mode
    try:
        repository = MarketDataRepository(
            data_dir=config.data_dir, cache_dir=config.data_cache_dir
        )
        logger.info(f"Created MarketDataRepository with data_dir: {config.data_dir}")
    except Exception as e:
        logger.error(f"Failed to create MarketDataRepository: {e}")

    return MarketDataService(
        client=client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_news_service(config: TradingAgentsConfig) -> NewsService:
    """
    Build NewsService with appropriate clients and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        NewsService: Configured service
    """
    finnhub_client = None
    google_client = None
    repository = None

    # Create clients for online mode
    if config.online_tools:
        # Finnhub client
        if config.finnhub_api_key:
            try:
                finnhub_client = FinnhubClient(config.finnhub_api_key)
                logger.info("Created FinnhubClient for live news data")
            except Exception as e:
                logger.warning(f"Failed to create FinnhubClient: {e}")
        else:
            logger.info("No Finnhub API key provided, skipping FinnhubClient")

        # Google News client
        try:
            google_client = GoogleNewsClient()
            logger.info("Created GoogleNewsClient for live news data")
        except Exception as e:
            logger.warning(f"Failed to create GoogleNewsClient: {e}")

    # Always create repository for fallback/offline mode
    try:
        repository = NewsRepository(
            data_dir=config.data_dir, cache_dir=config.data_cache_dir
        )
        logger.info(f"Created NewsRepository with data_dir: {config.data_dir}")
    except Exception as e:
        logger.error(f"Failed to create NewsRepository: {e}")

    return NewsService(
        finnhub_client=finnhub_client,
        google_client=google_client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_social_media_service(config: TradingAgentsConfig) -> SocialMediaService:
    """
    Build SocialMediaService with appropriate client and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        SocialMediaService: Configured service
    """
    client = None
    repository = None

    # Create client for online mode
    if config.online_tools:
        # Reddit client
        if config.reddit_client_id and config.reddit_client_secret:
            try:
                client = RedditClient(
                    client_id=config.reddit_client_id,
                    client_secret=config.reddit_client_secret,
                    user_agent=config.reddit_user_agent,
                )
                logger.info("Created RedditClient for live social media data")
            except Exception as e:
                logger.warning(f"Failed to create RedditClient: {e}")
        else:
            logger.info("No Reddit credentials provided, skipping RedditClient")

    # Always create repository for fallback/offline mode
    try:
        repository = SocialRepository(config.data_dir)
        logger.info(f"Created SocialRepository with data_dir: {config.data_dir}")
    except Exception as e:
        logger.error(f"Failed to create SocialRepository: {e}")

    return SocialMediaService(
        client=client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_fundamental_service(config: TradingAgentsConfig) -> FundamentalDataService:
    """
    Build FundamentalDataService with appropriate client and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        FundamentalDataService: Configured service
    """
    client = None
    repository = None

    # Create client for online mode
    if config.online_tools:
        # SimFin client (would be implemented when SimFinClient is available)
        # try:
        #     client = SimFinClient()  # This would need API key configuration
        #     logger.info("Created SimFinClient for live fundamental data")
        # except Exception as e:
        #     logger.warning(f"Failed to create SimFinClient: {e}")
        logger.info("SimFinClient not yet implemented, using repository only")

    # Always create repository for fallback/offline mode
    try:
        repository = FundamentalDataRepository(config.data_dir)
        logger.info(
            f"Created FundamentalDataRepository with data_dir: {config.data_dir}"
        )
    except Exception as e:
        logger.error(f"Failed to create FundamentalDataRepository: {e}")

    return FundamentalDataService(
        simfin_client=client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_insider_service(config: TradingAgentsConfig) -> InsiderDataService:
    """
    Build InsiderDataService with appropriate client and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        InsiderDataService: Configured service
    """
    client = None
    repository = None

    # Create client for online mode
    if config.online_tools:
        # Finnhub client for insider data
        if config.finnhub_api_key:
            try:
                client = FinnhubClient(config.finnhub_api_key)
                logger.info("Created FinnhubClient for live insider trading data")
            except Exception as e:
                logger.warning(f"Failed to create FinnhubClient for insider data: {e}")
        else:
            logger.info("No Finnhub API key provided, skipping insider data client")

    # Always create repository for fallback/offline mode
    try:
        repository = InsiderDataRepository(config.data_dir)
        logger.info(f"Created InsiderDataRepository with data_dir: {config.data_dir}")
    except Exception as e:
        logger.error(f"Failed to create InsiderDataRepository: {e}")

    return InsiderDataService(
        finnhub_client=client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_openai_service(config: TradingAgentsConfig) -> OpenAIDataService:
    """
    Build OpenAIDataService with appropriate client and repository.

    Args:
        config: TradingAgents configuration

    Returns:
        OpenAIDataService: Configured service
    """
    client = None
    repository = None

    # Create client for online mode
    if config.online_tools:
        # OpenAI client (would be implemented when OpenAIClient is available)
        # if config.openai_api_key:
        #     try:
        #         client = OpenAIClient(api_key=config.openai_api_key)
        #         logger.info("Created OpenAIClient for AI-powered analysis")
        #     except Exception as e:
        #         logger.warning(f"Failed to create OpenAIClient: {e}")
        # else:
        #     logger.info("No OpenAI API key provided, skipping OpenAI client")
        logger.info("OpenAIClient not yet implemented, using repository only")

    # Always create repository for fallback/offline mode
    try:
        repository = OpenAIRepository(config.data_dir)
        logger.info(f"Created OpenAIRepository with data_dir: {config.data_dir}")
    except Exception as e:
        logger.error(f"Failed to create OpenAIRepository: {e}")

    return OpenAIDataService(
        openai_client=client,
        repository=repository,
        online_mode=config.online_tools,
        data_dir=config.data_dir,
    )


def build_all_services(config: TradingAgentsConfig) -> dict:
    """
    Build all available services.

    Args:
        config: TradingAgents configuration

    Returns:
        dict: Dictionary of service name to service instance
    """
    services = {}

    # Build MarketDataService
    try:
        services["market_data"] = build_market_data_service(config)
        logger.info("Built MarketDataService")
    except Exception as e:
        logger.error(f"Failed to build MarketDataService: {e}")

    # Build NewsService
    try:
        services["news"] = build_news_service(config)
        logger.info("Built NewsService")
    except Exception as e:
        logger.error(f"Failed to build NewsService: {e}")

    # Build SocialMediaService
    try:
        services["social_media"] = build_social_media_service(config)
        logger.info("Built SocialMediaService")
    except Exception as e:
        logger.error(f"Failed to build SocialMediaService: {e}")

    # Build FundamentalDataService
    try:
        services["fundamental"] = build_fundamental_service(config)
        logger.info("Built FundamentalDataService")
    except Exception as e:
        logger.error(f"Failed to build FundamentalDataService: {e}")

    # Build InsiderDataService
    try:
        services["insider"] = build_insider_service(config)
        logger.info("Built InsiderDataService")
    except Exception as e:
        logger.error(f"Failed to build InsiderDataService: {e}")

    # Build OpenAIDataService
    try:
        services["openai"] = build_openai_service(config)
        logger.info("Built OpenAIDataService")
    except Exception as e:
        logger.error(f"Failed to build OpenAIDataService: {e}")

    logger.info(f"Built {len(services)} services: {list(services.keys())}")
    return services


def build_toolkit_services(config: TradingAgentsConfig) -> dict:
    """
    Build services specifically configured for Toolkit usage.

    Args:
        config: TradingAgents configuration

    Returns:
        dict: Dictionary of services for Toolkit
    """
    return build_all_services(config)


# Aliases for the service toolkit
create_market_data_service = build_market_data_service
create_news_service = build_news_service
create_social_media_service = build_social_media_service
create_fundamental_data_service = build_fundamental_service
create_insider_data_service = build_insider_service
create_openai_data_service = build_openai_service
