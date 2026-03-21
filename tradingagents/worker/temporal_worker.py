"""Temporal worker setup with dependency injection."""

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from tradingagents.activities.llm_activities import LLMActivities
from tradingagents.activities.news_activities import NewsActivities
from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.news.article_scraper_client import ArticleScraperClient
from tradingagents.domains.news.news_repository import NewsRepository
from tradingagents.lib.database import DatabaseManager
from tradingagents.lib.llm_client import LLMClient
from tradingagents.workflows.news_workflow import (
    BatchNewsProcessingWorkflow,
    NewsProcessingWorkflow,
)

logger = logging.getLogger(__name__)

NEWS_PROCESSING_TASK_QUEUE = "news-processing"


async def create_llm_activities(config: TradingAgentsConfig) -> LLMActivities:
    """Create LLM activities with injected dependencies.

    Args:
        config: Application configuration.

    Returns:
        Configured LLMActivities instance.
    """
    llm_client = LLMClient(config)
    return LLMActivities(llm_client)


async def create_news_activities(
    config: TradingAgentsConfig,
) -> NewsActivities:
    """Create News activities with injected dependencies.

    Args:
        config: Application configuration.

    Returns:
        Configured NewsActivities instance.
    """
    db_manager = DatabaseManager(config.database_url)
    repository = NewsRepository(db_manager)
    scraper = ArticleScraperClient("")
    return NewsActivities(repository, scraper)


async def run_worker(
    config: TradingAgentsConfig,
    host: str = "localhost",
    port: int = 7233,
) -> None:
    """Run Temporal worker with dependency injection.

    Args:
        config: Application configuration.
        host: Temporal server host.
        port: Temporal server port.
    """
    temporal_address = f"{host}:{port}"
    logger.info(f"Connecting to Temporal at {temporal_address}")

    client = await Client.connect(temporal_address)

    logger.info("Creating activities with dependency injection")
    llm_activities = await create_llm_activities(config)
    news_activities = await create_news_activities(config)

    logger.info(f"Starting worker on task queue: {NEWS_PROCESSING_TASK_QUEUE}")
    worker = Worker(
        client,
        task_queue=NEWS_PROCESSING_TASK_QUEUE,
        workflows=[NewsProcessingWorkflow, BatchNewsProcessingWorkflow],
        activities=[llm_activities, news_activities],
    )

    logger.info("Worker started, awaiting tasks...")
    await worker.run()


async def main() -> None:
    """Main entry point for running worker."""
    logging.basicConfig(level=logging.INFO)

    config = TradingAgentsConfig.from_env()
    await run_worker(config)


if __name__ == "__main__":
    asyncio.run(main())
