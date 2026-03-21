"""Temporal worker setup with dependency injection."""

import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.news.news_service import NewsService
from tradingagents.lib.database import DatabaseManager
from tradingagents.workflows.activities.news_activities import NewsActivities
from tradingagents.workflows.news_workflow import (
    BatchNewsProcessingWorkflow,
    NewsProcessingWorkflow,
)

logger = logging.getLogger(__name__)

NEWS_PROCESSING_TASK_QUEUE = "news-processing"


async def create_news_service(config: TradingAgentsConfig) -> NewsService:
    """Create NewsService with injected dependencies.

    Args:
        config: Application configuration.

    Returns:
        Configured NewsService instance.
    """
    db_manager = DatabaseManager(config.database_url)
    return NewsService.build(db_manager, config)


async def create_news_activities(
    news_service: NewsService,
) -> type[NewsActivities]:
    """Inject NewsService and return NewsActivities class for worker.

    Args:
        news_service: Configured NewsService with repository, scraper, and LLM.

    Returns:
        NewsActivities class ready for worker registration.
    """
    import tradingagents.workflows.activities.news_activities as activities_module

    activities_module._news_service = news_service
    return NewsActivities


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

    logger.info("Creating services with dependency injection")
    news_service = await create_news_service(config)
    news_activities = await create_news_activities(news_service)

    logger.info(f"Starting worker on task queue: {NEWS_PROCESSING_TASK_QUEUE}")
    worker = Worker(
        client,
        task_queue=NEWS_PROCESSING_TASK_QUEUE,
        workflows=[NewsProcessingWorkflow, BatchNewsProcessingWorkflow],
        activities=[news_activities],
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
