"""Temporal worker setup with dependency injection."""

import asyncio
import logging
import signal
from collections.abc import Callable, Sequence
from typing import cast

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


def create_news_service(config: TradingAgentsConfig) -> NewsService:
    """Create NewsService with injected dependencies.

    Args:
        config: Application configuration.

    Returns:
        Configured NewsService instance.
    """
    db_manager = DatabaseManager(config.database_url)
    return NewsService.build(db_manager, config)


def create_news_activities(news_service: NewsService) -> NewsActivities:
    """Create NewsActivities with constructor-injected NewsService.

    Args:
        news_service: Configured NewsService with repository, scraper, and LLM.

    Returns:
        NewsActivities instance with injected dependencies.
    """
    return NewsActivities(news_service)


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
    shutdown_event = asyncio.Event()

    def handle_shutdown() -> None:
        logger.info("Shutdown signal received, draining worker...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown)

    temporal_address = f"{host}:{port}"
    logger.info(f"Connecting to Temporal at {temporal_address}")

    client = await Client.connect(temporal_address)

    logger.info("Creating services with dependency injection")
    news_service = create_news_service(config)
    news_activities = create_news_activities(news_service)

    logger.info(f"Starting worker on task queue: {NEWS_PROCESSING_TASK_QUEUE}")
    worker = Worker(
        client,
        task_queue=NEWS_PROCESSING_TASK_QUEUE,
        workflows=[NewsProcessingWorkflow, BatchNewsProcessingWorkflow],
        activities=cast("Sequence[Callable[..., object]]", [news_activities]),
    )

    logger.info("Worker started, awaiting tasks...")

    async with worker:
        await shutdown_event.wait()

    logger.info("Worker shutdown complete")


async def main() -> None:
    """Main entry point for running worker."""
    logging.basicConfig(level=logging.INFO)

    config = TradingAgentsConfig.from_env()
    await run_worker(config)


if __name__ == "__main__":
    asyncio.run(main())
