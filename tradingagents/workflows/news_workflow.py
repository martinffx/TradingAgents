"""News Processing Workflow for Temporal - imperative shell coordination."""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.workflows.activities.news_activities import NewsActivities


@workflow.defn
class NewsProcessingWorkflow:
    """Workflow for processing news articles with LLM enrichment.

    This is an imperative shell - thin coordination layer.
    All business logic lives in domain entities (NewsArticle).

    Temporal handles:
    - Activity retry with exponential backoff on 429
    - Timeout management
    - Activity heartbeats
    - Workflow state persistence
    - Activity result caching
    """

    @workflow.run
    async def run(self, article_id: str) -> dict:
        """Process single article end-to-end.

        Steps:
        1. Fetch article from repository
        2. Scrape full content
        3. Add scraped content to entity
        4. Analyze sentiment (with Temporal retry on 429)
        5. Add sentiment to entity
        6. Generate embeddings in parallel (with Temporal retry)
        7. Add embeddings to entity
        8. Save enriched article

        Args:
            article_id: UUID string of article to process.

        Returns:
            Dict with article_id and status.
        """
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=120),
            maximum_attempts=5,
        )

        article_data = await workflow.execute_activity_method(
            NewsActivities.fetch_article,
            article_id,
            start_to_close_timeout=timedelta(seconds=30),
        )

        if article_data is None:
            raise ValueError(f"Article {article_id} not found")

        article = NewsArticle.from_dict(article_data)

        scraped = await workflow.execute_activity_method(
            NewsActivities.scrape_article,
            article.url,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        article = article.with_content(
            summary=scraped["content"],
            author=scraped.get("author"),
        )

        sentiment = await workflow.execute_activity_method(
            NewsActivities.analyze_sentiment,
            article.summary,
            start_to_close_timeout=timedelta(seconds=90),
            retry_policy=retry_policy,
        )

        article = article.with_sentiment(
            label=sentiment["sentiment"],
            confidence=sentiment["confidence"],
        )

        title_emb, content_emb = await asyncio.gather(
            workflow.execute_activity_method(
                NewsActivities.create_embedding,
                article.headline,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=retry_policy,
            ),
            workflow.execute_activity_method(
                NewsActivities.create_embedding,
                article.summary,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=retry_policy,
            ),
        )

        article = article.with_embeddings(title=title_emb, content=content_emb)

        saved_id = await workflow.execute_activity_method(
            NewsActivities.save_article,
            article.to_dict(),
            start_to_close_timeout=timedelta(seconds=30),
        )

        return {"article_id": saved_id, "status": "completed"}


@workflow.defn
class BatchNewsProcessingWorkflow:
    """Workflow for batch processing multiple news articles.

    Processes articles in parallel using asyncio.gather.
    Each article processing is independent and can run concurrently.
    """

    @workflow.run
    async def run(self, article_ids: list[str]) -> dict:
        """Process multiple articles in parallel.

        Args:
            article_ids: List of article UUID strings.

        Returns:
            Dict with success count and failed IDs.
        """
        tasks = [
            workflow.execute_child_workflow(
                NewsProcessingWorkflow.run,
                article_id,
            )
            for article_id in article_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = 0
        failed = []
        for article_id, result in zip(article_ids, results, strict=True):
            if isinstance(result, Exception):
                failed.append({"article_id": article_id, "error": str(result)})
            else:
                success += 1

        return {
            "total": len(article_ids),
            "success": success,
            "failed": failed,
        }
