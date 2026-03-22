"""News domain activities for Temporal workflow."""

import logging
from datetime import date
from uuid import UUID

from temporalio import activity

from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.domains.news.news_service import NewsService

logger = logging.getLogger(__name__)


@activity.defn
class NewsActivities:
    """News domain activities with constructor-injected NewsService.

    This class is instantiated by the worker with dependencies, then
    Temporal calls the decorated methods for each activity invocation.
    """

    def __init__(self, news_service: NewsService) -> None:
        """Initialize with injected NewsService.

        Args:
            news_service: NewsService with repository, scraper, and LLM client.
        """
        self._news_service = news_service

    @activity.defn
    async def fetch_article(self, article_id: str) -> dict | None:
        """Fetch article from repository by ID."""
        logger.info("fetch_article started", extra={"article_id": article_id})
        try:
            article = await self._news_service.repository.get(UUID(article_id))
            if article is None:
                logger.warning("Article not found", extra={"article_id": article_id})
                return None
            logger.info("fetch_article completed")
            return article.to_dict()
        except ValueError:
            logger.error("Invalid UUID format", extra={"article_id": article_id})
            return None
        except Exception:
            logger.exception("fetch_article failed", extra={"article_id": article_id})
            raise

    @activity.defn
    async def scrape_article(self, url: str) -> dict:
        """Scrape article content from URL."""
        logger.info("scrape_article started", extra={"url": url[:100]})
        try:
            result = self._news_service.scraper.scrape_article(url)
            logger.info("scrape_article completed", extra={"status": result.status})
            return {
                "content": result.content,
                "title": result.title,
                "author": result.author,
                "publish_date": result.publish_date,
            }
        except Exception:
            logger.exception("scrape_article failed", extra={"url": url[:100]})
            raise

    @activity.defn
    async def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using LLM."""
        logger.info("analyze_sentiment started", extra={"text_length": len(text)})
        try:
            result = await self._news_service.llm_client.analyze_sentiment(text)
            logger.info(
                "analyze_sentiment completed",
                extra={"sentiment": result.sentiment, "confidence": result.confidence},
            )
            return {
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }
        except Exception:
            logger.exception("analyze_sentiment failed")
            raise

    @activity.defn
    async def create_embedding(self, text: str) -> list[float]:
        """Create vector embedding for text."""
        logger.info("create_embedding started", extra={"text_length": len(text)})
        try:
            result = await self._news_service.llm_client.create_embedding(text)
            logger.info(
                "create_embedding completed", extra={"embedding_length": len(result)}
            )
            return result
        except Exception:
            logger.exception("create_embedding failed")
            raise

    @activity.defn
    async def save_article(self, article_data: dict) -> str:
        """Save processed article to repository."""
        logger.info("save_article started")
        try:
            article = NewsArticle.from_dict(article_data)
            saved = await self._news_service.repository.upsert(article)
            logger.info("save_article completed", extra={"article_id": str(saved.id)})
            return str(saved.id)
        except Exception:
            logger.exception("save_article failed")
            raise

    @activity.defn
    async def list_articles_for_processing(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """List articles that need processing."""
        logger.info(
            "list_articles_for_processing started",
            extra={"symbol": symbol, "limit": limit},
        )
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)

            articles = await self._news_service.repository.list_by_date_range(
                symbol=symbol,
                start_date=start,
                end_date=end,
                limit=limit,
            )

            logger.info(
                "list_articles_for_processing completed", extra={"count": len(articles)}
            )
            return [a.to_dict() for a in articles]
        except Exception:
            logger.exception("list_articles_for_processing failed")
            raise
