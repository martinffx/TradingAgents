"""News domain activities for Temporal workflow."""

from datetime import date
from uuid import UUID

from temporalio import activity

from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.domains.news.news_service import NewsService

_news_service: NewsService | None = None


def _get_news_service() -> NewsService:
    """Get the injected NewsService or raise if not configured."""
    if _news_service is None:
        raise RuntimeError("NewsService not injected. Call create_news_activities first.")
    return _news_service


@activity.defn
class NewsActivities:
    """All news-domain activities including LLM operations.

    Thin Temporal shell over NewsService.
    NewsService provides: repository, scraper, llm_client.
    """

    def __init__(self) -> None:
        """Initialize - NewsService injected via module-level state."""
        pass

    @activity.defn
    async def fetch_article(self, article_id: str) -> dict | None:
        """Fetch article from repository by ID.

        Args:
            article_id: UUID string of article to fetch.

        Returns:
            Article as dict, or None if not found.
        """
        news_service = _get_news_service()
        article = await news_service.repository.get(UUID(article_id))
        if article is None:
            return None
        return article.to_dict()

    @activity.defn
    async def scrape_article(self, url: str) -> dict:
        """Scrape article content from URL.

        Args:
            url: URL of article to scrape.

        Returns:
            Dict with content, title, author, publish_date.
        """
        news_service = _get_news_service()
        result = news_service.scraper.scrape_article(url)
        return {
            "content": result.content,
            "title": result.title,
            "author": result.author,
            "publish_date": result.publish_date,
        }

    @activity.defn
    async def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using LLM.

        Temporal retry policy handles 429 rate limits automatically.

        Args:
            text: Text content to analyze.

        Returns:
            Dict with sentiment label, confidence, and reasoning.
        """
        news_service = _get_news_service()
        result = await news_service.llm_client.analyze_sentiment(text)
        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    @activity.defn
    async def create_embedding(self, text: str) -> list[float]:
        """Create vector embedding for text.

        Temporal retry policy handles 429 rate limits automatically.

        Args:
            text: Text content to embed.

        Returns:
            List of float values representing embedding vector.
        """
        news_service = _get_news_service()
        return await news_service.llm_client.create_embedding(text)

    @activity.defn
    async def save_article(self, article_data: dict) -> str:
        """Save processed article to repository.

        Args:
            article_data: Article dict with all fields.

        Returns:
            Saved article ID as string.
        """
        news_service = _get_news_service()
        article = NewsArticle.from_dict(article_data)
        saved = await news_service.repository.upsert(article)
        return str(saved.id)

    @activity.defn
    async def list_articles_for_processing(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> list[dict]:
        """List articles that need processing.

        Args:
            symbol: Stock symbol to filter by.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            limit: Maximum articles to return.

        Returns:
            List of article dicts needing enrichment.
        """
        news_service = _get_news_service()
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        articles = await news_service.repository.list_by_date_range(
            symbol=symbol,
            start_date=start,
            end_date=end,
            limit=limit,
        )

        return [a.to_dict() for a in articles]
