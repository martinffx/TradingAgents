"""News Activities for Temporal workflow - thin shell over repository and scraper."""

from uuid import UUID

from temporalio import activity

from tradingagents.domains.news.article_scraper_client import ArticleScraperClient
from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.domains.news.news_repository import NewsRepository


@activity.defn
class NewsActivities:
    """Activity class for news operations with injected dependencies.

    This is an imperative shell - thin wrapper over repository and scraper.
    All business logic lives in domain entities.

    Temporal handles:
    - Retry on transient failures
    - Timeout management
    - Activity heartbeats
    """

    def __init__(
        self,
        repository: NewsRepository,
        scraper: ArticleScraperClient,
    ):
        """Initialize with repository and scraper dependencies.

        Args:
            repository: Injected news repository for persistence.
            scraper: Injected article scraper client.
        """
        self._repository = repository
        self._scraper = scraper

    @activity.defn
    async def fetch_article(self, article_id: str) -> dict | None:
        """Fetch article from repository by ID.

        Args:
            article_id: UUID string of article to fetch.

        Returns:
            Article as dict, or None if not found.
        """
        article = await self._repository.get(UUID(article_id))
        if article is None:
            return None
        return NewsArticle.from_record(article).to_dict()

    @activity.defn
    async def scrape_article(self, url: str) -> dict:
        """Scrape article content from URL.

        Args:
            url: URL of article to scrape.

        Returns:
            Dict with content, title, author, publish_date.
        """
        result = self._scraper.scrape_article(url)
        return {
            "content": result.content,
            "title": result.title,
            "author": result.author,
            "publish_date": result.publish_date,
        }

    @activity.defn
    async def save_article(self, article_data: dict) -> str:
        """Save processed article to repository.

        Args:
            article_data: Article dict with all fields.

        Returns:
            Saved article ID as string.
        """
        article = NewsArticle.from_dict(article_data)
        saved = await self._repository.upsert(
            article,
            symbol=article_data.get("symbol"),
        )
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
        from datetime import date

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        articles = await self._repository.list_by_date_range(
            symbol=symbol,
            start_date=start,
            end_date=end,
            limit=limit,
        )

        return [NewsArticle.from_record(a).to_dict() for a in articles]
