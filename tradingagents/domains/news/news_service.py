"""
News service that provides structured news context.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.news.google_news_client import (
    GoogleNewsClient,
)
from tradingagents.domains.news.news_repository import NewsArticle, NewsRepository
from tradingagents.lib.llm_client import LLMClient

from .article_scraper_client import ArticleScraperClient

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels for news data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SentimentScore:
    """Sentiment analysis score."""

    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    label: str  # positive/negative/neutral


@dataclass
class ArticleData:
    """News article data."""

    title: str
    content: str
    author: str
    source: str
    date: str  # YYYY-MM-DD format
    url: str
    sentiment: SentimentScore | None = None


@dataclass
class NewsContext:
    """News context for trading analysis."""

    query: str | None
    symbol: str | None
    categories: list[str] | None
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    articles: list[ArticleData]
    article_count: int
    sources: list[str]
    metadata: dict[str, Any]


@dataclass
class NewsUpdateResult:
    """Result of news update operation."""

    status: str
    articles_found: int
    articles_scraped: int
    articles_failed: int
    symbol: str | None = None
    categories: list[str] | None = None
    date_range: dict[str, str] | None = None


class NewsService:
    """Service for news data and sentiment analysis."""

    def __init__(
        self,
        google_client: GoogleNewsClient,
        repository: NewsRepository,
        article_scraper: ArticleScraperClient,
        llm_client: LLMClient,
    ):
        """
        Initialize news service.

        Args:
            google_client: Client for Google News data
            repository: Repository for cached news data
            article_scraper: Client for scraping article content
            openrouter_client: Client for LLM sentiment analysis (required)
        """
        self._google_client = google_client
        self._repository = repository
        self._article_scraper = article_scraper
        self._llm_client = llm_client

    @staticmethod
    def build(database_manager, config: TradingAgentsConfig):
        google_client = GoogleNewsClient()
        repository = NewsRepository(database_manager)
        article_scraper = ArticleScraperClient("")
        llm_client = LLMClient(config)

        return NewsService(google_client, repository, article_scraper, llm_client)

    async def get_company_news_context(
        self, symbol: str, start_date: str, end_date: str
    ) -> NewsContext:
        """
        Get news context specific to a company from repository (no API calls).

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            NewsContext: Company-specific news context
        """
        logger.info(f"Getting company news context for {symbol} from repository")

        # Convert date strings to date objects
        start_date_obj = date.fromisoformat(start_date)
        end_date_obj = date.fromisoformat(end_date)

        # Get articles directly from repository
        news_articles = await self._repository.list_by_date_range(
            symbol=symbol,
            start_date=start_date_obj,
            end_date=end_date_obj,
        )

        articles = [
            ArticleData(
                title=article.headline,
                content=article.summary or "",  # Use summary as fallback for content
                author=article.author or "",
                source=article.source,
                date=article.published_date.isoformat(),
                url=article.url,
                sentiment=None,  # Will be calculated later
            )
            for article in news_articles
        ]

        # Extract unique sources
        sources = list(
            {article.source for article in articles if hasattr(article, "source")}
        )

        return NewsContext(
            query=symbol,
            symbol=symbol,
            categories=None,
            period={"start": start_date, "end": end_date},
            articles=articles,
            article_count=len(articles),
            sources=sources,
            metadata={
                "service": "news",
                "data_source": "repository",
                "method": "get_company_news_context",
            },
        )

    async def get_global_news_context(
        self,
        start_date: str,
        end_date: str,
        categories: list[str] | None = None,
    ) -> NewsContext:
        """
        Get global/macro news context from repository (no API calls).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            categories: News categories to search

        Returns:
            GlobalNewsContext: Global news context
        """
        if categories is None:
            categories = ["general", "business", "politics"]

        logger.info(
            f"Getting global news context from repository for categories: {categories}"
        )

        # Convert date strings to date objects
        start_date_obj = date.fromisoformat(start_date)
        end_date_obj = date.fromisoformat(end_date)

        # Get articles for each category
        articles = []
        for category in categories:
            news_articles = await self._repository.list_by_date_range(
                symbol=category,  # Use category as symbol for global news
                start_date=start_date_obj,
                end_date=end_date_obj,
            )

            # Convert NewsArticle objects to ArticleData objects
            for article in news_articles:
                articles.append(
                    ArticleData(
                        title=article.headline,
                        content=article.summary or "",
                        author=article.author or "",
                        source=article.source,
                        date=article.published_date.isoformat(),
                        url=article.url,
                        sentiment=None,
                    )
                )

        # Extract unique sources
        sources = list(
            {article.source for article in articles if hasattr(article, "source")}
        )

        return NewsContext(
            query=None,
            symbol=None,
            period={"start": start_date, "end": end_date},
            categories=categories,
            articles=articles,
            article_count=len(articles),
            sources=sources,
            metadata={
                "service": "news",
                "data_source": "repository",
                "method": "get_global_news_context",
            },
        )

    async def fetch_articles(self, symbol: str) -> list[NewsArticle]:
        """
        Fetch RSS feeds from Google News and save basic article metadata to repository.

        Args:
            symbol: Stock ticker symbol

        Returns:
            list[GoogleNewsArticle]: List of Google News articles with basic metadata
        """
        try:
            logger.info(f"Fetching RSS feeds for {symbol}")

            if not self._google_client:
                raise ValueError("Google client not configured")

            # 1. Get RSS feed data
            google_articles = self._google_client.get_company_news(symbol)

            if not google_articles:
                logger.warning(f"No articles found in RSS feed for {symbol}")
                return []

            # 2. Save basic article metadata to repository immediately
            logger.info(f"Saving {len(google_articles)} basic articles for {symbol}")

            # Convert GoogleNewsArticle to NewsArticle for repository storage
            news_articles = [
                NewsArticle(
                    headline=google_article.title,
                    url=google_article.link,
                    source=google_article.source,
                    published_date=google_article.published.date(),  # Convert datetime to date
                    summary=google_article.summary,  # Use RSS summary as initial content
                    author="",  # No author info from RSS
                )
                for google_article in google_articles
            ]

            # Store all articles in batch with basic metadata
            await self._repository.upsert_batch(news_articles, symbol)

            return news_articles
        except Exception as e:
            logger.error(f"Error fetching RSS feeds for {symbol}: {e}")
            raise

    async def process_article(self, article_id: uuid.UUID) -> NewsArticle:
        """
        Process a single article by scraping content, running LLM sentiment analysis,
        generating embeddings, and saving the enriched data.

        Args:
            article_id: UUID of the article to process

        Returns:
            NewsArticle: The processed article with enriched data
        """
        try:
            logger.info(f"Processing article {article_id}")

            # 1. Get article from repository
            article = await self._repository.get(article_id)
            if not article:
                raise ValueError(f"Article with ID {article_id} not found")

            if not article.url:
                raise ValueError(f"Article {article_id} has no URL to process")

            # 2. Scrape article content
            logger.info(f"Scraping article content from {article.url}")
            scrape_result = self._article_scraper.scrape_article(article.url)

            if scrape_result.status not in ["SUCCESS", "ARCHIVE_SUCCESS"]:
                logger.warning(
                    f"Failed to scrape article {article.url}: {scrape_result.status}"
                )
                # Keep existing data but mark as failed
                return article

            # 3. Run LLM-based sentiment analysis
            logger.info("Running LLM sentiment analysis")
            try:
                llm_sentiment = await self._llm_client.analyze_sentiment(
                    scrape_result.content
                )

                # Convert to our sentiment format
                sentiment_score = (
                    llm_sentiment.confidence
                    if llm_sentiment.sentiment == "positive"
                    else -llm_sentiment.confidence
                    if llm_sentiment.sentiment == "negative"
                    else 0.0
                )

                # Update article with sentiment data
                article = NewsArticle(
                    headline=article.headline,
                    url=article.url,
                    source=article.source,
                    published_date=article.published_date,
                    summary=article.summary,
                    entities=article.entities,
                    sentiment_score=sentiment_score,
                    sentiment_confidence=llm_sentiment.confidence,
                    sentiment_label=llm_sentiment.sentiment,
                    author=article.author,
                    category=article.category,
                    title_embedding=article.title_embedding,
                    content_embedding=article.content_embedding,
                )
            except Exception as e:
                logger.error(
                    f"LLM sentiment analysis failed for article {article.url}: {e}"
                )
                # Continue without sentiment data

            # 4. Generate vector embeddings
            logger.info("Generating vector embeddings")
            try:
                (
                    title_embedding,
                    content_embedding,
                ) = await self._generate_article_embeddings(
                    scrape_result.title or article.headline, scrape_result.content
                )
                # Update article with embedding data
                article = NewsArticle(
                    headline=article.headline,
                    url=article.url,
                    source=article.source,
                    published_date=article.published_date,
                    summary=article.summary,
                    entities=article.entities,
                    sentiment_score=article.sentiment_score,
                    sentiment_confidence=article.sentiment_confidence,
                    sentiment_label=article.sentiment_label,
                    author=article.author,
                    category=article.category,
                    title_embedding=title_embedding,
                    content_embedding=content_embedding,
                )
            except Exception as e:
                logger.error(
                    f"Embedding generation failed for article {article.url}: {e}"
                )
                # Continue without embeddings

            # 5. Update article with scraped content
            article = NewsArticle(
                headline=scrape_result.title or article.headline,
                url=article.url,
                source=article.source,
                published_date=article.published_date,
                summary=scrape_result.content,
                entities=article.entities,
                sentiment_score=article.sentiment_score,
                sentiment_confidence=article.sentiment_confidence,
                sentiment_label=article.sentiment_label,
                author=article.author,
                category=article.category,
                title_embedding=article.title_embedding,
                content_embedding=article.content_embedding,
            )
            article.author = scrape_result.author
            # Keep the published_date from RSS if we don't have a better one from scraping
            if scrape_result.publish_date:
                try:
                    from datetime import date as date_class

                    article.published_date = date_class.fromisoformat(
                        scrape_result.publish_date
                    )
                except Exception:
                    pass  # Keep existing date if parsing fails

            # 6. Save updated article
            logger.info(f"Saving processed article {article_id}")
            # Note: We need to determine the symbol/category for upsert
            symbol_or_category = article.category or ""
            updated_article = await self._repository.upsert(article, symbol_or_category)

            logger.info(f"Successfully processed article {article_id}")
            return updated_article

        except Exception as e:
            logger.error(f"Error processing article {article_id}: {e}")
            raise

    async def _generate_article_embeddings(
        self, title: str, content: str
    ) -> tuple[list[float], list[float]]:
        """
        Generate vector embeddings for article title and content.

        Args:
            title: Article title
            content: Article content

        Returns:
            Tuple of (title_embedding, content_embedding)
        """
        try:
            title_embedding = await self._llm_client.create_embedding(title)
            content_embedding = await self._llm_client.create_embedding(content)
            return title_embedding, content_embedding
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
