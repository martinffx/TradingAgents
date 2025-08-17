"""
News service that provides structured news context.
"""

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.news.google_news_client import GoogleNewsClient
from tradingagents.domains.news.news_repository import NewsArticle, NewsRepository

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

    query: str
    symbol: str | None
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    articles: list[ArticleData]
    sentiment_summary: SentimentScore
    article_count: int
    sources: list[str]
    metadata: dict[str, Any]


@dataclass
class GlobalNewsContext:
    """Global news context for macro analysis."""

    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    categories: list[str]
    articles: list[ArticleData]
    sentiment_summary: SentimentScore
    article_count: int
    sources: list[str]
    trending_topics: list[str]
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
    ):
        """
        Initialize news service.

        Args:
            google_client: Client for Google News data
            repository: Repository for cached news data
            article_scraper: Client for scraping article content
        """
        self.google_client = google_client
        self.repository = repository
        self.article_scraper = article_scraper

    @staticmethod
    def build(database_manager, _config: TradingAgentsConfig):
        google_client = GoogleNewsClient()
        repository = NewsRepository(database_manager)
        article_scraper = ArticleScraperClient("")
        return NewsService(google_client, repository, article_scraper)

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
        try:
            logger.info(f"Getting company news context for {symbol} from repository")

            # Get articles from repository (READ PATH - no API calls)
            articles = []
            if self.repository:
                try:
                    # Convert date strings to date objects
                    start_date_obj = date.fromisoformat(start_date)
                    end_date_obj = date.fromisoformat(end_date)

                    # Get articles directly from repository
                    news_articles = await self.repository.list_by_date_range(
                        symbol=symbol,
                        start_date=start_date_obj,
                        end_date=end_date_obj,
                    )

                    # Convert NewsArticle objects to ArticleData objects
                    for article in news_articles:
                        articles.append(
                            ArticleData(
                                title=article.headline,
                                content=article.summary
                                or "",  # Use summary as fallback for content
                                author=article.author or "",
                                source=article.source,
                                date=article.published_date.isoformat(),
                                url=article.url,
                                sentiment=None,  # Will be calculated later
                            )
                        )

                    logger.debug(
                        f"Retrieved {len(articles)} articles from repository for {symbol}"
                    )
                except Exception as e:
                    logger.warning(f"Error retrieving articles from repository: {e}")
                    articles = []

            # Calculate sentiment summary from articles
            sentiment_summary = self._calculate_sentiment_summary(articles)

            # Extract unique sources
            sources = list(
                {article.source for article in articles if hasattr(article, "source")}
            )

            return NewsContext(
                query=symbol,
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                articles=articles,
                sentiment_summary=sentiment_summary,
                article_count=len(articles),
                sources=sources,
                metadata={
                    "service": "news",
                    "data_source": "repository",
                    "method": "get_company_news_context",
                },
            )

        except Exception as e:
            logger.error(f"Error getting company news context for {symbol}: {e}")
            # Return empty context on error
            return NewsContext(
                query=symbol,
                symbol=symbol,
                period={"start": start_date, "end": end_date},
                articles=[],
                sentiment_summary=SentimentScore(
                    score=0.0, confidence=0.0, label="neutral"
                ),
                article_count=0,
                sources=[],
                metadata={
                    "service": "news",
                    "data_source": "repository",
                    "error": str(e),
                },
            )

    async def get_global_news_context(
        self,
        start_date: str,
        end_date: str,
        categories: list[str] | None = None,
    ) -> GlobalNewsContext:
        """
        Get global/macro news context from repository (no API calls).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            categories: News categories to search

        Returns:
            GlobalNewsContext: Global news context
        """
        try:
            if categories is None:
                categories = ["general", "business", "politics"]

            logger.info(
                f"Getting global news context from repository for categories: {categories}"
            )

            # Get articles from repository (READ PATH - no API calls)
            articles = []
            if self.repository:
                try:
                    # Convert date strings to date objects
                    start_date_obj = date.fromisoformat(start_date)
                    end_date_obj = date.fromisoformat(end_date)

                    # Get articles for each category
                    for category in categories:
                        news_articles = await self.repository.list_by_date_range(
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

                    logger.debug(
                        f"Retrieved {len(articles)} global articles from repository"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error retrieving global articles from repository: {e}"
                    )
                    articles = []

            # Calculate sentiment summary from articles
            sentiment_summary = self._calculate_sentiment_summary(articles)

            # Extract unique sources
            sources = list(
                {article.source for article in articles if hasattr(article, "source")}
            )

            # Extract trending topics (simplified implementation)
            trending_topics = self._extract_trending_topics(articles)

            return GlobalNewsContext(
                period={"start": start_date, "end": end_date},
                categories=categories,
                articles=articles,
                sentiment_summary=sentiment_summary,
                article_count=len(articles),
                sources=sources,
                trending_topics=trending_topics,
                metadata={
                    "service": "news",
                    "data_source": "repository",
                    "method": "get_global_news_context",
                },
            )

        except Exception as e:
            logger.error(f"Error getting global news context: {e}")
            # Return empty context on error
            return GlobalNewsContext(
                period={"start": start_date, "end": end_date},
                categories=categories or [],
                articles=[],
                sentiment_summary=SentimentScore(
                    score=0.0, confidence=0.0, label="neutral"
                ),
                article_count=0,
                sources=[],
                trending_topics=[],
                metadata={
                    "service": "news",
                    "data_source": "repository",
                    "error": str(e),
                },
            )

    async def update_company_news(self, symbol: str) -> NewsUpdateResult:
        """
        Update company news by fetching RSS feeds and scraping article content.

        Args:
            symbol: Stock ticker symbol

        Returns:
            NewsUpdateResult with update status and statistics
        """
        try:
            logger.info(f"Updating company news for {symbol}")

            if not self.google_client:
                raise ValueError("Google client not configured")

            # 1. Get RSS feed data
            google_articles = self.google_client.get_company_news(symbol)

            if not google_articles:
                logger.warning(f"No articles found in RSS feed for {symbol}")
                return NewsUpdateResult(
                    status="completed",
                    articles_found=0,
                    articles_scraped=0,
                    articles_failed=0,
                    symbol=symbol,
                )

            # 2. Scrape each article content and convert to ArticleData
            article_data_list = []
            scraped_count = 0
            failed_count = 0

            for i, google_article in enumerate(google_articles):
                if not google_article.link:
                    failed_count += 1
                    continue

                logger.info(
                    f"Scraping article {i + 1}/{len(google_articles)}: {google_article.link}"
                )
                scrape_result = self.article_scraper.scrape_article(google_article.link)

                # Create ArticleData with scraped content
                if scrape_result.status in ["SUCCESS", "ARCHIVE_SUCCESS"]:
                    article_data = ArticleData(
                        title=scrape_result.title or google_article.title,
                        content=scrape_result.content,
                        author=scrape_result.author,
                        source=google_article.source,
                        date=scrape_result.publish_date
                        or google_article.published.strftime("%Y-%m-%d"),
                        url=google_article.link,
                        sentiment=None,  # Will be calculated later
                    )
                    scraped_count += 1
                else:
                    # Create ArticleData with just RSS data if scraping failed
                    article_data = ArticleData(
                        title=google_article.title,
                        content=google_article.summary,  # Use summary as fallback content
                        author="",
                        source=google_article.source,
                        date=google_article.published.strftime("%Y-%m-%d"),
                        url=google_article.link,
                        sentiment=None,
                    )
                    failed_count += 1

                article_data_list.append(article_data)

            # 3. Store in repository
            try:
                logger.info(f"Storing {len(article_data_list)} articles for {symbol}")

                # Convert ArticleData to NewsArticle for repository storage
                news_articles = []
                for article_data in article_data_list:
                    news_article = NewsArticle(
                        headline=article_data.title,
                        url=article_data.url,
                        source=article_data.source,
                        published_date=date.fromisoformat(article_data.date),
                        summary=article_data.content,
                        author=article_data.author,
                    )
                    news_articles.append(news_article)

                # Store all articles in batch
                await self.repository.upsert_batch(news_articles, symbol)

            except Exception as e:
                logger.error(f"Error storing articles in repository: {e}")

            logger.info(
                f"Company news update completed for {symbol}: {scraped_count} scraped, {failed_count} failed"
            )

            return NewsUpdateResult(
                status="completed",
                articles_found=len(google_articles),
                articles_scraped=scraped_count,
                articles_failed=failed_count,
                symbol=symbol,
            )

        except Exception as e:
            logger.error(f"Error updating company news for {symbol}: {e}")
            raise

    async def update_global_news(
        self, start_date: str, end_date: str, categories: list[str] | None = None
    ) -> NewsUpdateResult:
        """
        Update global/macro news by fetching RSS feeds and scraping article content.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            categories: List of news categories to search

        Returns:
            NewsUpdateResult with update status and statistics
        """
        try:
            if categories is None:
                categories = ["general", "business", "politics"]

            logger.info(
                f"Updating global news from {start_date} to {end_date} for categories: {categories}"
            )

            if not self.google_client:
                raise ValueError("Google client not configured")

            # 1. Get RSS feed data for all categories
            google_articles = self.google_client.get_global_news(categories)

            if not google_articles:
                logger.warning("No articles found in RSS feeds for global news")
                return NewsUpdateResult(
                    status="completed",
                    articles_found=0,
                    articles_scraped=0,
                    articles_failed=0,
                    categories=categories,
                    date_range={"start": start_date, "end": end_date},
                )

            # 2. Scrape each article content and convert to ArticleData
            article_data_list = []
            scraped_count = 0
            failed_count = 0

            for i, google_article in enumerate(google_articles):
                if not google_article.link:
                    failed_count += 1
                    continue

                logger.info(
                    f"Scraping global article {i + 1}/{len(google_articles)}: {google_article.link}"
                )
                scrape_result = self.article_scraper.scrape_article(google_article.link)

                # Create ArticleData with scraped content
                if scrape_result.status in ["SUCCESS", "ARCHIVE_SUCCESS"]:
                    article_data = ArticleData(
                        title=scrape_result.title or google_article.title,
                        content=scrape_result.content,
                        author=scrape_result.author,
                        source=google_article.source,
                        date=scrape_result.publish_date
                        or google_article.published.strftime("%Y-%m-%d"),
                        url=google_article.link,
                        sentiment=None,  # Will be calculated later
                    )
                    scraped_count += 1
                else:
                    # Create ArticleData with just RSS data if scraping failed
                    article_data = ArticleData(
                        title=google_article.title,
                        content=google_article.summary,  # Use summary as fallback content
                        author="",
                        source=google_article.source,
                        date=google_article.published.strftime("%Y-%m-%d"),
                        url=google_article.link,
                        sentiment=None,
                    )
                    failed_count += 1

                article_data_list.append(article_data)

            # 3. Store in repository
            try:
                logger.info(f"Storing {len(article_data_list)} global articles")

                # Convert ArticleData to NewsArticle for repository storage
                news_articles = []
                for article_data in article_data_list:
                    news_article = NewsArticle(
                        headline=article_data.title,
                        url=article_data.url,
                        source=article_data.source,
                        published_date=date.fromisoformat(article_data.date),
                        summary=article_data.content,
                        author=article_data.author,
                        category="global",  # Mark as global news
                    )
                    news_articles.append(news_article)

                # Store all articles in batch (use "global" as symbol for global news)
                await self.repository.upsert_batch(news_articles, "global")

            except Exception as e:
                logger.error(f"Error storing global articles in repository: {e}")

            logger.info(
                f"Global news update completed: {scraped_count} scraped, {failed_count} failed"
            )

            return NewsUpdateResult(
                status="completed",
                articles_found=len(google_articles),
                articles_scraped=scraped_count,
                articles_failed=failed_count,
                categories=categories,
                date_range={"start": start_date, "end": end_date},
            )

        except Exception as e:
            logger.error(f"Error updating global news: {e}")
            raise

    def _calculate_sentiment_summary(
        self, articles: list[ArticleData]
    ) -> SentimentScore:
        """
        Calculate aggregate sentiment from article list.

        Args:
            articles: List of ArticleData objects

        Returns:
            SentimentScore: Aggregate sentiment score
        """
        if not articles:
            return SentimentScore(score=0.0, confidence=0.0, label="neutral")

        # Simple keyword-based sentiment analysis
        positive_words = {
            "good",
            "great",
            "excellent",
            "positive",
            "up",
            "rise",
            "gain",
            "profit",
            "growth",
            "success",
            "strong",
            "bullish",
            "optimistic",
            "boost",
            "surge",
        }
        negative_words = {
            "bad",
            "terrible",
            "negative",
            "down",
            "fall",
            "loss",
            "decline",
            "weak",
            "bearish",
            "pessimistic",
            "crash",
            "drop",
            "plunge",
            "concern",
        }

        total_score = 0.0
        scored_articles = 0

        for article in articles:
            if not hasattr(article, "content") or not article.content:
                continue

            content_lower = article.content.lower()
            words = content_lower.split()

            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)

            if positive_count + negative_count > 0:
                article_score = (positive_count - negative_count) / len(words)
                total_score += article_score
                scored_articles += 1

        if scored_articles == 0:
            return SentimentScore(score=0.0, confidence=0.0, label="neutral")

        avg_score = total_score / scored_articles
        confidence = min(scored_articles / len(articles), 1.0)

        # Normalize score to -1.0 to 1.0 range
        normalized_score = max(-1.0, min(1.0, avg_score * 10))

        # Determine label
        if normalized_score > 0.1:
            label = "positive"
        elif normalized_score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScore(
            score=normalized_score, confidence=confidence, label=label
        )

    def _extract_trending_topics(self, articles: list[ArticleData]) -> list[str]:
        """
        Extract trending topics from article titles and content.

        Args:
            articles: List of ArticleData objects

        Returns:
            List of trending topic strings
        """
        if not articles:
            return []

        # Simple keyword extraction from titles
        word_counts: dict[str, int] = {}
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
        }

        for article in articles:
            if hasattr(article, "title") and article.title:
                words = article.title.lower().split()
                for word in words:
                    # Clean word
                    word = "".join(c for c in word if c.isalnum())
                    if len(word) > 3 and word not in stop_words:
                        if word in word_counts:
                            word_counts[word] += 1
                        else:
                            word_counts[word] = 1

        # Get top trending words
        trending = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, count in trending if count > 1]
