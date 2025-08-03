"""
Repository for historical news data (cached files).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""

    headline: str
    url: str  # Unique identifier for deduplication
    source: str  # "Finnhub", "Google News", etc.
    published_date: date

    # Optional fields
    summary: str | None = None
    entities: list[str] = field(default_factory=list)
    sentiment_score: float | None = None
    author: str | None = None
    category: str | None = None


@dataclass
class NewsData:
    """Container for news data with metadata."""

    query: str
    date: date
    source: str  # "finnhub", "google_news"
    articles: list[NewsArticle]


class NewsRepository:
    """Repository for accessing cached news data with source separation."""

    def __init__(self, data_dir: str):
        """
        Initialize news repository.

        Args:
            data_dir: Base directory for news data storage
            **kwargs: Additional configuration
        """
        self.news_data_dir = Path(data_dir) / "news_data"
        self.news_data_dir.mkdir(parents=True, exist_ok=True)

    def get_news_data(
        self,
        query: str,
        start_date: date,
        end_date: date,
        sources: list[str] | None = None,
    ) -> dict[date, list[NewsData]]:
        """
        Get cached news data for a query and date range across sources.

        Args:
            query: Search query or symbol
            start_date: Start date
            end_date: End date
            sources: List of sources to check (default: ["finnhub", "google_news"])

        Returns:
            Dict[date, list[NewsData]]: News data keyed by date, with list of source data
        """
        if sources is None:
            sources = ["finnhub", "google_news"]

        news_data = {}

        for source in sources:
            source_dir = self.news_data_dir / source / query

            if not source_dir.exists():
                logger.debug(f"No data directory found for {source}/{query}")
                continue

            # Scan for JSON files in the source/query directory
            for json_file in source_dir.glob("*.json"):
                try:
                    # Parse date from filename (YYYY-MM-DD.json)
                    date_str = json_file.stem
                    file_date = date.fromisoformat(date_str)

                    # Filter by date range
                    if start_date <= file_date <= end_date:
                        with open(json_file) as f:
                            data = json.load(f)

                        # Create NewsArticle objects from JSON data
                        articles = []
                        for article_data in data.get("articles", []):
                            # Convert date strings back to date objects
                            article_data_copy = article_data.copy()
                            if "published_date" in article_data_copy:
                                article_data_copy["published_date"] = (
                                    date.fromisoformat(
                                        article_data_copy["published_date"]
                                    )
                                )

                            article = NewsArticle(**article_data_copy)
                            articles.append(article)

                        # Create NewsData container
                        news_data_item = NewsData(
                            query=query,
                            date=file_date,
                            source=source,
                            articles=articles,
                        )

                        # Group by date (multiple sources per date)
                        if file_date not in news_data:
                            news_data[file_date] = []
                        news_data[file_date].append(news_data_item)

                except (ValueError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f"Error reading news data from {json_file}: {e}")
                    continue

        logger.info(
            f"Retrieved news data for {len(news_data)} dates for query '{query}'"
        )
        return news_data

    def store_news_articles(
        self,
        query: str,
        date: date,
        source: str,
        articles: list[NewsArticle],
    ) -> tuple[date, NewsData]:
        """
        Store news articles for a query, date, and source, merging with existing data.

        Args:
            query: Search query or symbol
            date: Date of the news articles
            source: News source ("finnhub", "google_news", etc.)
            articles: List of news articles

        Returns:
            Tuple[date, NewsData]: The stored date and news data
        """
        # Create source/query directory
        source_dir = self.news_data_dir / source / query

        # Create JSON file path
        file_path = source_dir / f"{date.isoformat()}.json"

        try:
            # Merge with existing articles if file exists
            merged_articles = self._merge_articles_with_existing(file_path, articles)

            # Prepare data for JSON serialization
            articles_data = []
            for article in merged_articles:
                article_dict = asdict(article)
                # Convert date objects to ISO format strings for JSON
                if article_dict.get("published_date"):
                    article_dict["published_date"] = article_dict[
                        "published_date"
                    ].isoformat()
                articles_data.append(article_dict)

            data = {
                "query": query,
                "date": date.isoformat(),
                "source": source,
                "articles": articles_data,
                "metadata": {
                    "article_count": len(merged_articles),
                    "stored_at": date.today().isoformat(),
                    "repository": "news_repository",
                },
            }

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Create NewsData result
            news_data = NewsData(
                query=query, date=date, source=source, articles=merged_articles
            )

            logger.info(
                f"Stored {len(articles)} new articles for {query} on {date} from {source} (total: {len(merged_articles)})"
            )
            return (date, news_data)

        except Exception as e:
            logger.error(
                f"Error storing news articles for {query} on {date} from {source}: {e}"
            )
            raise

    def store_news_data_batch(
        self,
        query: str,
        news_data_by_source: dict[str, dict[date, list[NewsArticle]]],
    ) -> dict[date, list[NewsData]]:
        """
        Store multiple news data sets for a query across sources.

        Args:
            query: Search query or symbol
            news_data_by_source: Nested dict of {source: {date: [articles]}}

        Returns:
            Dict[date, list[NewsData]]: The stored news data organized by date
        """
        stored_data = {}

        for source, date_articles in news_data_by_source.items():
            for article_date, articles in date_articles.items():
                try:
                    stored_date, stored_news_data = self.store_news_articles(
                        query, article_date, source, articles
                    )

                    # Group by date
                    if stored_date not in stored_data:
                        stored_data[stored_date] = []
                    stored_data[stored_date].append(stored_news_data)

                except Exception as e:
                    logger.error(
                        f"Failed to store news data for {query} on {article_date} from {source}: {e}"
                    )
                    continue

        total_dates = len(stored_data)
        total_sources = sum(len(news_list) for news_list in stored_data.values())
        logger.info(
            f"Stored news data for {total_dates} dates, {total_sources} source entries for query '{query}'"
        )
        return stored_data

    def _merge_articles_with_existing(
        self, file_path: Path, new_articles: list[NewsArticle]
    ) -> list[NewsArticle]:
        """
        Merge new articles with existing articles, deduplicating by URL.

        Args:
            file_path: Path to existing JSON file
            new_articles: New articles to merge

        Returns:
            List[NewsArticle]: Merged and deduplicated articles
        """
        existing_articles = []

        # Load existing articles if file exists
        if file_path.exists():
            try:
                with open(file_path) as f:
                    data = json.load(f)

                for existing_data in data.get("articles", []):
                    # Convert date strings back to date objects
                    existing_data_copy = existing_data.copy()
                    if "published_date" in existing_data_copy:
                        existing_data_copy["published_date"] = date.fromisoformat(
                            existing_data_copy["published_date"]
                        )

                    existing_article = NewsArticle(**existing_data_copy)
                    existing_articles.append(existing_article)

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error reading existing file {file_path}: {e}")
                existing_articles = []

        # Merge articles, deduplicating by URL (keep newer data)
        articles_by_url = {}

        # Add existing articles
        for article in existing_articles:
            articles_by_url[article.url] = article

        # Add/update with new articles (they take precedence)
        for article in new_articles:
            articles_by_url[article.url] = article

        # Return as sorted list
        merged_articles = list(articles_by_url.values())
        merged_articles.sort(
            key=lambda x: x.published_date, reverse=True
        )  # Newest first

        return merged_articles
