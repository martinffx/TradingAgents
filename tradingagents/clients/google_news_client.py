"""
Google News client for live news data via web scraping.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from tradingagents.dataflows.googlenews_utils import getNewsData

from .base import BaseClient

logger = logging.getLogger(__name__)


class GoogleNewsClient(BaseClient):
    """Client for Google News data via web scraping."""

    def __init__(self, **kwargs):
        """
        Initialize Google News client.

        Args:
            **kwargs: Configuration options including rate limits
        """
        super().__init__(**kwargs)
        self.max_retries = kwargs.get("max_retries", 3)
        self.delay_between_requests = kwargs.get("delay_between_requests", 1.0)

    def test_connection(self) -> bool:
        """Test Google News connection by fetching a simple query."""
        try:
            # Test with a simple query for recent news
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            test_data = getNewsData("technology", start_date, end_date)
            return isinstance(test_data, list)

        except Exception as e:
            logger.error(f"Google News connection test failed: {e}")
            return False

    def get_data(
        self, query: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """
        Get news data for a query and date range.

        Args:
            query: Search query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: News data with metadata
        """
        if not self.validate_date_range(start_date, end_date):
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")

        try:
            # Replace spaces with + for URL encoding
            formatted_query = query.replace(" ", "+")

            logger.info(
                f"Fetching Google News for query: {query} from {start_date} to {end_date}"
            )

            news_results = getNewsData(formatted_query, start_date, end_date)

            if not news_results:
                logger.warning(f"No news found for query: {query}")
                return {
                    "query": query,
                    "period": {"start": start_date, "end": end_date},
                    "articles": [],
                    "metadata": {
                        "source": "google_news",
                        "empty": True,
                        "reason": "no_articles_found",
                    },
                }

            # Process and standardize article data
            processed_articles = []
            for article in news_results:
                processed_article = {
                    "headline": article.get("title", ""),
                    "summary": article.get("snippet", ""),
                    "url": article.get("link", ""),
                    "source": article.get("source", "Unknown"),
                    "date": article.get(
                        "date", end_date
                    ),  # Fallback to end_date if no date
                    "entities": article.get("entities", []),
                }
                processed_articles.append(processed_article)

            return {
                "query": query,
                "period": {"start": start_date, "end": end_date},
                "articles": processed_articles,
                "metadata": {
                    "source": "google_news",
                    "article_count": len(processed_articles),
                    "retrieved_at": datetime.utcnow().isoformat(),
                    "search_query": formatted_query,
                },
            }

        except Exception as e:
            logger.error(f"Error fetching Google News for query '{query}': {e}")
            raise

    def get_company_news(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """
        Get news data specific to a company symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Company-specific news data
        """
        # Create company-focused search query
        company_query = f"{symbol} stock"

        result = self.get_data(company_query, start_date, end_date, **kwargs)
        result["symbol"] = symbol
        result["metadata"]["query_type"] = "company_specific"

        return result

    def get_global_news(
        self,
        start_date: str,
        end_date: str,
        categories: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Get global/macro news that might affect markets.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            categories: List of news categories to search
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Global news data
        """
        if categories is None:
            categories = ["economy", "finance", "markets", "business"]

        all_articles = []

        for category in categories:
            try:
                category_data = self.get_data(category, start_date, end_date, **kwargs)

                # Add category tag to each article
                for article in category_data.get("articles", []):
                    article["category"] = category

                all_articles.extend(category_data.get("articles", []))

            except Exception as e:
                logger.warning(f"Failed to fetch news for category '{category}': {e}")
                continue

        return {
            "query": "global_news",
            "categories": categories,
            "period": {"start": start_date, "end": end_date},
            "articles": all_articles,
            "metadata": {
                "source": "google_news",
                "article_count": len(all_articles),
                "categories_searched": categories,
                "retrieved_at": datetime.utcnow().isoformat(),
                "query_type": "global_news",
            },
        }

    def get_available_categories(self) -> list[str]:
        """
        Get list of commonly used news categories.

        Returns:
            List[str]: News categories
        """
        return [
            "business",
            "economy",
            "finance",
            "markets",
            "technology",
            "politics",
            "world",
            "healthcare",
            "energy",
            "crypto",
        ]
