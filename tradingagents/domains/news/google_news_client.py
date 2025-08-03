"""
Google News client for live news data via RSS feeds.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

import feedparser
import requests
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


@dataclass
class FeedEntry:
    """Structured representation of a feedparser entry."""

    title: str
    link: str
    published: str
    published_parsed: time.struct_time | None
    summary: str
    guid: str

    @classmethod
    def from_feedparser_dict(cls, entry: feedparser.FeedParserDict) -> "FeedEntry":
        """Convert a FeedParserDict to a structured FeedEntry."""
        return cls(
            title=getattr(entry, "title", "Untitled"),
            link=getattr(entry, "link", ""),
            published=getattr(entry, "published", ""),
            published_parsed=getattr(entry, "published_parsed", None),
            summary=getattr(entry, "summary", ""),
            guid=getattr(entry, "id", getattr(entry, "link", "")),
        )


@dataclass
class GoogleNewsArticle:
    """Represents a news article from Google News RSS feed."""

    title: str
    link: str
    published: datetime
    summary: str
    source: str
    guid: str


class GoogleNewsClient:
    """Client for Google News data via web scraping."""

    def get_company_news(self, symbol: str) -> list[GoogleNewsArticle]:
        """
        Get news data specific to a company symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            list[GoogleNewsArticle]: Company-specific news data
        """
        return self._get_rss_feed(symbol)

    def get_global_news(
        self,
        categories: list[str],
    ) -> list[GoogleNewsArticle]:
        """
        Get global/macro news that might affect markets.

        Args:
            categories: List of news categories to search

        Returns:
            list[GoogleNewsArticle]: Global news data
        """
        # Get RSS queries for categories
        all_articles = []

        for category in categories:
            try:
                articles = self._get_rss_feed(category)
                all_articles.extend(articles)

            except Exception as e:
                logger.warning(f"Failed to fetch news for category '{category}': {e}")
                continue

        return all_articles

    def _get_rss_feed(self, query: str) -> list[GoogleNewsArticle]:
        """
        Fetch RSS feed from Google News for a given query.

        Args:
            query: Search query (company symbol or news category)

        Returns:
            list[GoogleNewsArticle]: Parsed articles from RSS feed
        """
        try:
            # Construct Google News RSS URL
            encoded_query = quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            logger.info(f"Fetching RSS feed for query: {query}")

            # Use requests with timeout and User-Agent header
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = requests.get(rss_url, timeout=10, headers=headers)
            response.raise_for_status()

            # Use feedparser to parse the fetched content
            feed = feedparser.parse(response.content)

            # Check if feed was parsed successfully
            if feed.bozo:
                logger.warning(
                    f"Feed parsing had issues for query '{query}': {feed.bozo_exception}"
                )

            articles = []
            for raw_entry in feed.entries:
                try:
                    # Convert FeedParserDict to structured dataclass
                    entry = FeedEntry.from_feedparser_dict(raw_entry)
                    article = self._convert_entry_to_article(entry)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article entry: {e}")
                    continue

            logger.info(
                f"Successfully fetched {len(articles)} articles for query: {query}"
            )
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching RSS feed for query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(
                f"Unexpected error fetching RSS feed for query '{query}': {e}",
                exc_info=True,
            )
            return []

    def _convert_entry_to_article(self, entry: FeedEntry) -> GoogleNewsArticle:
        """
        Convert a structured FeedEntry to a GoogleNewsArticle.

        Args:
            entry: Structured FeedEntry dataclass

        Returns:
            GoogleNewsArticle: Converted article object
        """
        # Parse published date with fallback to current time
        try:
            published = (
                date_parser.parse(entry.published)
                if entry.published
                else datetime.utcnow()
            )
        except (ValueError, OverflowError, TypeError):
            published = datetime.utcnow()

        # Extract source from title (Google News format: "Title - Source")
        title_parts = entry.title.split(" - ")
        title = title_parts[0] if title_parts else entry.title
        source = title_parts[-1] if len(title_parts) > 1 else "Unknown"

        return GoogleNewsArticle(
            title=title,
            link=entry.link,
            published=published,
            summary=entry.summary,
            source=source,
            guid=entry.guid,
        )
