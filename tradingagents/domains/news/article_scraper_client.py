"""
Article scraper client for extracting full content from news URLs.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from newspaper import Article, Config

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of article scraping operation."""

    status: str  # 'SUCCESS', 'SCRAPE_FAILED', 'ARCHIVE_SUCCESS', 'NOT_FOUND'
    content: str = ""
    author: str = ""
    final_url: str = ""
    title: str = ""
    publish_date: str = ""


class ArticleScraperClient:
    """Client for scraping article content with Internet Archive fallback."""

    def __init__(self, user_agent: str | None = None, delay: float = 1.0):
        """
        Initialize article scraper.

        Args:
            user_agent: User agent string for requests (None for default)
            delay: Delay between requests in seconds
        """
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.delay = delay

    def scrape_article(self, url: str) -> ScrapeResult:
        """
        Scrape article content from URL with fallback to Internet Archive.

        Args:
            url: Article URL to scrape

        Returns:
            ScrapeResult: Scraping result with content and metadata
        """
        if not url or not self._is_valid_url(url):
            return ScrapeResult(status="NOT_FOUND", final_url=url)

        # Try original source first
        result = self._scrape_from_source(url)
        if result.status == "SUCCESS":
            return result

        # Fallback to Internet Archive
        logger.info(f"Original scraping failed for {url}, trying Internet Archive")
        return self._scrape_from_wayback(url)

    def _scrape_from_source(self, url: str) -> ScrapeResult:
        """Scrape article from original source using newspaper4k."""
        try:
            # Add delay to be respectful
            time.sleep(self.delay)

            # Configure newspaper4k with optimizations
            config = Config()
            config.browser_user_agent = self.user_agent
            config.request_timeout = 10
            config.fetch_images = False

            article = Article(url, config=config)
            article.download()
            article.parse()

            # Validate content
            if not article.text or len(article.text.strip()) < 100:
                logger.warning(f"Article content too short or empty for {url}")
                return ScrapeResult(status="SCRAPE_FAILED", final_url=url)

            # Handle publish_date which can be datetime or string
            publish_date_str = ""
            if article.publish_date:
                if isinstance(article.publish_date, datetime):
                    publish_date_str = article.publish_date.strftime("%Y-%m-%d")
                elif isinstance(article.publish_date, str):
                    publish_date_str = article.publish_date
                else:
                    # Try to convert to string
                    publish_date_str = str(article.publish_date)

            return ScrapeResult(
                status="SUCCESS",
                content=article.text.strip(),
                author=", ".join(article.authors) if article.authors else "",
                final_url=url,
                title=article.title or "",
                publish_date=publish_date_str,
            )

        except Exception as e:
            logger.warning(f"Error scraping article from {url}: {e}")
            return ScrapeResult(status="SCRAPE_FAILED", final_url=url)

    def _scrape_from_wayback(self, url: str) -> ScrapeResult:
        """Scrape article from Internet Archive Wayback Machine."""
        try:
            import requests
        except ImportError:
            logger.error("requests not installed. Install with: pip install requests")
            return ScrapeResult(status="NOT_FOUND", final_url=url)

        try:
            # Query Wayback Machine CDX API for snapshots
            cdx_url = "http://web.archive.org/cdx/search/cdx"
            params = {
                "url": url,
                "output": "json",
                "fl": "timestamp,original",
                "filter": "statuscode:200",
                "limit": "1",
            }

            response = requests.get(cdx_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if len(data) < 2:  # First row is headers
                logger.warning(f"No archived snapshots found for {url}")
                return ScrapeResult(status="NOT_FOUND", final_url=url)

            # Get the most recent snapshot
            timestamp, original_url = data[1]
            archive_url = f"https://web.archive.org/web/{timestamp}/{original_url}"

            logger.info(f"Found archived snapshot: {archive_url}")

            # Scrape from archive URL
            result = self._scrape_from_source(archive_url)
            if result.status == "SUCCESS":
                result.status = "ARCHIVE_SUCCESS"
                result.final_url = archive_url

            return result

        except Exception as e:
            logger.warning(f"Error accessing Internet Archive for {url}: {e}")
            return ScrapeResult(status="NOT_FOUND", final_url=url)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible."""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ("http", "https")
        except Exception:
            return False

    def scrape_multiple_articles(self, urls: list[str]) -> dict[str, ScrapeResult]:
        """
        Scrape multiple articles sequentially.

        Args:
            urls: List of article URLs to scrape

        Returns:
            Dict mapping URLs to ScrapeResults
        """
        results = {}

        for i, url in enumerate(urls):
            logger.info(f"Scraping article {i + 1}/{len(urls)}: {url}")
            results[url] = self.scrape_article(url)

            # Add delay between requests
            if i < len(urls) - 1:
                time.sleep(self.delay)

        return results
