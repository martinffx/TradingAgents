"""
Article scraper client for extracting full content from news URLs.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

from newspaper import Article
from newspaper.configuration import Configuration

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of article scraping operation."""

    status: str  # 'SUCCESS', 'SCRAPE_FAILED', 'PAYWALL_DETECTED', 'NOT_FOUND'
    content: str = ""
    author: str = ""
    final_url: str = ""
    title: str = ""
    publish_date: str = ""
    is_paywall: bool = False
    keywords: list[str] | None = None  # Extracted keywords from newspaper4k
    summary: str = ""  # Article summary from newspaper4k


class ArticleScraperClient:
    """Client for scraping article content using newspaper4k."""

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

        # Download NLTK data for newspaper4k NLP
        try:
            import nltk

            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except ImportError:
            logger.warning("NLTK not available - NLP features will be limited")

        # Common paywall indicators
        self.paywall_indicators = {
            "subscribe",
            "subscription",
            "premium",
            "paywall",
            "sign in to read",
            "log in to continue",
            "register to read",
            "become a member",
            "upgrade to premium",
            "this article is for subscribers",
            "limited free articles",
            "subscribe now",
            "create a free account",
            "read more with subscription",
            "unlock full access",
            "premium content",
            "subscriber exclusive",
            "behind paywall",
            "free trial",
        }

    def scrape_article(self, url: str) -> ScrapeResult:
        """
        Scrape article content from URL.

        Args:
            url: Article URL to scrape

        Returns:
            ScrapeResult: Scraping result with content and metadata
        """
        if not url or not self._is_valid_url(url):
            return ScrapeResult(status="NOT_FOUND", final_url=url)

        # Scrape from original source
        return self._scrape_from_source(url)

    def _scrape_from_source(self, url: str) -> ScrapeResult:
        """Scrape article from original source using newspaper4k."""
        try:
            # Add delay to be respectful
            time.sleep(self.delay)

            # Configure newspaper4k with optimizations
            config = Configuration()
            config.browser_user_agent = self.user_agent
            config.request_timeout = 10
            config.fetch_images = False

            article = Article(url, config=config)
            article.download()
            article.parse()
            article.nlp()

            # Validate content and check for paywall
            content = article.text.strip() if article.text else ""
            is_paywall = self._detect_paywall(content, article.title or "")

            if not content or len(content) < 100:
                if is_paywall:
                    logger.info(f"Paywall detected for {url}")
                    return ScrapeResult(
                        status="PAYWALL_DETECTED",
                        final_url=url,
                        is_paywall=True,
                        title=article.title or "",
                        content=content,  # Include partial content
                    )
                else:
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
                content=content,
                author=", ".join(article.authors) if article.authors else "",
                final_url=url,
                title=article.title or "",
                publish_date=publish_date_str,
                is_paywall=is_paywall,
                keywords=list(article.keywords) if article.keywords else [],
                summary=article.summary or "",
            )

        except Exception as e:
            logger.warning(f"Error scraping article from {url}: {e}")
            return ScrapeResult(status="SCRAPE_FAILED", final_url=url)

    def _detect_paywall(self, content: str, title: str) -> bool:
        """
        Detect if article is behind a paywall.

        Args:
            content: Article content text
            title: Article title

        Returns:
            bool: True if paywall indicators are found
        """
        if not content and not title:
            return False

        # Combine content and title for analysis
        text_to_check = f"{title} {content}".lower()

        # Check for paywall indicators
        for indicator in self.paywall_indicators:
            if indicator in text_to_check:
                return True

        # Additional heuristics
        # Very short content with subscription-related words
        if len(content) < 200 and any(
            word in text_to_check
            for word in ["subscription", "subscribe", "member", "premium"]
        ):
            return True

        # Content that ends abruptly with subscription prompts
        content_end = content[-200:].lower() if len(content) > 200 else content.lower()
        return any(
            phrase in content_end
            for phrase in ["to continue reading", "subscribe to", "become a member"]
        )

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
