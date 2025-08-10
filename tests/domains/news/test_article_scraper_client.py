"""
Test ArticleScraperClient with pytest-vcr for HTTP recording/replay.

Following pragmatic TDD principles:
- Mock HTTP boundaries with VCR cassettes
- Test real business logic and data transformations
- Fast, deterministic tests
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tradingagents.domains.news.article_scraper_client import (
    ArticleScraperClient,
    ScrapeResult,
)


@pytest.fixture
def cassette_dir():
    """Directory for VCR cassettes."""
    return (
        Path(__file__).parent.parent.parent
        / "fixtures"
        / "vcr_cassettes"
        / "article_scraper"
    )


@pytest.fixture
def scraper():
    """ArticleScraperClient instance for testing."""
    return ArticleScraperClient(
        user_agent="Test-Agent/1.0",
        delay=0.1,  # Faster tests
    )


@pytest.fixture
def valid_urls():
    """Valid test URLs."""
    return [
        "https://www.reuters.com/business/finance/",
        "https://www.bloomberg.com/markets/stocks",
        "https://techcrunch.com/2024/01/15/tech-news/",
    ]


@pytest.fixture
def invalid_urls():
    """Invalid test URLs."""
    return [
        "",
        "not-a-url",
        "http://",
        "https://",
        "ftp://example.com/file.txt",
        "https://non-existent-domain-123456.com/article",
    ]


class TestArticleScraperClient:
    """Test ArticleScraperClient functionality."""

    def test_initialization(self):
        """Test scraper initializes with correct configuration."""
        # Test with custom user agent
        scraper = ArticleScraperClient("Custom-Agent/1.0", delay=2.0)
        assert scraper.user_agent == "Custom-Agent/1.0"
        assert scraper.delay == 2.0

        # Test with default user agent (None/empty)
        scraper_default = ArticleScraperClient(None)
        assert "Chrome" in scraper_default.user_agent
        assert scraper_default.delay == 1.0

    def test_is_valid_url(self, scraper):
        """Test URL validation logic."""
        # Valid URLs
        assert scraper._is_valid_url("https://example.com/article") is True
        assert scraper._is_valid_url("http://example.com/article") is True
        assert scraper._is_valid_url("https://sub.domain.com/path?query=value") is True

        # Invalid URLs
        assert scraper._is_valid_url("") is False
        assert scraper._is_valid_url("not-a-url") is False
        assert scraper._is_valid_url("ftp://example.com") is False
        assert scraper._is_valid_url("http://") is False
        assert scraper._is_valid_url("https://") is False

    def test_scrape_article_invalid_url(self, scraper, invalid_urls):
        """Test scraping with invalid URLs returns NOT_FOUND."""
        for url in invalid_urls:
            result = scraper.scrape_article(url)
            assert result.status == "NOT_FOUND"
            assert result.content == ""
            assert result.final_url == url


class TestArticleScrapingSuccess:
    """Test successful article scraping scenarios."""

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_success(self, mock_article_class, mock_sleep, scraper):
        """Test successful article scraping with mocked newspaper4k."""
        # Setup mock article
        mock_article = Mock()
        mock_article.text = "This is a long article content that is definitely over 100 characters in length and should pass the validation check."
        mock_article.title = "Test Article Title"
        mock_article.authors = ["John Doe", "Jane Smith"]
        mock_article.publish_date = "2024-01-15"
        mock_article.download.return_value = None
        mock_article.parse.return_value = None

        mock_article_class.return_value = mock_article

        # Test scraping
        result = scraper.scrape_article("https://example.com/article")

        # Verify results
        assert result.status == "SUCCESS"
        assert result.content == mock_article.text
        assert result.title == "Test Article Title"
        assert result.author == "John Doe, Jane Smith"
        assert result.publish_date == "2024-01-15"
        assert result.final_url == "https://example.com/article"

        # Verify newspaper4k was configured correctly
        mock_article_class.assert_called_once()
        args, kwargs = mock_article_class.call_args
        assert args[0] == "https://example.com/article"
        config = (
            kwargs["config"]
            if "config" in kwargs
            else args[1]
            if len(args) > 1
            else None
        )
        assert config is not None
        assert config.browser_user_agent == "Test-Agent/1.0"
        assert config.request_timeout == 10

        # Verify delay was applied
        mock_sleep.assert_called_once_with(0.1)

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_with_datetime_publish_date(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test successful scraping with datetime publish_date."""
        from datetime import datetime

        mock_article = Mock()
        mock_article.text = "Long article content over 100 characters for testing publish date handling in the newspaper4k client."
        mock_article.title = "DateTime Test Article"
        mock_article.authors = []
        mock_article.publish_date = datetime(2024, 1, 15, 14, 30, 0)

        mock_article_class.return_value = mock_article

        result = scraper.scrape_article("https://example.com/datetime-article")

        assert result.status == "SUCCESS"
        assert result.publish_date == "2024-01-15"
        assert result.author == ""  # Empty authors list

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_short_content_fails(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test that articles with content under 100 chars are rejected."""
        mock_article = Mock()
        mock_article.text = "Short content"  # Under 100 characters
        mock_article.title = "Short Article"
        mock_article.authors = []
        mock_article.publish_date = None

        mock_article_class.return_value = mock_article

        result = scraper.scrape_article("https://example.com/short-article")

        assert result.status == "SCRAPE_FAILED"
        assert result.content == ""

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_empty_content_fails(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test that articles with empty content are rejected."""
        mock_article = Mock()
        mock_article.text = ""  # Empty content
        mock_article.title = ""
        mock_article.authors = []
        mock_article.publish_date = None

        mock_article_class.return_value = mock_article

        result = scraper.scrape_article("https://example.com/empty-article")

        assert result.status == "SCRAPE_FAILED"
        assert result.content == ""


class TestArticleScrapingFailure:
    """Test article scraping failure scenarios."""

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_download_exception(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test scraping when newspaper4k download fails."""
        mock_article = Mock()
        mock_article.download.side_effect = Exception("Download failed")

        mock_article_class.return_value = mock_article

        result = scraper.scrape_article("https://example.com/failing-article")

        assert result.status == "SCRAPE_FAILED"
        assert result.content == ""
        assert result.final_url == "https://example.com/failing-article"

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_parse_exception(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test scraping when newspaper4k parse fails."""
        mock_article = Mock()
        mock_article.download.return_value = None
        mock_article.parse.side_effect = Exception("Parse failed")

        mock_article_class.return_value = mock_article

        result = scraper.scrape_article("https://example.com/parse-fail-article")

        assert result.status == "SCRAPE_FAILED"
        assert result.content == ""


class TestWaybackMachineFallback:
    """Test Internet Archive Wayback Machine fallback functionality."""

    @patch("tradingagents.domains.news.article_scraper_client.requests.get")
    def test_scrape_from_wayback_no_requests(self, mock_get, scraper):
        """Test Wayback fallback when requests is not available."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'requests'")
        ):
            result = scraper._scrape_from_wayback("https://example.com/article")

        assert result.status == "NOT_FOUND"
        assert result.final_url == "https://example.com/article"

    @patch("tradingagents.domains.news.article_scraper_client.requests.get")
    def test_scrape_from_wayback_no_snapshots(self, mock_get, scraper):
        """Test Wayback fallback when no archived snapshots exist."""
        # Mock CDX API response with only headers (no snapshots)
        mock_response = Mock()
        mock_response.json.return_value = [["timestamp", "original"]]  # Only headers
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = scraper._scrape_from_wayback("https://example.com/no-archive")

        assert result.status == "NOT_FOUND"
        assert result.final_url == "https://example.com/no-archive"

    @patch("tradingagents.domains.news.article_scraper_client.requests.get")
    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_from_wayback_success(
        self, mock_article_class, mock_sleep, mock_get, scraper
    ):
        """Test successful Wayback Machine scraping."""
        # Mock CDX API response
        mock_response = Mock()
        mock_response.json.return_value = [
            ["timestamp", "original"],  # Headers
            ["20240115120000", "https://example.com/article"],  # Snapshot data
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock successful article scraping from archive
        mock_article = Mock()
        mock_article.text = "Archived article content that is long enough to pass validation checks and contains meaningful information."
        mock_article.title = "Archived Article"
        mock_article.authors = ["Archive Author"]
        mock_article.publish_date = "2024-01-15"
        mock_article_class.return_value = mock_article

        result = scraper._scrape_from_wayback("https://example.com/article")

        assert result.status == "ARCHIVE_SUCCESS"
        assert result.content == mock_article.text
        assert result.title == "Archived Article"
        assert (
            result.final_url
            == "https://web.archive.org/web/20240115120000/https://example.com/article"
        )

        # Verify CDX API was called correctly
        mock_get.assert_called_with(
            "http://web.archive.org/cdx/search/cdx",
            params={
                "url": "https://example.com/article",
                "output": "json",
                "fl": "timestamp,original",
                "filter": "statuscode:200",
                "limit": "1",
            },
            timeout=10,
        )

    @patch("tradingagents.domains.news.article_scraper_client.requests.get")
    def test_scrape_from_wayback_requests_exception(self, mock_get, scraper):
        """Test Wayback fallback when requests fails."""
        mock_get.side_effect = Exception("Request timeout")

        result = scraper._scrape_from_wayback("https://example.com/timeout")

        assert result.status == "NOT_FOUND"
        assert result.final_url == "https://example.com/timeout"

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_scrape_article_fallback_to_wayback(
        self, mock_article_class, mock_sleep, scraper
    ):
        """Test full workflow: source fails, fallback to Wayback succeeds."""
        # First call (original source) fails
        # Second call (Wayback source) succeeds
        mock_article_fail = Mock()
        mock_article_fail.download.side_effect = Exception("Download failed")

        mock_article_success = Mock()
        mock_article_success.text = "Successfully scraped content from Wayback Machine with enough length to pass validation tests."
        mock_article_success.title = "Wayback Success"
        mock_article_success.authors = ["Wayback Author"]
        mock_article_success.publish_date = "2024-01-15"
        mock_article_success.download.return_value = None
        mock_article_success.parse.return_value = None

        mock_article_class.side_effect = [mock_article_fail, mock_article_success]

        with patch(
            "tradingagents.domains.news.article_scraper_client.requests.get"
        ) as mock_get:
            # Mock successful CDX API response
            mock_response = Mock()
            mock_response.json.return_value = [
                ["timestamp", "original"],
                ["20240115120000", "https://example.com/article"],
            ]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = scraper.scrape_article("https://example.com/article")

        assert result.status == "ARCHIVE_SUCCESS"
        assert (
            result.content
            == "Successfully scraped content from Wayback Machine with enough length to pass validation tests."
        )
        assert "web.archive.org" in result.final_url


class TestMultipleArticles:
    """Test scraping multiple articles functionality."""

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    def test_scrape_multiple_articles_empty_list(self, mock_sleep, scraper):
        """Test scraping empty list returns empty dict."""
        results = scraper.scrape_multiple_articles([])
        assert results == {}
        mock_sleep.assert_not_called()

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    def test_scrape_multiple_articles_single_url(self, mock_sleep, scraper):
        """Test scraping single URL in list."""
        urls = ["https://example.com/single"]

        with patch.object(scraper, "scrape_article") as mock_scrape:
            mock_scrape.return_value = ScrapeResult(
                status="SUCCESS", content="Single article content"
            )

            results = scraper.scrape_multiple_articles(urls)

            assert len(results) == 1
            assert results["https://example.com/single"].status == "SUCCESS"
            mock_scrape.assert_called_once_with("https://example.com/single")
            # No delay needed for single article
            mock_sleep.assert_not_called()

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    def test_scrape_multiple_articles_with_delays(self, mock_sleep, scraper):
        """Test scraping multiple URLs with delays between requests."""
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]

        with patch.object(scraper, "scrape_article") as mock_scrape:
            mock_scrape.side_effect = [
                ScrapeResult(status="SUCCESS", content="Article 1"),
                ScrapeResult(status="SUCCESS", content="Article 2"),
                ScrapeResult(status="SCRAPE_FAILED", content=""),
            ]

            results = scraper.scrape_multiple_articles(urls)

            assert len(results) == 3
            assert results["https://example.com/article1"].status == "SUCCESS"
            assert results["https://example.com/article2"].status == "SUCCESS"
            assert results["https://example.com/article3"].status == "SCRAPE_FAILED"

            # Verify delay called between requests (n-1 times)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_called_with(0.1)


class TestDataTransformation:
    """Test data transformation and edge cases."""

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_publish_date_edge_cases(self, mock_article_class, mock_sleep, scraper):
        """Test various publish_date formats are handled correctly."""
        from datetime import datetime

        test_cases = [
            (None, ""),
            ("", ""),
            ("2024-01-15", "2024-01-15"),
            (datetime(2024, 1, 15), "2024-01-15"),
            (12345, "12345"),  # Numeric conversion
            ({"year": 2024}, "{'year': 2024}"),  # Dict conversion
        ]

        for pub_date, expected in test_cases:
            mock_article = Mock()
            mock_article.text = "Long enough content for validation testing with various publish date formats and edge cases."
            mock_article.title = "Date Test"
            mock_article.authors = []
            mock_article.publish_date = pub_date

            mock_article_class.return_value = mock_article

            result = scraper.scrape_article("https://example.com/date-test")
            assert result.status == "SUCCESS"
            assert result.publish_date == expected

    def test_scrape_result_dataclass_defaults(self):
        """Test ScrapeResult dataclass has correct defaults."""
        result = ScrapeResult(status="TEST")

        assert result.status == "TEST"
        assert result.content == ""
        assert result.author == ""
        assert result.final_url == ""
        assert result.title == ""
        assert result.publish_date == ""

    def test_scrape_result_all_fields(self):
        """Test ScrapeResult with all fields populated."""
        result = ScrapeResult(
            status="SUCCESS",
            content="Full article content",
            author="Test Author",
            final_url="https://final.com/url",
            title="Test Title",
            publish_date="2024-01-15",
        )

        assert result.status == "SUCCESS"
        assert result.content == "Full article content"
        assert result.author == "Test Author"
        assert result.final_url == "https://final.com/url"
        assert result.title == "Test Title"
        assert result.publish_date == "2024-01-15"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_user_agent_fallback(self):
        """Test user agent fallback when None or empty is provided."""
        scraper_none = ArticleScraperClient(None)
        scraper_empty = ArticleScraperClient("")

        # Both should use default Chrome user agent
        default_ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        assert scraper_none.user_agent == default_ua
        assert scraper_empty.user_agent == default_ua

    @patch("tradingagents.domains.news.article_scraper_client.time.sleep")
    @patch("tradingagents.domains.news.article_scraper_client.Article")
    def test_config_applied_correctly(self, mock_article_class, mock_sleep):
        """Test that newspaper4k Config is applied with correct settings."""
        scraper = ArticleScraperClient("Custom-Agent/2.0", delay=0.5)

        mock_article = Mock()
        mock_article.text = "Test content that meets minimum length requirements for successful article scraping validation."
        mock_article_class.return_value = mock_article

        scraper.scrape_article("https://example.com/config-test")

        # Verify Article was created with correct config
        mock_article_class.assert_called_once()
        args, kwargs = mock_article_class.call_args

        assert args[0] == "https://example.com/config-test"
        config = kwargs.get("config") or (args[1] if len(args) > 1 else None)
        assert config is not None
        assert config.browser_user_agent == "Custom-Agent/2.0"
        assert config.request_timeout == 10
        assert config.keep_article_html is True
        assert config.fetch_images is False
