"""
Test suite for NewsService following pragmatic outside-in TDD methodology.

This test suite follows the CLAUDE.md testing principles:
- Mock I/O boundaries (Repository calls, HTTP clients, external systems)
- Real objects for logic (Data transformations, validation, business logic)
- Outside-in but practical - Start with service tests, work inward
"""

from datetime import date
from unittest.mock import Mock

import pytest

from tradingagents.domains.news.news_repository import (
    NewsData,
)
from tradingagents.domains.news.news_service import (
    ArticleData,
    NewsContext,
    NewsService,
    NewsUpdateResult,
    SentimentScore,
)

# Import mock ScrapeResult from conftest to avoid newspaper3k import issues
from ...conftest import ScrapeResult


class TestNewsServiceCollaboratorInteractions:
    """Test NewsService interactions with its collaborators (I/O boundaries)."""

    def test_get_company_news_context_calls_repository_with_correct_params(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test that get_company_news_context calls repository with correct parameters."""
        # Arrange - Mock the I/O boundary
        mock_repository.get_news_data.return_value = {}

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act - Call the service method
        result = service.get_company_news_context("AAPL", "2024-01-01", "2024-01-31")

        # Assert - Repository should be called with converted date objects
        mock_repository.get_news_data.assert_called_once_with(
            query="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            sources=["finnhub", "google_news"],
        )

        # Assert - Result should have correct structure (real object logic)
        assert isinstance(result, NewsContext)
        assert result.query == "AAPL"
        assert result.symbol == "AAPL"
        assert result.period == {"start": "2024-01-01", "end": "2024-01-31"}

    def test_get_global_news_context_calls_repository_for_each_category(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test that get_global_news_context calls repository for each category."""
        # Arrange - Mock the I/O boundary
        mock_repository.get_news_data.return_value = {}

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)
        categories = ["business", "politics", "technology"]

        # Act
        service.get_global_news_context(
            "2024-01-01", "2024-01-31", categories=categories
        )

        # Assert - Repository should be called once for each category
        assert mock_repository.get_news_data.call_count == 3

        for call_args in mock_repository.get_news_data.call_args_list:
            args, kwargs = call_args
            assert args[0] in categories  # query should be one of the categories
            assert args[1] == date(2024, 1, 1)  # start_date
            assert args[2] == date(2024, 1, 31)  # end_date
            assert kwargs["sources"] == ["google_news"]

    def test_update_company_news_calls_google_client(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test that update_company_news calls GoogleNewsClient correctly."""
        # Arrange - Mock the I/O boundary
        mock_google_client.get_company_news.return_value = []

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act
        result = service.update_company_news("AAPL")

        # Assert - Google client should be called
        mock_google_client.get_company_news.assert_called_once_with("AAPL")
        assert isinstance(result, NewsUpdateResult)
        assert result.symbol == "AAPL"
        assert result.articles_found == 0

    def test_update_company_news_scrapes_each_article_url(
        self,
        mock_repository,
        mock_google_client,
        mock_article_scraper,
        sample_google_articles,
    ):
        """Test that update_company_news calls scraper for each article URL."""
        # Arrange - Mock I/O boundaries with real data objects
        mock_google_client.get_company_news.return_value = sample_google_articles
        mock_article_scraper.scrape_article.return_value = ScrapeResult(
            status="SUCCESS",
            content="Full article content",
            author="Test Author",
            title="Test Title",
            publish_date="2024-01-15",
        )

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act
        result = service.update_company_news("AAPL")

        # Assert - Scraper should be called for each article
        assert mock_article_scraper.scrape_article.call_count == 2
        mock_article_scraper.scrape_article.assert_any_call(
            "https://example.com/apple-soars"
        )
        mock_article_scraper.scrape_article.assert_any_call(
            "https://example.com/apple-products"
        )

        # Assert - Real object logic for result
        assert result.articles_found == 2
        assert result.articles_scraped == 2
        assert result.articles_failed == 0

    def test_repository_failure_returns_empty_context_with_error_metadata(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test that repository failure is handled gracefully."""
        # Arrange - Mock repository failure (I/O boundary)
        mock_repository.get_news_data.side_effect = Exception(
            "Database connection failed"
        )

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act
        result = service.get_company_news_context("AAPL", "2024-01-01", "2024-01-31")

        # Assert - Should return empty context with error metadata (real object logic)
        assert isinstance(result, NewsContext)
        assert result.articles == []
        assert result.article_count == 0
        assert "error" in result.metadata
        assert "Database connection failed" in result.metadata["error"]


class TestNewsServiceDataTransformations:
    """Test data transformations using real objects (no mocking)."""

    def test_converts_repository_articles_to_article_data(
        self, mock_google_client, mock_article_scraper, sample_news_articles
    ):
        """Test conversion of NewsRepository.NewsArticle to ArticleData."""
        # Arrange - Create real repository with sample data
        mock_repo = Mock()
        news_data = NewsData(
            query="AAPL",
            date=date(2024, 1, 15),
            source="finnhub",
            articles=sample_news_articles,
        )
        mock_repo.get_news_data.return_value = {date(2024, 1, 15): [news_data]}

        service = NewsService(mock_google_client, mock_repo, mock_article_scraper)

        # Act - Test real data transformation logic
        result = service.get_company_news_context("AAPL", "2024-01-01", "2024-01-31")

        # Assert - Real object data transformation
        assert len(result.articles) == 2
        assert result.articles[0].title == "Apple Stock Rises 5% on Strong Earnings"
        assert (
            result.articles[0].content
            == "Apple reports strong quarterly earnings beating expectations"
        )
        assert result.articles[0].date == "2024-01-15"
        assert result.articles[0].source == "CNBC"
        assert result.articles[0].url == "https://example.com/apple-earnings"

    def test_calculates_sentiment_summary_from_articles(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test sentiment summary calculation from article list."""
        # Arrange - Create articles with sentiment-bearing content (real objects)
        articles = [
            ArticleData(
                title="Great News for Apple",
                content="Apple stock is performing excellent with strong growth and positive outlook",
                author="Analyst",
                source="CNBC",
                date="2024-01-15",
                url="https://example.com/positive",
            ),
            ArticleData(
                title="Apple Faces Challenges",
                content="Apple stock is declining due to bad earnings and negative market sentiment",
                author="Reporter",
                source="Reuters",
                date="2024-01-16",
                url="https://example.com/negative",
            ),
        ]

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act - Test real sentiment calculation logic (private method)
        sentiment = service._calculate_sentiment_summary(articles)

        # Assert - Real sentiment calculation
        assert isinstance(sentiment, SentimentScore)
        assert -1.0 <= sentiment.score <= 1.0
        assert 0.0 <= sentiment.confidence <= 1.0
        assert sentiment.label in ["positive", "negative", "neutral"]

    def test_extracts_trending_topics_from_articles(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test trending topic extraction."""
        # Arrange - Create articles with repeated keywords (real objects)
        articles = [
            ArticleData(
                title="Apple iPhone Sales Surge",
                content="Content about iPhone",
                author="Reporter",
                source="TechNews",
                date="2024-01-15",
                url="https://example.com/iphone1",
            ),
            ArticleData(
                title="iPhone Market Share Growth",
                content="More iPhone content",
                author="Analyst",
                source="MarketWatch",
                date="2024-01-16",
                url="https://example.com/iphone2",
            ),
            ArticleData(
                title="Apple Revenue from Services",
                content="Services revenue content",
                author="Finance Writer",
                source="Bloomberg",
                date="2024-01-17",
                url="https://example.com/services",
            ),
        ]

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act - Test real trending topic extraction logic
        topics = service._extract_trending_topics(articles)

        # Assert - Should identify repeated keywords
        assert isinstance(topics, list)
        assert "iphone" in topics  # Should appear twice
        assert "apple" in topics  # Should appear multiple times


class TestNewsServiceErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_handles_google_client_failure(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test handling of GoogleNewsClient failure."""
        # Arrange - Mock client failure (I/O boundary)
        mock_google_client.get_company_news.side_effect = Exception(
            "API rate limit exceeded"
        )

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act & Assert - Should raise the exception
        with pytest.raises(Exception, match="API rate limit exceeded"):
            service.update_company_news("AAPL")

    def test_handles_article_scraper_failure(
        self,
        mock_repository,
        mock_google_client,
        mock_article_scraper,
        sample_google_articles,
    ):
        """Test handling of article scraper failure."""
        # Arrange - Mock scraper returning failure status
        mock_google_client.get_company_news.return_value = sample_google_articles
        mock_article_scraper.scrape_article.return_value = ScrapeResult(
            status="SCRAPE_FAILED", content="", author="", title="", publish_date=""
        )

        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act
        result = service.update_company_news("AAPL")

        # Assert - Should handle scraper failures gracefully
        assert result.articles_found == 2
        assert result.articles_scraped == 0
        assert result.articles_failed == 2

    def test_handles_invalid_date_formats(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test validation of date formats."""
        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act & Assert - Should raise ValueError for invalid date format
        with pytest.raises(ValueError):
            service.get_company_news_context("AAPL", "invalid-date", "2024-01-31")

    def test_handles_empty_articles_gracefully(
        self, mock_repository, mock_google_client, mock_article_scraper
    ):
        """Test handling of empty article list."""
        service = NewsService(mock_google_client, mock_repository, mock_article_scraper)

        # Act - Test sentiment calculation with empty list
        sentiment = service._calculate_sentiment_summary([])

        # Assert - Should return neutral sentiment
        assert sentiment.score == 0.0
        assert sentiment.confidence == 0.0
        assert sentiment.label == "neutral"
