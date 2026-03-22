"""
Test suite for NewsService following pragmatic outside-in TDD methodology.

This test suite follows the CLAUDE.md testing principles:
- Mock I/O boundaries (Repository calls, HTTP clients, external systems)
- Real objects for logic (Data transformations, validation, business logic)
- Outside-in but practical - Start with service tests, work inward
"""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from tradingagents.domains.news.news_service import (
    NewsContext,
    NewsService,
)


class TestNewsServiceCollaboratorInteractions:
    """Test NewsService interactions with its collaborators (I/O boundaries)."""

    @pytest.mark.asyncio
    async def test_get_company_news_context_calls_repository_with_correct_params(
        self,
        mock_repository,
        mock_google_client,
        mock_article_scraper,
        mock_openrouter_client,
    ):
        """Test that get_company_news_context calls repository with correct parameters."""
        # Arrange - Mock the I/O boundary
        mock_repository.list_by_date_range.return_value = []

        service = NewsService(
            mock_google_client,
            mock_repository,
            mock_article_scraper,
            mock_openrouter_client,
        )

        # Act - Call the service method
        result = await service.get_company_news_context(
            "AAPL", "2024-01-01", "2024-01-31"
        )

        # Assert - Repository should be called with converted date objects
        mock_repository.list_by_date_range.assert_called_once_with(
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # Assert - Result should have correct structure (real object logic)
        assert isinstance(result, NewsContext)
        assert result.query == "AAPL"
        assert result.symbol == "AAPL"
        assert result.period == {"start": "2024-01-01", "end": "2024-01-31"}

    @pytest.mark.asyncio
    async def test_get_global_news_context_calls_repository_for_each_category(
        self,
        mock_repository,
        mock_google_client,
        mock_article_scraper,
        mock_openrouter_client,
    ):
        """Test that get_global_news_context calls repository for each category."""
        # Arrange - Mock the I/O boundary
        mock_repository.list_by_date_range.return_value = []

        service = NewsService(
            mock_google_client,
            mock_repository,
            mock_article_scraper,
            mock_openrouter_client,
        )
        categories = ["business", "politics", "technology"]

        # Act
        await service.get_global_news_context(
            "2024-01-01", "2024-01-31", categories=categories
        )

        # Assert - Repository should be called once for each category
        assert mock_repository.list_by_date_range.call_count == 3

        for call_args in mock_repository.list_by_date_range.call_args_list:
            args, kwargs = call_args
            assert (
                kwargs["symbol"] in categories
            )  # symbol should be one of the categories
            assert kwargs["start_date"] == date(2024, 1, 1)  # start_date
            assert kwargs["end_date"] == date(2024, 1, 31)  # end_date


class TestNewsServiceDataTransformations:
    """Test data transformations using real objects (no mocking)."""

    @pytest.mark.asyncio
    async def test_converts_repository_articles_to_article_data(
        self,
        mock_google_client,
        mock_article_scraper,
        mock_openrouter_client,
        sample_news_articles,
    ):
        """Test conversion of NewsRepository.NewsArticle to ArticleData."""
        # Arrange - Create real repository with sample data
        mock_repo = AsyncMock()
        mock_repo.list_by_date_range.return_value = sample_news_articles

        service = NewsService(
            mock_google_client, mock_repo, mock_article_scraper, mock_openrouter_client
        )

        # Act - Test real data transformation logic
        result = await service.get_company_news_context(
            "AAPL", "2024-01-01", "2024-01-31"
        )

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
