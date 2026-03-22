"""Unit tests for NewsActivities.

Tests focus on activity method calls - mocking NewsService.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.workflows.activities.news_activities import NewsActivities


@pytest.fixture
def mock_news_service():
    """Create mock NewsService with async methods."""
    mock = MagicMock()
    mock.repository.get = AsyncMock()
    mock.repository.upsert = AsyncMock()
    mock.repository.list_by_date_range = AsyncMock()
    mock.scraper.scrape_article = MagicMock()
    mock.llm_client.analyze_sentiment = AsyncMock()
    mock.llm_client.create_embedding = AsyncMock()
    return mock


@pytest.fixture
def news_activities(mock_news_service):
    """Create NewsActivities with mocked NewsService via constructor DI."""
    return NewsActivities(mock_news_service)


class TestFetchArticleActivity:
    """Tests for fetch_article activity."""

    @pytest.mark.asyncio
    async def test_returns_article_dict(self, news_activities, mock_news_service):
        """Activity returns article as dict."""
        article = NewsArticle(
            id=uuid4(),
            url="https://example.com/test",
            headline="Test Article",
            source="TestSource",
            published_date=date(2024, 1, 15),
            entities=[],
        )
        mock_news_service.repository.get.return_value = article

        result = await news_activities.fetch_article(str(article.id))

        assert result["headline"] == "Test Article"
        assert result["url"] == "https://example.com/test"
        mock_news_service.repository.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_article(
        self, news_activities, mock_news_service
    ):
        """Activity returns None when article not found."""
        mock_news_service.repository.get.return_value = None

        result = await news_activities.fetch_article(
            "00000000-0000-0000-0000-000000000000"
        )

        assert result is None


class TestScrapeArticleActivity:
    """Tests for scrape_article activity."""

    @pytest.mark.asyncio
    async def test_returns_scraped_content(self, news_activities, mock_news_service):
        """Activity returns scraped content dict."""
        mock_result = MagicMock()
        mock_result.content = "Full article content"
        mock_result.title = "Article Title"
        mock_result.author = "John Doe"
        mock_result.publish_date = "2024-01-15"
        mock_news_service.scraper.scrape_article.return_value = mock_result

        result = await news_activities.scrape_article("https://example.com/test")

        assert result["content"] == "Full article content"
        assert result["title"] == "Article Title"
        assert result["author"] == "John Doe"


class TestAnalyzeSentimentActivity:
    """Tests for analyze_sentiment activity."""

    @pytest.mark.asyncio
    async def test_returns_sentiment_dict(self, news_activities, mock_news_service):
        """Activity returns properly formatted sentiment dict."""
        mock_result = MagicMock()
        mock_result.sentiment = "positive"
        mock_result.confidence = 0.85
        mock_result.reasoning = "Strong earnings"
        mock_news_service.llm_client.analyze_sentiment.return_value = mock_result

        result = await news_activities.analyze_sentiment("Test text")

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Strong earnings"

    @pytest.mark.asyncio
    async def test_propagates_llm_errors(self, news_activities, mock_news_service):
        """Activity propagates LLM client errors."""
        from tradingagents.lib.llm_client import SentimentAnalysisError

        mock_news_service.llm_client.analyze_sentiment.side_effect = (
            SentimentAnalysisError("API failed")
        )

        with pytest.raises(SentimentAnalysisError):
            await news_activities.analyze_sentiment("Test text")


class TestCreateEmbeddingActivity:
    """Tests for create_embedding activity."""

    @pytest.mark.asyncio
    async def test_returns_embedding_list(self, news_activities, mock_news_service):
        """Activity returns embedding as list."""
        mock_embedding = [0.1] * 1536
        mock_news_service.llm_client.create_embedding.return_value = mock_embedding

        result = await news_activities.create_embedding("Test text")

        assert len(result) == 1536
        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_propagates_embedding_errors(
        self, news_activities, mock_news_service
    ):
        """Activity propagates embedding generation errors."""
        from tradingagents.lib.llm_client import EmbeddingGenerationError

        mock_news_service.llm_client.create_embedding.side_effect = (
            EmbeddingGenerationError("Generation failed")
        )

        with pytest.raises(EmbeddingGenerationError):
            await news_activities.create_embedding("Test text")


class TestSaveArticleActivity:
    """Tests for save_article activity."""

    @pytest.mark.asyncio
    async def test_saves_and_returns_id(self, news_activities, mock_news_service):
        """Activity saves article and returns ID."""
        saved_article = NewsArticle(
            id=uuid4(),
            url="https://example.com/test",
            headline="Test",
            source="Test",
            published_date=date(2024, 1, 15),
            entities=[],
        )
        mock_news_service.repository.upsert.return_value = saved_article

        article_data = {
            "id": str(uuid4()),
            "url": "https://example.com/test",
            "headline": "Test",
            "source": "Test",
            "published_date": "2024-01-15",
        }

        result = await news_activities.save_article(article_data)

        assert result == str(saved_article.id)
        mock_news_service.repository.upsert.assert_called_once()


class TestListArticlesForProcessingActivity:
    """Tests for list_articles_for_processing activity."""

    @pytest.mark.asyncio
    async def test_returns_article_list(self, news_activities, mock_news_service):
        """Activity returns list of article dicts."""
        articles = [
            NewsArticle(
                id=uuid4(),
                url=f"https://example.com/{i}",
                headline=f"Article {i}",
                source="Test",
                published_date=date(2024, 1, 15),
                entities=[],
            )
            for i in range(3)
        ]
        mock_news_service.repository.list_by_date_range.return_value = articles

        result = await news_activities.list_articles_for_processing(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-31",
            limit=10,
        )

        assert len(result) == 3
        assert result[0]["headline"] == "Article 0"
