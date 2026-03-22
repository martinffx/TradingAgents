"""
Tests for vector similarity search in NewsRepository.

Uses Docker-based PostgreSQL test container with pgvector extension.
"""

from datetime import date
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from uuid_utils import uuid7

from tradingagents.domains.news.news_article import NewsArticle
from tradingagents.domains.news.news_repository import NewsRepository
from tradingagents.lib.database import create_test_database_manager


@pytest.fixture
async def test_db_manager():
    """Create test database manager with TimescaleDB container."""
    db_manager = create_test_database_manager()

    health = await db_manager.health_check()
    if not health:
        pytest.skip("TimescaleDB test container not available")

    await db_manager.create_tables()

    yield db_manager

    await db_manager.drop_tables()
    await db_manager.close()


@pytest.fixture
async def repository(test_db_manager):
    """Create repository instance with test database."""
    repo = NewsRepository(test_db_manager)

    async with test_db_manager.get_session() as session:
        await session.execute(text("DELETE FROM news_articles"))
        await session.commit()

    return repo


@pytest.fixture
def sample_article_with_embeddings():
    """Create a sample news article with embeddings for testing."""
    base_embedding = [0.1] * 1536
    return NewsArticle(
        id=UUID(str(uuid7())),
        headline="Apple Quarterly Earnings Beat Expectations",
        url="https://example.com/apple-earnings-q1-2024",
        source="TechCrunch",
        published_date=date(2024, 1, 15),
        summary="Apple reported strong quarterly earnings with iPhone sales exceeding analyst predictions.",
        entities=["Apple", "iPhone", "earnings"],
        sentiment_score=0.8,
        sentiment_confidence=0.9,
        sentiment_label="positive",
        author="Jane Tech Reporter",
        category="earnings",
        title_embedding=base_embedding,
        content_embedding=base_embedding,
    )


class TestVectorSimilaritySearch:
    """Test vector similarity search methods."""

    async def test_find_similar_by_title(
        self, repository, sample_article_with_embeddings
    ):
        """Test finding similar articles by title embedding."""
        article = sample_article_with_embeddings

        similar_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Apple Stock Rises on Strong iPhone Sales",
            url="https://example.com/apple-stock-rises-2024",
            source="Reuters",
            published_date=date(2024, 1, 16),
            summary="Apple shares increased following positive earnings news.",
            title_embedding=article.title_embedding,
        )

        await repository.upsert(article)
        await repository.upsert(similar_article)

        results = await repository.find_similar_by_title(
            query_embedding=article.title_embedding,
            limit=5,
        )

        assert len(results) >= 2
        urls = [a.url for a in results]
        assert article.url in urls
        assert similar_article.url in urls

    async def test_find_similar_by_content(
        self, repository, sample_article_with_embeddings
    ):
        """Test finding similar articles by content embedding."""
        article = sample_article_with_embeddings

        similar_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Different Headline But Similar Content",
            url="https://example.com/similar-content-2024",
            source="Bloomberg",
            published_date=date(2024, 1, 17),
            summary=article.summary,
            content_embedding=article.content_embedding,
        )

        await repository.upsert(article)
        await repository.upsert(similar_article)

        results = await repository.find_similar_by_content(
            query_embedding=article.content_embedding,
            limit=5,
        )

        assert len(results) >= 2
        urls = [a.url for a in results]
        assert article.url in urls
        assert similar_article.url in urls

    async def test_find_similar_by_title_with_symbol_filter(
        self, repository, sample_article_with_embeddings
    ):
        """Test similarity search with symbol filter."""
        article = sample_article_with_embeddings
        article_aapl = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Apple News",
            url="https://example.com/apple-news-1",
            source="Test",
            published_date=date(2024, 1, 16),
            title_embedding=article.title_embedding,
        )
        article_googl = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Google News",
            url="https://example.com/google-news-2",
            source="Test",
            published_date=date(2024, 1, 17),
            title_embedding=article.title_embedding,
        )

        await repository.upsert(article_aapl, symbol="AAPL")
        await repository.upsert(article_googl, symbol="GOOGL")

        results = await repository.find_similar_by_title(
            query_embedding=article.title_embedding,
            limit=5,
            symbol="AAPL",
        )

        urls = [r.url for r in results]
        assert "https://example.com/apple-news-1" in urls
        assert "https://example.com/google-news-2" not in urls

    async def test_find_similar_by_title_excludes_no_embedding(
        self, repository, sample_article_with_embeddings
    ):
        """Test that articles without title embeddings are excluded."""
        article_with_embedding = sample_article_with_embeddings

        article_without_embedding = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Article Without Embedding",
            url="https://example.com/no-embedding",
            source="Test",
            published_date=date(2024, 1, 18),
        )

        await repository.upsert(article_with_embedding)
        await repository.upsert(article_without_embedding)

        results = await repository.find_similar_by_title(
            query_embedding=article_with_embedding.title_embedding,
            limit=5,
        )

        urls = [a.url for a in results]
        assert article_with_embedding.url in urls
        assert article_without_embedding.url not in urls

    async def test_find_similar_to_article(self, repository):
        """Test finding articles similar to an existing article."""
        base_embedding = [0.5] * 1536
        reference_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Tech Stock Analysis",
            url="https://example.com/tech-analysis",
            source="WSJ",
            published_date=date(2024, 1, 19),
            summary="Analysis of tech sector performance.",
            title_embedding=base_embedding,
        )

        similar_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Similar Tech Article",
            url="https://example.com/similar-tech",
            source="CNBC",
            published_date=date(2024, 1, 20),
            title_embedding=base_embedding,
        )

        unrelated_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Unrelated Sports News",
            url="https://example.com/sports",
            source="ESPN",
            published_date=date(2024, 1, 21),
            title_embedding=[0.9] * 1536,
        )

        await repository.upsert(reference_article)
        await repository.upsert(similar_article)
        await repository.upsert(unrelated_article)

        results = await repository.find_similar_to_article(
            article_id=reference_article.id,
            limit=10,
            use_title=True,
        )

        urls = [a.url for a in results]
        assert reference_article.url not in urls
        assert similar_article.url in urls

    async def test_find_similar_to_article_not_found(self, repository):
        """Test finding similar articles when reference article doesn't exist."""
        results = await repository.find_similar_to_article(
            article_id=uuid4(),
            limit=5,
        )
        assert results == []

    async def test_find_by_sentiment_and_similarity(self, repository):
        """Test combined sentiment and similarity search."""
        positive_embedding = [0.1] * 1536
        negative_embedding = [0.9] * 1536

        positive_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Positive Tech News",
            url="https://example.com/positive-tech",
            source="TechCrunch",
            published_date=date(2024, 1, 22),
            sentiment_label="positive",
            sentiment_score=0.8,
            sentiment_confidence=0.9,
            content_embedding=positive_embedding,
        )

        negative_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Negative Tech News",
            url="https://example.com/negative-tech",
            source="Reuters",
            published_date=date(2024, 1, 23),
            sentiment_label="negative",
            sentiment_score=-0.7,
            sentiment_confidence=0.8,
            content_embedding=negative_embedding,
        )

        await repository.upsert(positive_article)
        await repository.upsert(negative_article)

        results = await repository.find_by_sentiment_and_similarity(
            sentiment_label="positive",
            query_embedding=positive_embedding,
            limit=5,
            use_title=False,
        )

        for result in results:
            assert result.sentiment_label == "positive"

    async def test_similarity_search_returns_empty_when_no_embeddings(
        self, repository
    ):
        """Test that similarity search returns empty list when no articles have embeddings."""
        article_no_embedding = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Article Without Embedding",
            url="https://example.com/no-embedding-test",
            source="Test",
            published_date=date(2024, 1, 24),
        )

        await repository.upsert(article_no_embedding)

        results = await repository.find_similar_by_title(
            query_embedding=[0.1] * 1536,
            limit=5,
        )

        assert len(results) == 0


class TestVectorSearchEdgeCases:
    """Test edge cases for vector similarity search."""

    async def test_limit_parameter(self, repository, sample_article_with_embeddings):
        """Test that limit parameter is respected."""
        base_embedding = [0.1] * 1536

        for i in range(5):
            article = NewsArticle(
                id=UUID(str(uuid7())),
                headline=f"Article {i}",
                url=f"https://example.com/article-{i}",
                source="Test",
                published_date=date(2024, 1, 1),
                title_embedding=base_embedding,
            )
            await repository.upsert(article)

        results = await repository.find_similar_by_title(
            query_embedding=base_embedding,
            limit=2,
        )

        assert len(results) <= 2

    async def test_empty_repository(self, repository):
        """Test similarity search on empty repository."""
        results = await repository.find_similar_by_title(
            query_embedding=[0.1] * 1536,
            limit=5,
        )

        assert results == []

    async def test_threshold_filters_dissimilar(
        self, repository, sample_article_with_embeddings
    ):
        """Test that threshold filters out dissimilar articles."""
        article = sample_article_with_embeddings

        dissimilar_article = NewsArticle(
            id=UUID(str(uuid7())),
            headline="Very Different Topic",
            url="https://example.com/different",
            source="Test",
            published_date=date(2024, 1, 25),
            title_embedding=[0.99] * 1536,
        )

        await repository.upsert(article)
        await repository.upsert(dissimilar_article)

        results = await repository.find_similar_by_title(
            query_embedding=article.title_embedding,
            threshold=0.99,
            limit=5,
        )

        urls = [a.url for a in results]
        assert article.url in urls
