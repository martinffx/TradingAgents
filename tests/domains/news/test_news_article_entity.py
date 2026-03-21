"""Unit tests for NewsArticle domain entity.

Tests focus on pure functions - no mocks needed for entity logic.
"""

from datetime import date
from uuid import uuid4

import pytest

from tradingagents.domains.news.news_article import NewsArticle


@pytest.fixture
def base_article():
    """Create a base article for testing transformations."""
    return NewsArticle(
        id=uuid4(),
        url="https://example.com/article",
        headline="Test Article Headline",
        source="TestSource",
        published_date=date(2024, 1, 15),
        entities=[],
    )


class TestImmutableEntity:
    """Tests for immutability."""

    def test_entity_is_frozen(self, base_article):
        """Entity should be frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            base_article.headline = "Changed"

    def test_with_methods_return_new_instance(self, base_article):
        """with_* methods should return new instances."""
        updated = base_article.with_content(summary="New content")

        assert base_article.summary is None
        assert updated.summary == "New content"
        assert updated.headline == base_article.headline


class TestHasReliableSentiment:
    """Tests for has_reliable_sentiment business logic."""

    def test_no_sentiment_returns_false(self, base_article):
        """Article without sentiment should return False."""
        assert base_article.has_reliable_sentiment() is False

    def test_low_confidence_returns_false(self, base_article):
        """Article with confidence < 0.6 should return False."""
        article = NewsArticle(
            id=base_article.id,
            url=base_article.url,
            headline=base_article.headline,
            source=base_article.source,
            published_date=base_article.published_date,
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            sentiment_label="positive",
        )
        assert article.has_reliable_sentiment() is False

    def test_high_confidence_returns_true(self, base_article):
        """Article with confidence >= 0.6 should return True."""
        article = NewsArticle(
            id=base_article.id,
            url=base_article.url,
            headline=base_article.headline,
            source=base_article.source,
            published_date=base_article.published_date,
            sentiment_score=0.8,
            sentiment_confidence=0.8,
            sentiment_label="positive",
        )
        assert article.has_reliable_sentiment() is True

    def test_exactly_06_returns_true(self, base_article):
        """Article with confidence exactly 0.6 should return True."""
        article = base_article.with_sentiment(label="neutral", confidence=0.6)
        assert article.has_reliable_sentiment() is True


class TestCalculateSentimentScore:
    """Tests for calculate_sentiment_score pure function."""

    def test_positive_sentiment(self, base_article):
        """Positive sentiment returns positive score."""
        score = base_article.calculate_sentiment_score("positive", 0.85)
        assert score == 0.85

    def test_negative_sentiment(self, base_article):
        """Negative sentiment returns negative score."""
        score = base_article.calculate_sentiment_score("negative", 0.75)
        assert score == -0.75

    def test_neutral_sentiment(self, base_article):
        """Neutral sentiment returns 0.0."""
        score = base_article.calculate_sentiment_score("neutral", 0.6)
        assert score == 0.0


class TestWithContent:
    """Tests for with_content transformation."""

    def test_adds_summary(self, base_article):
        """with_content adds summary to new instance."""
        updated = base_article.with_content(summary="Full article content here...")

        assert updated.summary == "Full article content here..."
        assert base_article.summary is None

    def test_preserves_author_if_none_provided(self, base_article):
        """Author is preserved if not provided."""
        article_with_author = NewsArticle(
            id=base_article.id,
            url=base_article.url,
            headline=base_article.headline,
            source=base_article.source,
            published_date=base_article.published_date,
            author="John Doe",
        )

        updated = article_with_author.with_content(summary="New content")

        assert updated.author == "John Doe"

    def test_updates_author_if_provided(self, base_article):
        """Author is updated if provided."""
        updated = base_article.with_content(summary="Content", author="Jane Doe")

        assert updated.author == "Jane Doe"


class TestWithSentiment:
    """Tests for with_sentiment transformation."""

    def test_adds_sentiment_data(self, base_article):
        """with_sentiment adds all sentiment fields."""
        updated = base_article.with_sentiment(label="positive", confidence=0.85)

        assert updated.sentiment_label == "positive"
        assert updated.sentiment_confidence == 0.85
        assert updated.sentiment_score == 0.85

    def test_calculates_negative_score(self, base_article):
        """Negative sentiment has negative score."""
        updated = base_article.with_sentiment(label="negative", confidence=0.7)

        assert updated.sentiment_score == -0.7

    def test_calculates_neutral_score(self, base_article):
        """Neutral sentiment has zero score."""
        updated = base_article.with_sentiment(label="neutral", confidence=0.6)

        assert updated.sentiment_score == 0.0


class TestWithEmbeddings:
    """Tests for with_embeddings transformation."""

    def test_adds_embeddings(self, base_article):
        """with_embeddings adds both embeddings."""
        title_emb = [0.1] * 1536
        content_emb = [0.2] * 1536

        updated = base_article.with_embeddings(title=title_emb, content=content_emb)

        assert updated.title_embedding == title_emb
        assert updated.content_embedding == content_emb

    def test_preserves_other_fields(self, base_article):
        """Embeddings transformation preserves other fields."""
        article_with_sentiment = base_article.with_sentiment(
            label="positive", confidence=0.8
        )
        embeddings = ([0.1] * 1536, [0.2] * 1536)

        updated = article_with_sentiment.with_embeddings(
            title=embeddings[0], content=embeddings[1]
        )

        assert updated.sentiment_label == "positive"
        assert updated.sentiment_confidence == 0.8
        assert updated.title_embedding == embeddings[0]


class TestChainedTransformations:
    """Tests for chained with_* transformations."""

    def test_full_processing_pipeline(self, base_article):
        """Chain full processing: content -> sentiment -> embeddings."""
        processed = (
            base_article.with_content(summary="Article content", author="Author")
            .with_sentiment(label="positive", confidence=0.85)
            .with_embeddings(title=[0.1, 0.2], content=[0.3, 0.4])
        )

        assert processed.summary == "Article content"
        assert processed.author == "Author"
        assert processed.sentiment_label == "positive"
        assert processed.sentiment_confidence == 0.85
        assert processed.sentiment_score == 0.85
        assert processed.title_embedding == [0.1, 0.2]
        assert processed.content_embedding == [0.3, 0.4]

    def test_original_unchanged_after_chain(self, base_article):
        """Original article is unchanged after chained transformations."""
        _ = (
            base_article.with_content(summary="Content")
            .with_sentiment(label="positive", confidence=0.8)
            .with_embeddings(title=[0.1], content=[0.2])
        )

        assert base_article.summary is None
        assert base_article.sentiment_label is None
        assert base_article.title_embedding is None


class TestSerialization:
    """Tests for to_dict and from_dict serialization."""

    def test_to_dict_roundtrip(self, base_article):
        """to_dict and from_dict are inverses."""
        with_sentiment = base_article.with_sentiment(label="positive", confidence=0.85)

        data = with_sentiment.to_dict()
        restored = NewsArticle.from_dict(data)

        assert restored.id == with_sentiment.id
        assert restored.headline == with_sentiment.headline
        assert restored.sentiment_label == with_sentiment.sentiment_label
        assert restored.sentiment_confidence == with_sentiment.sentiment_confidence

    def test_from_dict_with_all_fields(self):
        """from_dict handles all fields correctly."""
        data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "url": "https://example.com/test",
            "headline": "Full Article",
            "source": "NewsSource",
            "published_date": "2024-03-15",
            "summary": "Article summary",
            "author": "Author Name",
            "sentiment_score": 0.75,
            "sentiment_confidence": 0.9,
            "sentiment_label": "positive",
            "title_embedding": [0.1, 0.2],
            "content_embedding": [0.3, 0.4],
        }

        article = NewsArticle.from_dict(data)

        assert str(article.id) == data["id"]
        assert article.url == data["url"]
        assert article.headline == data["headline"]
        assert article.sentiment_score == 0.75
        assert article.sentiment_label == "positive"

    def test_from_dict_handles_none_optional_fields(self):
        """from_dict handles missing optional fields."""
        data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "url": "https://example.com/test",
            "headline": "Article",
            "source": "Source",
            "published_date": "2024-01-01",
        }

        article = NewsArticle.from_dict(data)

        assert article.summary is None
        assert article.sentiment_score is None
        assert article.title_embedding is None
