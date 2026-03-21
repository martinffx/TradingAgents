"""Domain entity for news articles - immutable with transformation methods."""

from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Self
from uuid import UUID

if TYPE_CHECKING:
    from tradingagents.domains.news.news_repository import NewsArticleEntity


@dataclass(frozen=True)
class NewsArticle:
    """Immutable domain entity for news articles.

    Follows functional core / imperative shell pattern:
    - Pure transformation methods (with_*)
    - Business logic encapsulated in entity
    - Immutable - all mutations return new instances
    """

    id: UUID
    url: str
    headline: str
    source: str
    published_date: date

    summary: str | None = None
    author: str | None = None
    category: str | None = None
    entities: list[str] = field(default_factory=list)

    sentiment_score: float | None = None
    sentiment_confidence: float | None = None
    sentiment_label: str | None = None

    title_embedding: list[float] | None = None
    content_embedding: list[float] | None = None

    def has_reliable_sentiment(self) -> bool:
        """Check if article has reliable sentiment data.

        Returns True when confidence >= 0.6 (domain rule).
        """
        return (
            self.sentiment_score is not None
            and self.sentiment_confidence is not None
            and self.sentiment_confidence >= 0.6
        )

    def calculate_sentiment_score(self, label: str, confidence: float) -> float:
        """Calculate sentiment score from label and confidence.

        Pure function - no side effects.

        Args:
            label: Sentiment label (positive/negative/neutral)
            confidence: Confidence score (0.0-1.0)

        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if label == "positive":
            return confidence
        elif label == "negative":
            return -confidence
        return 0.0

    def with_content(self, summary: str, author: str | None = None) -> "NewsArticle":
        """Return new article with scraped content.

        Pure transformation - returns new immutable instance.
        """
        return replace(
            self,
            summary=summary,
            author=author if author else self.author,
        )

    def with_sentiment(self, label: str, confidence: float) -> "NewsArticle":
        """Return new article with sentiment data.

        Pure transformation - calculates score and returns new instance.
        """
        score = self.calculate_sentiment_score(label, confidence)
        return replace(
            self,
            sentiment_label=label,
            sentiment_confidence=confidence,
            sentiment_score=score,
        )

    def with_embeddings(
        self, title: list[float], content: list[float]
    ) -> "NewsArticle":
        """Return new article with vector embeddings.

        Pure transformation - returns new immutable instance.
        """
        return replace(
            self,
            title_embedding=title,
            content_embedding=content,
        )

    @classmethod
    def from_record(cls, record: "NewsArticleEntity") -> Self:
        """Transform database record to domain entity.

        Factory method - creates immutable entity from ORM model.
        """
        return cls(
            id=record.id,
            url=record.url,
            headline=record.headline,
            source=record.source,
            published_date=record.published_date,
            summary=record.summary,
            author=record.author,
            category=record.category,
            entities=record.entities or [],
            sentiment_score=record.sentiment_score,
            sentiment_confidence=record.sentiment_confidence,
            sentiment_label=record.sentiment_label,
            title_embedding=record.title_embedding,
            content_embedding=record.content_embedding,
        )

    def to_record(self, symbol: str | None = None) -> "NewsArticleEntity":
        """Transform domain entity to database record.

        Args:
            symbol: Optional symbol for the article record.

        Returns:
            ORM model for persistence.
        """
        from tradingagents.domains.news.news_repository import NewsArticleEntity

        return NewsArticleEntity(
            id=self.id,
            url=self.url,
            headline=self.headline,
            source=self.source,
            published_date=self.published_date,
            summary=self.summary,
            author=self.author,
            category=self.category,
            entities=self.entities,
            symbol=symbol,
            sentiment_score=self.sentiment_score,
            sentiment_confidence=self.sentiment_confidence,
            sentiment_label=self.sentiment_label,
            title_embedding=self.title_embedding,
            content_embedding=self.content_embedding,
        )

    def to_dict(self) -> dict:
        """Serialize for Temporal workflow transmission."""
        return {
            "id": str(self.id),
            "url": self.url,
            "headline": self.headline,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "summary": self.summary,
            "author": self.author,
            "category": self.category,
            "entities": self.entities,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_label": self.sentiment_label,
            "title_embedding": self.title_embedding,
            "content_embedding": self.content_embedding,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from Temporal workflow transmission."""
        return cls(
            id=UUID(data["id"]),
            url=data["url"],
            headline=data["headline"],
            source=data["source"],
            published_date=date.fromisoformat(data["published_date"]),
            summary=data.get("summary"),
            author=data.get("author"),
            category=data.get("category"),
            entities=data.get("entities", []),
            sentiment_score=data.get("sentiment_score"),
            sentiment_confidence=data.get("sentiment_confidence"),
            sentiment_label=data.get("sentiment_label"),
            title_embedding=data.get("title_embedding"),
            content_embedding=data.get("content_embedding"),
        )
