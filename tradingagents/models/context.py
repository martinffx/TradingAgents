"""
Pydantic models for structured context objects in TradingAgents.

These models define the schema for JSON context objects that services
provide to agents, replacing the previous markdown string approach.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class DataQuality(str, Enum):
    """Data quality indicator for context metadata."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SentimentScore(BaseModel):
    """Sentiment analysis result with confidence."""

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score from -1 (negative) to 1 (positive)",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in sentiment score"
    )
    label: str | None = Field(
        default=None, description="Human-readable sentiment label"
    )

    @validator("label", pre=True, always=True)
    def set_sentiment_label(cls, v, values):
        if v is not None:
            return v

        score = values.get("score", 0)
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"


class TechnicalIndicatorData(BaseModel):
    """Technical indicator data point."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    value: float | dict[str, float] = Field(
        ..., description="Indicator value or values"
    )
    indicator_type: str = Field(
        ..., description="Type of indicator (e.g., 'rsi', 'macd', 'sma')"
    )


class ArticleData(BaseModel):
    """News article data."""

    headline: str = Field(..., description="Article headline")
    summary: str | None = Field(default=None, description="Article summary or snippet")
    url: str | None = Field(default=None, description="Article URL")
    source: str = Field(..., description="News source")
    date: str = Field(..., description="Publication date in YYYY-MM-DD format")
    sentiment: SentimentScore | None = Field(
        default=None, description="Article sentiment analysis"
    )
    entities: list[str] = Field(
        default_factory=list, description="Named entities mentioned"
    )


class PostData(BaseModel):
    """Social media post data."""

    title: str = Field(..., description="Post title")
    content: str | None = Field(default=None, description="Post content")
    author: str = Field(..., description="Post author")
    source: str = Field(..., description="Post source (e.g., subreddit name)")
    date: str = Field(..., description="Post date in YYYY-MM-DD format")
    url: str | None = Field(default=None, description="Post URL")
    score: int = Field(default=0, description="Post score/upvotes")
    comments: int = Field(default=0, description="Number of comments")
    engagement_score: int = Field(default=0, description="Combined engagement metric")
    subreddit: str | None = Field(default=None, description="Subreddit name")
    sentiment: SentimentScore | None = Field(
        default=None, description="Post sentiment analysis"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional post metadata"
    )


class FinancialStatement(BaseModel):
    """Financial statement data."""

    period: str = Field(..., description="Reporting period (e.g., '2024-Q1', '2024')")
    report_date: str = Field(..., description="Report date in YYYY-MM-DD format")
    publish_date: str = Field(..., description="Publish date in YYYY-MM-DD format")
    currency: str = Field(default="USD", description="Currency of financial data")
    data: dict[str, float] = Field(..., description="Financial statement line items")


class InsiderTransaction(BaseModel):
    """Insider trading transaction data."""

    filing_date: str = Field(..., description="Filing date in YYYY-MM-DD format")
    name: str = Field(..., description="Insider name")
    change: float = Field(..., description="Change in shares")
    shares: float = Field(..., description="Total shares after transaction")
    transaction_price: float = Field(..., description="Price per share")
    transaction_code: str = Field(
        ..., description="Transaction code (e.g., 'S' for sale)"
    )


class MarketDataContext(BaseModel):
    """Market data context with price and technical indicators."""

    symbol: str = Field(..., description="Stock ticker symbol")
    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    price_data: list[dict[str, Any]] = Field(..., description="Historical price data")
    technical_indicators: dict[str, list[TechnicalIndicatorData]] = Field(
        default_factory=dict, description="Technical indicators organized by type"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata including data quality and source info",
    )

    @validator("period")
    def validate_period(cls, v):
        required_keys = {"start", "end"}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"Period must contain keys: {required_keys}")
        return v


class NewsContext(BaseModel):
    """News context with articles and sentiment analysis."""

    symbol: str | None = Field(
        default=None, description="Stock ticker if company-specific"
    )
    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    articles: list[ArticleData] = Field(..., description="News articles")
    sentiment_summary: SentimentScore = Field(
        ..., description="Overall sentiment across articles"
    )
    article_count: int = Field(..., description="Total number of articles")
    sources: list[str] = Field(default_factory=list, description="Unique news sources")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata including data quality and coverage info",
    )

    @validator("article_count", pre=True, always=True)
    def set_article_count(cls, v, values):
        if v is not None:
            return v
        return len(values.get("articles", []))

    @validator("sources", pre=True, always=True)
    def set_sources(cls, v, values):
        if v:
            return v
        articles = values.get("articles", [])
        return list({article.source for article in articles})


class SocialContext(BaseModel):
    """Social media context with posts and engagement metrics."""

    symbol: str | None = Field(
        default=None, description="Stock ticker if company-specific"
    )
    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    posts: list[PostData] = Field(..., description="Social media posts")
    engagement_metrics: dict[str, float] = Field(
        default_factory=dict, description="Aggregated engagement metrics"
    )
    sentiment_summary: SentimentScore = Field(
        ..., description="Overall sentiment across posts"
    )
    post_count: int = Field(..., description="Total number of posts")
    platforms: list[str] = Field(
        default_factory=list, description="Social media platforms"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata including data quality and platform info",
    )

    @validator("post_count", pre=True, always=True)
    def set_post_count(cls, v, values):
        if v is not None:
            return v
        return len(values.get("posts", []))

    @property
    def platform(self) -> str | None:
        """Primary platform for backward compatibility."""
        return self.platforms[0] if self.platforms else None


class FundamentalContext(BaseModel):
    """Fundamental analysis context with financial statements."""

    symbol: str = Field(..., description="Stock ticker symbol")
    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    balance_sheet: FinancialStatement | None = Field(
        default=None, description="Balance sheet data"
    )
    income_statement: FinancialStatement | None = Field(
        default=None, description="Income statement data"
    )
    cash_flow: FinancialStatement | None = Field(
        default=None, description="Cash flow statement data"
    )
    key_ratios: dict[str, float] = Field(
        default_factory=dict, description="Calculated financial ratios"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata including data quality and completeness",
    )


class InsiderContext(BaseModel):
    """Insider trading context with transaction data and sentiment."""

    symbol: str = Field(..., description="Stock ticker symbol")
    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    transactions: list[InsiderTransaction] = Field(
        ..., description="Insider transactions"
    )
    sentiment_data: dict[str, Any] = Field(
        default_factory=dict, description="Insider sentiment metrics"
    )
    transaction_count: int = Field(..., description="Total number of transactions")
    net_activity: dict[str, float] = Field(
        default_factory=dict, description="Net buying/selling activity metrics"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context metadata including data quality and coverage",
    )

    @validator("transaction_count", pre=True, always=True)
    def set_transaction_count(cls, v, values):
        if v is not None:
            return v
        return len(values.get("transactions", []))


# Base context for extensibility
class BaseContext(BaseModel):
    """Base context model for common fields."""

    period: dict[str, str] = Field(
        ..., description="Date range with start and end keys"
    )
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "data_quality": DataQuality.MEDIUM,
            "created_at": datetime.utcnow().isoformat(),
            "source": "unknown",
        },
        description="Context metadata",
    )

    class Config:
        use_enum_values = True  # Serialize enums as values
        json_encoders = {datetime: lambda v: v.isoformat()}
