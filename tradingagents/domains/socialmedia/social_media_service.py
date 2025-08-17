"""
Social Media Service for aggregating and analyzing social media data.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tradingagents.config import TradingAgentsConfig

from .reddit_client import RedditClient
from .social_media_repository import SocialMediaRepository

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels for social media data."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SentimentScore:
    """Sentiment analysis score."""

    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    label: str  # positive/negative/neutral


@dataclass
class PostMetadata:
    upvotes: int
    num_comments: int
    subreddit: str


@dataclass
class PostData:
    title: str
    content: str
    author: str
    source: str  # subreddit name or "reddit"
    date: str  # YYYY-MM-DD format
    url: str
    score: int  # Reddit score/upvotes
    comments: int  # Number of comments
    engagement_score: int  # Calculated: upvotes + comments
    subreddit: str | None
    sentiment: SentimentScore | None = None  # Added by sentiment analysis
    metadata: PostMetadata | None = None


@dataclass
class EngagementMetrics:
    """Engagement metrics for social media posts."""

    total_engagement: float
    average_engagement: float
    max_engagement: float
    total_posts: int


@dataclass
class SocialContext:
    """Social media context data for trading analysis."""

    symbol: str | None
    period: tuple[str, str]  # (start_date, end_date)
    posts: list[PostData]
    engagement_metrics: EngagementMetrics
    sentiment_summary: SentimentScore
    post_count: int
    platforms: list[str]
    metadata: dict[str, Any]


@dataclass
class StockSocialContext:
    """Stock-specific social media context for targeted analysis."""

    symbol: str
    period: dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    posts: list[PostData]
    engagement_metrics: EngagementMetrics
    sentiment_summary: SentimentScore
    post_count: int
    platforms: list[str]
    trending_topics: list[str]
    metadata: dict[str, Any]


class SocialMediaService:
    """Service for social media data aggregation and analysis."""

    def __init__(
        self,
        reddit_client: RedditClient,
        repository: SocialMediaRepository,
    ):
        """Initialize Social Media Service.

        Args:
            reddit_client: Client for Reddit API access
            repository: Repository for cached social data
            online_mode: Whether to fetch live data
            data_dir: Directory for data storage
        """
        self.reddit_client = reddit_client
        self.repository = repository

    @staticmethod
    def build(_config: TradingAgentsConfig):
        client = RedditClient()
        repo = SocialMediaRepository("")
        return SocialMediaService(client, repo)

    def get_context(
        self,
        query: str,
        start_date: str,
        end_date: str,
        symbol: str,
        subreddits: list[str],
        force_refresh: bool = False,
    ) -> SocialContext:
        """Get social media context for a query.

        Args:
            query: Search query
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbol: Optional stock symbol
            subreddits: Optional list of subreddits to search
            force_refresh: If True, skip local data and fetch fresh from APIs

        Returns:
            SocialContext with posts and sentiment analysis
        """
        _ = query  # Acknowledge unused parameter
        posts = []
        data_source = "unknown"

        # Create structured engagement metrics
        structured_metrics = EngagementMetrics(
            total_engagement=0,
            average_engagement=0,
            max_engagement=0,
            total_posts=0,
        )

        return SocialContext(
            symbol=symbol,
            period=(start_date, end_date),
            posts=posts,
            engagement_metrics=structured_metrics,
            sentiment_summary=SentimentScore(score=0, confidence=0, label=""),
            post_count=len(posts),
            platforms=["reddit"],
            metadata={
                "service": "social_media",
                "subreddits": subreddits or [],
                "data_source": data_source,
                "force_refresh": force_refresh,
            },
        )

    def get_stock_social_context(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        subreddits: list[str] | None = None,
        **kwargs,
    ) -> StockSocialContext:
        """
        Get stock-specific social media context with trending topics.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            subreddits: List of subreddits to search
            **kwargs: Additional parameters

        Returns:
            StockSocialContext: Focused stock social media context
        """
        # Use existing get_context method
        base_context = self.get_context(
            query=symbol,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            subreddits=subreddits or [],
            **kwargs,
        )

        # TODO: Extract trending topics from posts
        trending_topics = []

        return StockSocialContext(
            symbol=symbol,
            period={"start": start_date, "end": end_date},
            posts=base_context.posts,
            engagement_metrics=base_context.engagement_metrics,
            sentiment_summary=base_context.sentiment_summary,
            post_count=base_context.post_count,
            platforms=base_context.platforms,
            trending_topics=trending_topics,
            metadata=base_context.metadata,
        )
