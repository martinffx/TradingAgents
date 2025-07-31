"""
Social Media Service for aggregating and analyzing social media data.
"""

import logging
from datetime import datetime
from typing import Any

from tradingagents.clients.base import BaseClient
from tradingagents.models.context import (
    DataQuality,
    PostData,
    SentimentScore,
    SocialContext,
)
from tradingagents.repositories.base import BaseRepository
from tradingagents.services.base import BaseService

logger = logging.getLogger(__name__)


class SocialMediaService(BaseService):
    """Service for social media data aggregation and analysis."""

    def __init__(
        self,
        reddit_client: BaseClient | None = None,
        repository: BaseRepository | None = None,
        online_mode: bool = True,
        data_dir: str = "data",
        **kwargs,
    ):
        """Initialize Social Media Service.

        Args:
            reddit_client: Client for Reddit API access
            repository: Repository for cached social data
            online_mode: Whether to fetch live data
            data_dir: Directory for data storage
        """
        super().__init__(online_mode=online_mode, data_dir=data_dir, **kwargs)
        self.reddit_client = reddit_client
        self.repository = repository

    def get_context(
        self,
        query: str,
        start_date: str,
        end_date: str,
        symbol: str | None = None,
        subreddits: list[str] | None = None,
        force_refresh: bool = False,
        **kwargs,
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
        posts = []
        error_info = {}
        data_source = "unknown"

        try:
            # Local-first data strategy with force refresh option
            if force_refresh:
                # Skip local data, fetch fresh from APIs
                posts, data_source = self._fetch_and_cache_fresh_social_data(
                    query, start_date, end_date, symbol, subreddits
                )
            else:
                # Check local data first, fetch missing if needed
                posts, data_source = self._get_social_data_local_first(
                    query, start_date, end_date, symbol, subreddits
                )

        except Exception as e:
            logger.error(f"Error fetching social media data: {e}")
            error_info = {"error": str(e)}

        # Calculate sentiment and engagement metrics
        sentiment_summary = self._calculate_sentiment(posts)
        engagement_metrics = self._calculate_engagement_metrics(posts)

        # Determine data quality based on data source
        data_quality = self._determine_data_quality(
            data_source=data_source,
            record_count=len(posts),
            has_errors=bool(error_info),
        )

        # Separate float metrics from metadata
        float_metrics = {
            k: v for k, v in engagement_metrics.items() if isinstance(v, int | float)
        }
        metadata_info = {
            k: v
            for k, v in engagement_metrics.items()
            if not isinstance(v, int | float)
        }

        return SocialContext(
            symbol=symbol,
            period={"start": start_date, "end": end_date},
            posts=posts,
            engagement_metrics=float_metrics,
            sentiment_summary=sentiment_summary,
            post_count=len(posts),
            platforms=["reddit"],
            metadata={
                "data_quality": data_quality,
                "service": "social_media",
                "online_mode": self.is_online(),
                "subreddits": subreddits or [],
                "data_source": data_source,
                "force_refresh": force_refresh,
                **metadata_info,
                **error_info,
            },
        )

    def get_company_social_context(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        subreddits: list[str] | None = None,
        **kwargs,
    ) -> SocialContext:
        """Get company-specific social media context.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            subreddits: Optional list of subreddits

        Returns:
            SocialContext for the company
        """
        return self.get_context(
            query=symbol,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            subreddits=subreddits,
            **kwargs,
        )

    def get_global_trends(
        self,
        start_date: str,
        end_date: str,
        subreddits: list[str] | None = None,
        **kwargs,
    ) -> SocialContext:
        """Get global social media trends.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            subreddits: Optional list of subreddits

        Returns:
            SocialContext with global trends
        """
        posts = []

        try:
            if self.is_online() and self.reddit_client:
                subreddit_list = subreddits or ["news", "worldnews", "Economics"]

                # Get top posts from subreddits
                raw_posts = self.reddit_client.get_top_posts(
                    subreddit_names=subreddit_list, limit=50, time_filter="week"
                )

                # Filter by date
                if hasattr(self.reddit_client, "filter_posts_by_date"):
                    raw_posts = self.reddit_client.filter_posts_by_date(
                        raw_posts, start_date, end_date
                    )

                posts = self._convert_to_post_data(raw_posts)

        except Exception as e:
            logger.error(f"Error fetching global trends: {e}")

        sentiment_summary = self._calculate_sentiment(posts)
        engagement_metrics = self._calculate_engagement_metrics(posts)

        # Separate float metrics from metadata
        float_metrics = {
            k: v for k, v in engagement_metrics.items() if isinstance(v, int | float)
        }
        metadata_info = {
            k: v
            for k, v in engagement_metrics.items()
            if not isinstance(v, int | float)
        }

        return SocialContext(
            symbol=None,  # No specific symbol for global trends
            period={"start": start_date, "end": end_date},
            posts=posts,
            engagement_metrics=float_metrics,
            sentiment_summary=sentiment_summary,
            post_count=len(posts),
            platforms=["reddit"],
            metadata={
                "data_quality": self._determine_data_quality(
                    data_source="live_api" if self.is_online() else "offline",
                    record_count=len(posts),
                    has_errors=False,
                ),
                "service": "social_media",
                "type": "global_trends",
                "subreddits": subreddits or [],
                **metadata_info,
            },
        )

    def _convert_to_post_data(self, raw_posts: list[dict[str, Any]]) -> list[PostData]:
        """Convert raw Reddit posts to PostData objects."""
        posts = []

        for post in raw_posts:
            try:
                # Calculate engagement score
                engagement = post.get("upvotes", 0) + post.get("num_comments", 0)

                # Get posted date
                if "posted_date" in post:
                    date_str = post["posted_date"]
                elif "created_utc" in post:
                    date_str = datetime.fromtimestamp(post["created_utc"]).strftime(
                        "%Y-%m-%d"
                    )
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")

                post_data = PostData(
                    title=post.get("title", ""),
                    content=post.get("content", ""),
                    author=post.get("author", "unknown"),
                    source=post.get("subreddit", "reddit"),
                    date=date_str,
                    url=post.get("url", ""),
                    score=post.get("score", 0),
                    comments=post.get("num_comments", 0),
                    engagement_score=engagement,
                    subreddit=post.get("subreddit"),
                    metadata={
                        "upvotes": post.get("upvotes", 0),
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": post.get("subreddit", ""),
                    },
                )
                posts.append(post_data)

            except Exception as e:
                logger.warning(f"Error converting post: {e}")
                continue

        return posts

    def _convert_cached_to_posts(self, cached_data: dict[str, Any]) -> list[PostData]:
        """Convert cached repository data to PostData objects."""
        posts = []

        if not cached_data or "posts" not in cached_data:
            return posts

        for post in cached_data.get("posts", []):
            try:
                posts.append(PostData(**post))
            except Exception as e:
                logger.warning(f"Error converting cached post: {e}")

        return posts

    def _get_social_data_local_first(
        self,
        query: str,
        start_date: str,
        end_date: str,
        symbol: str | None,
        subreddits: list[str] | None,
    ) -> tuple[list[PostData], str]:
        """Get social data using local-first strategy: check local data first, fetch missing if needed."""
        try:
            # Check if we have sufficient local data
            search_key = symbol or query
            if self.repository and self.repository.has_data_for_period(
                search_key, start_date, end_date, symbol=symbol
            ):
                logger.info(
                    f"Using local social data for {search_key} ({start_date} to {end_date})"
                )
                cached_data = self.repository.get_data(
                    query=search_key, start_date=start_date, end_date=end_date
                )
                posts = self._convert_cached_to_posts(cached_data)
                return posts, "local_cache"

            # We don't have sufficient local data - need to fetch from APIs
            logger.info(
                f"Local data insufficient, fetching from APIs for {search_key} ({start_date} to {end_date})"
            )
            posts, _ = self._fetch_fresh_social_data(
                query, start_date, end_date, symbol, subreddits
            )

            # Cache the fresh data if we have a repository
            if posts and self.repository:
                try:
                    posts_data = [post.model_dump() for post in posts]
                    cache_data = {
                        "query": query,
                        "symbol": symbol,
                        "posts": posts_data,
                        "subreddits": subreddits,
                        "metadata": {"cached_at": datetime.utcnow().isoformat()},
                    }
                    self.repository.store_data(search_key, cache_data, symbol=symbol)
                    logger.debug(f"Cached fresh social data for {search_key}")
                except Exception as e:
                    logger.warning(f"Failed to cache social data for {search_key}: {e}")

            return posts, "live_api"

        except Exception as e:
            logger.error(f"Error fetching social data for {query}: {e}")
            return [], "error"

    def _fetch_and_cache_fresh_social_data(
        self,
        query: str,
        start_date: str,
        end_date: str,
        symbol: str | None,
        subreddits: list[str] | None,
    ) -> tuple[list[PostData], str]:
        """Force fetch fresh social data from APIs and cache it, bypassing local data."""
        try:
            search_key = symbol or query
            logger.info(
                f"Force refreshing social data from APIs for {search_key} ({start_date} to {end_date})"
            )

            # Clear existing data if we have a repository
            if self.repository:
                try:
                    self.repository.clear_data(
                        search_key, start_date, end_date, symbol=symbol
                    )
                    logger.debug(f"Cleared existing social data for {search_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clear existing social data for {search_key}: {e}"
                    )

            # Fetch fresh data
            posts, _ = self._fetch_fresh_social_data(
                query, start_date, end_date, symbol, subreddits
            )

            # Cache the fresh data
            if posts and self.repository:
                try:
                    posts_data = [post.model_dump() for post in posts]
                    cache_data = {
                        "query": query,
                        "symbol": symbol,
                        "posts": posts_data,
                        "subreddits": subreddits,
                        "metadata": {"refreshed_at": datetime.utcnow().isoformat()},
                    }
                    self.repository.store_data(
                        search_key, cache_data, symbol=symbol, overwrite=True
                    )
                    logger.debug(f"Cached refreshed social data for {search_key}")
                except Exception as e:
                    logger.warning(
                        f"Failed to cache refreshed social data for {search_key}: {e}"
                    )

            return posts, "live_api_refresh"

        except Exception as e:
            logger.error(f"Error force refreshing social data for {query}: {e}")
            return [], "refresh_error"

    def _fetch_fresh_social_data(
        self,
        query: str,
        start_date: str,
        end_date: str,
        symbol: str | None,
        subreddits: list[str] | None,
    ) -> tuple[list[PostData], str]:
        """Fetch fresh social data from APIs."""
        posts = []

        if self.is_online() and self.reddit_client:
            # Get live Reddit data
            subreddit_list = subreddits or ["investing", "stocks", "wallstreetbets"]

            # Search for posts
            raw_posts = self.reddit_client.search_posts(
                query=query,
                subreddit_names=subreddit_list,
                limit=50,
                time_filter="week",
            )

            # Filter by date
            if hasattr(self.reddit_client, "filter_posts_by_date"):
                raw_posts = self.reddit_client.filter_posts_by_date(
                    raw_posts, start_date, end_date
                )

            # Convert to PostData objects
            posts = self._convert_to_post_data(raw_posts)

        return posts, "live_api"

    def _calculate_sentiment(self, posts: list[PostData]) -> SentimentScore:
        """Calculate overall sentiment from posts."""
        if not posts:
            return SentimentScore(score=0.0, confidence=0.0, label="neutral")

        total_score = 0.0
        total_weight = 0.0

        for post in posts:
            # Simple sentiment analysis based on keywords and engagement
            sentiment_score = self._analyze_post_sentiment(post)

            # Weight by engagement
            weight = 1 + (
                post.engagement_score / 1000
            )  # Higher engagement = more weight
            total_score += sentiment_score * weight
            total_weight += weight

            # Set individual post sentiment
            post.sentiment = SentimentScore(
                score=sentiment_score,
                confidence=0.7,  # Moderate confidence for keyword-based analysis
                label="positive"
                if sentiment_score > 0.2
                else "negative"
                if sentiment_score < -0.2
                else "neutral",
            )

        # Calculate weighted average
        avg_score = total_score / total_weight if total_weight > 0 else 0.0

        # Determine label
        if avg_score > 0.2:
            label = "positive"
        elif avg_score < -0.2:
            label = "negative"
        else:
            label = "neutral"

        # Confidence based on number of posts
        confidence = min(0.9, 0.5 + (len(posts) / 100))

        return SentimentScore(score=avg_score, confidence=confidence, label=label)

    def _analyze_post_sentiment(self, post: PostData) -> float:
        """Analyze sentiment of a single post."""
        text = f"{post.title} {post.content or ''}".lower()

        # Simple keyword-based sentiment
        positive_words = [
            "bullish",
            "moon",
            "gains",
            "buy",
            "hold",
            "amazing",
            "great",
            "excellent",
            "positive",
            "growth",
            "beat",
            "upgrade",
            "🚀",
        ]
        negative_words = [
            "bearish",
            "crash",
            "sell",
            "loss",
            "decline",
            "terrible",
            "bad",
            "negative",
            "downgrade",
            "warning",
            "overvalued",
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        # Score from -1 to 1
        if positive_count + negative_count == 0:
            return 0.0

        score = (positive_count - negative_count) / (positive_count + negative_count)

        # Adjust for score ratio (upvotes vs downvotes implied)
        if post.score > 0:
            score_adjustment = min(0.2, post.score / 1000)
            score = score * 0.8 + score_adjustment * 0.2

        return max(-1.0, min(1.0, score))

    def _calculate_engagement_metrics(self, posts: list[PostData]) -> dict[str, float]:
        """Calculate engagement metrics from posts."""
        if not posts:
            return {
                "total_engagement": 0,
                "average_engagement": 0,
                "max_engagement": 0,
                "total_posts": 0,
            }

        engagements = [post.engagement_score for post in posts]

        metrics = {
            "total_engagement": sum(engagements),
            "average_engagement": sum(engagements) / len(engagements),
            "max_engagement": max(engagements),
            "total_posts": len(posts),
        }

        # Add top posts info
        sorted_posts = sorted(posts, key=lambda p: p.engagement_score, reverse=True)
        metrics["top_posts"] = [
            {"title": p.title[:100], "engagement": p.engagement_score}
            for p in sorted_posts[:3]
        ]

        return metrics

    def _determine_data_quality(
        self, data_source: str, record_count: int, has_errors: bool = False
    ) -> DataQuality:
        """Determine data quality based on source, record count, and errors."""
        if has_errors or record_count == 0:
            return DataQuality.LOW

        if data_source in ["local_cache", "error", "refresh_error"]:
            return DataQuality.LOW
        elif data_source in ["live_api", "live_api_refresh"]:
            if record_count >= 20:
                return DataQuality.HIGH
            elif record_count >= 5:
                return DataQuality.MEDIUM
            else:
                return DataQuality.LOW
        else:
            return DataQuality.MEDIUM
