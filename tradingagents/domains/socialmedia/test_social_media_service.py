#!/usr/bin/env python3
"""
Test SocialMediaService with mock RedditClient and real SocialRepository.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from tradingagents.clients.base import BaseClient
from tradingagents.domains.socialmedia.social_media_service import (
    DataQuality,
    PostData,
    SentimentScore,
    SocialContext,
    SocialMediaService,
)
from tradingagents.repositories.social_repository import SocialRepository


class MockRedditClient(BaseClient):
    """Mock Reddit client that returns sample social media data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_works = True

    def test_connection(self) -> bool:
        return self.connection_works

    def get_data(self, *args, **kwargs) -> dict[str, Any]:
        """Not used directly by SocialMediaService."""
        return {}

    def search_posts(
        self,
        query: str,
        subreddit_names: list[str],
        limit: int = 25,
        time_filter: str = "week",
    ) -> list[dict[str, Any]]:
        """Return mock Reddit search results."""
        posts = []
        # Use fixed dates that will work with our date filter
        base_date = datetime(2024, 1, 2)  # Within our test range

        for i, subreddit in enumerate(
            subreddit_names[:2]
        ):  # Limit to 2 subreddits for testing
            posts.extend(
                [
                    {
                        "title": f"{query} to the moon! 🚀",
                        "content": f"DD on {query}: Strong fundamentals, great earnings beat. Buy and hold!",
                        "url": f"https://reddit.com/r/{subreddit}/post1",
                        "upvotes": 1500 - (i * 100),
                        "score": 1450 - (i * 100),
                        "num_comments": 234,
                        "created_utc": (base_date + timedelta(hours=i)).timestamp(),
                        "subreddit": subreddit,
                        "author": f"WSBtrader{i}",
                        "posted_date": (base_date + timedelta(hours=i)).strftime(
                            "%Y-%m-%d"
                        ),
                    },
                    {
                        "title": f"Why I'm bearish on {query}",
                        "content": f"Overvalued, competition increasing, margins declining. Time to sell {query}.",
                        "url": f"https://reddit.com/r/{subreddit}/post2",
                        "upvotes": 800 - (i * 50),
                        "score": 750 - (i * 50),
                        "num_comments": 156,
                        "created_utc": (base_date + timedelta(hours=i + 1)).timestamp(),
                        "subreddit": subreddit,
                        "author": f"BearishTrader{i}",
                        "posted_date": (base_date + timedelta(hours=i + 1)).strftime(
                            "%Y-%m-%d"
                        ),
                    },
                ]
            )
        return posts

    def get_top_posts(
        self, subreddit_names: list[str], limit: int = 25, time_filter: str = "week"
    ) -> list[dict[str, Any]]:
        """Return mock top posts from subreddits."""
        posts = []
        # Use fixed dates that will work with our date filter
        base_date = datetime(2024, 1, 2)  # Within our test range

        for subreddit in subreddit_names[:2]:
            posts.append(
                {
                    "title": "Market Update: Tech stocks rally continues",
                    "content": "FAANG stocks leading the charge. SPY hit new ATH. Bull market confirmed.",
                    "url": f"https://reddit.com/r/{subreddit}/top1",
                    "upvotes": 2500,
                    "score": 2400,
                    "num_comments": 456,
                    "created_utc": base_date.timestamp(),
                    "subreddit": subreddit,
                    "author": "MarketWatcher",
                    "posted_date": base_date.strftime("%Y-%m-%d"),
                }
            )
        return posts

    def filter_posts_by_date(
        self, posts: list[dict[str, Any]], start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """Filter posts by date range."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        filtered = []
        for post in posts:
            if "posted_date" in post:
                post_dt = datetime.strptime(post["posted_date"], "%Y-%m-%d")
                if start_dt <= post_dt <= end_dt:
                    filtered.append(post)
        return filtered


def test_online_mode_with_mock_reddit():
    """Test SocialMediaService in online mode with mock Reddit client."""
    # Create mock client and real repository
    mock_reddit = MockRedditClient()
    real_repo = SocialRepository("test_data")

    # Create service in online mode
    service = SocialMediaService(
        reddit_client=mock_reddit,
        repository=real_repo,
        online_mode=True,
        data_dir="test_data",
    )

    # Test company-specific social context
    context = service.get_company_social_context(
        symbol="TSLA",
        start_date="2024-01-01",
        end_date="2024-01-05",
        subreddits=["wallstreetbets", "stocks"],
        force_refresh=True,
    )

    # Validate context structure
    assert isinstance(context, SocialContext)
    assert context.symbol == "TSLA"
    assert context.period["start"] == "2024-01-01"
    assert context.period["end"] == "2024-01-05"
    assert len(context.posts) > 0
    assert isinstance(context.sentiment_summary, SentimentScore)
    assert context.post_count == len(context.posts)
    assert "data_quality" in context.metadata

    # Test JSON serialization
    json_output = context.model_dump_json(indent=2)
    assert len(json_output) > 0

    # Validate individual posts
    for post in context.posts:
        assert isinstance(post, PostData)
        assert post.title
        assert post.author
        assert post.date
        assert post.score >= 0


def test_global_social_trends():
    """Test global social media trends functionality."""
    mock_reddit = MockRedditClient()
    real_repo = SocialRepository("test_data")

    service = SocialMediaService(
        reddit_client=mock_reddit, repository=real_repo, online_mode=True
    )

    # Test global trends
    context = service.get_global_trends(
        start_date="2024-01-01",
        end_date="2024-01-03",
        subreddits=["investing", "stocks", "wallstreetbets"],
        force_refresh=True,
    )

    # Validate global context
    assert context.symbol is None  # Global trends have no specific symbol
    assert len(context.posts) > 0
    assert "reddit" in context.platforms
    assert "subreddits" in context.metadata


def test_sentiment_analysis():
    """Test sentiment analysis on social posts."""

    # Create service with posts that have clear sentiment
    class SentimentTestClient(MockRedditClient):
        def search_posts(self, query, subreddit_names, limit=25, time_filter="week"):
            return [
                {
                    "title": f"{query} is the best investment ever! 🚀🚀🚀",
                    "content": "Amazing earnings, incredible growth, bullish AF!",
                    "upvotes": 5000,
                    "score": 4900,
                    "num_comments": 500,
                    "subreddit": "wallstreetbets",
                    "author": "BullishTrader",
                    "posted_date": "2024-01-01",
                },
                {
                    "title": f"WARNING: {query} is about to crash hard",
                    "content": "Terrible fundamentals, overvalued, sell now before it's too late!",
                    "upvotes": 100,
                    "score": 50,
                    "num_comments": 30,
                    "subreddit": "stocks",
                    "author": "BearishAnalyst",
                    "posted_date": "2024-01-01",
                },
            ]

    sentiment_client = SentimentTestClient()
    service = SocialMediaService(
        reddit_client=sentiment_client, repository=None, online_mode=True
    )

    context = service.get_context("GME", "2024-01-01", "2024-01-02")

    # Check sentiment analysis
    assert context.sentiment_summary.score != 0  # Should have some sentiment
    assert context.sentiment_summary.confidence > 0
    assert context.sentiment_summary.label in ["positive", "negative", "neutral"]

    # Check individual post sentiments
    for post in context.posts:
        if post.sentiment:
            assert -1.0 <= post.sentiment.score <= 1.0


def test_offline_mode():
    """Test SocialMediaService in offline mode."""
    real_repo = SocialRepository("test_data")

    service = SocialMediaService(
        reddit_client=None, repository=real_repo, online_mode=False
    )

    # Should handle offline gracefully
    context = service.get_context("AAPL", "2024-01-01", "2024-01-05", symbol="AAPL")

    assert context.symbol == "AAPL"
    assert isinstance(context.posts, list)
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_engagement_metrics():
    """Test calculation of engagement metrics."""
    mock_reddit = MockRedditClient()
    service = SocialMediaService(
        reddit_client=mock_reddit, repository=None, online_mode=True
    )

    context = service.get_company_social_context(
        symbol="NVDA",
        start_date="2024-01-01",
        end_date="2024-01-02",
        subreddits=["nvidia", "stocks"],
    )

    # Check engagement metrics in the context
    assert len(context.engagement_metrics) > 0
    assert (
        "total_engagement" in context.engagement_metrics
        or "total_engagement" in context.metadata
    )

    # Verify post scores
    for post in context.posts:
        # Posts should have score and comments
        assert post.score >= 0
        assert post.comments >= 0


def test_subreddit_filtering():
    """Test filtering by specific subreddits."""
    mock_reddit = MockRedditClient()
    service = SocialMediaService(
        reddit_client=mock_reddit, repository=None, online_mode=True
    )

    # Test with specific subreddits
    context = service.get_company_social_context(
        symbol="AMD",
        start_date="2024-01-01",
        end_date="2024-01-02",
        subreddits=["AMD_Stock", "wallstreetbets"],
    )

    # Check that posts are from requested subreddits
    subreddit_set = set()
    for post in context.posts:
        if post.subreddit:
            subreddit_set.add(post.subreddit)

    assert len(subreddit_set) > 0
    assert all(sub in ["AMD_Stock", "wallstreetbets"] for sub in subreddit_set)


def test_error_handling():
    """Test error handling with broken client."""

    class BrokenRedditClient(BaseClient):
        def test_connection(self):
            return False

        def get_data(self, *args, **kwargs):
            raise Exception("Reddit API error")

        def search_posts(self, *args, **kwargs):
            raise Exception("Reddit API error")

        def get_top_posts(self, *args, **kwargs):
            raise Exception("Reddit API error")

    broken_client = BrokenRedditClient()
    service = SocialMediaService(
        reddit_client=broken_client, repository=None, online_mode=True
    )

    # Should handle errors gracefully
    context = service.get_context("TSLA", "2024-01-01", "2024-01-02", symbol="TSLA")

    assert context.symbol == "TSLA"
    assert len(context.posts) == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_json_structure():
    """Test JSON structure of social context."""
    mock_reddit = MockRedditClient()
    service = SocialMediaService(
        reddit_client=mock_reddit, repository=None, online_mode=True
    )

    context = service.get_context("PLTR", "2024-01-01", "2024-01-02")
    json_data = context.model_dump()

    # Validate required fields
    required_fields = [
        "symbol",
        "period",
        "posts",
        "sentiment_summary",
        "post_count",
        "platforms",
        "metadata",
    ]
    for field in required_fields:
        assert field in json_data

    # Validate posts structure
    if json_data["posts"]:
        first_post = json_data["posts"][0]
        post_fields = ["title", "author", "date", "score"]
        for field in post_fields:
            assert field in first_post

    # Validate sentiment structure
    sentiment = json_data["sentiment_summary"]
    assert "score" in sentiment
    assert "confidence" in sentiment
    assert "label" in sentiment
