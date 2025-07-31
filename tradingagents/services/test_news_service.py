#!/usr/bin/env python3
"""
Test NewsService with mock clients and real NewsRepository.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from tradingagents.clients.base import BaseClient
from tradingagents.models.context import NewsContext, SentimentScore
from tradingagents.repositories.news_repository import NewsRepository
from tradingagents.services.news_service import NewsService


class MockFinnhubClient(BaseClient):
    """Mock Finnhub client that returns sample news data."""

    def test_connection(self) -> bool:
        return True

    def get_data(self, *args, **kwargs) -> dict[str, Any]:
        """Not used directly by NewsService."""
        return {}

    def get_company_news(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """Return mock Finnhub company news."""
        return {
            "symbol": symbol,
            "period": {"start": start_date, "end": end_date},
            "articles": [
                {
                    "headline": f"{symbol} Beats Q4 Earnings Expectations",
                    "summary": f"{symbol} reported earnings of $2.50 per share, beating analyst estimates of $2.25.",
                    "url": f"https://example.com/finnhub/{symbol.lower()}-earnings",
                    "source": "Finnhub Financial",
                    "date": start_date,
                    "entities": [symbol],
                },
                {
                    "headline": f"Insider Trading Activity at {symbol}",
                    "summary": f"Company executives at {symbol} have increased their holdings by 15% this quarter.",
                    "url": f"https://example.com/finnhub/{symbol.lower()}-insider",
                    "source": "Finnhub SEC Filings",
                    "date": end_date,
                    "entities": [symbol, "insider trading"],
                },
            ],
            "metadata": {
                "source": "mock_finnhub",
                "article_count": 2,
                "retrieved_at": datetime.utcnow().isoformat(),
            },
        }


class MockGoogleNewsClient(BaseClient):
    """Mock Google News client that returns sample articles."""

    def test_connection(self) -> bool:
        return True

    def get_data(
        self, query: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """Return mock Google News data."""
        article_templates = [
            {
                "template": "{query} Stock Surges on Positive Outlook",
                "summary": "Shares of {query} rose 5% in after-hours trading following strong guidance for next quarter.",
                "source": "Mock Market News",
            },
            {
                "template": "Analysts Recommend Buy Rating for {query}",
                "summary": "Three major investment firms upgraded {query} to 'Buy' with improved price targets.",
                "source": "Mock Investment Daily",
            },
            {
                "template": "{query} Announces Strategic Partnership",
                "summary": "The company revealed a new collaboration that could expand market reach significantly.",
                "source": "Mock Business Wire",
            },
        ]

        articles = []
        for i, template in enumerate(article_templates):
            current_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i)

            articles.append(
                {
                    "headline": template["template"].format(query=query),
                    "summary": template["summary"].format(query=query),
                    "url": f"https://example.com/google/{query.lower()}-{i}",
                    "source": template["source"],
                    "date": current_date.strftime("%Y-%m-%d"),
                    "entities": [query],
                }
            )

        return {
            "query": query,
            "period": {"start": start_date, "end": end_date},
            "articles": articles,
            "metadata": {
                "source": "mock_google_news",
                "article_count": len(articles),
                "retrieved_at": datetime.utcnow().isoformat(),
            },
        }


def test_online_mode_with_mock_clients():
    """Test NewsService in online mode with mock clients."""
    print("📰 Testing NewsService - Online Mode")

    # Create mock clients and real repository
    mock_finnhub = MockFinnhubClient()
    mock_google = MockGoogleNewsClient()
    real_repo = NewsRepository("test_data")

    # Create service in online mode
    service = NewsService(
        finnhub_client=mock_finnhub,
        google_client=mock_google,
        repository=real_repo,
        online_mode=True,
        data_dir="test_data",
    )

    try:
        # Test company news context
        context = service.get_company_news_context(
            symbol="AAPL", start_date="2024-01-01", end_date="2024-01-05"
        )

        print(f"✅ Company news context created: {context.__class__.__name__}")
        print(f"   Symbol: {context.symbol}")
        print(f"   Period: {context.period}")
        print(f"   Articles: {len(context.articles)}")
        print(f"   Sentiment score: {context.sentiment_summary.score:.3f}")
        print(f"   Sentiment confidence: {context.sentiment_summary.confidence:.3f}")
        print(f"   Sources: {context.sources}")

        # Validate required fields
        assert context.symbol == "AAPL"
        assert context.period["start"] == "2024-01-01"
        assert context.period["end"] == "2024-01-05"
        assert len(context.articles) > 0
        assert (
            context.sentiment_summary.score >= -1.0
            and context.sentiment_summary.score <= 1.0
        )
        assert "data_quality" in context.metadata

        print("✅ Basic validation passed")

        # Test JSON serialization
        json_output = context.model_dump_json(indent=2)
        parsed = json.loads(json_output)

        print(f"✅ JSON serialization: {len(json_output)} characters")
        print(f"   Top-level keys: {list(parsed.keys())}")

        return True

    except Exception as e:
        print(f"❌ Online mode test failed: {e}")
        return False


def test_global_news_context():
    """Test global news functionality."""
    print("\n🌍 Testing Global News Context")

    mock_google = MockGoogleNewsClient()
    real_repo = NewsRepository("test_data")

    service = NewsService(
        finnhub_client=None,
        google_client=mock_google,
        repository=real_repo,
        online_mode=True,
        data_dir="test_data",
    )

    try:
        # Test global news with categories
        context = service.get_global_news_context(
            start_date="2024-01-01",
            end_date="2024-01-03",
            categories=["economy", "markets"],
        )

        print("✅ Global news context created")
        print(f"   Symbol: {context.symbol}")  # Should be None for global news
        print(f"   Articles: {len(context.articles)}")
        print(f"   Categories searched: {context.metadata.get('categories', [])}")
        print(f"   Sentiment score: {context.sentiment_summary.score:.3f}")

        # Validate global news structure
        assert context.symbol is None  # Global news shouldn't have a symbol
        assert len(context.articles) > 0
        assert "categories" in context.metadata

        print("✅ Global news validation passed")

        return True

    except Exception as e:
        print(f"❌ Global news test failed: {e}")
        return False


def test_offline_mode_with_real_repository():
    """Test NewsService in offline mode with real repository."""
    print("\n💾 Testing NewsService - Offline Mode")

    # Create service in offline mode (no clients)
    real_repo = NewsRepository("test_data")
    service = NewsService(
        finnhub_client=None,
        google_client=None,
        repository=real_repo,
        online_mode=False,
        data_dir="test_data",
    )

    try:
        # Test offline context (will likely return empty data)
        context = service.get_company_news_context(
            symbol="AAPL", start_date="2024-01-01", end_date="2024-01-05"
        )

        print(f"✅ Offline context created: {context.__class__.__name__}")
        print(f"   Symbol: {context.symbol}")
        print(f"   Articles: {len(context.articles)}")
        print(f"   Data quality: {context.metadata.get('data_quality')}")
        print(f"   Service mode: online={service.is_online()}")

        # Should handle empty data gracefully
        assert context.symbol == "AAPL"
        assert isinstance(context.articles, list)
        assert isinstance(context.sentiment_summary, SentimentScore)
        assert "data_quality" in context.metadata

        print("✅ Offline mode graceful handling verified")

        return True

    except Exception as e:
        print(f"❌ Offline mode test failed: {e}")
        return False


def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("\n😊 Testing Sentiment Analysis")

    # Create service with custom articles for sentiment testing
    class SentimentTestClient(BaseClient):
        def test_connection(self):
            return True

        def get_data(self, query, start_date, end_date, **kwargs):
            return {
                "query": query,
                "articles": [
                    {
                        "headline": f"{query} Soars on Excellent Earnings Report",
                        "summary": "Great performance with strong growth and positive outlook for investors.",
                        "source": "Positive News",
                        "date": start_date,
                        "entities": [query],
                    },
                    {
                        "headline": f"{query} Faces Challenges in Market Downturn",
                        "summary": "Concerns about declining revenue and poor market conditions affecting performance.",
                        "source": "Negative News",
                        "date": end_date,
                        "entities": [query],
                    },
                ],
            }

    sentiment_client = SentimentTestClient()
    service = NewsService(
        finnhub_client=None,
        google_client=sentiment_client,
        repository=None,
        online_mode=True,
    )

    try:
        context = service.get_context(
            "TEST", "2024-01-01", "2024-01-02", sources=["google"]
        )

        print("✅ Sentiment analysis completed")
        print(f"   Articles analyzed: {len(context.articles)}")
        print(f"   Overall sentiment: {context.sentiment_summary.score:.3f}")
        print(f"   Confidence: {context.sentiment_summary.confidence:.3f}")
        print(f"   Label: {context.sentiment_summary.label}")

        # Validate sentiment processing
        assert len(context.articles) == 2
        assert (
            context.sentiment_summary.score >= -1.0
            and context.sentiment_summary.score <= 1.0
        )
        assert (
            context.sentiment_summary.confidence >= 0.0
            and context.sentiment_summary.confidence <= 1.0
        )
        assert context.sentiment_summary.label in ["positive", "negative", "neutral"]

        # Check individual article sentiments
        for article in context.articles:
            if article.sentiment:
                assert (
                    article.sentiment.score >= -1.0 and article.sentiment.score <= 1.0
                )

        print("✅ Sentiment analysis validation passed")

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False


def test_multiple_source_aggregation():
    """Test aggregation from multiple news sources."""
    print("\n🔄 Testing Multiple Source Aggregation")

    mock_finnhub = MockFinnhubClient()
    mock_google = MockGoogleNewsClient()
    real_repo = NewsRepository("test_data")

    service = NewsService(
        finnhub_client=mock_finnhub,
        google_client=mock_google,
        repository=real_repo,
        online_mode=True,
    )

    try:
        # Test with both sources
        context = service.get_context(
            query="MSFT",
            start_date="2024-01-01",
            end_date="2024-01-03",
            symbol="MSFT",
            sources=["finnhub", "google"],
        )

        print("✅ Multi-source aggregation completed")
        print(f"   Total articles: {len(context.articles)}")
        print(f"   Unique sources: {context.sources}")
        print(f"   Sources used: {context.metadata.get('sources_used', [])}")

        # Should have articles from both sources
        assert len(context.articles) > 0
        assert len(context.sources) > 0

        # Check that articles from different sources are present
        source_counts = {}
        for article in context.articles:
            source = article.source
            source_counts[source] = source_counts.get(source, 0) + 1

        print(f"   Source distribution: {source_counts}")

        print("✅ Multi-source aggregation validated")

        return True

    except Exception as e:
        print(f"❌ Multi-source test failed: {e}")
        return False


def test_json_structure_validation():
    """Test detailed JSON structure validation."""
    print("\n📄 Testing JSON Structure")

    mock_google = MockGoogleNewsClient()
    service = NewsService(
        finnhub_client=None,
        google_client=mock_google,
        repository=None,
        online_mode=True,
    )

    try:
        context = service.get_context(
            "TSLA", "2024-01-01", "2024-01-03", sources=["google"]
        )
        json_str = context.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Validate required structure
        required_fields = [
            "symbol",
            "period",
            "articles",
            "sentiment_summary",
            "article_count",
            "sources",
            "metadata",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Validate period structure
        period = data["period"]
        assert "start" in period and "end" in period

        # Validate articles structure
        assert isinstance(data["articles"], list)
        if data["articles"]:
            first_article = data["articles"][0]
            required_article_fields = ["headline", "source", "date"]
            for field in required_article_fields:
                assert field in first_article, f"Missing article field: {field}"

        # Validate sentiment structure
        sentiment = data["sentiment_summary"]
        assert (
            "score" in sentiment and "confidence" in sentiment and "label" in sentiment
        )
        assert -1.0 <= sentiment["score"] <= 1.0
        assert 0.0 <= sentiment["confidence"] <= 1.0

        # Validate metadata
        metadata = data["metadata"]
        assert "data_quality" in metadata
        assert "service" in metadata

        print("✅ JSON structure validation passed")
        print(f"   Fields: {list(data.keys())}")
        print(f"   Articles: {len(data['articles'])}")
        print(f"   Sentiment score: {sentiment['score']:.3f}")

        return True

    except Exception as e:
        print(f"❌ JSON structure test failed: {e}")
        return False


def test_force_refresh_parameter():
    """Test the force_refresh parameter functionality."""
    print("\n🔄 Testing Force Refresh Parameter")

    try:
        mock_google = MockGoogleNewsClient()
        real_repo = NewsRepository("test_data")

        service = NewsService(
            finnhub_client=None,
            google_client=mock_google,
            repository=real_repo,
            online_mode=True,
        )

        # Test normal flow (should use repository if available)
        normal_context = service.get_context(
            "AAPL", "2024-01-01", "2024-01-31", sources=["google"], force_refresh=False
        )

        # Test force refresh (should bypass repository and use client)
        refresh_context = service.get_context(
            "AAPL", "2024-01-01", "2024-01-31", sources=["google"], force_refresh=True
        )

        # Both should return valid contexts
        assert isinstance(normal_context, NewsContext)
        assert isinstance(refresh_context, NewsContext)
        assert normal_context.symbol == "AAPL"
        assert refresh_context.symbol == "AAPL"

        # Check metadata indicates source
        refresh_metadata = refresh_context.metadata
        assert "force_refresh" in refresh_metadata
        assert refresh_metadata["force_refresh"]

        print("✅ Force refresh parameter test passed")
        return True

    except Exception as e:
        print(f"❌ Force refresh test failed: {e}")
        return False


def test_local_first_strategy():
    """Test that the service checks local data first when available."""
    print("\n🏠 Testing Local-First Strategy")

    try:

        class MockRepositoryWithData(NewsRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return True  # Pretend we have the data

            def get_data(
                self, query: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "query": kwargs.get("query", "TEST"),
                    "symbol": kwargs.get("symbol"),
                    "articles": [
                        {
                            "headline": "Test Article from Local Cache",
                            "summary": "This article came from local repository",
                            "source": "Local Cache",
                            "date": "2024-01-01",
                            "url": "https://local.cache/test",
                            "entities": ["TEST"],
                        }
                    ],
                    "metadata": {"source": "test_repository"},
                }

        mock_client = MockGoogleNewsClient()
        mock_repo = MockRepositoryWithData("test_data")

        service = NewsService(
            finnhub_client=None,
            google_client=mock_client,
            repository=mock_repo,
            online_mode=True,
        )

        # Should use local data since repository has_data_for_period returns True
        context = service.get_context(
            "TEST", "2024-01-01", "2024-01-31", sources=["google"]
        )

        # Verify we used local data
        assert context.metadata.get("data_source") == "local_cache"
        assert len(context.articles) == 1  # From mock repository
        assert context.articles[0].headline == "Test Article from Local Cache"

        print("✅ Local-first strategy test passed")
        return True

    except Exception as e:
        print(f"❌ Local-first strategy test failed: {e}")
        return False


def test_local_first_fallback_to_api():
    """Test that service falls back to API when local data is insufficient."""
    print("\n🔄 Testing Local-First Fallback to API")

    try:

        class MockRepositoryWithoutData(NewsRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return False  # Pretend we don't have the data

            def get_data(
                self, query: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "query": kwargs.get("query", "TEST"),
                    "articles": [],
                    "metadata": {},
                }

            def store_data(
                self,
                symbol: str,
                data: dict[str, Any],
                overwrite: bool = False,
                **kwargs,
            ) -> bool:
                return True  # Pretend storage was successful

        mock_client = MockGoogleNewsClient()
        mock_repo = MockRepositoryWithoutData("test_data")

        service = NewsService(
            finnhub_client=None,
            google_client=mock_client,
            repository=mock_repo,
            online_mode=True,
        )

        # Should fall back to API since repository doesn't have data
        context = service.get_context(
            "TEST", "2024-01-01", "2024-01-31", sources=["google"]
        )

        # Verify we used API data
        assert context.metadata.get("data_source") == "live_api"
        assert len(context.articles) > 0  # From mock client

        print("✅ Local-first fallback to API test passed")
        return True

    except Exception as e:
        print(f"❌ Local-first fallback test failed: {e}")
        return False


def test_force_refresh_bypasses_local_data():
    """Test that force_refresh=True bypasses local data even when available."""
    print("\n⚡ Testing Force Refresh Bypasses Local Data")

    try:

        class MockRepositoryAlwaysHasData(NewsRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return True  # Always claim we have data

            def get_data(
                self, query: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "query": kwargs.get("query", "TEST"),
                    "symbol": kwargs.get("symbol"),
                    "articles": [
                        {
                            "headline": "Old Cached Article",
                            "summary": "This is from local cache",
                            "source": "Local Cache",
                            "date": "2024-01-01",
                            "url": "https://cache.local/old",
                            "entities": ["TEST"],
                        }
                    ],
                    "metadata": {"source": "local"},
                }

            def clear_data(
                self, symbol: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return True

            def store_data(
                self,
                symbol: str,
                data: dict[str, Any],
                overwrite: bool = False,
                **kwargs,
            ) -> bool:
                return True

        mock_client = MockGoogleNewsClient()
        mock_repo = MockRepositoryAlwaysHasData("test_data")

        service = NewsService(
            finnhub_client=None,
            google_client=mock_client,
            repository=mock_repo,
            online_mode=True,
        )

        # Force refresh should bypass local data
        context = service.get_context(
            "TEST", "2024-01-01", "2024-01-31", sources=["google"], force_refresh=True
        )

        # Verify we used API data (force refresh)
        assert context.metadata.get("data_source") == "live_api_refresh"
        assert context.metadata.get("force_refresh")
        # Should have fresh data from client, not the old cached article
        assert len(context.articles) > 1  # Client returns multiple articles

        print("✅ Force refresh bypasses local data test passed")
        return True

    except Exception as e:
        print(f"❌ Force refresh bypass test failed: {e}")
        return False


def main():
    """Run all NewsService tests."""
    print("🧪 Testing NewsService\n")

    tests = [
        test_online_mode_with_mock_clients,
        test_global_news_context,
        test_offline_mode_with_real_repository,
        test_sentiment_analysis,
        test_multiple_source_aggregation,
        test_json_structure_validation,
        test_force_refresh_parameter,
        test_local_first_strategy,
        test_local_first_fallback_to_api,
        test_force_refresh_bypasses_local_data,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n📊 NewsService Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")

    if failed == 0:
        print("🎉 All NewsService tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
