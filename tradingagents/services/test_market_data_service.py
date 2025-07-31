#!/usr/bin/env python3
"""
Test MarketDataService with mock YFinanceClient and real MarketDataRepository.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from tradingagents.clients.base import BaseClient
from tradingagents.models.context import DataQuality, MarketDataContext
from tradingagents.repositories.market_data_repository import MarketDataRepository
from tradingagents.services.market_data_service import MarketDataService


class MockYFinanceClient(BaseClient):
    """Mock Yahoo Finance client that returns predictable test data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_works = True

    def test_connection(self) -> bool:
        return self.connection_works

    def get_data(
        self, symbol: str, start_date: str, end_date: str, **kwargs
    ) -> dict[str, Any]:
        """Return realistic mock market data."""
        # Generate realistic price data
        base_price = {"AAPL": 180.0, "TSLA": 250.0, "MSFT": 400.0}.get(symbol, 100.0)

        mock_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        price = base_price
        while current_date <= end_date_dt:
            # Simulate some price movement
            price_change = (
                hash(current_date.strftime("%Y-%m-%d")) % 10 - 5
            ) / 100  # -5% to +5%
            price *= 1 + price_change * 0.01

            mock_data.append(
                {
                    "Date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Open": round(price * 0.99, 2),
                    "High": round(price * 1.02, 2),
                    "Low": round(price * 0.98, 2),
                    "Close": round(price, 2),
                    "Adj Close": round(price, 2),
                    "Volume": 45000000 + (hash(symbol) % 20000000),
                }
            )

            current_date += timedelta(days=1)

        return {
            "symbol": symbol,
            "period": {"start": start_date, "end": end_date},
            "data": mock_data,
            "metadata": {
                "source": "mock_yahoo_finance",
                "record_count": len(mock_data),
                "columns": [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                ],
                "retrieved_at": datetime.utcnow().isoformat(),
            },
        }


def test_online_mode_with_mock_client():
    """Test MarketDataService in online mode with mock client."""
    print("📈 Testing MarketDataService - Online Mode")

    # Create mock client and real repository
    mock_client = MockYFinanceClient()
    real_repo = MarketDataRepository("test_data")

    # Create service in online mode
    service = MarketDataService(
        client=mock_client, repository=real_repo, online_mode=True, data_dir="test_data"
    )

    try:
        # Test basic price context
        context = service.get_price_context(
            symbol="AAPL", start_date="2024-01-01", end_date="2024-01-05"
        )

        print(f"✅ Price context created: {context.__class__.__name__}")
        print(f"   Symbol: {context.symbol}")
        print(f"   Period: {context.period}")
        print(f"   Price data records: {len(context.price_data)}")
        print(f"   Technical indicators: {len(context.technical_indicators)}")

        # Validate required fields
        assert context.symbol == "AAPL"
        assert context.period["start"] == "2024-01-01"
        assert context.period["end"] == "2024-01-05"
        assert len(context.price_data) > 0
        assert "data_quality" in context.metadata

        print("✅ Basic validation passed")

        # Test JSON serialization
        json_output = context.model_dump_json(indent=2)
        parsed = json.loads(json_output)

        print(f"✅ JSON serialization: {len(json_output)} characters")
        print(f"   Top-level keys: {list(parsed.keys())}")

        # Test with technical indicators
        context_with_indicators = service.get_context(
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-01-03",
            indicators=["rsi", "macd"],
        )

        print("✅ Context with indicators created")
        print("   Requested indicators: ['rsi', 'macd']")
        print(
            f"   Available indicators: {list(context_with_indicators.technical_indicators.keys())}"
        )

        return True

    except Exception as e:
        print(f"❌ Online mode test failed: {e}")
        return False


def test_offline_mode_with_real_repository():
    """Test MarketDataService in offline mode with real repository."""
    print("\n💾 Testing MarketDataService - Offline Mode")

    # Create service in offline mode (no client)
    real_repo = MarketDataRepository("test_data")
    service = MarketDataService(
        client=None, repository=real_repo, online_mode=False, data_dir="test_data"
    )

    try:
        # Test offline context (will likely return empty data)
        context = service.get_price_context(
            symbol="AAPL", start_date="2024-01-01", end_date="2024-01-05"
        )

        print(f"✅ Offline context created: {context.__class__.__name__}")
        print(f"   Symbol: {context.symbol}")
        print(f"   Price data records: {len(context.price_data)}")
        print(f"   Data quality: {context.metadata.get('data_quality')}")
        print(f"   Service mode: online={service.is_online()}")

        # Should handle empty data gracefully
        assert context.symbol == "AAPL"
        assert isinstance(context.price_data, list)
        assert "data_quality" in context.metadata

        print("✅ Offline mode graceful handling verified")

        return True

    except Exception as e:
        print(f"❌ Offline mode test failed: {e}")
        return False


def test_error_handling():
    """Test error handling scenarios."""
    print("\n⚠️  Testing Error Handling")

    # Test with broken client
    class BrokenClient(BaseClient):
        def test_connection(self):
            return False

        def get_data(self, *args, **kwargs):
            raise Exception("Simulated client failure")

    broken_client = BrokenClient()
    real_repo = MarketDataRepository("test_data")

    service = MarketDataService(
        client=broken_client,
        repository=real_repo,
        online_mode=True,  # Online mode but client will fail
        data_dir="test_data",
    )

    try:
        context = service.get_price_context("AAPL", "2024-01-01", "2024-01-05")

        print("✅ Error handling worked")
        print(f"   Symbol: {context.symbol}")
        print(f"   Price data records: {len(context.price_data)}")
        print(f"   Data quality: {context.metadata.get('data_quality')}")

        # Should fallback to repository or return empty data
        assert context.symbol == "AAPL"
        assert isinstance(context.price_data, list)

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_data_quality_assessment():
    """Test data quality determination logic."""
    print("\n🔍 Testing Data Quality Assessment")

    mock_client = MockYFinanceClient()
    real_repo = MarketDataRepository("test_data")

    service = MarketDataService(
        client=mock_client, repository=real_repo, online_mode=True, data_dir="test_data"
    )

    try:
        # Test with good data
        context = service.get_context("AAPL", "2024-01-01", "2024-01-10")

        data_quality = context.metadata.get("data_quality")
        print(f"✅ Data quality assessment: {data_quality}")
        print(f"   Records: {len(context.price_data)}")
        print(f"   Online mode: {service.is_online()}")

        # Should be medium or high quality for mock data
        assert data_quality in [DataQuality.MEDIUM, DataQuality.HIGH]

        return True

    except Exception as e:
        print(f"❌ Data quality test failed: {e}")
        return False


def test_json_structure_validation():
    """Test detailed JSON structure validation."""
    print("\n📄 Testing JSON Structure")

    mock_client = MockYFinanceClient()
    service = MarketDataService(client=mock_client, repository=None, online_mode=True)

    try:
        context = service.get_price_context("MSFT", "2024-01-01", "2024-01-03")
        json_str = context.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Validate required structure
        required_fields = [
            "symbol",
            "period",
            "price_data",
            "technical_indicators",
            "metadata",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # Validate period structure
        period = data["period"]
        assert "start" in period and "end" in period

        # Validate price data structure
        assert isinstance(data["price_data"], list)
        if data["price_data"]:
            first_record = data["price_data"][0]
            required_price_fields = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for field in required_price_fields:
                assert field in first_record, f"Missing price field: {field}"

        # Validate metadata
        metadata = data["metadata"]
        assert "data_quality" in metadata
        assert "service" in metadata

        print("✅ JSON structure validation passed")
        print(f"   Fields: {list(data.keys())}")
        print(f"   Price records: {len(data['price_data'])}")
        print(f"   Metadata keys: {list(metadata.keys())}")

        return True

    except Exception as e:
        print(f"❌ JSON structure test failed: {e}")
        return False


def test_force_refresh_parameter():
    """Test the force_refresh parameter functionality."""
    try:
        mock_client = MockYFinanceClient()
        real_repo = MarketDataRepository("test_data")

        service = MarketDataService(
            client=mock_client, repository=real_repo, online_mode=True
        )

        # Test normal flow (should use repository if available)
        normal_context = service.get_context(
            "AAPL", "2024-01-01", "2024-01-31", force_refresh=False
        )

        # Test force refresh (should bypass repository and use client)
        refresh_context = service.get_context(
            "AAPL", "2024-01-01", "2024-01-31", force_refresh=True
        )

        # Both should return valid contexts
        assert isinstance(normal_context, MarketDataContext)
        assert isinstance(refresh_context, MarketDataContext)
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
    try:

        class MockRepositoryWithData(MarketDataRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return True  # Pretend we have the data

            def get_data(
                self, symbol: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "symbol": kwargs.get("symbol", "TEST"),
                    "data": [
                        {"date": "2024-01-01", "close": 150.0},
                        {"date": "2024-01-02", "close": 151.0},
                    ],
                    "metadata": {"source": "test_repository"},
                }

        mock_client = MockYFinanceClient()
        mock_repo = MockRepositoryWithData("test_data")

        service = MarketDataService(
            client=mock_client, repository=mock_repo, online_mode=True
        )

        # Should use local data since repository has_data_for_period returns True
        context = service.get_context("TEST", "2024-01-01", "2024-01-31")

        # Verify we used local data
        assert context.metadata.get("price_data_source") == "local_cache"
        assert len(context.price_data) == 2  # From mock repository

        print("✅ Local-first strategy test passed")
        return True

    except Exception as e:
        print(f"❌ Local-first strategy test failed: {e}")
        return False


def test_local_first_fallback_to_api():
    """Test that service falls back to API when local data is insufficient."""
    try:

        class MockRepositoryWithoutData(MarketDataRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return False  # Pretend we don't have the data

            def get_data(
                self, symbol: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "symbol": kwargs.get("symbol", "TEST"),
                    "data": [],
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

        mock_client = MockYFinanceClient()
        mock_repo = MockRepositoryWithoutData("test_data")

        service = MarketDataService(
            client=mock_client, repository=mock_repo, online_mode=True
        )

        # Should fall back to API since repository doesn't have data
        context = service.get_context("TEST", "2024-01-01", "2024-01-31")

        # Verify we used API data
        assert context.metadata.get("price_data_source") == "live_api"
        assert len(context.price_data) > 0  # From mock client

        print("✅ Local-first fallback to API test passed")
        return True

    except Exception as e:
        print(f"❌ Local-first fallback test failed: {e}")
        return False


def test_force_refresh_bypasses_local_data():
    """Test that force_refresh=True bypasses local data even when available."""
    try:

        class MockRepositoryAlwaysHasData(MarketDataRepository):
            def has_data_for_period(
                self, identifier: str, start_date: str, end_date: str, **kwargs
            ) -> bool:
                return True  # Always claim we have data

            def get_data(
                self, symbol: str, start_date: str, end_date: str, **kwargs
            ) -> dict[str, Any]:
                return {
                    "symbol": kwargs.get("symbol", "TEST"),
                    "data": [
                        {"date": "2024-01-01", "close": 100.0}
                    ],  # Different from client
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

        mock_client = MockYFinanceClient()
        mock_repo = MockRepositoryAlwaysHasData("test_data")

        service = MarketDataService(
            client=mock_client, repository=mock_repo, online_mode=True
        )

        # Force refresh should bypass local data
        context = service.get_context(
            "TEST", "2024-01-01", "2024-01-31", force_refresh=True
        )

        # Verify we used API data (force refresh)
        assert context.metadata.get("price_data_source") == "live_api_refresh"
        assert context.metadata.get("force_refresh")
        # Should have more data from client than the single point from repository
        assert len(context.price_data) > 1

        print("✅ Force refresh bypasses local data test passed")
        return True

    except Exception as e:
        print(f"❌ Force refresh bypass test failed: {e}")
        return False


def main():
    """Run all MarketDataService tests."""
    print("🧪 Testing MarketDataService\n")

    tests = [
        test_online_mode_with_mock_client,
        test_offline_mode_with_real_repository,
        test_error_handling,
        test_data_quality_assessment,
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

    print("\n📊 MarketDataService Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")

    if failed == 0:
        print("🎉 All MarketDataService tests passed!")
    else:
        print("⚠️  Some tests failed - check output above")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
