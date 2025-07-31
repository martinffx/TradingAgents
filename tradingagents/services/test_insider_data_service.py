#!/usr/bin/env python3
"""
Test InsiderDataService with mock Finnhub client and real InsiderDataRepository.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

from tradingagents.clients.base import BaseClient
from tradingagents.models.context import DataQuality, InsiderContext, InsiderTransaction
from tradingagents.repositories.insider_repository import InsiderDataRepository
from tradingagents.services.insider_data_service import InsiderDataService


class MockFinnhubClient(BaseClient):
    """Mock Finnhub client that returns sample insider trading data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_works = True

    def test_connection(self) -> bool:
        return self.connection_works

    def get_data(self, *args, **kwargs) -> dict[str, Any]:
        """Not used directly by InsiderDataService."""
        return {}

    def get_insider_trading(
        self, ticker: str, start_date: str, end_date: str
    ) -> dict[str, Any]:
        """Return mock insider trading data."""
        # Use fixed dates within test range for predictable filtering
        base_date = datetime(2024, 6, 15)  # Within our test range

        return {
            "ticker": ticker,
            "data_type": "insider_trading",
            "transactions": [
                {
                    "filingDate": (base_date - timedelta(days=30)).strftime("%Y-%m-%d"),
                    "name": "John Smith",
                    "change": -50000,
                    "sharesTotal": 150000,
                    "transactionPrice": 180.50,
                    "transactionCode": "S",  # Sale
                },
                {
                    "filingDate": (base_date - timedelta(days=20)).strftime("%Y-%m-%d"),
                    "name": "Jane Doe",
                    "change": 25000,
                    "sharesTotal": 75000,
                    "transactionPrice": 185.25,
                    "transactionCode": "P",  # Purchase
                },
                {
                    "filingDate": (base_date - timedelta(days=10)).strftime("%Y-%m-%d"),
                    "name": "Robert Johnson",
                    "change": -10000,
                    "sharesTotal": 40000,
                    "transactionPrice": 178.75,
                    "transactionCode": "S",  # Sale
                },
                {
                    "filingDate": (base_date - timedelta(days=5)).strftime("%Y-%m-%d"),
                    "name": "Mary Wilson",
                    "change": 15000,
                    "sharesTotal": 65000,
                    "transactionPrice": 182.00,
                    "transactionCode": "P",  # Purchase
                },
            ],
            "metadata": {
                "source": "mock_finnhub",
                "retrieved_at": datetime(2024, 1, 2).isoformat(),
                "symbol": ticker,
            },
        }


def test_online_mode_with_mock_finnhub():
    """Test InsiderDataService in online mode with mock Finnhub client."""
    # Create mock client and real repository
    mock_finnhub = MockFinnhubClient()
    real_repo = InsiderDataRepository("test_data")

    # Create service in online mode
    service = InsiderDataService(
        finnhub_client=mock_finnhub,
        repository=real_repo,
        online_mode=True,
        data_dir="test_data",
    )

    # Test getting insider context
    context = service.get_insider_context(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
        force_refresh=True,
    )

    # Validate context structure
    assert isinstance(context, InsiderContext)
    assert context.symbol == "AAPL"
    assert context.period["start"] == "2024-01-01"
    assert context.period["end"] == "2024-12-31"

    # Validate transactions
    assert len(context.transactions) == 4
    assert all(isinstance(tx, InsiderTransaction) for tx in context.transactions)

    # Check transaction details
    first_tx = context.transactions[0]
    assert first_tx.name == "John Smith"
    assert first_tx.change == -50000  # Sale
    assert first_tx.transaction_code == "S"
    assert first_tx.transaction_price == 180.50

    # Validate sentiment data and net activity
    assert "buy_sell_ratio" in context.sentiment_data
    assert "insider_sentiment_score" in context.sentiment_data

    assert "net_shares_change" in context.net_activity
    assert "net_transaction_value" in context.net_activity
    assert "buy_transactions" in context.net_activity
    assert "sell_transactions" in context.net_activity

    # Validate metadata
    assert context.transaction_count == 4
    assert "data_quality" in context.metadata
    assert context.metadata["service"] == "insider_data"

    # Test JSON serialization
    json_output = context.model_dump_json(indent=2)
    assert len(json_output) > 0


def test_insider_sentiment_analysis():
    """Test insider sentiment calculation based on transactions."""
    mock_finnhub = MockFinnhubClient()
    service = InsiderDataService(
        finnhub_client=mock_finnhub, repository=None, online_mode=True
    )

    context = service.get_insider_context("TSLA", "2024-01-01", "2024-12-31")

    # Check sentiment calculations
    sentiment = context.sentiment_data

    # Should have buy/sell ratio
    assert "buy_sell_ratio" in sentiment
    assert sentiment["buy_sell_ratio"] > 0

    # Should have insider sentiment score (-1 to 1)
    assert "insider_sentiment_score" in sentiment
    assert -1.0 <= sentiment["insider_sentiment_score"] <= 1.0

    # Check net activity calculations
    net_activity = context.net_activity

    # Net shares change: sum of all changes
    expected_net_change = -50000 + 25000 + (-10000) + 15000  # -20000
    assert net_activity["net_shares_change"] == expected_net_change

    # Should have buy/sell transaction counts
    assert net_activity["buy_transactions"] == 2  # Jane Doe and Mary Wilson
    assert net_activity["sell_transactions"] == 2  # John Smith and Robert Johnson


def test_offline_mode():
    """Test InsiderDataService in offline mode."""
    real_repo = InsiderDataRepository("test_data")

    service = InsiderDataService(
        finnhub_client=None, repository=real_repo, online_mode=False
    )

    # Should handle offline gracefully
    context = service.get_insider_context("AAPL", "2024-01-01", "2024-12-31")

    assert context.symbol == "AAPL"
    assert len(context.transactions) == 0  # No data available offline
    assert context.transaction_count == 0
    # Sentiment data should have default values even with no transactions
    assert context.sentiment_data.get("insider_sentiment_score", 0) == 0
    assert context.sentiment_data.get("buy_sell_ratio", 0) == 0
    # Net activity should have default values
    assert context.net_activity.get("net_shares_change", 0) == 0
    assert context.net_activity.get("net_transaction_value", 0) == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_empty_data_handling():
    """Test handling when no insider transactions are available."""

    class EmptyDataClient(MockFinnhubClient):
        def get_insider_trading(self, ticker, start_date, end_date):
            return {
                "ticker": ticker,
                "data_type": "insider_trading",
                "transactions": [],
                "metadata": {"source": "mock_finnhub", "empty": True},
            }

    empty_client = EmptyDataClient()
    service = InsiderDataService(
        finnhub_client=empty_client, repository=None, online_mode=True
    )

    context = service.get_insider_context("XYZ", "2024-01-01", "2024-12-31")

    # Should handle empty data gracefully
    assert context.symbol == "XYZ"
    assert len(context.transactions) == 0
    assert context.transaction_count == 0

    # Sentiment should be neutral with no data
    assert context.sentiment_data.get("insider_sentiment_score", 0) == 0
    assert context.net_activity.get("net_shares_change", 0) == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_error_handling():
    """Test error handling with broken client."""

    class BrokenFinnhubClient(BaseClient):
        def test_connection(self):
            return False

        def get_data(self, *args, **kwargs):
            raise Exception("Finnhub API error")

        def get_insider_trading(self, *args, **kwargs):
            raise Exception("Finnhub API error")

    broken_client = BrokenFinnhubClient()
    service = InsiderDataService(
        finnhub_client=broken_client, repository=None, online_mode=True
    )

    # Should handle errors gracefully
    context = service.get_insider_context(
        "FAIL", "2024-01-01", "2024-12-31", force_refresh=True
    )

    assert context.symbol == "FAIL"
    assert len(context.transactions) == 0
    assert context.transaction_count == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW
    # Service logs errors but doesn't include them in metadata


def test_transaction_filtering():
    """Test filtering transactions by date range."""

    # Create a client that returns transactions outside the date range
    class DateFilterTestClient(MockFinnhubClient):
        def get_insider_trading(self, ticker, start_date, end_date):
            return {
                "ticker": ticker,
                "data_type": "insider_trading",
                "transactions": [
                    {
                        "filingDate": "2023-12-15",  # Before start date
                        "name": "Old Transaction",
                        "change": -1000,
                        "sharesTotal": 10000,
                        "transactionPrice": 100.0,
                        "transactionCode": "S",
                    },
                    {
                        "filingDate": "2024-06-15",  # Within range
                        "name": "Valid Transaction",
                        "change": 5000,
                        "sharesTotal": 15000,
                        "transactionPrice": 110.0,
                        "transactionCode": "P",
                    },
                    {
                        "filingDate": "2025-01-15",  # After end date
                        "name": "Future Transaction",
                        "change": -2000,
                        "sharesTotal": 8000,
                        "transactionPrice": 120.0,
                        "transactionCode": "S",
                    },
                ],
                "metadata": {"source": "mock_finnhub"},
            }

    filter_client = DateFilterTestClient()
    service = InsiderDataService(
        finnhub_client=filter_client, repository=None, online_mode=True
    )

    context = service.get_insider_context("TEST", "2024-01-01", "2024-12-31")

    # Should only include the transaction within the date range
    assert len(context.transactions) == 1
    assert context.transactions[0].name == "Valid Transaction"
    assert context.transaction_count == 1


def test_json_structure():
    """Test JSON structure of insider context."""
    mock_finnhub = MockFinnhubClient()
    service = InsiderDataService(
        finnhub_client=mock_finnhub, repository=None, online_mode=True
    )

    context = service.get_insider_context("NVDA", "2024-01-01", "2024-12-31")
    json_data = context.model_dump()

    # Validate required fields
    required_fields = [
        "symbol",
        "period",
        "transactions",
        "sentiment_data",
        "transaction_count",
        "net_activity",
        "metadata",
    ]
    for field in required_fields:
        assert field in json_data

    # Validate transaction structure
    if json_data["transactions"]:
        transaction = json_data["transactions"][0]
        required_tx_fields = [
            "filing_date",
            "name",
            "change",
            "shares",
            "transaction_price",
            "transaction_code",
        ]
        for field in required_tx_fields:
            assert field in transaction

    # Validate sentiment data structure
    sentiment = json_data["sentiment_data"]
    assert "buy_sell_ratio" in sentiment
    assert "insider_sentiment_score" in sentiment

    # Validate net activity structure
    net_activity = json_data["net_activity"]
    expected_net_fields = [
        "net_shares_change",
        "net_transaction_value",
        "buy_transactions",
        "sell_transactions",
    ]
    for field in expected_net_fields:
        assert field in net_activity

    # Validate metadata
    metadata = json_data["metadata"]
    assert "data_quality" in metadata
    assert "service" in metadata


def test_comprehensive_sentiment_calculation():
    """Test comprehensive insider sentiment calculation."""
    mock_finnhub = MockFinnhubClient()
    service = InsiderDataService(
        finnhub_client=mock_finnhub, repository=None, online_mode=True
    )

    context = service.get_insider_context("COMP", "2024-01-01", "2024-12-31")

    # Validate sentiment calculations are reasonable
    sentiment = context.sentiment_data
    net_activity = context.net_activity

    # Buy/sell ratio should be positive (we have both buys and sells)
    assert sentiment["buy_sell_ratio"] >= 0

    # Insider sentiment score should be between -1 and 1
    assert -1.0 <= sentiment["insider_sentiment_score"] <= 1.0

    # Net transaction value should be calculated correctly
    expected_value = (
        (-50000 * 180.50)  # John Smith sale
        + (25000 * 185.25)  # Jane Doe purchase
        + (-10000 * 178.75)  # Robert Johnson sale
        + (15000 * 182.00)  # Mary Wilson purchase
    )
    assert abs(net_activity["net_transaction_value"] - expected_value) < 0.01

    # Transaction counts should match
    assert net_activity["buy_transactions"] == 2
    assert net_activity["sell_transactions"] == 2

    # Net shares change
    expected_net_shares = -50000 + 25000 + (-10000) + 15000  # -20000
    assert net_activity["net_shares_change"] == expected_net_shares
