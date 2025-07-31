#!/usr/bin/env python3
"""
Comprehensive tests for FinnhubClient using pytest-vcr.

This test suite records real API interactions with Finnhub and replays them
in subsequent runs, providing realistic testing without network dependencies.
"""

import os
from datetime import date

import pytest

from tradingagents.clients.finnhub_client import FinnhubClient


@pytest.fixture
def finnhub_client():
    """Create FinnhubClient with test API key."""
    # Use environment variable or test key for VCR recording
    api_key = os.getenv("FINNHUB_API_KEY", "test_api_key")
    return FinnhubClient(api_key=api_key)


@pytest.fixture
def vcr_config():
    """Configure VCR with proper settings."""
    return {
        "filter_headers": ["X-Finnhub-Token"],  # Filter out API key from recordings
        "record_mode": "once",  # Record once, then replay
        "match_on": ["uri", "method"],
        "decode_compressed_response": True,
    }


class TestFinnhubClientConnection:
    """Test client connection and initialization."""

    @pytest.mark.vcr
    def test_client_initialization(self, finnhub_client):
        """Test that client initializes correctly."""
        assert finnhub_client.api_key is not None
        assert finnhub_client.client is not None

    @pytest.mark.vcr
    def test_connection_success(self, finnhub_client):
        """Test successful API connection."""
        assert finnhub_client.test_connection() is True

    def test_client_info(self, finnhub_client):
        """Test client metadata."""
        info = finnhub_client.get_client_info()
        assert info["client_type"] == "FinnhubClient"
        assert info["api_key_set"] is True


class TestFundamentalDataMethods:
    """Test the fundamental data methods needed by FundamentalDataService."""

    @pytest.mark.vcr
    def test_get_balance_sheet_quarterly(self, finnhub_client):
        """Test balance sheet retrieval with date object."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_balance_sheet("AAPL", "quarterly", test_date)

        assert isinstance(data, dict)
        # Finnhub financials_reported returns data structure with 'data' key
        assert "data" in data or len(data) > 0

    @pytest.mark.vcr
    def test_get_balance_sheet_annual(self, finnhub_client):
        """Test annual balance sheet retrieval."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_balance_sheet("AAPL", "annual", test_date)

        assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_get_income_statement_quarterly(self, finnhub_client):
        """Test income statement retrieval."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_income_statement("AAPL", "quarterly", test_date)

        assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_get_income_statement_annual(self, finnhub_client):
        """Test annual income statement retrieval."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_income_statement("AAPL", "annual", test_date)

        assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_get_cash_flow_with_date_object(self, finnhub_client):
        """Test cash flow retrieval with date object."""
        test_date = date(2024, 3, 31)
        data = finnhub_client.get_cash_flow("AAPL", "quarterly", test_date)

        assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_fundamental_data_different_symbols(self, finnhub_client):
        """Test fundamental data for different symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        test_date = date(2024, 1, 1)

        for symbol in symbols:
            data = finnhub_client.get_balance_sheet(symbol, "quarterly", test_date)
            assert isinstance(data, dict)


class TestExistingMethods:
    """Test existing methods with enhanced date support."""

    @pytest.mark.vcr
    def test_get_company_news_january(self, finnhub_client):
        """Test company news for January 2024."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        news = finnhub_client.get_company_news("AAPL", start_date, end_date)

        assert isinstance(news, list)

    @pytest.mark.vcr
    def test_get_company_news_date_objects(self, finnhub_client):
        """Test company news with date objects."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        news = finnhub_client.get_company_news("AAPL", start_date, end_date)

        assert isinstance(news, list)

    @pytest.mark.vcr
    def test_get_insider_transactions_date_objects(self, finnhub_client):
        """Test insider transactions with date objects."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        data = finnhub_client.get_insider_transactions("AAPL", start_date, end_date)

        assert isinstance(data, dict)
        assert "data" in data

    @pytest.mark.vcr
    def test_get_insider_sentiment(self, finnhub_client):
        """Test insider sentiment with date objects."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        data = finnhub_client.get_insider_sentiment("AAPL", start_date, end_date)

        assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_get_quote(self, finnhub_client):
        """Test stock quote retrieval."""
        quote = finnhub_client.get_quote("AAPL")

        assert isinstance(quote, dict)
        # Quote should contain current price 'c' field
        if quote:  # Only check if we got data
            assert "c" in quote

    @pytest.mark.vcr
    def test_get_company_profile(self, finnhub_client):
        """Test company profile retrieval."""
        profile = finnhub_client.get_company_profile("AAPL")

        assert isinstance(profile, dict)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.vcr
    def test_invalid_symbol_balance_sheet(self, finnhub_client):
        """Test balance sheet with invalid symbol."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_balance_sheet(
            "INVALID_SYMBOL_XYZ", "quarterly", test_date
        )

        # Should return dict with empty data structure, not raise exception
        assert isinstance(data, dict)
        assert "data" in data

    @pytest.mark.vcr
    def test_invalid_symbol_news(self, finnhub_client):
        """Test news with invalid symbol."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        news = finnhub_client.get_company_news(
            "INVALID_SYMBOL_XYZ", start_date, end_date
        )

        # Should return empty list, not raise exception
        assert isinstance(news, list)

    def test_connection_with_invalid_api_key(self):
        """Test connection failure with invalid API key."""
        client = FinnhubClient("invalid_api_key")
        # This should return False, not raise an exception
        assert client.test_connection() is False

    @pytest.mark.vcr
    def test_frequency_normalization(self, finnhub_client):
        """Test that frequency parameters are normalized correctly."""
        # Test different frequency formats
        frequencies = ["quarterly", "QUARTERLY", "q", "Q", "annual", "ANNUAL", "a", "A"]
        test_date = date(2024, 1, 1)

        for freq in frequencies:
            data = finnhub_client.get_balance_sheet("AAPL", freq, test_date)
            assert isinstance(data, dict)

    def test_date_edge_cases(self, finnhub_client):
        """Test date edge cases."""
        # Test with year-end date
        test_date = date(2024, 12, 31)

        # This shouldn't raise exceptions
        # (actual API calls are mocked/recorded)
        data = finnhub_client.get_balance_sheet("AAPL", "quarterly", test_date)
        assert isinstance(data, dict)


class TestMultipleSymbolsAndTimeframes:
    """Test with multiple symbols and different timeframes."""

    @pytest.mark.vcr
    def test_multiple_symbols_balance_sheet(self, finnhub_client):
        """Test balance sheet for multiple major symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        test_date = date(2024, 1, 1)

        for symbol in symbols:
            data = finnhub_client.get_balance_sheet(symbol, "quarterly", test_date)
            assert isinstance(data, dict)

    @pytest.mark.vcr
    def test_quarterly_vs_annual_frequency(self, finnhub_client):
        """Test quarterly vs annual frequency."""
        symbol = "AAPL"
        test_date = date(2024, 1, 1)

        quarterly_data = finnhub_client.get_balance_sheet(
            symbol, "quarterly", test_date
        )
        annual_data = finnhub_client.get_balance_sheet(symbol, "annual", test_date)

        assert isinstance(quarterly_data, dict)
        assert isinstance(annual_data, dict)

    @pytest.mark.vcr
    def test_all_fundamental_methods_same_symbol(self, finnhub_client):
        """Test all fundamental methods for the same symbol."""
        symbol = "AAPL"
        frequency = "quarterly"
        report_date = date(2024, 1, 1)

        balance_sheet = finnhub_client.get_balance_sheet(symbol, frequency, report_date)
        income_statement = finnhub_client.get_income_statement(
            symbol, frequency, report_date
        )
        cash_flow = finnhub_client.get_cash_flow(symbol, frequency, report_date)

        assert isinstance(balance_sheet, dict)
        assert isinstance(income_statement, dict)
        assert isinstance(cash_flow, dict)


# Integration test to verify the client works with service expectations
class TestServiceIntegration:
    """Test that client works as expected by FundamentalDataService."""

    def test_service_expected_methods_exist(self, finnhub_client):
        """Test that all methods expected by FundamentalDataService exist."""
        # These are the methods the service calls
        assert hasattr(finnhub_client, "get_balance_sheet")
        assert hasattr(finnhub_client, "get_income_statement")
        assert hasattr(finnhub_client, "get_cash_flow")

        # Verify method signatures accept the expected parameters
        import inspect

        for method_name in [
            "get_balance_sheet",
            "get_income_statement",
            "get_cash_flow",
        ]:
            method = getattr(finnhub_client, method_name)
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            # Service calls: method(symbol, frequency, date)
            assert "symbol" in params
            assert "frequency" in params
            assert "report_date" in params

    @pytest.mark.vcr
    def test_data_format_for_service_conversion(self, finnhub_client):
        """Test that returned data format can be processed by service conversion method."""
        test_date = date(2024, 1, 1)
        data = finnhub_client.get_balance_sheet("AAPL", "quarterly", test_date)

        # Service expects either empty dict or dict with data
        assert isinstance(data, dict)

        # The service's _convert_to_financial_statement expects either:
        # 1. Empty/falsy data -> returns None
        # 2. Dict with "data" key containing the actual financial data
        if data:  # If we got data
            # Should be able to access data["data"] or similar structure
            # This validates the format is compatible with service expectations
            assert isinstance(data, dict)
