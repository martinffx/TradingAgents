"""
Test FundamentalDataService with mock SimFin clients and real FundamentalDataRepository.
"""

import tempfile
from datetime import datetime
from typing import Any

import pytest

from tradingagents.clients.base import BaseClient
from tradingagents.models.context import (
    DataQuality,
    FinancialStatement,
    FundamentalContext,
)
from tradingagents.repositories.fundamental_repository import FundamentalDataRepository
from tradingagents.services.fundamental_data_service import FundamentalDataService


class MockSimFinClient(BaseClient):
    """Mock SimFin client that returns sample financial statement data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_works = True

    def test_connection(self) -> bool:
        return self.connection_works

    def get_data(self, *args, **kwargs) -> dict[str, Any]:
        """Not used directly by FundamentalDataService."""
        return {}

    def get_balance_sheet(
        self, ticker: str, freq: str, curr_date: str
    ) -> dict[str, Any]:
        """Return mock balance sheet data."""
        return {
            "ticker": ticker,
            "statement_type": "balance_sheet",
            "frequency": freq,
            "period": "Q3-2024" if freq == "quarterly" else "2024",
            "report_date": "2024-09-30",
            "publish_date": "2024-10-30",
            "currency": "USD",
            "data": {
                "Total Assets": 365725000000.0,
                "Total Current Assets": 143566000000.0,
                "Cash and Cash Equivalents": 28965000000.0,
                "Short-term Investments": 31590000000.0,
                "Accounts Receivable": 13348000000.0,
                "Inventory": 6511000000.0,
                "Total Non-Current Assets": 222159000000.0,
                "Property, Plant & Equipment": 43715000000.0,
                "Intangible Assets": 11235000000.0,
                "Total Liabilities": 279414000000.0,
                "Total Current Liabilities": 136817000000.0,
                "Accounts Payable": 58146000000.0,
                "Short-term Debt": 20208000000.0,
                "Total Non-Current Liabilities": 142597000000.0,
                "Long-term Debt": 106550000000.0,
                "Total Shareholders Equity": 86311000000.0,
                "Retained Earnings": 672000000.0,
                "Common Stock": 77958000000.0,
            },
            "metadata": {
                "source": "mock_simfin",
                "retrieved_at": datetime(2024, 1, 2).isoformat(),
            },
        }

    def get_income_statement(
        self, ticker: str, freq: str, curr_date: str
    ) -> dict[str, Any]:
        """Return mock income statement data."""
        return {
            "ticker": ticker,
            "statement_type": "income_statement",
            "frequency": freq,
            "period": "Q3-2024" if freq == "quarterly" else "2024",
            "report_date": "2024-09-30",
            "publish_date": "2024-10-30",
            "currency": "USD",
            "data": {
                "Total Revenue": 94930000000.0,
                "Cost of Revenue": 55720000000.0,
                "Gross Profit": 39210000000.0,
                "Operating Expenses": 15706000000.0,
                "Research and Development": 8067000000.0,
                "Sales, General & Administrative": 7639000000.0,
                "Operating Income": 23504000000.0,
                "Interest Expense": 1013000000.0,
                "Other Income": 269000000.0,
                "Income Before Tax": 22760000000.0,
                "Tax Provision": 4438000000.0,
                "Net Income": 18322000000.0,
                "Basic EPS": 1.18,
                "Diluted EPS": 1.15,
                "Shares Outstanding": 15550193000,
            },
            "metadata": {
                "source": "mock_simfin",
                "retrieved_at": datetime(2024, 1, 2).isoformat(),
            },
        }

    def get_cash_flow(self, ticker: str, freq: str, curr_date: str) -> dict[str, Any]:
        """Return mock cash flow statement data."""
        return {
            "ticker": ticker,
            "statement_type": "cash_flow",
            "frequency": freq,
            "period": "Q3-2024" if freq == "quarterly" else "2024",
            "report_date": "2024-09-30",
            "publish_date": "2024-10-30",
            "currency": "USD",
            "data": {
                "Net Income": 18322000000.0,
                "Depreciation & Amortization": 2871000000.0,
                "Changes in Working Capital": -1684000000.0,
                "Operating Cash Flow": 23302000000.0,
                "Capital Expenditures": -2736000000.0,
                "Acquisitions": -1800000000.0,
                "Asset Sales": 234000000.0,
                "Investing Cash Flow": -4302000000.0,
                "Dividends Paid": -3746000000.0,
                "Share Repurchases": -24979000000.0,
                "Debt Proceeds": 750000000.0,
                "Debt Repayment": -1500000000.0,
                "Financing Cash Flow": -28475000000.0,
                "Free Cash Flow": 20566000000.0,
                "Net Change in Cash": -9475000000.0,
            },
            "metadata": {
                "source": "mock_simfin",
                "retrieved_at": datetime(2024, 1, 2).isoformat(),
            },
        }


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data and clean up after test."""
    with tempfile.TemporaryDirectory(prefix="fundamental_test_") as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_simfin_client():
    """Create a mock SimFin client for testing."""
    return MockSimFinClient()


@pytest.fixture
def broken_simfin_client():
    """Create a broken SimFin client for error testing."""

    class BrokenSimFinClient(BaseClient):
        def test_connection(self):
            return False

        def get_data(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_balance_sheet(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_income_statement(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_cash_flow(self, *args, **kwargs):
            raise Exception("SimFin API error")

    return BrokenSimFinClient()


@pytest.fixture
def partial_data_client():
    """Create a client that returns partial data for testing."""

    class PartialDataClient(MockSimFinClient):
        def get_cash_flow(self, ticker, freq, curr_date):
            # Simulate missing cash flow data
            raise Exception("Cash flow data not available")

        def get_income_statement(self, ticker, freq, curr_date):
            # Simulate missing income statement
            return {"data": {}, "metadata": {"error": "No data found"}}

    return PartialDataClient()


def test_online_mode_with_mock_simfin(temp_data_dir, mock_simfin_client):
    """Test FundamentalDataService in online mode with mock SimFin client."""
    # Create real repository with temporary directory
    real_repo = FundamentalDataRepository(temp_data_dir)

    # Create service with mock client and real repository
    service = FundamentalDataService(
        simfin_client=mock_simfin_client,
        repository=real_repo,
        data_dir=temp_data_dir,
    )

    # Test getting fundamental context with all three statements
    context = service.get_fundamental_context(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="quarterly",
        force_refresh=True,  # Force using mock client instead of cache
    )

    # Validate context structure
    assert isinstance(context, FundamentalContext)
    assert context.symbol == "AAPL"
    assert context.period["start"] == "2024-01-01"
    assert context.period["end"] == "2024-12-31"

    # Validate financial statements
    assert context.balance_sheet is not None
    assert isinstance(context.balance_sheet, FinancialStatement)
    assert context.balance_sheet.period == "Q3-2024"
    assert context.balance_sheet.currency == "USD"
    assert "Total Assets" in context.balance_sheet.data

    assert context.income_statement is not None
    assert isinstance(context.income_statement, FinancialStatement)
    assert "Total Revenue" in context.income_statement.data
    assert "Net Income" in context.income_statement.data

    assert context.cash_flow is not None
    assert isinstance(context.cash_flow, FinancialStatement)
    assert "Operating Cash Flow" in context.cash_flow.data
    assert "Free Cash Flow" in context.cash_flow.data

    # Validate key ratios calculation
    assert len(context.key_ratios) > 0
    assert "current_ratio" in context.key_ratios
    assert "debt_to_equity" in context.key_ratios
    assert "roe" in context.key_ratios  # Return on Equity
    assert "gross_margin" in context.key_ratios

    # Validate metadata
    assert "data_quality" in context.metadata
    assert context.metadata["service"] == "fundamental_data"

    # Test JSON serialization
    json_output = context.model_dump_json(indent=2)
    assert len(json_output) > 0


def test_annual_vs_quarterly_frequency(temp_data_dir, mock_simfin_client):
    """Test different reporting frequencies."""
    real_repo = FundamentalDataRepository(temp_data_dir)
    service = FundamentalDataService(
        simfin_client=mock_simfin_client, repository=real_repo, data_dir=temp_data_dir
    )

    # Test quarterly
    quarterly_context = service.get_fundamental_context(
        symbol="MSFT",
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="quarterly",
    )

    assert quarterly_context.balance_sheet is not None
    assert quarterly_context.balance_sheet.period == "Q3-2024"

    # Test annual
    annual_context = service.get_fundamental_context(
        symbol="MSFT",
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="annual",
    )

    assert annual_context.balance_sheet is not None
    assert annual_context.balance_sheet.period == "2024"


def test_financial_ratio_calculations(temp_data_dir, mock_simfin_client):
    """Test calculation of key financial ratios."""
    real_repo = FundamentalDataRepository(temp_data_dir)
    service = FundamentalDataService(
        simfin_client=mock_simfin_client, repository=real_repo, data_dir=temp_data_dir
    )

    context = service.get_fundamental_context("TSLA", "2024-01-01", "2024-12-31")

    # Check that key ratios are calculated
    ratios = context.key_ratios

    # Liquidity ratios
    assert "current_ratio" in ratios
    assert ratios["current_ratio"] > 0

    # Leverage ratios
    assert "debt_to_equity" in ratios
    assert ratios["debt_to_equity"] >= 0

    # Profitability ratios
    assert "gross_margin" in ratios
    assert "operating_margin" in ratios
    assert "net_margin" in ratios
    assert "roe" in ratios  # Return on Equity
    assert "roa" in ratios  # Return on Assets

    # Efficiency ratios
    assert "asset_turnover" in ratios

    # Validate ratio calculations are reasonable
    assert 0 <= ratios["gross_margin"] <= 1
    assert 0 <= ratios["net_margin"] <= 1


def test_offline_mode(temp_data_dir):
    """Test FundamentalDataService without a client (offline mode)."""
    real_repo = FundamentalDataRepository(temp_data_dir)

    service = FundamentalDataService(
        simfin_client=None, repository=real_repo, data_dir=temp_data_dir
    )

    # Should handle offline gracefully
    context = service.get_fundamental_context("AAPL", "2024-01-01", "2024-12-31")

    assert context.symbol == "AAPL"
    assert context.balance_sheet is None  # No data available offline
    assert context.income_statement is None
    assert context.cash_flow is None
    assert len(context.key_ratios) == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_partial_data_handling():
    """Test handling when only some financial statements are available."""

    class PartialDataClient(MockSimFinClient):
        def get_cash_flow(self, ticker, freq, curr_date):
            # Simulate missing cash flow data
            raise Exception("Cash flow data not available")

        def get_income_statement(self, ticker, freq, curr_date):
            # Simulate missing income statement
            return {"data": {}, "metadata": {"error": "No data found"}}

    partial_client = PartialDataClient()
    service = FundamentalDataService(
        simfin_client=partial_client, repository=None, online_mode=True
    )

    context = service.get_fundamental_context("XYZ", "2024-01-01", "2024-12-31")

    # Should have balance sheet but not others
    assert context.balance_sheet is not None
    assert context.income_statement is None  # Failed to load
    assert context.cash_flow is None  # Failed to load

    # Ratios should be limited without full data (only balance sheet ratios available)
    assert (
        len(context.key_ratios) <= 8
    )  # Only balance sheet ratios possible, no profitability ratios
    assert context.metadata.get("data_quality") == DataQuality.LOW


def test_error_handling():
    """Test error handling with broken client."""

    class BrokenSimFinClient(BaseClient):
        def test_connection(self):
            return False

        def get_data(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_balance_sheet(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_income_statement(self, *args, **kwargs):
            raise Exception("SimFin API error")

        def get_cash_flow(self, *args, **kwargs):
            raise Exception("SimFin API error")

    broken_client = BrokenSimFinClient()
    service = FundamentalDataService(
        simfin_client=broken_client, repository=None, online_mode=True
    )

    # Should handle errors gracefully
    context = service.get_fundamental_context(
        "FAIL", "2024-01-01", "2024-12-31", force_refresh=True
    )

    assert context.symbol == "FAIL"
    assert context.balance_sheet is None
    assert context.income_statement is None
    assert context.cash_flow is None
    assert len(context.key_ratios) == 0
    assert context.metadata.get("data_quality") == DataQuality.LOW
    # Service logs errors but doesn't include them in metadata


def test_json_structure():
    """Test JSON structure of fundamental context."""
    mock_simfin = MockSimFinClient()
    service = FundamentalDataService(
        simfin_client=mock_simfin, repository=None, online_mode=True
    )

    context = service.get_fundamental_context("NVDA", "2024-01-01", "2024-12-31")
    json_data = context.model_dump()

    # Validate required fields
    required_fields = [
        "symbol",
        "period",
        "balance_sheet",
        "income_statement",
        "cash_flow",
        "key_ratios",
        "metadata",
    ]
    for field in required_fields:
        assert field in json_data

    # Validate financial statement structure
    if json_data["balance_sheet"]:
        balance_sheet = json_data["balance_sheet"]
        required_statement_fields = [
            "period",
            "report_date",
            "publish_date",
            "currency",
            "data",
        ]
        for field in required_statement_fields:
            assert field in balance_sheet

        # Check some key balance sheet items
        bs_data = balance_sheet["data"]
        assert "Total Assets" in bs_data
        assert "Total Liabilities" in bs_data
        assert "Total Shareholders Equity" in bs_data

    # Validate key ratios
    ratios = json_data["key_ratios"]
    assert isinstance(ratios, dict)
    assert len(ratios) > 0

    # Validate metadata
    metadata = json_data["metadata"]
    assert "data_quality" in metadata
    assert "service" in metadata


def test_comprehensive_ratio_calculation():
    """Test comprehensive financial ratio calculations."""
    mock_simfin = MockSimFinClient()
    service = FundamentalDataService(
        simfin_client=mock_simfin, repository=None, online_mode=True
    )

    context = service.get_fundamental_context("COMP", "2024-01-01", "2024-12-31")
    ratios = context.key_ratios

    # Liquidity ratios

    # Not all ratios may be calculable depending on available data
    calculated_ratios = set(ratios.keys())
    core_ratios = {
        "current_ratio",
        "debt_to_equity",
        "gross_margin",
        "net_margin",
        "roe",
        "roa",
    }

    # At least the core ratios should be present
    assert core_ratios.issubset(calculated_ratios), (
        f"Missing core ratios: {core_ratios - calculated_ratios}"
    )

    # All ratio values should be numbers
    for ratio_name, ratio_value in ratios.items():
        assert isinstance(ratio_value, int | float), (
            f"{ratio_name} should be numeric, got {type(ratio_value)}"
        )
        assert ratio_value == ratio_value, (
            f"{ratio_name} should not be NaN"
        )  # NaN check
