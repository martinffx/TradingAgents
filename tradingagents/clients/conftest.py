"""
Pytest configuration for FinnhubClient tests with VCR.
"""

import pytest


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for recording/replaying HTTP interactions."""
    return {
        # Don't record the API key in cassettes
        "filter_headers": ["X-Finnhub-Token", "Authorization"],
        # Record once, then replay from cassettes
        "record_mode": "once",
        # Match requests on URI and method
        "match_on": ["uri", "method"],
        # Decode compressed responses for better readability
        "decode_compressed_response": True,
        # Store cassettes in the cassettes subdirectory
        "cassette_library_dir": "tradingagents/clients/cassettes",
        # Ignore localhost requests
        "ignore_localhost": True,
        # Custom serializer for better readability
        "serializer": "yaml",
    }


@pytest.fixture(scope="session")
def vcr_cassette_dir(tmp_path_factory):
    """Create temporary directory for VCR cassettes during testing."""
    return tmp_path_factory.mktemp("cassettes")
