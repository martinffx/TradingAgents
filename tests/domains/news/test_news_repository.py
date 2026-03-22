"""
Integration tests for NewsRepository.

Tests the PostgreSQL repository with TimescaleDB using Docker.
Follows pragmatic TDD principles - test real persistence with Docker container.
"""

from datetime import date
from uuid import UUID

import pytest
from sqlalchemy import text
from uuid_utils import uuid7

from tradingagents.domains.news.news_repository import (
    NewsArticle,
    NewsRepository,
)
from tradingagents.lib.database import create_test_database_manager


@pytest.fixture
async def test_db_manager():
    """Create test database manager with TimescaleDB container."""
    db_manager = create_test_database_manager()

    # Verify database health
    health = await db_manager.health_check()
    if not health:
        pytest.skip("TimescaleDB test container not available")

    # Create tables
    await db_manager.create_tables()

    yield db_manager

    # Cleanup
    await db_manager.drop_tables()
    await db_manager.close()


@pytest.fixture
async def repository(test_db_manager):
    """Create repository instance with test database."""
    repo = NewsRepository(test_db_manager)

    # Clean up any existing test data
    async with test_db_manager.get_session() as session:
        await session.execute(text("DELETE FROM news_articles"))
        await session.commit()

    return repo


@pytest.fixture
def sample_article():
    """Create a sample news article for testing."""
    return NewsArticle(
        id=UUID(str(uuid7())),
        headline="Apple Quarterly Earnings Beat Expectations",
        url="https://example.com/apple-earnings-q1-2024",
        source="TechCrunch",
        published_date=date(2024, 1, 15),
        summary="Apple reported strong quarterly earnings with iPhone sales exceeding analyst predictions.",
        entities=["Apple", "iPhone", "earnings"],
        sentiment_score=0.8,
        author="Jane Tech Reporter",
        category="earnings",
    )


class TestDatabaseConnectionManagement:
    """Test database connection and session management."""

    async def test_database_health_check(self, test_db_manager):
        """Test database health check functionality."""
        # Act
        health = await test_db_manager.health_check()

        # Assert
        assert health is True

    async def test_session_context_manager(self, test_db_manager):
        """Test that session context manager handles transactions correctly."""
        # Act & Assert - No exceptions should be raised
        async with test_db_manager.get_session() as session:
            await session.execute(text("SELECT 1"))
            # Session should auto-commit on successful exit

    async def test_session_rollback_on_exception(self, test_db_manager):
        """Test that session rolls back on exceptions."""
        with pytest.raises(Exception, match="Test exception"):
            async with test_db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
                raise Exception("Test exception")
                # Session should auto-rollback due to exception
