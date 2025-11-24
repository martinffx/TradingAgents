# Coding Standards - TradingAgents

## Stub-Driven TDD Approach

**3-Step Process:**
1. **Create Stub** - Method signatures that raise `NotImplementedError("ClassName.method_name")`
2. **Write Test** - Test against stub expecting NotImplementedError, then write tests for actual behavior
3. **Implement** - Replace stub with working code to make tests pass

**Benefits:**
- Clear interface design before implementation
- Tests written before code (true TDD)
- Incremental development with continuous validation
- Natural async/await pattern development

## Implementation Order

Always implement in dependency order (bottom-up):

```
Entity → Repository → Service → Client → Agent
```

**Why this order?**
- Entity has no dependencies (pure domain logic)
- Repository depends on Entity (data transformations)
- Service depends on Repository + Entity (business logic)
- Client depends on Service (external API integration)
- Agent depends on Service + Client (LLM orchestration)

## Stub Pattern

Every layer follows the same pattern with async/await:

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import date
import logging

class NewsArticle:
    """Domain entity for news articles with business rules."""
    
    def __init__(self, headline: str, url: str, source: str, published_date: date):
        self.headline = headline
        self.url = url
        self.source = source
        self.published_date = published_date
        self.sentiment_score: Optional[float] = None
        self.embedding: Optional[List[float]] = None
    
    @classmethod
    def from_entity(cls, entity: 'NewsArticleEntity') -> 'NewsArticle':
        """Transform from database entity to domain model."""
        raise NotImplementedError(f"NewsArticle.from_entity")
    
    def to_entity(self, symbol: Optional[str] = None) -> 'NewsArticleEntity':
        """Transform to database entity for storage."""
        raise NotImplementedError(f"NewsArticle.to_entity")
    
    def to_dict(self) -> Dict[str, Any]:
        """Transform to dictionary for API responses."""
        raise NotImplementedError(f"NewsArticle.to_dict")
    
    def validate(self) -> List[str]:
        """Validate business rules and return list of errors."""
        raise NotImplementedError(f"NewsArticle.validate")
```

## Testing Strategy

### Pragmatic Outside-In TDD

**Philosophy**: Mock I/O boundaries, test real logic, optimize for fast feedback.

**Core Principle**: Test behavior, not implementation. Focus on public interfaces and data transformations while mocking external dependencies (HTTP, database, filesystem).

### Testing Strategy by Layer

#### 1. Services (Business Logic) - Mock Boundaries
```python
# tests/domains/news/test_news_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from tradingagents.domains.news.news_service import NewsService
from tradingagents.domains.news.news_repository import NewsArticle

@pytest.fixture
def mock_repository():
    return AsyncMock(spec=NewsRepository)

async def test_get_articles_returns_empty_on_repository_error(mock_repository):
    # Mock repository failure
    mock_repository.list.side_effect = Exception("Database connection failed")
    
    service = NewsService(repository=mock_repository, clients={})
    
    # Service should handle error gracefully
    articles = await service.get_articles("AAPL", date(2024, 1, 15))
    
    assert articles == []
    mock_repository.list.assert_called_once_with("AAPL", date(2024, 1, 15))
```

#### 2. Repositories (Data Access) - Real Persistence
```python
# tests/domains/news/test_news_repository.py
import pytest
from tradingagents.lib.database import create_test_database_manager
from tradingagents.domains.news.news_repository import NewsRepository, NewsArticle

@pytest.fixture
async def db_manager():
    """Use real PostgreSQL for repository tests"""
    manager = create_test_database_manager()
    await manager.create_tables()
    yield manager
    await manager.drop_tables()
    await manager.close()

async def test_upsert_batch_handles_duplicates_correctly(db_manager):
    """Test actual database behavior with real SQL operations"""
    repository = NewsRepository(db_manager)
    
    # Insert initial articles
    articles = [
        NewsArticle("Apple Earnings Beat", "https://cnn.com/1", "CNN", date(2024, 1, 15)),
        NewsArticle("Apple Stock Rises", "https://cnn.com/2", "CNN", date(2024, 1, 15))
    ]
    
    result1 = await repository.upsert_batch(articles, "AAPL")
    assert len(result1) == 2
    
    # Update one article (same URL)
    updated_articles = [
        NewsArticle("Apple Earnings Beat Expectations", "https://cnn.com/1", "CNN", date(2024, 1, 15))
    ]
    
    result2 = await repository.upsert_batch(updated_articles, "AAPL")
    
    # Should update existing, not create duplicate
    all_articles = await repository.list("AAPL", date(2024, 1, 15))
    assert len(all_articles) == 2
    assert any("Beat Expectations" in a.headline for a in all_articles)
```

#### 3. Clients (External APIs) - pytest-vcr
```python
# tests/domains/news/test_google_news_client.py
import pytest
import pytest_vcr
from tradingagents.domains.news.google_news_client import GoogleNewsClient

class TestGoogleNewsClient:
    @pytest_vcr.use_cassette("google_news_apple_search.yaml")
    async def test_search_returns_structured_articles(self):
        """Real HTTP calls recorded with VCR cassettes"""
        client = GoogleNewsClient()
        
        articles = await client.search("AAPL", max_results=5)
        
        # Test real API response structure
        assert len(articles) > 0
        assert all(article.title for article in articles)
        assert all(article.link.startswith("http") for article in articles)
        assert all(article.source for article in articles)
```

### Quality Standards

#### Coverage Requirements
- **85% minimum coverage** across all domains
- **100% coverage** for critical financial calculations
- **Branch coverage** for error handling paths

#### Performance Standards
- **< 100ms per unit test** (fast feedback)
- **< 5s for integration test suite** (rapid development)
- **< 30s for full test suite** (CI/CD efficiency)

## Code Style

### Formatting with Ruff

**Configuration** (pyproject.toml):
```toml
[tool.ruff]
target-version = "py313"
line-length = 88
fix = true
extend-exclude = [
    "migrations/",
    "alembic/versions/",
    ".env",
    "venv/",
    ".venv/",
]

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # Pyflakes
    "I",     # isort
    "B",     # flake8-bugbear
    "C4",    # flake8-comprehensions
    "UP",    # pyupgrade
    "ERA",   # eradicate
    "PIE",   # flake8-pie
    "SIM",   # flake8-simplify
    "TCH",   # flake8-type-checking
    "ARG",   # flake8-unused-arguments
    "PTH",   # flake8-use-pathlib
    "FIX",   # flake8-fixme
    "TD",    # flake8-todos
]

ignore = [
    "E501",  # Line too long (handled by formatter)
    "B008",  # Do not perform function calls in argument defaults
    "B904",  # Use `raise ... from ...` for exception chaining
    "TD002", # Missing author in TODO
    "TD003", # Missing issue link on line following TODO
    "FIX002", # Line contains TODO
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = [
    "S101",    # Use of assert detected
    "ARG001",  # Unused function argument
    "FBT001",  # Boolean positional arg
    "PLR2004", # Magic value used in comparison
]

[tool.ruff.lint.isort]
known-first-party = ["tradingagents"]
force-sort-within-sections = true
```

### Type Hints and Annotations

**Modern Type Syntax** (Python 3.13):
```python
# Use built-in generics (no typing.List, typing.Dict)
def process_articles(articles: list[NewsArticle]) -> dict[str, int]:
    """Process articles and return symbol counts"""
    counts: dict[str, int] = {}
    for article in articles:
        symbol = article.symbol or "UNKNOWN"
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts

# Union types with |
def get_article(article_id: str | int) -> NewsArticle | None:
    """Get article by ID (string or integer)"""
    if isinstance(article_id, str):
        return get_by_url(article_id)
    return get_by_id(article_id)

# Optional with explicit None
def calculate_sentiment(text: str, model: str | None = None) -> float | None:
    """Calculate sentiment score"""
    if not text.strip():
        return None
    # Implementation
    return 0.5
```

### Naming Conventions

- **Classes**: PascalCase (`NewsArticle`, `MarketDataService`, `GoogleNewsClient`)
- **Methods/Functions**: snake_case (`get_articles`, `find_by_symbol`, `create_embedding`)
- **Variables**: snake_case (`article_id`, `market_data`, `client_config`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_RETRY_ATTEMPTS`, `API_BASE_URL`, `DEFAULT_LOOKBACK_DAYS`)
- **Private Members**: Prefix with underscore (`_validate_data`, `_connection_pool`)

### Import Organization

**Import Order**:
```python
# 1. Standard library imports
import asyncio
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

# 2. Third-party imports  
import aiohttp
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import pytest

# 3. First-party imports
from tradingagents.config import TradingAgentsConfig
from tradingagents.domains.news.news_repository import NewsArticle, NewsRepository
from tradingagents.lib.database import DatabaseManager

# 4. Relative imports (avoid when possible)
from .google_news_client import GoogleNewsClient
```

## Error Handling

### Service Layer Error Handling

```python
class NewsServiceError(Exception):
    """Base exception for news service errors."""
    pass

class ArticleNotFoundError(NewsServiceError):
    """Raised when article is not found."""
    pass

class ValidationError(NewsServiceError):
    """Raised when data validation fails."""
    pass

async def get_article_by_id(self, article_id: str) -> NewsArticle:
    try:
        article = await self.repository.get_by_id(article_id)
        if not article:
            raise ArticleNotFoundError(f"Article {article_id} not found")
        return article
    except DatabaseError as e:
        logger.error(f"Database error retrieving article {article_id}: {e}")
        raise NewsServiceError("Failed to retrieve article") from e
```

### Graceful Degradation

```python
async def get_articles(self, symbol: str, target_date: date) -> List[NewsArticle]:
    """Service-level error handling with fallbacks."""
    try:
        # Try primary repository
        articles = await self.repository.list(symbol, target_date)
        logger.info(f"Retrieved {len(articles)} articles from database")
        return articles
        
    except DatabaseConnectionError:
        logger.warning("Database unavailable, trying cache fallback")
        # Fallback to file cache
        return await self.cache_repository.list(symbol, target_date)
        
    except Exception as e:
        logger.error(f"Failed to retrieve articles for {symbol}: {e}")
        # Graceful degradation - return empty list rather than crash
        return []
```

## Async Testing Patterns

**Use pytest-asyncio** for async test methods:

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_service_method():
    """Test async service method with proper await."""
    service = NewsService(mock_repository, mock_clients)
    
    result = await service.get_articles("AAPL", date(2024, 1, 15))
    
    assert isinstance(result, list)
```

**Mock async methods** using AsyncMock:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_repository():
    repository = AsyncMock()
    repository.list.return_value = []
    return repository
```

## Development Workflow

### Daily Development Commands

```bash
# 1. Start development environment
mise run docker    # Start PostgreSQL + TimescaleDB

# 2. Install/update dependencies
mise run install   # uv sync --dev

# 3. Development iteration
mise run format    # Auto-format with ruff
mise run lint      # Check code quality
mise run typecheck # Type checking with pyright
mise run test      # Run test suite

# 4. Run application
mise run dev       # Interactive CLI
mise run run       # Direct execution
```

### Quality Assurance

```bash
# Run all quality checks before commit
mise run check     # format + lint + typecheck

# Coverage analysis
mise run coverage
```

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Technology Stack**: Python 3.13, asyncio, SQLAlchemy, pytest