# Architecture Standards - TradingAgents

## Layered Architecture Pattern

**Data Flow**: `Request → Agent → Service → Repository → Entity → Database`

This pattern enforces clean separation of concerns and dependency inversion, making the system testable and maintainable.

## Component Responsibilities

### 1. Entity Layer (Domain Models)

**Purpose**: Pure domain objects with business rules and data transformations.

**Location**: `tradingagents/domains/{feature}/entities/`

**Characteristics**:
- No external dependencies
- Rich domain models with validation
- Data transformation methods
- Business rule enforcement

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import date
from uuid import UUID

@dataclass
class NewsArticle:
    """Domain entity for news articles with business rules."""
    
    headline: str
    url: str
    source: str
    published_date: date
    sentiment_score: Optional[float] = None
    embedding: Optional[List[float]] = None
    id: Optional[UUID] = None
    
    def to_entity(self, symbol: Optional[str] = None) -> 'NewsArticleEntity':
        """Transform to database entity for persistence."""
        return NewsArticleEntity(
            headline=self.headline,
            url=self.url,
            source=self.source,
            published_date=self.published_date,
            symbol=symbol,
            sentiment_score=self.sentiment_score,
            embedding=self.embedding
        )
    
    @classmethod
    def from_entity(cls, entity: 'NewsArticleEntity') -> 'NewsArticle':
        """Transform from database entity to domain model."""
        return cls(
            headline=entity.headline,
            url=entity.url,
            source=entity.source,
            published_date=entity.published_date,
            sentiment_score=entity.sentiment_score,
            embedding=entity.embedding,
            id=entity.id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Transform to dictionary for API responses."""
        return {
            "id": str(self.id) if self.id else None,
            "headline": self.headline,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat(),
            "sentiment_score": self.sentiment_score,
            "embedding_provided": self.embedding is not None
        }
    
    def validate(self) -> List[str]:
        """Validate business rules and return list of errors."""
        errors = []
        
        if not self.headline or not self.headline.strip():
            errors.append("Headline cannot be empty")
        
        if len(self.headline) > 500:
            errors.append("Headline cannot exceed 500 characters")
        
        if not self.url or not self.url.startswith(("http://", "https://")):
            errors.append("URL must be valid and start with http:// or https://")
        
        if not self.source or not self.source.strip():
            errors.append("Source cannot be empty")
        
        if self.published_date > date.today():
            errors.append("Published date cannot be in the future")
        
        if self.sentiment_score is not None and not (-1 <= self.sentiment_score <= 1):
            errors.append("Sentiment score must be between -1 and 1")
        
        return errors
```

### 2. Repository Layer (Data Access)

**Purpose**: Handle all database operations with proper async patterns and error handling.

**Location**: `tradingagents/domains/{feature}/repositories/`

**Characteristics**:
- Async database operations
- SQLAlchemy integration
- Connection management
- Error handling and logging

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

class NewsRepository:
    """Repository for news article data access operations."""
    
    def __init__(self, db_manager: 'DatabaseManager'):
        self.db_manager = db_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with proper error handling."""
        session = self.db_manager.AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def list(self, symbol: str, target_date: date) -> List[NewsArticle]:
        """Retrieve news articles for symbol and date."""
        async with self.get_session() as session:
            result = await session.execute(
                select(NewsArticleEntity)
                .filter(and_(
                    NewsArticleEntity.symbol == symbol,
                    NewsArticleEntity.published_date == target_date
                ))
                .order_by(NewsArticleEntity.published_date.desc())
            )
            entities = result.scalars().all()
            return [NewsArticle.from_entity(e) for e in entities]
    
    async def upsert_batch(self, articles: List[NewsArticle], symbol: str) -> List[NewsArticle]:
        """Bulk insert/update articles with duplicate handling."""
        if not articles:
            return []
        
        async with self.get_session() as session:
            # Convert to entities
            entities = [article.to_entity(symbol) for article in articles]
            
            # Use PostgreSQL ON CONFLICT for atomic upserts
            stmt = insert(NewsArticleEntity).values([
                entity.__dict__ for entity in entities
            ])
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=["url"],
                set_={
                    "headline": stmt.excluded.headline,
                    "source": stmt.excluded.source,
                    "published_date": stmt.excluded.published_date,
                    "sentiment_score": stmt.excluded.sentiment_score,
                    "embedding": stmt.excluded.embedding,
                    "updated_at": func.now()
                }
            ).returning(NewsArticleEntity)
            
            result = await session.execute(upsert_stmt)
            upserted_entities = result.scalars().all()
            
            return [NewsArticle.from_entity(e) for e in upserted_entities]
    
    async def find_similar(self, query_embedding: List[float], limit: int = 10) -> List[NewsArticle]:
        """Find articles with similar embeddings using vector search."""
        async with self.get_session() as session:
            result = await session.execute(
                select(NewsArticleEntity)
                .filter(NewsArticleEntity.embedding.isnot(None))
                .order_by(NewsArticleEntity.embedding.cosine_distance(query_embedding))
                .limit(limit)
            )
            entities = result.scalars().all()
            return [NewsArticle.from_entity(e) for e in entities]
```

### 3. Service Layer (Business Logic)

**Purpose**: Orchestrate business operations and coordinate between repositories and clients.

**Location**: `tradingagents/domains/{feature}/services/`

**Characteristics**:
- Business logic orchestration
- External client coordination
- Error handling and resilience
- Logging and monitoring

```python
from typing import List, Dict, Any, Optional
from datetime import date, timedelta
import logging
from asyncio import gather, create_task

class NewsService:
    """Service for news article business logic and coordination."""
    
    def __init__(self, repository: NewsRepository, clients: Dict[str, Any]):
        self.repository = repository
        self.clients = clients
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_articles(self, symbol: str, target_date: date) -> List[NewsArticle]:
        """Retrieve articles with error handling and logging."""
        extra = {"symbol": symbol, "target_date": target_date.isoformat()}
        
        self.logger.info("Starting article retrieval", extra=extra)
        
        try:
            articles = await self.repository.list(symbol, target_date)
            
            self.logger.info(
                f"Successfully retrieved {len(articles)} articles",
                extra={**extra, "count": len(articles)}
            )
            
            return articles
            
        except Exception as e:
            self.logger.error(
                f"Failed to retrieve articles: {e}",
                extra=extra,
                exc_info=True
            )
            raise NewsServiceError(f"Failed to retrieve articles for {symbol}") from e
    
    async def update_articles(self, symbol: str, target_date: date) -> Dict[str, Any]:
        """Fetch and store articles from external sources with comprehensive error handling."""
        extra = {"symbol": symbol, "target_date": target_date.isoformat()}
        
        self.logger.info("Starting article update process", extra=extra)
        
        results = {
            "total_fetched": 0,
            "total_stored": 0,
            "sources": {},
            "errors": []
        }
        
        try:
            # Fetch from multiple sources concurrently
            source_tasks = [
                create_task(self._fetch_from_source(source, symbol, target_date))
                for source in self.clients.keys()
            ]
            
            source_results = await gather(*source_tasks, return_exceptions=True)
            
            # Process results from each source
            all_articles = []
            for i, (source, result) in enumerate(zip(self.clients.keys(), source_results)):
                if isinstance(result, Exception):
                    error_msg = f"Failed to fetch from {source}: {str(result)}"
                    results["errors"].append(error_msg)
                    self.logger.warning(error_msg, extra=extra)
                    continue
                
                source_articles, source_count = result
                results["sources"][source] = source_count
                results["total_fetched"] += source_count
                all_articles.extend(source_articles)
            
            # Validate and store articles
            if all_articles:
                validated_articles = []
                for article in all_articles:
                    validation_errors = article.validate()
                    if not validation_errors:
                        validated_articles.append(article)
                    else:
                        self.logger.warning(
                            f"Article validation failed: {validation_errors}",
                            extra={**extra, "article_url": article.url}
                        )
                
                # Store in database
                stored_articles = await self.repository.upsert_batch(validated_articles, symbol)
                results["total_stored"] = len(stored_articles)
            
            return results
            
        except Exception as e:
            self.logger.error(
                f"Critical error in article update: {e}",
                extra=extra,
                exc_info=True
            )
            raise NewsServiceError(f"Failed to update articles for {symbol}") from e
```

### 4. Client Layer (External API Integration)

**Purpose**: Handle external API integrations with proper error handling and rate limiting.

**Location**: `tradingagents/domains/{feature}/clients/`

**Characteristics**:
- HTTP client management
- Rate limiting and retry logic
- Data transformation
- pytest-vcr recording

```python
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from datetime import date, timedelta
import feedparser
from urllib.parse import quote

class BaseAPIClient(ABC):
    """Base client for external API integrations with retry logic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._rate_limiter = asyncio.Semaphore(config.get("max_concurrent", 5))
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class GoogleNewsClient(BaseAPIClient):
    """Client for Google News RSS feed parsing with caching."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://news.google.com/rss"
    
    async def search(self, query: str, max_results: int = 10, days: int = 7) -> List[Dict[str, Any]]:
        """Search Google News for articles matching query."""
        try:
            # Build RSS URL
            encoded_query = quote(query)
            search_url = f"{self.base_url}/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            # Fetch RSS feed
            async with self._rate_limiter:
                async with self.session.get(search_url) as response:
                    if response.status != 200:
                        raise aiohttp.ClientError(f"Failed to fetch RSS: {response.status}")
                    
                    rss_content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(rss_content)
            
            # Transform to standardized format
            articles = []
            for entry in feed.entries[:max_results]:
                article = {
                    "headline": entry.get("title", "").strip(),
                    "url": entry.get("link", ""),
                    "source": self._extract_source(entry),
                    "published_date": self._parse_date(entry.get("published")),
                    "summary": entry.get("summary", "")
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Google News search failed: {e}", exc_info=True)
            raise
```

## Database Architecture

### Core Stack: PostgreSQL + TimescaleDB + pgvectorscale

**Primary Database**: PostgreSQL 16+ with TimescaleDB and pgvector extensions
- **TimescaleDB**: Optimized for time-series financial data (prices, volumes, news timestamps)
- **pgvector/pgvectorscale**: Vector embeddings for RAG-powered agents
- **Connection**: asyncpg driver for high-performance async operations

**Database URL Pattern**:
```python
# Development
DATABASE_URL = "postgresql+asyncpg://postgres:tradingagents@localhost:5432/tradingagents"

# Production
DATABASE_URL = "postgresql+asyncpg://username:password@host:port/database"
```

**Required Extensions**:
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### Schema Design Standards

**Time-Series Tables (TimescaleDB)**:
```sql
-- Market data with time-based partitioning
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid7(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DECIMAL(18,8),
    volume BIGINT,
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data', 'timestamp');

-- Indexes for common query patterns
CREATE INDEX ON market_data (symbol, timestamp DESC);
```

**Vector-Enabled Tables**:
```sql
-- News articles with embeddings
CREATE TABLE news_articles (
    id UUID PRIMARY KEY DEFAULT uuid7(),
    headline TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,  -- Deduplication key
    published_date DATE NOT NULL,
    title_embedding VECTOR(1536),  -- OpenAI embedding size
    content_embedding VECTOR(1536),
    -- TimescaleDB partitioning on published_date
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX ON news_articles USING ivfflat (title_embedding vector_cosine_ops);
```

### Connection Management

**Async Session Factory**:
```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

class DatabaseManager:
    def __init__(self, database_url: str, echo: bool = False):
        # Ensure asyncpg driver
        if not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        self.engine = create_async_engine(
            database_url,
            echo=echo,
            pool_recycle=3600,  # 1-hour connection recycling
            pool_pre_ping=True,  # Connection health checks
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
        )
```

## LLM Integration Standards

### OpenRouter as Unified Provider

**Configuration**:
```python
# Environment variables
OPENROUTER_API_KEY = "your_openrouter_key"
LLM_PROVIDER = "openrouter"
DEEP_THINK_LLM = "openai/gpt-4o"      # Complex analysis
QUICK_THINK_LLM = "openai/gpt-4o-mini" # Fast responses
BACKEND_URL = "https://openrouter.ai/api/v1"
```

**Model Selection Strategy**:
- **Deep Think**: Complex reasoning, debates, risk analysis (`openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)
- **Quick Think**: Data formatting, simple queries (`openai/gpt-4o-mini`, `anthropic/claude-3-haiku`)

**Cost Optimization**:
```python
# Development/testing configuration
config = TradingAgentsConfig(
    llm_provider="openrouter",
    deep_think_llm="openai/gpt-4o-mini",     # Lower cost
    quick_think_llm="openai/gpt-4o-mini",    # Consistent model
    max_debate_rounds=1,                     # Reduce API calls
    online_tools=False,                      # Use cached data
)
```

### Agent Integration Patterns

**Anti-Corruption Layer**:
```python
class AgentToolkit:
    """Mediates between LLM agents and domain services"""
    
    def __init__(self, config: TradingAgentsConfig):
        self.config = config
        self.services = self._initialize_services()
    
    async def get_news_context(self, symbol: str, date: date) -> dict:
        """Convert domain models to structured LLM context"""
        articles = await self.news_service.get_articles(symbol, date)
        
        return {
            "articles": [article.to_dict() for article in articles],
            "count": len(articles),
            "data_quality": self._assess_data_quality(articles),
            "source_distribution": self._analyze_sources(articles)
        }
```

## Domain Isolation

### Three Core Domains

1. **News Domain** (`tradingagents/domains/news/`)
   - Article collection and processing
   - Sentiment analysis
   - Content scraping

2. **Market Data Domain** (`tradingagents/domains/marketdata/`)
   - Price and volume data
   - Technical indicators
   - Fundamental data

3. **Social Media Domain** (`tradingagents/domains/socialmedia/`)
   - Reddit and Twitter data
   - Social sentiment analysis
   - Discussion tracking

### Domain Boundary Rules

- **Service Interfaces Only**: Domains communicate through service interfaces
- **No Direct Database Access**: Cross-domain database queries prohibited
- **Shared Types**: Common types in `tradingagents/types/`
- **Domain Events**: Loose coupling through event publishing
- **Independent Testing**: Each domain has isolated test suite

## Dependency Injection

### Service Container Pattern

```python
from typing import Dict, Any
from tradingagents.lib.database import DatabaseManager
from tradingagents.lib.llm_client import LLMClient

class ServiceContainer:
    """Dependency injection container for domain services."""
    
    def __init__(self, config: TradingAgentsConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.database_url)
        self.llm_client = LLMClient(config)
        self._services: Dict[str, Any] = {}
    
    def get_news_service(self) -> NewsService:
        """Get or create news service with all dependencies."""
        if "news_service" not in self._services:
            repository = NewsRepository(self.db_manager)
            clients = {
                "google_news": GoogleNewsClient(self.config.google_news_config),
                "article_scraper": ArticleScraperClient(self.config.scraper_config),
                "llm": self.llm_client
            }
            self._services["news_service"] = NewsService(repository, clients)
        
        return self._services["news_service"]
```

## Error Handling Strategy

### Exception Hierarchy

```python
class TradingAgentsError(Exception):
    """Base exception for all TradingAgents errors."""
    pass

class DomainError(TradingAgentsError):
    """Base exception for domain-specific errors."""
    pass

class NewsServiceError(DomainError):
    """News domain specific errors."""
    pass

class ValidationError(DomainError):
    """Data validation errors."""
    pass

class DatabaseError(TradingAgentsError):
    """Database operation errors."""
    pass

class ExternalAPIError(TradingAgentsError):
    """External API integration errors."""
    pass
```

### Error Handling Patterns

```python
# Service layer error handling
async def get_articles(self, symbol: str, target_date: date) -> List[NewsArticle]:
    try:
        return await self.repository.list(symbol, target_date)
    except DatabaseError as e:
        self.logger.error(f"Database error retrieving articles: {e}")
        raise NewsServiceError("Failed to retrieve articles due to database error") from e
    except Exception as e:
        self.logger.error(f"Unexpected error retrieving articles: {e}")
        raise NewsServiceError("Failed to retrieve articles") from e

# Repository layer error handling
async def list(self, symbol: str, target_date: date) -> List[NewsArticle]:
    try:
        async with self.get_session() as session:
            # Database operations
            pass
    except SQLAlchemyError as e:
        self.logger.error(f"SQLAlchemy error: {e}")
        raise DatabaseError(f"Database query failed: {e}") from e
```

This architecture ensures clean separation of concerns, proper dependency management, and comprehensive error handling throughout the TradingAgents system.

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Technology Stack**: Python 3.13, asyncio, SQLAlchemy, Domain-Driven Design