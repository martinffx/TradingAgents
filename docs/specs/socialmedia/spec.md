# Social Media Domain Specification

## Context

**Product**: Multi-agent LLM financial trading framework that mirrors real-world trading firm dynamics for research-based market analysis and trading decisions.

**Domain**: Social Media (greenfield - complete implementation from stubs)

**Stack**: PostgreSQL + TimescaleDB + pgvectorscale + OpenRouter + PRAW

**Current Status**: Basic stub implementation with empty RedditClient, file-based JSON storage. Complete rebuild needed for production.

---

## User Story

**Primary Actor**: Dagster Pipeline + AI Agents

> As a Dagster pipeline, I want to collect Reddit posts from financial subreddits with LLM sentiment analysis and vector embeddings, so that AI Agents can access comprehensive social media context for ticker-specific trading decisions through RAG-powered queries.

---

## Acceptance Criteria

### AC1: Daily Data Collection
**GIVEN** a scheduled Dagster pipeline **WHEN** it executes daily **THEN** it collects Reddit posts from configured financial subreddits without manual intervention

### AC2: PostgreSQL Storage
**GIVEN** Reddit posts are collected **WHEN** processed **THEN** they are stored in PostgreSQL with TimescaleDB optimization and vector embeddings for semantic search

### AC3: LLM Sentiment Analysis
**GIVEN** social media posts **WHEN** processed **THEN** each post receives OpenRouter LLM sentiment analysis with structured scores (positive/negative/neutral with confidence)

### AC4: Fast Agent Queries
**GIVEN** a ticker symbol **WHEN** AI agents request social context **THEN** they receive relevant Reddit posts with sentiment scores and vector similarity ranking within 2 seconds

### AC5: RAG Integration
**GIVEN** social media data **WHEN** agents query **THEN** AgentToolkit provides RAG-enhanced context including post content, sentiment trends, and engagement metrics

---

## Business Rules

### BR1: Data Collection
- Daily automated collection from configured financial subreddits (wallstreetbets, investing, stocks, SecurityAnalysis)
- Rate limiting compliance with Reddit API terms of service (1 request/second)

### BR2: LLM Integration
- OpenRouter LLM sentiment analysis for all posts with confidence scoring
- Best-effort processing: API failures don't block other posts

### BR3: Data Quality
- Post deduplication by Reddit post_id
- Data retention policy: 90 days for social media posts

### BR4: Vector Search
- Vector embeddings generation for semantic similarity search
- 1536-dimension embeddings using text-embedding-3-large

---

## Scope

### Included
- PostgreSQL migration from current file-based storage
- Reddit API integration using PRAW
- OpenRouter LLM sentiment analysis integration
- Vector embeddings generation and similarity search
- AgentToolkit integration with `get_reddit_news` and `get_reddit_stock_info` methods
- Dagster pipeline for scheduled daily collection
- SQLAlchemy entities with TimescaleDB and pgvectorscale support

### Excluded
- Other social media platforms beyond Reddit (Twitter, LinkedIn)
- Real-time social media streaming (batch processing only)
- Custom sentiment models (use OpenRouter LLMs only)
- Multi-language post support (English only)
- Historical Reddit data backfilling beyond 30 days

---

## Technical Design

### Architecture Pattern

**Router → Service → Repository → Entity → Database** (matching news domain)

### Data Flow

```
Dagster Pipeline → RedditClient → SocialMediaService → SocialRepository → PostgreSQL + pgvectorscale
                                                  ↓
                                         AgentToolkit (RAG queries)
```

### Domain Model

#### SocialPost (Domain Entity)
Core entity for Reddit posts with sentiment and engagement data:
- Fields: post_id, title, content, author, subreddit, created_utc, upvotes, downvotes, comments_count, url
- Enhanced: sentiment_score, sentiment_label, sentiment_confidence, tickers, embeddings

#### SentimentScore
Structured sentiment analysis result from OpenRouter LLM:
- sentiment: positive | negative | neutral
- confidence: 0.0-1.0
- reasoning: brief explanation

#### SocialMediaPostEntity (SQLAlchemy)
PostgreSQL persistence entity with vector fields

### Database Schema

```sql
CREATE TABLE social_media_posts (
    id UUID PRIMARY KEY DEFAULT uuid7(),
    post_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    author VARCHAR(100) NOT NULL,
    subreddit VARCHAR(50) NOT NULL,
    created_utc TIMESTAMPTZ NOT NULL,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    url TEXT NOT NULL,
    sentiment_score JSONB,
    sentiment_label VARCHAR(20),
    tickers TEXT[],
    title_embedding VECTOR(1536),
    content_embedding VECTOR(1536),
    inserted_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('social_media_posts', 'created_utc', chunk_time_interval => INTERVAL '1 day');

CREATE INDEX idx_social_posts_tickers_gin ON social_media_posts USING GIN (tickers);
CREATE INDEX idx_social_posts_title_embedding ON social_media_posts USING vectors (title_embedding vector_cosine_ops);
CREATE INDEX idx_social_posts_content_embedding ON social_media_posts USING vectors (content_embedding vector_cosine_ops);
```

### Components

#### RedditClient
- PRAW integration with rate limiting
- Financial subreddit targeting
- Ticker-specific post filtering

#### SocialRepository
- PostgreSQL with deduplication by post_id
- Vector similarity search using pgvectorscale
- TimescaleDB time-series optimization

#### SocialMediaService
- Orchestrates: Reddit → Sentiment → Embeddings → Storage
- Ticker-specific social context
- Aggregate sentiment metrics

#### AgentToolkit Integration
- `get_reddit_news(ticker, days)` - Returns formatted social media context
- `get_reddit_stock_info(ticker, query)` - Semantic search with sentiment aggregation

---

## Implementation Phases

### Phase 1: Foundation (12 hours)
- Database Schema Migration (3h)
- SQLAlchemy Entity Implementation (3h)
- Domain Model Enhancement (3h)
- Repository Implementation (3h)

### Phase 2: API Integration (12 hours)
- Reddit Client Implementation (4h)
- Sentiment Analyzer (3h)
- Embedding Generator (2h)
- Service Layer Implementation (3h)

### Phase 3: Integration (8 hours)
- AgentToolkit Integration (3h)
- Dagster Pipeline (2h)
- Comprehensive Testing (3h)

---

## Configuration

### Environment Variables
```bash
REDDIT_CLIENT_ID="..."
REDDIT_CLIENT_SECRET="..."
REDDIT_USER_AGENT="TradingAgents/1.0"
OPENROUTER_API_KEY="..."
DATABASE_URL="postgresql://..."
```

### Dependencies
```toml
praw = "*"
asyncpg = "*"
psycopg2-binary = "*"
sqlalchemy = {extras = ["asyncio"]}
```

---

## Success Metrics

### Performance Targets
- **Agent Queries**: < 2 seconds for social context retrieval
- **Repository Operations**: < 100ms for common queries
- **Batch Processing**: < 5s for 100 posts with sentiment and embeddings

### Quality Targets
- **Test Coverage**: > 85% matching project standards
- **Sentiment Reliability**: > 80% posts with reliable sentiment (confidence >= 0.5)
- **Data Collection**: 400+ posts daily from financial subreddits

### Integration Targets
- Seamless AgentToolkit RAG integration
- Architecture matches news domain patterns
- Daily automated collection without manual intervention
