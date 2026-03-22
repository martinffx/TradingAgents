# News Domain Completion Specification

## Context

**Product**: Multi-agent LLM financial trading framework that mirrors real-world trading firm dynamics for research-based market analysis and trading decisions.

**Domain**: News (95% complete → finalize 5%)

**Stack**: PostgreSQL + TimescaleDB + pgvectorscale + OpenRouter

**Current Status**: Core infrastructure exists (NewsService, GoogleNewsClient, ArticleScraperClient, NewsRepository). Missing: scheduled execution, LLM sentiment analysis, vector embeddings.

---

## User Story

**Primary Actor**: Dagster Job (automated system)  
**Secondary Actor**: News Analysts (LLM agents)

> As a Dagster Job, I want to automatically fetch Google News articles for tracked tickers, extract content, perform LLM sentiment analysis, and store with embeddings in the database, so that News Analysts can access comprehensive, up-to-date news data for trading decisions.

---

## Acceptance Criteria

### AC1: Scheduled Execution
**GIVEN** a scheduled job runs daily  
**WHEN** it executes  
**THEN** it fetches news for all configured tickers without manual intervention

**Validation**:
- Job executes at configured time (default: daily at 6 AM UTC)
- All tickers in configuration are processed
- Job completion status is logged with metrics

### AC2: Content Resilience
**GIVEN** a news article is found  
**WHEN** content extraction fails due to paywall  
**THEN** a warning is logged and processing continues with available metadata

**Validation**:
- Paywall detection doesn't halt processing
- Warning messages include article URL and error reason
- Metadata (title, source, publish_date) is still stored

### AC3: Fast News Retrieval
**GIVEN** a ticker symbol  
**WHEN** a News Analyst requests news data  
**THEN** they receive articles with sentiment scores and embeddings within 2 seconds

**Validation**:
- Database queries return results in < 2 seconds
- Results include sentiment scores and vector embeddings
- Pagination supports large result sets

### AC4: LLM Sentiment Analysis
**GIVEN** news articles are processed  
**WHEN** LLM sentiment analysis runs  
**THEN** each article gets a structured sentiment score (positive/negative/neutral with confidence)

**Validation**:
- Sentiment scores use structured format: `{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}`
- LLM integration uses OpenRouter unified provider
- Failed sentiment analysis doesn't prevent article storage

### AC5: Vector Embeddings Storage
**GIVEN** news articles are stored  
**WHEN** saved to database  
**THEN** they include vector embeddings for both title and content for semantic search

**Validation**:
- 1536-dimension embeddings generated for title and content
- Embeddings stored in pgvectorscale-optimized columns
- Semantic similarity search returns relevant results

---

## Business Rules

### BR1: Best Effort Processing
- Log warnings for paywalled/blocked content but continue processing
- Network failures don't halt entire job execution
- API rate limits are respected with exponential backoff

### BR2: Daily Schedule Execution
- Configurable ticker list supports adding/removing symbols
- Job execution time is configurable (default: daily at 6 AM UTC)
- Manual job execution available for testing and backfill

### BR3: Data Quality Standards
- URL-based deduplication prevents duplicate articles
- Article publish dates must be within last 30 days
- Source URLs must be valid and accessible

### BR4: LLM Integration Standards
- Use OpenRouter unified provider for sentiment analysis
- Quick-think LLM for sentiment processing (cost optimization)
- Structured prompts ensure consistent sentiment format

### BR5: Vector Search Optimization
- Embeddings enable semantic similarity search for agents
- Vector indexes optimize query performance
- Embedding generation uses consistent model for coherence

### BR6: Graceful Error Handling
- Individual article failures don't stop batch processing
- Comprehensive logging for monitoring and debugging
- Database transactions ensure data consistency

---

## Scope

### Included
- Scheduled news collection job using existing NewsService
- LLM-based sentiment analysis replacing current keyword approach
- Vector embedding generation for articles
- Configuration management for ticker lists and schedules
- Integration with existing GoogleNewsClient and ArticleScraperClient
- Database storage using existing NewsRepository patterns

### Excluded
- Other news sources beyond Google News XML feed
- Real-time news streaming (daily batch processing only)
- Custom sentiment models (use OpenRouter LLMs only)
- News source reliability scoring
- Multi-language news support

---

## Technical Design

### Architecture Pattern

Follows established **Router → Service → Repository → Entity → Database** pattern:

```
ScheduledNewsCollector → NewsService → NewsRepository → NewsArticle → PostgreSQL+pgvectorscale
```

### Data Flow

1. **Scheduled Collection Flow**
   ```
   APScheduler → ScheduledNewsCollector → NewsService.update_company_news()
   → GoogleNewsClient → ArticleScraperClient → OpenRouter (sentiment + embeddings)
   → NewsRepository.upsert_batch() → PostgreSQL
   ```

2. **Agent Query Flow**
   ```
   News Analyst → AgentToolkit → NewsService.find_relevant_articles()
   → NewsRepository (semantic search) → pgvectorscale vector similarity
   ```

### Domain Model Changes

#### NewsArticle (enhance existing)

New fields:
- `sentiment_score: JSONB` - structured sentiment: `{sentiment, confidence, reasoning}`
- `title_embedding: vector(1536)` - semantic embedding for headline
- `content_embedding: vector(1536)` - semantic embedding for article content

Validation rules:
- Sentiment confidence must be ≥ 0.5 for reliable classification
- Embeddings must be exactly 1536 dimensions

#### NewsJobConfig (new entity)

Fields:
- `id: UUID` - primary key
- `name: str` - job name (max 255 chars)
- `symbols: List[str]` - ticker symbols to track (uppercase)
- `categories: List[str]` - news categories (optional)
- `frequency_cron: str` - cron expression for schedule
- `enabled: bool` - job active state
- `last_run: datetime` - timestamp of last execution

---

## Database Schema

### news_articles Enhancements

```sql
-- Add sentiment and embedding columns
ALTER TABLE news_articles ADD COLUMN sentiment_score JSONB;
ALTER TABLE news_articles ADD COLUMN title_embedding vector(1536);
ALTER TABLE news_articles ADD COLUMN content_embedding vector(1536);

-- Vector similarity indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_articles_title_embedding 
    ON news_articles USING vectors (title_embedding vector_cosine_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_articles_content_embedding 
    ON news_articles USING vectors (content_embedding vector_cosine_ops);

-- Sentiment filter index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_articles_sentiment 
    ON news_articles (((sentiment_score->>'sentiment'))) 
    WHERE sentiment_score IS NOT NULL;
```

### news_job_configs Table

```sql
CREATE TABLE news_job_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    symbols JSONB NOT NULL,
    categories JSONB DEFAULT '[]',
    frequency_cron VARCHAR(100) NOT NULL,
    enabled BOOLEAN DEFAULT true,
    last_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_news_jobs_enabled_frequency ON news_job_configs (enabled, frequency_cron);
CREATE INDEX idx_news_jobs_last_run ON news_job_configs (last_run) WHERE enabled = true;
```

---

## External APIs

### OpenRouter (unified LLM provider)

**Sentiment Analysis**:
- Model: quick_think_llm (default: anthropic/claude-3.5-haiku)
- Structured output: JSON with sentiment, confidence, reasoning

**Embeddings**:
- Model: text-embedding-3-small (1536 dimensions)
- Input: article title and content (truncated to 8000 chars)

---

## Implementation Phases

### Phase 1: Foundation (4-7 hours)
- Database migration for news_job_configs table
- Enhance NewsArticle entity with sentiment and embedding fields
- Create NewsJobConfig entity

### Phase 2: Data Access (2-3 hours)
- Enhance NewsRepository with vector similarity search
- Add NewsJobConfig CRUD operations

### Phase 3: LLM Integration (5-8 hours)
- OpenRouter sentiment analysis client
- OpenRouter embeddings client
- Integrate into NewsService

### Phase 4: Scheduling (4-6 hours)
- APScheduler integration with job management
- CLI commands for job management

### Phase 5: Validation (3-5 hours)
- Integration tests
- Performance benchmarks
- Documentation updates

---

## Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY="sk-or-..."
OPENAI_API_KEY="sk-..."  # For embeddings via OpenRouter
DATABASE_URL="postgresql://..."
NEWS_SCHEDULE_HOUR=6     # UTC hour for daily execution
NEWS_TICKERS="AAPL,GOOGL,MSFT,TSLA"
```

### Dependencies

```toml
# Add to pyproject.toml
apscheduler = "^3.10"
```

---

## Success Metrics

### Performance Targets
- **Query Response Time**: < 2 seconds for news retrieval with sentiment
- **Job Execution Time**: < 30 minutes for daily collection (4 tickers)
- **Success Rate**: > 95% article processing success rate

### Quality Targets
- **Test Coverage**: Maintain > 85% including new components
- **Zero Breaking Changes**: AgentToolkit integration unchanged

### Operational Metrics
- Daily job completion status and execution time
- Article processing success/failure rates per ticker
- LLM sentiment analysis success rates
- Vector embedding generation performance
