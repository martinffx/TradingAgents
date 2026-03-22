# News Domain Completion Specification

## Context

**Product**: Multi-agent LLM financial trading framework that mirrors real-world trading firm dynamics for research-based market analysis and trading decisions.

**Domain**: News (95% complete → finalize 5%)

**Stack**: PostgreSQL + TimescaleDB + pgvectorscale + OpenRouter

**Current Status**: Core infrastructure exists (NewsService, GoogleNewsClient, ArticleScraperClient, NewsRepository). Implemented: Temporal workflow orchestration, LLM sentiment analysis, vector embeddings. Missing: vector similarity search, CLI workflow triggers.

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

Follows established **Router → Service → Repository → Entity → Database** pattern with Temporal orchestration:

```
Temporal Worker → NewsProcessingWorkflow → NewsActivities → NewsService → NewsRepository → NewsArticle → PostgreSQL+pgvectorscale
```

### Data Flow

1. **News Collection Flow**
   ```
   Temporal Worker → NewsProcessingWorkflow
   → NewsActivities.fetch_article() / scrape_article()
   → NewsActivities.analyze_sentiment() + create_embedding()
   → NewsActivities.save_article() → NewsRepository
   → PostgreSQL+pgvectorscale
   ```

2. **Agent Query Flow**
   ```
   News Analyst → AgentToolkit → NewsService
   → NewsRepository (semantic search) → pgvectorscale vector similarity
   ```

### Domain Model Changes

#### NewsArticle (enhance existing)

New fields:
- `sentiment_score: JSONB` - structured sentiment: `{sentiment, confidence, reasoning}`
- `title_embedding: vector(1536)` - semantic embedding for headline
- `content_embedding: vector(1536)` - semantic embedding for article content

Validation rules:
- Sentiment confidence must be ≥ 0.6 for reliable classification
- Embeddings must be exactly 1536 dimensions

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
    ON news_articles USING btree (sentiment_label) 
    WHERE sentiment_label IS NOT NULL;
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

### Temporal (workflow orchestration)

**NewsProcessingWorkflow**:
- Single article processing workflow
- Handles fetch, scrape, sentiment, embeddings, save
- Activity retry with exponential backoff on 429

**BatchNewsProcessingWorkflow**:
- Parallel processing of multiple articles
- Uses asyncio.gather for concurrent execution
- Returns success/failed counts

---

## Implementation Phases

### Phase 1: Foundation ✅
- [x] Temporal workflow orchestration (NewsProcessingWorkflow)
- [x] NewsArticle entity with sentiment and embedding fields
- [x] NewsActivities for LLM calls
- [x] NewsArticleEntity with Vector(1536) columns

### Phase 2: Data Access ⚠️
- [x] NewsRepository CRUD operations
- [x] Batch upsert operations
- [ ] Vector similarity search (cosine distance queries)
- [ ] Semantic article search endpoint

### Phase 3: LLM Integration ✅
- [x] OpenRouter sentiment analysis client
- [x] OpenRouter embeddings client
- [x] NewsService LLM integration

### Phase 4: Workflow Orchestration ⚠️
- [x] Temporal workflow execution
- [ ] CLI commands for workflow triggers
- [ ] Temporal worker configuration

### Phase 5: Validation (3-5 hours)
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Documentation updates

---

## Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY="sk-or-..."
DATABASE_URL="postgresql://..."
TEMPORAL_HOST="localhost:7233"  # Temporal server address
TEMPORAL_NAMESPACE="default"
```

### Dependencies

```toml
# Already in pyproject.toml
temporalio = "^1.0"
pgvector = "^0.2"
openai = "^1.0"
```

---

## Success Metrics

### Performance Targets
- **Query Response Time**: < 2 seconds for news retrieval with sentiment
- **Workflow Execution Time**: < 5 minutes per article (including LLM calls)
- **Success Rate**: > 95% article processing success rate

### Quality Targets
- **Test Coverage**: Maintain > 85% including new components
- **Zero Breaking Changes**: AgentToolkit integration unchanged

### Operational Metrics
- Workflow completion status and execution time
- Article processing success/failure rates
- LLM sentiment analysis success rates
- Vector embedding generation performance
