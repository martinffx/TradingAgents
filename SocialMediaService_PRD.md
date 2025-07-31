# Product Requirements Document: SocialMediaService Completion

## Overview

Complete the `SocialMediaService` to provide strongly-typed social media data and sentiment analysis to trading agents using a local-first data strategy with gap detection and intelligent caching.

## Current State Analysis

### Issues to Fix
- **CRITICAL**: Missing `RedditClient` implementation - service calls non-existent client methods
- **CRITICAL**: Service uses `BaseClient` inheritance but needs typed `RedditClient`
- **CRITICAL**: `SocialRepository` has different interface than standard service pattern
- **CRITICAL**: Repository uses `date` objects internally but service expects string date interface
- Missing strongly-typed interfaces between components
- Service calls `reddit_client.search_posts()`, `get_top_posts()`, `filter_posts_by_date()` methods that don't exist

### What Works
- ✅ Local-first data strategy implementation (`_get_social_data_local_first`)
- ✅ Force refresh logic (`_fetch_and_cache_fresh_social_data`)
- ✅ `SocialContext` Pydantic model for agent consumption
- ✅ Comprehensive sentiment analysis with keyword-based scoring
- ✅ Engagement metrics calculation and post ranking
- ✅ Error handling and metadata creation patterns
- ✅ `SocialRepository` with JSON storage and post deduplication
- ✅ `PostData` and `SentimentScore` models for structured data
- ✅ Real-time sentiment analysis with weighted scoring

## Technical Requirements

### 1. Strongly-Typed Interfaces

#### Client → Service Interface
```python
# RedditClient methods (to be implemented)
def search_posts(query: str, subreddit_names: list[str], start_date: date, end_date: date, limit: int, time_filter: str) -> dict[str, Any]
def get_top_posts(subreddit_names: list[str], start_date: date, end_date: date, limit: int, time_filter: str) -> dict[str, Any]
def get_company_posts(symbol: str, subreddit_names: list[str], start_date: date, end_date: date, limit: int) -> dict[str, Any]
```

#### Service → Repository Interface
```python
# SocialRepository methods (to be implemented/bridged)
def has_data_for_period(query: str, start_date: str, end_date: str, symbol: str | None) -> bool
def get_data(query: str, start_date: str, end_date: str, symbol: str | None) -> dict[str, Any]
def store_data(query: str, cache_data: dict, symbol: str | None, overwrite: bool) -> bool
def clear_data(query: str, start_date: str, end_date: str, symbol: str | None) -> bool
```

#### Service → Agent Interface
```python
# Service output (already defined)
def get_context(query: str, start_date: str, end_date: str, symbol: str | None, subreddits: list[str], force_refresh: bool) -> SocialContext
def get_company_social_context(symbol: str, start_date: str, end_date: str, subreddits: list[str]) -> SocialContext
def get_global_trends(start_date: str, end_date: str, subreddits: list[str]) -> SocialContext
```

### 2. Local-First Data Strategy

#### Flow
1. **Repository Lookup**: Check `SocialRepository.has_data_for_period()`
2. **Gap Detection**: Identify missing social media data periods
3. **Selective Fetching**: Fetch only missing data from `RedditClient`
4. **Cache Updates**: Store new data via `repository.store_data()`
5. **Context Assembly**: Return validated `SocialContext`

#### Force Refresh Support
- `force_refresh=True` bypasses local data completely
- Clears existing cache before fetching fresh data
- Stores refreshed data with metadata indicating refresh

### 3. Date Object Conversion

#### Service Boundary Conversion
```python
# Service receives string dates from agents
def get_context(self, query: str, start_date: str, end_date: str, ...) -> SocialContext:
    # Convert to date objects for client calls
    start_dt = date.fromisoformat(start_date)
    end_dt = date.fromisoformat(end_date)
    
    # Use date objects when calling RedditClient
    posts_data = self.reddit_client.search_posts(query, subreddits, start_dt, end_dt, limit, time_filter)
    
    # Repository bridge handles string to date conversion internally
    cached_data = self.repository.get_data(query, start_date, end_date, symbol)
```

### 4. Reddit API Integration

#### RedditClient Implementation Strategy
```python
# RedditClient following FinnhubClient standard
class RedditClient:
    """Client for Reddit API access with PRAW library integration."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """Initialize Reddit client with PRAW."""
        import praw
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def search_posts(self, query: str, subreddit_names: list[str], 
                    start_date: date, end_date: date, limit: int = 50, 
                    time_filter: str = "week") -> dict[str, Any]:
        """Search for posts across subreddits within date range."""
        
    def get_top_posts(self, subreddit_names: list[str], 
                     start_date: date, end_date: date, limit: int = 50, 
                     time_filter: str = "week") -> dict[str, Any]:
        """Get top posts from subreddits within date range."""
        
    def get_company_posts(self, symbol: str, subreddit_names: list[str],
                         start_date: date, end_date: date, limit: int = 50) -> dict[str, Any]:
        """Get company-specific posts from subreddits."""
```

#### Reddit Response Format
```python
{
    "query": "AAPL",
    "period": {"start": "2024-01-01", "end": "2024-01-31"},
    "posts": [
        {
            "title": "Apple earnings discussion",
            "content": "What do you think about...",
            "author": "redditor123",
            "subreddit": "investing",
            "created_utc": 1704067200,
            "score": 125,
            "num_comments": 45,
            "upvote_ratio": 0.87,
            "url": "https://reddit.com/r/investing/comments/abc123",
            "id": "abc123"
        }
    ],
    "metadata": {
        "source": "reddit",
        "retrieved_at": "2024-01-31T10:00:00Z",
        "data_quality": "HIGH",
        "subreddits": ["investing", "stocks"],
        "total_posts": 25
    }
}
```

### 5. Sentiment Analysis Enhancement

#### Advanced Sentiment Features
- **Weighted Scoring**: High-engagement posts have more influence on overall sentiment
- **Keyword Analysis**: Comprehensive positive/negative keyword detection
- **Score Adjustment**: Reddit score (upvotes) influences sentiment confidence
- **Confidence Metrics**: Based on post count and engagement levels
- **Multi-level Analysis**: Individual post sentiment + overall summary sentiment

#### Sentiment Calculation Strategy
```python
def _calculate_advanced_sentiment(self, posts: list[PostData]) -> SentimentScore:
    """Enhanced sentiment analysis with multiple factors."""
    # Weight by engagement score (upvotes + comments)
    # Adjust for subreddit context (WSB vs investing)
    # Consider temporal patterns (recent posts weighted higher)
    # Apply confidence scoring based on data volume
```

### 6. Pydantic Validation

#### Context Structure
```python
@dataclass 
class SocialContext(BaseModel):
    symbol: str | None
    period: dict[str, str]  # {"start": "2024-01-01", "end": "2024-01-31"}
    posts: list[PostData]
    engagement_metrics: dict[str, float]
    sentiment_summary: SentimentScore
    post_count: int
    platforms: list[str]  # ["reddit"]
    metadata: dict[str, Any]
```

#### PostData Format
```python
@dataclass
class PostData(BaseModel):
    title: str
    content: str
    author: str
    source: str  # subreddit name
    date: str
    url: str
    score: int
    comments: int
    engagement_score: int
    subreddit: str | None
    sentiment: SentimentScore | None
    metadata: dict[str, Any]
```

## Implementation Tasks

### Phase 1: Create RedditClient

1. **RedditClient Implementation**
   - Create `tradingagents/clients/reddit_client.py`
   - Follow FinnhubClient standard: no BaseClient inheritance, date objects, proper error handling
   - Use PRAW (Python Reddit API Wrapper) library for Reddit API access
   - Methods: `search_posts()`, `get_top_posts()`, `get_company_posts()`
   - Implement date filtering for posts within specified ranges
   - Handle Reddit API rate limits and authentication

2. **Comprehensive Testing**
   - Create `tradingagents/clients/test_reddit_client.py`
   - Use pytest-vcr for Reddit API interaction recording
   - Test all client methods with multiple queries and subreddits
   - Test error handling and API rate limit scenarios
   - Mock Reddit API responses for consistent testing

### Phase 2: Bridge SocialRepository Interface

3. **Repository Interface Standardization**
   - Add standard service interface methods to `SocialRepository`
   - Bridge existing `get_social_data()` with `get_data()`
   - Bridge existing `store_social_posts()` with `store_data()`
   - Add missing `has_data_for_period()` and `clear_data()` methods
   - File: `tradingagents/repositories/social_repository.py`
   - Maintain existing dataclass functionality while adding service compatibility

4. **Repository Method Implementation**
   ```python
   # Add these methods to SocialRepository
   def has_data_for_period(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> bool
   def get_data(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> dict[str, Any]
   def store_data(self, query: str, cache_data: dict, symbol: str | None = None, overwrite: bool = False) -> bool
   def clear_data(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> bool
   ```

### Phase 3: Update SocialMediaService

5. **Client Integration Fix**
   - Replace `BaseClient` dependency with `RedditClient`
   - File: `tradingagents/services/social_media_service.py:27`
   - Update constructor: `reddit_client: RedditClient`

6. **Date Conversion Fix**
   - Add `date.fromisoformat()` conversion in service methods
   - Update all client calls to use date objects instead of strings
   - File: `tradingagents/services/social_media_service.py:182-190, 418-429`

7. **Repository Interface Integration**
   - Update repository method calls to use new standard interface
   - Ensure proper error handling for repository operations
   - File: `tradingagents/services/social_media_service.py:302-311, 325-337`

### Phase 4: Type Safety & Validation

8. **Comprehensive Type Checking**
   - Run `mise run typecheck` - must pass with 0 errors
   - Validate all date object conversions
   - Ensure SocialContext compliance

9. **Enhanced Testing**
   - Update existing service tests for new RedditClient interface
   - Add gap detection test scenarios
   - Test sentiment analysis accuracy with known datasets
   - Test multi-subreddit aggregation and deduplication

## Success Criteria

### Functional Requirements
- ✅ Service successfully calls `RedditClient` with `date` objects
- ✅ Local-first strategy works: checks cache → identifies gaps → fetches missing → stores updates
- ✅ Returns properly validated `SocialContext` to agents
- ✅ Sentiment analysis provides accurate scores with confidence metrics
- ✅ Multi-subreddit support with post deduplication
- ✅ Force refresh bypasses cache and refreshes data

### Technical Requirements
- ✅ Zero type checking errors: `mise run typecheck`
- ✅ Zero linting errors: `mise run lint`
- ✅ All existing tests pass with updated architecture
- ✅ No runtime errors with date conversions

### Quality Requirements
- ✅ Strongly-typed interfaces between all components
- ✅ PRAW library integration for reliable Reddit API access
- ✅ Comprehensive error handling and logging
- ✅ Efficient caching with minimal API calls
- ✅ Clear separation of concerns between service, client, and repository
- ✅ Accurate sentiment analysis with engagement weighting

## Data Architecture

### RedditClient Response Format
```python
{
    "query": "Tesla",
    "period": {"start": "2024-01-01", "end": "2024-01-31"},
    "posts": [
        {
            "title": "Tesla Q4 earnings beat expectations",
            "content": "Tesla reported strong Q4 results...",
            "author": "teslaInvestor",
            "subreddit": "TeslaInvestors",
            "created_utc": 1704067200,
            "score": 245,
            "num_comments": 67,
            "upvote_ratio": 0.92,
            "url": "https://reddit.com/r/TeslaInvestors/comments/xyz789",
            "id": "xyz789"
        }
    ],
    "metadata": {
        "source": "reddit",
        "retrieved_at": "2024-01-31T10:00:00Z",
        "data_quality": "HIGH",
        "subreddits": ["TeslaInvestors", "stocks"],
        "post_count": 25,
        "api_calls": 3
    }
}
```

### SocialRepository Data Bridge Format
```python
# Repository stores data in existing SocialPost format but provides service interface
{
    "query": "Tesla",
    "symbol": "TSLA",
    "posts": [
        {
            "title": "Tesla Q4 earnings beat expectations",
            "content": "Tesla reported strong Q4 results...",
            "author": "teslaInvestor",
            "source": "TeslaInvestors",
            "date": "2024-01-15",
            "url": "https://reddit.com/r/TeslaInvestors/comments/xyz789",
            "score": 245,
            "comments": 67,
            "engagement_score": 312,
            "subreddit": "TeslaInvestors",
            "sentiment": {
                "score": 0.7,
                "confidence": 0.8,
                "label": "positive"
            },
            "metadata": {
                "platform_id": "xyz789",
                "upvote_ratio": 0.92
            }
        }
    ],
    "metadata": {
        "cached_at": "2024-01-31T10:00:00Z",
        "post_count": 25,
        "sources": ["reddit"]
    }
}
```

## Dependencies

### Missing Components (Need Creation)
- ⏳ `RedditClient` needs full implementation from scratch
- ⏳ Service interface bridge methods for `SocialRepository`
- ⏳ Comprehensive pytest-vcr test suites for Reddit API

### Existing Components (Ready)
- ✅ `SocialRepository` with JSON storage and deduplication
- ✅ `SocialContext` and `PostData` Pydantic models
- ✅ Sentiment analysis and engagement metrics logic

### Required
- PRAW (Python Reddit API Wrapper) library for Reddit integration
- Valid Reddit API credentials (client_id, client_secret, user_agent)
- Working internet connection for live data fetching
- Writable data directory for repository storage

## Timeline

### Immediate (Phase 1)
- Create RedditClient following FinnhubClient standard with PRAW integration
- Implement comprehensive testing with pytest-vcr for Reddit API
- Validate client functionality with multiple subreddits and queries

### Phase 2-3
- Add standard service interface methods to SocialRepository
- Update SocialMediaService to use RedditClient with date objects
- Bridge repository interfaces while maintaining existing functionality

### Phase 4
- Comprehensive type checking and validation
- Integration testing with sentiment analysis workflows
- Performance optimization and caching efficiency

## Acceptance Criteria

### Must Have
1. **Type Safety**: Service passes `mise run typecheck` with zero errors
2. **Client Integration**: All `RedditClient` calls use `date` objects correctly
3. **Local-First**: Service checks repository before Reddit API calls
4. **Context Validation**: Returns valid `SocialContext` with Pydantic validation
5. **Sentiment Analysis**: Provides accurate sentiment scores with confidence metrics
6. **Multi-Platform**: Seamlessly aggregates social data from Reddit with extensibility

### Should Have
1. **Gap Detection**: Intelligent identification of missing data periods
2. **Cache Efficiency**: Minimal redundant API calls to Reddit
3. **Force Refresh**: Complete cache bypass when requested
4. **Data Quality**: Metadata indicating data source and quality metrics
5. **Deduplication**: Automatic removal of duplicate posts by platform_id

### Nice to Have
1. **Performance Metrics**: Timing and cache hit rate logging
2. **Data Staleness**: Automatic refresh of old cached social data
3. **Enhanced Sentiment**: Integration with advanced NLP libraries (TextBlob, VADER)
4. **Real-time Social**: Support for live social media feeds and alerts
5. **Platform Expansion**: Easy addition of Twitter, Discord, other social platforms

---

This PRD focuses on completing the `SocialMediaService` as a strongly-typed, local-first data service that integrates Reddit social media data through a new `RedditClient` following the established FinnhubClient standard patterns, while providing comprehensive sentiment analysis and engagement metrics to trading agents.