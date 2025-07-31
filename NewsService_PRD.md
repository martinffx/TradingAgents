# Product Requirements Document: NewsService Completion

## Overview

Complete the `NewsService` to provide strongly-typed news data and sentiment analysis to trading agents using a local-first data strategy with RSS feed integration, article content extraction, and LLM-powered sentiment analysis.

## Current State Analysis

### Issues to Fix
- **CRITICAL**: Service is currently empty placeholder with only method stubs
- **CRITICAL**: Need to implement GoogleNewsClient to read RSS feeds
- **CRITICAL**: Need RSS article fetching with fallback to Internet Archive
- **CRITICAL**: Need LLM-powered sentiment analysis integration
- **CRITICAL**: Service uses `BaseClient` inheritance instead of typed clients
- **CRITICAL**: `NewsRepository` has different interface than service expectations
- Missing strongly-typed interfaces between components
- No concrete approach for article content extraction

### What Works
- ✅ `NewsContext` and `ArticleData` Pydantic models for agent consumption
- ✅ `SentimentScore` model for structured sentiment data
- ✅ `FinnhubClient` with `get_company_news()` method using date objects
- ✅ `NewsRepository` with dataclass-based storage and deduplication
- ✅ Service structure placeholder ready for implementation

## Technical Requirements

### 1. Strongly-Typed Interfaces

#### Client → Service Interface
```python
# FinnhubClient methods (already implemented)
def get_company_news(symbol: str, start_date: date, end_date: date) -> dict[str, Any]

# GoogleNewsClient methods (to be implemented)
def fetch_rss_feed(query: str, start_date: date, end_date: date) -> dict[str, Any]
def fetch_article_content(url: str, use_archive_fallback: bool = True) -> dict[str, Any]
def get_company_news(symbol: str, start_date: date, end_date: date) -> dict[str, Any]
def get_global_news(start_date: date, end_date: date, categories: list[str]) -> dict[str, Any]
```

#### Service → Repository Interface
```python
# NewsRepository methods (to be implemented/bridged)
def has_data_for_period(query: str, start_date: str, end_date: str, symbol: str | None) -> bool
def get_data(query: str, start_date: str, end_date: str, symbol: str | None) -> dict[str, Any]
def store_data(query: str, cache_data: dict, symbol: str | None, overwrite: bool) -> bool
def clear_data(query: str, start_date: str, end_date: str, symbol: str | None) -> bool
```

#### Service → Agent Interface
```python
# Service output (already defined)
def get_context(query: str, start_date: str, end_date: str, symbol: str | None, sources: list[str], force_refresh: bool) -> NewsContext
```

### 2. Local-First Data Strategy

#### Flow
1. **Repository Lookup**: Check `NewsRepository.has_data_for_period()`
2. **Freshness Check**: Determine if cache needs updating (news is append-only)
3. **RSS Feed Fetching**: Fetch RSS feeds from Google News
4. **Content Extraction**: Extract full article content with Internet Archive fallback
5. **LLM Analysis**: Perform sentiment analysis using LLM
6. **Cache Updates**: Store enriched articles via `repository.store_data()`
7. **Context Assembly**: Return validated `NewsContext`

#### News-Specific Gap Detection
```python
def should_fetch_new_articles(self, last_fetch_time: datetime, current_time: datetime) -> bool:
    """
    News doesn't have "gaps" - it's append-only. Check if enough time passed for new articles.

    Returns True if:
    - Last fetch was more than 6 hours ago
    - User requested force_refresh
    - No data exists for the query/period
    """
    if not last_fetch_time:
        return True

    hours_since_fetch = (current_time - last_fetch_time).total_seconds() / 3600
    return hours_since_fetch >= 6  # Fetch new articles every 6 hours
```

#### Force Refresh Support
- `force_refresh=True` fetches all articles fresh from sources
- Does NOT clear existing cache (news is immutable)
- Deduplicates against existing articles before storing

#### Cache Invalidation Strategy
- **Articles are immutable**: Once published, articles don't change
- **Cache grows append-only**: New articles are added, old ones retained
- **Freshness check**: Re-fetch every 6 hours for new articles
- **No deletion**: Articles are never removed from cache

### 3. RSS Feed Processing & Article Fetching

#### GoogleNewsClient RSS Implementation
```python
import feedparser
from newspaper import Article
import requests
from datetime import date, datetime
from typing import Any, Optional

class GoogleNewsClient:
    """Google News RSS client following FinnhubClient standard."""

    def __init__(self):
        self.base_rss_url = "https://news.google.com/rss"
        self.archive_base_url = "https://archive.org/wayback/available"

    def fetch_rss_feed(self, query: str, start_date: date, end_date: date) -> dict[str, Any]:
        """
        Fetch RSS feed data for news articles.

        Args:
            query: Search query or company symbol
            start_date: Start date for filtering articles
            end_date: End date for filtering articles

        Returns:
            Dict containing RSS feed articles with metadata
        """
        # Construct RSS feed URL
        rss_url = f"{self.base_rss_url}/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        # Parse RSS feed
        feed = feedparser.parse(rss_url)

        # Filter and structure articles
        articles = []
        for entry in feed.entries:
            # Parse publication date
            pub_date = datetime(*entry.published_parsed[:6]).date()

            # Filter by date range
            if start_date <= pub_date <= end_date:
                articles.append({
                    "headline": entry.title,
                    "url": entry.link,
                    "source": entry.source.get('title', 'Google News'),
                    "date": pub_date.isoformat(),
                    "summary": entry.get('summary', ''),
                })

        return {
            "query": query,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "articles": articles,
            "metadata": {
                "source": "google_news_rss",
                "rss_feed_url": rss_url,
                "article_count": len(articles)
            }
        }

    def fetch_article_content(self, url: str, use_archive_fallback: bool = True) -> dict[str, Any]:
        """
        Fetch full article content from URL with Internet Archive fallback.

        Args:
            url: Article URL to fetch
            use_archive_fallback: Whether to try Internet Archive if direct fetch fails

        Returns:
            Dict containing article content, title, publication date
        """
        try:
            # Try direct fetch
            article = Article(url)
            article.download()
            article.parse()

            return {
                "content": article.text,
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "extracted_via": "direct_fetch",
                "extraction_success": True
            }

        except Exception as e:
            if use_archive_fallback:
                # Try Internet Archive
                archive_url = self._get_archive_url(url)
                if archive_url:
                    try:
                        article = Article(archive_url)
                        article.download()
                        article.parse()

                        return {
                            "content": article.text,
                            "title": article.title,
                            "authors": article.authors,
                            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                            "extracted_via": "internet_archive",
                            "extraction_success": True
                        }
                    except Exception:
                        pass

            # Return failure
            return {
                "content": "",
                "title": "",
                "extracted_via": "failed",
                "extraction_success": False,
                "error": str(e)
            }

    def _get_archive_url(self, url: str) -> Optional[str]:
        """Get Internet Archive URL for a given URL."""
        try:
            response = requests.get(f"{self.archive_base_url}?url={url}")
            data = response.json()
            if data.get("archived_snapshots", {}).get("closest", {}).get("available"):
                return data["archived_snapshots"]["closest"]["url"]
        except Exception:
            pass
        return None
```

### 4. LLM-Powered Sentiment Analysis

#### Sentiment Analysis Integration
```python
class LLMSentimentAnalyzer:
    """LLM-based sentiment analyzer for financial news."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.sentiment_prompt = """
        Analyze the sentiment of this financial news article for trading purposes.

        Article:
        Title: {headline}
        Content: {content}

        Provide your analysis in the following JSON format:
        {{
            "score": <float between -1.0 (very negative) and 1.0 (very positive)>,
            "confidence": <float between 0.0 and 1.0>,
            "label": <"positive", "negative", or "neutral">,
            "reasoning": <brief explanation>,
            "key_themes": <list of key financial themes>,
            "financial_entities": <list of mentioned companies/tickers>
        }}

        Focus on the financial and market implications of the news.
        """

    def analyze_sentiment(self, article: ArticleData) -> SentimentScore:
        """
        Analyze article sentiment using LLM.

        Args:
            article: Article data with headline and content

        Returns:
            SentimentScore with score, confidence, and label
        """
        # Prepare prompt
        prompt = self.sentiment_prompt.format(
            headline=article.headline,
            content=article.content[:2000]  # Limit content length
        )

        # Get LLM response
        response = self.llm_client.complete(prompt)

        # Parse response
        try:
            result = json.loads(response)

            # Convert to SentimentScore
            score = result.get("score", 0.0)
            return SentimentScore(
                positive=max(0, score),
                negative=abs(min(0, score)),
                neutral=1.0 - abs(score),
                metadata={
                    "confidence": result.get("confidence", 0.5),
                    "label": result.get("label", "neutral"),
                    "reasoning": result.get("reasoning", ""),
                    "key_themes": result.get("key_themes", []),
                    "financial_entities": result.get("financial_entities", [])
                }
            )
        except Exception as e:
            # Return neutral sentiment on error
            return SentimentScore(
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                metadata={"error": str(e)}
            )

    def batch_analyze(self, articles: list[ArticleData], batch_size: int = 5) -> list[SentimentScore]:
        """
        Batch process sentiment analysis for multiple articles.

        Args:
            articles: List of articles to analyze
            batch_size: Number of articles to process in parallel

        Returns:
            List of sentiment scores corresponding to input articles
        """
        results = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]

            # Process batch (could be parallelized)
            for article in batch:
                sentiment = self.analyze_sentiment(article)
                results.append(sentiment)

                # Add small delay to respect rate limits
                time.sleep(0.1)

        return results
```

### 5. Date Object Conversion

#### Service Boundary Conversion
```python
# Service receives string dates from agents
def get_context(self, query: str, start_date: str, end_date: str, ...) -> NewsContext:
    # Validate date strings
    try:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")

    # Check date order
    if end_dt < start_dt:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    # Fetch from multiple sources
    finnhub_data = self.finnhub_client.get_company_news(symbol, start_dt, end_dt) if symbol else None
    google_rss = self.google_client.fetch_rss_feed(query, start_dt, end_dt)

    # Fetch full article content for RSS articles
    for article in google_rss.get('articles', []):
        content_data = self.google_client.fetch_article_content(article['url'])
        article.update(content_data)

    # Combine all articles
    all_articles = self._combine_and_deduplicate(finnhub_data, google_rss)

    # Perform LLM sentiment analysis
    enriched_articles = []
    for article in all_articles:
        article_data = ArticleData(**article)
        article_data.sentiment = self.sentiment_analyzer.analyze_sentiment(article_data)
        enriched_articles.append(article_data)

    # Create and return context
    return self._create_news_context(enriched_articles, start_date, end_date)
```

### 6. Error Recovery and Partial Data

```python
def handle_source_failure(
    self,
    finnhub_data: dict | None,
    google_data: dict | None,
    errors: dict[str, Exception]
) -> NewsContext:
    """
    Handle cases where one or more news sources fail.

    - If all sources fail: Raise exception
    - If some sources succeed: Return partial data with metadata
    - Track content extraction failures separately
    """
    if not finnhub_data and not google_data:
        raise ValueError("All news sources failed to return data")

    # Track extraction statistics
    extraction_stats = {
        "total_articles": 0,
        "successful_extractions": 0,
        "archive_fallbacks": 0,
        "failed_extractions": 0
    }

    # Process available articles
    all_articles = []
    successful_sources = []

    if finnhub_data:
        all_articles.extend(finnhub_data.get('articles', []))
        successful_sources.append('finnhub')

    if google_data:
        articles = google_data.get('articles', [])
        for article in articles:
            extraction_stats["total_articles"] += 1
            if article.get("extraction_success"):
                extraction_stats["successful_extractions"] += 1
                if article.get("extracted_via") == "internet_archive":
                    extraction_stats["archive_fallbacks"] += 1
            else:
                extraction_stats["failed_extractions"] += 1

        all_articles.extend(articles)
        successful_sources.append('google_news')

    metadata = {
        "sources_requested": ["finnhub", "google_news"],
        "sources_successful": successful_sources,
        "sources_failed": {source: str(error) for source, error in errors.items()},
        "extraction_stats": extraction_stats,
        "partial_data": len(successful_sources) < 2
    }

    # Deduplicate and return context
    return self._create_context(all_articles, metadata)
```

### 7. Repository Method Bridging

```python
# Add these bridge methods to NewsRepository
def has_data_for_period(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> bool:
    """Bridge to existing get_news_data method."""
    existing_data = self.get_news_data(
        symbol=symbol or query,
        start_date=start_date,
        end_date=end_date
    )
    return len(existing_data.get('articles', [])) > 0

def get_data(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> dict[str, Any]:
    """Bridge to existing get_news_data method."""
    return self.get_news_data(
        symbol=symbol or query,
        start_date=start_date,
        end_date=end_date
    )

def store_data(self, query: str, cache_data: dict, symbol: str | None = None, overwrite: bool = False) -> bool:
    """Bridge to existing store_news_articles method."""
    articles = cache_data.get('articles', [])
    if not articles:
        return False

    # Convert to expected format
    news_articles = [
        NewsArticle(
            symbol=symbol or query,
            headline=a['headline'],
            summary=a.get('summary', ''),
            content=a.get('content', ''),
            url=a['url'],
            source=a['source'],
            date=a['date'],
            entities=a.get('entities', []),
            sentiment_score=a.get('sentiment', {}).get('score', 0.0),
            sentiment_metadata=a.get('sentiment', {})
        )
        for a in articles
    ]

    return self.store_news_articles(news_articles)

def clear_data(self, query: str, start_date: str, end_date: str, symbol: str | None = None) -> bool:
    """News is append-only, so this just marks data as stale for re-fetch."""
    # Implementation depends on repository design
    # Could update metadata to trigger re-fetch
    return True
```

### 8. Pydantic Validation

#### Context Structure
```python
@dataclass
class NewsContext(BaseModel):
    symbol: str | None
    period: dict[str, str]  # {"start": "2024-01-01", "end": "2024-01-31"}
    articles: list[ArticleData]
    sentiment_summary: SentimentScore
    article_count: int
    sources: list[str]
    metadata: dict[str, Any]

    @validator('period')
    def validate_period(cls, v):
        # Ensure start and end dates are present and valid
        if 'start' not in v or 'end' not in v:
            raise ValueError("Period must have 'start' and 'end' dates")
        return v

    @validator('articles')
    def validate_articles(cls, v):
        # Ensure no duplicate URLs
        urls = [a.url for a in v]
        if len(urls) != len(set(urls)):
            raise ValueError("Duplicate articles detected")
        return v
```

## Implementation Tasks

### Phase 1: Create GoogleNewsClient

1. **GoogleNewsClient Implementation**
   - Create `tradingagents/clients/google_news_client.py` following FinnhubClient standard
   - Implement RSS feed parsing using `feedparser` library
   - Add `fetch_rss_feed()` method with Google News RSS integration
   - Add `fetch_article_content()` method with `newspaper3k` and Internet Archive fallback
   - Use `date` objects for all date parameters
   - No BaseClient inheritance

2. **Article Content Extraction**
   - Implement robust article content extraction using `newspaper3k`
   - Add fallback to Internet Archive Wayback Machine for failed fetches
   - Handle paywall detection and alternative content sources
   - Extract clean text, title, publication date, and metadata

3. **Comprehensive Testing**
   - Create test suite for GoogleNewsClient
   - Test RSS parsing with various queries
   - Test content extraction with real and archived URLs
   - Use pytest-vcr for HTTP interaction recording

### Phase 2: Bridge NewsRepository Interface

4. **Repository Interface Standardization**
   - Add standard service interface methods to `NewsRepository`
   - Bridge existing methods without changing underlying storage
   - File: `tradingagents/repositories/news_repository.py`
   - Maintain backward compatibility

### Phase 3: Implement NewsService

5. **Service Core Implementation**
   - Replace method stubs with full implementation
   - Implement `get_context()`, `get_company_news_context()`, `get_global_news_context()`
   - Add local-first data strategy with freshness checking
   - Replace `BaseClient` dependencies with typed clients
   - File: `tradingagents/services/news_service.py`

6. **LLM Sentiment Analysis Integration**
   - Implement `LLMSentimentAnalyzer` class
   - Create financial news sentiment prompts
   - Add batch processing for efficiency
   - Handle LLM rate limiting and errors

7. **Date Conversion and Article Processing**
   - Add date validation and conversion
   - Implement RSS article fetching pipeline
   - Add content extraction with fallback
   - Combine articles from multiple sources
   - Implement deduplication by URL

### Phase 4: Type Safety & Validation

8. **Comprehensive Type Checking**
   - Run `mise run typecheck` - must pass with 0 errors
   - Validate all date object conversions
   - Ensure NewsContext compliance

9. **Enhanced Testing**
   - Test RSS feed parsing edge cases
   - Test content extraction failures and fallbacks
   - Test LLM sentiment analysis with various article types
   - Test multi-source aggregation and deduplication

## Testing Scenarios

### Integration Tests

1. **RSS Feed Processing**
   - Test with various search queries
   - Test date filtering in RSS results
   - Test handling of malformed RSS feeds

2. **Content Extraction**
   - Test direct fetch success
   - Test Internet Archive fallback
   - Test paywall detection
   - Test extraction failure handling

3. **LLM Sentiment Analysis**
   - Test positive news sentiment
   - Test negative earnings reports
   - Test neutral market updates
   - Test batch processing
   - Test LLM error handling

4. **Multi-Source Aggregation**
   - Test both sources succeed
   - Test Finnhub fails, Google succeeds
   - Test Google fails, Finnhub succeeds
   - Test both sources fail

5. **Date Handling**
   - Test invalid date formats
   - Test end_date < start_date
   - Test date filtering in RSS feeds

## Success Criteria

### Functional Requirements
- ✅ Service successfully implements all placeholder methods
- ✅ GoogleNewsClient reads and parses RSS feeds correctly
- ✅ Article content extraction works with Internet Archive fallback
- ✅ LLM sentiment analysis provides structured financial sentiment
- ✅ Local-first strategy with proper freshness checking
- ✅ Multi-source aggregation with deduplication
- ✅ Returns properly validated `NewsContext` to agents
- ✅ Force refresh fetches fresh articles without clearing cache

### Technical Requirements
- ✅ Zero type checking errors: `mise run typecheck`
- ✅ Zero linting errors: `mise run lint`
- ✅ All tests pass with new implementation
- ✅ No runtime errors with date conversions
- ✅ Proper error messages for validation failures

### Quality Requirements
- ✅ Strongly-typed interfaces between all components
- ✅ RSS feed parsing with robust error handling
- ✅ Article content extraction with fallback strategy
- ✅ LLM integration with proper prompt engineering
- ✅ Efficient caching with minimal external calls
- ✅ Clear separation of concerns

## Data Architecture

### GoogleNewsClient RSS Response Format
```python
{
    "query": "Apple stock",
    "period": {"start": "2024-01-01", "end": "2024-01-31"},
    "articles": [
        {
            "headline": "Apple Stock Soars on New Product Launch",
            "summary": "Brief summary from RSS feed...",
            "content": "Full article text extracted from source...",
            "url": "https://www.cnbc.com/2024/01/20/apple-stock.html",
            "source": "CNBC",
            "date": "2024-01-20",
            "authors": ["Tech Reporter"],
            "publish_date": "2024-01-20T14:30:00Z",
            "extracted_via": "direct_fetch",  # or "internet_archive"
            "extraction_success": true
        }
    ],
    "metadata": {
        "source": "google_news_rss",
        "article_count": 25,
        "rss_feed_url": "https://news.google.com/rss/search?q=Apple+stock",
        "extraction_stats": {
            "successful": 22,
            "archive_fallback": 2,
            "failed": 3
        }
    }
}
```

### LLM Sentiment Analysis Response Format
```python
{
    "article_url": "https://www.cnbc.com/2024/01/20/apple-stock.html",
    "sentiment": {
        "positive": 0.7,
        "negative": 0.1,
        "neutral": 0.2,
        "metadata": {
            "score": 0.7,
            "confidence": 0.85,
            "label": "positive",
            "reasoning": "Article discusses positive earnings and growth outlook",
            "key_themes": ["earnings_beat", "product_launch", "revenue_growth"],
            "financial_entities": ["AAPL", "Apple Inc.", "iPhone 15"]
        }
    }
}
```

### Aggregate Sentiment Summary
```python
{
    "sentiment_summary": {
        "positive": 0.65,  # Average across all articles
        "negative": 0.20,
        "neutral": 0.15,
        "metadata": {
            "dominant_sentiment": "positive",
            "confidence": 0.82,
            "article_count": 25,
            "themes": {
                "earnings": 8,
                "product_launch": 5,
                "market_analysis": 12
            }
        }
    }
}
```

## Dependencies

### Components to Create
- ⏳ `GoogleNewsClient` - Full implementation with RSS and content extraction
- ⏳ `LLMSentimentAnalyzer` - LLM integration for sentiment analysis
- ⏳ `NewsService` - Replace stubs with full implementation

### Existing Components
- ✅ `FinnhubClient` with company news using date objects
- ✅ `NewsRepository` with dataclass storage
- ✅ `NewsContext` and related Pydantic models

### Required Libraries
- `feedparser` - RSS feed parsing
- `newspaper3k` - Article content extraction
- `requests` - HTTP requests and Internet Archive API
- `beautifulsoup4` - HTML parsing fallback
- LLM client library (OpenAI, Anthropic, etc.)

## Timeline

### Immediate (Phase 1)
- Create GoogleNewsClient with RSS and content extraction
- Implement feedparser integration
- Add Internet Archive fallback
- Create comprehensive test suite

### Phase 2-3
- Add repository bridge methods
- Implement full NewsService
- Integrate LLM sentiment analysis
- Handle multi-source aggregation

### Phase 4
- Type checking and validation
- Integration testing
- Performance optimization
- Documentation

## Acceptance Criteria

### Must Have
1. **Type Safety**: Service passes `mise run typecheck` with zero errors
2. **RSS Integration**: Successfully parse Google News RSS feeds
3. **Content Extraction**: Extract full articles with fallback
4. **LLM Sentiment**: Financial sentiment analysis for all articles
5. **Service Implementation**: All stubs replaced with working code
6. **Local-First**: Check cache before fetching new data
7. **Multi-Source**: Aggregate Finnhub and Google News

### Should Have
1. **Extraction Stats**: Track success/failure rates
2. **Batch Processing**: Efficient LLM sentiment analysis
3. **Force Refresh**: Fetch new articles on demand
4. **Error Recovery**: Handle partial failures gracefully

### Nice to Have
1. **Additional Sources**: Support more news providers
2. **Real-time Monitoring**: WebSocket for breaking news
3. **Advanced Extraction**: Handle PDFs, videos
4. **Sentiment Trends**: Track sentiment over time

---

This PRD focuses on completing the currently empty `NewsService` with a full implementation including RSS feed integration, article content extraction with Internet Archive fallback, and LLM-powered sentiment analysis for financial news.
