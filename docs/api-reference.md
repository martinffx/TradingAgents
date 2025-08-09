# TradingAgents API Reference

This document provides comprehensive API documentation for the TradingAgents framework.

## Core Classes

### TradingAgentsGraph

Main orchestrator class for running trading analysis workflows.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.config import TradingAgentsConfig

# Initialize with default configuration
ta = TradingAgentsGraph(debug=True, config=TradingAgentsConfig.from_env())

# Run analysis for a specific stock and date
result, decision = ta.propagate("AAPL", "2024-01-15")
```

**Methods:**

- `propagate(symbol: str, date: str) -> tuple[dict, str]`: Execute full trading analysis workflow
- `get_memory()`: Access the vector memory system for past decisions

### TradingAgentsConfig

Configuration management class with environment variable support.

```python
from tradingagents.config import TradingAgentsConfig

# Create from environment variables
config = TradingAgentsConfig.from_env()

# Create with custom values
config = TradingAgentsConfig(
    llm_provider="anthropic",
    deep_think_llm="claude-3-5-sonnet-latest",
    max_debate_rounds=3,
    online_tools=True
)
```

**Configuration Options:**

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|---------------------|-------------|
| `project_dir` | str | Current directory | `TRADINGAGENTS_PROJECT_DIR` | Base project directory |
| `results_dir` | str | "./results" | `TRADINGAGENTS_RESULTS_DIR` | Output directory for analysis results |
| `data_dir` | str | "/Users/yluo/Documents/Code/ScAI/FR1-data" | `TRADINGAGENTS_DATA_DIR` | Directory for local data storage |
| `llm_provider` | Literal | "openai" | `LLM_PROVIDER` | LLM provider (openai, anthropic, google, ollama, openrouter) |
| `deep_think_llm` | str | "o4-mini" | `DEEP_THINK_LLM` | Model for complex reasoning tasks |
| `quick_think_llm` | str | "gpt-4o-mini" | `QUICK_THINK_LLM` | Model for fast responses |
| `backend_url` | str | "https://api.openai.com/v1" | `BACKEND_URL` | API endpoint for LLM providers |
| `max_debate_rounds` | int | 1 | `MAX_DEBATE_ROUNDS` | Maximum rounds in investment debates |
| `max_risk_discuss_rounds` | int | 1 | `MAX_RISK_DISCUSS_ROUNDS` | Maximum rounds in risk discussions |
| `max_recur_limit` | int | 100 | `MAX_RECUR_LIMIT` | Maximum recursion depth for workflows |
| `online_tools` | bool | True | `ONLINE_TOOLS` | Enable live API calls vs cached data |
| `default_lookback_days` | int | 30 | `DEFAULT_LOOKBACK_DAYS` | Historical data range for analysis |
| `default_ta_lookback_days` | int | 30 | `DEFAULT_TA_LOOKBACK_DAYS` | Technical analysis data range |

## Domain Services

### MarketDataService

Provides market data and technical analysis.

```python
from tradingagents.domains.marketdata.market_data_service import MarketDataService

service = MarketDataService.build(config)

# Get market data context
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")
print(f"Latest price: ${context.latest_price}")
print(f"Price change: {context.price_change_percent}%")

# Update market data
service.update_market_data("AAPL", "2024-01-01", "2024-01-31")
```

**Methods:**

- `get_market_data_context(symbol, start_date, end_date) -> PriceDataContext`: Get price and volume data
- `get_ta_report(symbol, start_date, end_date) -> TAReportContext`: Get technical analysis indicators
- `update_market_data(symbol, start_date, end_date)`: Fetch and cache fresh market data

### NewsService

Provides news analysis and sentiment scoring.

```python
from tradingagents.domains.news.news_service import NewsService

service = NewsService.build(config)

# Get stock-specific news
context = service.get_news_context("AAPL", "2024-01-01", "2024-01-31")
print(f"Articles found: {context.article_count}")
print(f"Overall sentiment: {context.sentiment_summary.label}")

# Get global news context
global_context = service.get_global_news_context("2024-01-15")
```

**Methods:**

- `get_news_context(symbol, start_date, end_date) -> NewsContext`: Get stock-specific news
- `get_global_news_context(date) -> GlobalNewsContext`: Get general market news
- `update_news(symbol, start_date, end_date)`: Fetch and cache fresh news articles

### SocialMediaService

Provides social media sentiment analysis.

```python
from tradingagents.domains.socialmedia.social_media_service import SocialMediaService

service = SocialMediaService.build(config)

# Get social media sentiment
context = service.get_socialmedia_stock_info("AAPL", "2024-01-15")
print(f"Posts analyzed: {context.post_count}")
print(f"Sentiment score: {context.sentiment_summary.score}")
```

**Methods:**

- `get_socialmedia_stock_info(symbol, date) -> StockSocialContext`: Get social media analysis
- `update_socialmedia_data(symbol, date)`: Fetch and cache fresh social media data

### FundamentalDataService

Provides financial statements and fundamental analysis.

```python
from tradingagents.domains.marketdata.fundamental_data_service import FundamentalDataService

service = FundamentalDataService.build(config)

# Get financial statements
income_stmt = service.get_income_statement_context("AAPL", "2024-01-15")
balance_sheet = service.get_balance_sheet_context("AAPL", "2024-01-15")
cash_flow = service.get_cash_flow_context("AAPL", "2024-01-15")
```

**Methods:**

- `get_income_statement_context(symbol, date) -> IncomeStatementContext`: Get income statement data
- `get_balance_sheet_context(symbol, date) -> BalanceSheetContext`: Get balance sheet data
- `get_cash_flow_context(symbol, date) -> CashFlowContext`: Get cash flow data

### InsiderDataService

Provides insider trading data and sentiment analysis.

```python
from tradingagents.domains.marketdata.insider_data_service import InsiderDataService

service = InsiderDataService.build(config)

# Get insider transactions
transactions = service.get_insider_transaction_context("AAPL", "2024-01-01", "2024-01-31")
sentiment = service.get_insider_sentiment_context("AAPL", "2024-01-15")
```

**Methods:**

- `get_insider_transaction_context(symbol, start_date, end_date) -> InsiderTransactionContext`: Get insider trading data
- `get_insider_sentiment_context(symbol, date) -> InsiderSentimentContext`: Get insider sentiment analysis

## AgentToolkit (Anti-Corruption Layer)

The AgentToolkit mediates between agents and domain services, providing LangChain tool decorators.

```python
from tradingagents.agents.libs.agent_toolkit import AgentToolkit

# AgentToolkit is injected into agents automatically
# Provides @tool decorated methods for LangChain agent consumption

# Available tools:
# - get_market_data(symbol, start_date, end_date)
# - get_ta_report(symbol, start_date, end_date) 
# - get_news(symbol, start_date, end_date)
# - get_global_news(date)
# - get_socialmedia_stock_info(symbol, date)
# - get_income_statement(symbol, date)
# - get_balance_sheet(symbol, date)
# - get_cash_flow_statement(symbol, date)
# - get_insider_transactions(symbol, start_date, end_date)
# - get_insider_sentiment(symbol, date)
```

## Context Models

### PriceDataContext

Market data context returned by MarketDataService.

```python
@dataclass
class PriceDataContext:
    symbol: str
    start_date: str
    end_date: str
    price_data: list[dict]
    latest_price: float
    price_change_percent: float
    volume_data: list[dict]
    metadata: dict[str, Any]
```

### TAReportContext

Technical analysis context with indicators.

```python
@dataclass 
class TAReportContext:
    symbol: str
    start_date: str
    end_date: str
    indicators: dict[str, Any]
    signals: list[str]
    metadata: dict[str, Any]
```

### NewsContext

News analysis context with sentiment scoring.

```python
@dataclass
class NewsContext:
    symbol: str
    start_date: str
    end_date: str
    articles: list[NewsArticle]
    article_count: int
    sentiment_summary: SentimentScore
    metadata: dict[str, Any]
```

### StockSocialContext

Social media analysis context.

```python
@dataclass
class StockSocialContext:
    symbol: str
    date: str
    posts: list[PostData]
    post_count: int
    sentiment_summary: SentimentScore
    engagement_metrics: EngagementMetrics
    metadata: dict[str, Any]
```

## Error Handling

All services implement consistent error handling patterns:

```python
try:
    context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")
except ServiceException as e:
    # Service-level errors (API failures, data validation)
    logger.error(f"Service error: {e}")
except ClientException as e:
    # Client-level errors (network, authentication)
    logger.error(f"Client error: {e}")
except RepositoryException as e:
    # Repository-level errors (file I/O, cache corruption)
    logger.error(f"Repository error: {e}")
```

## Multi-LLM Provider Support

### OpenAI Configuration

```bash
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your_openai_api_key"
export DEEP_THINK_LLM="gpt-4"
export QUICK_THINK_LLM="gpt-4o-mini"
export BACKEND_URL="https://api.openai.com/v1"
```

### Anthropic Configuration

```bash
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export DEEP_THINK_LLM="claude-3-5-sonnet-latest"
export QUICK_THINK_LLM="claude-3-haiku-latest"
export BACKEND_URL="https://api.anthropic.com"
```

### Google Configuration

```bash
export LLM_PROVIDER="google"
export GOOGLE_API_KEY="your_google_api_key"
export DEEP_THINK_LLM="gemini-1.5-pro"
export QUICK_THINK_LLM="gemini-1.5-flash"
export BACKEND_URL="https://generativelanguage.googleapis.com"
```

## Memory System

The framework includes a ChromaDB-based vector memory system for learning from past decisions.

```python
# Memory is automatically managed by TradingAgentsGraph
graph = TradingAgentsGraph(debug=True, config=config)

# Access memory directly if needed
memory = graph.get_memory()
similar_situations = memory.search("AAPL analysis", k=5)
```

**Memory Features:**

- Automatic storage of decision contexts and outcomes
- Vector similarity search for similar market situations
- Learning from historical performance to improve future decisions
- Configurable retention policies

## Best Practices

### Service Usage Patterns

```python
# Always read from repository first (cached data)
context = service.get_context(symbol, start_date, end_date)

# Use separate update operations for fresh data
if context.metadata.get("data_quality") == "LOW":
    service.update_data(symbol, start_date, end_date)
    context = service.get_context(symbol, start_date, end_date)
```

### Error Resilience

```python
# Services return contexts with quality metadata
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")

if context.metadata.get("data_quality") == "HIGH":
    # Use data confidently
    latest_price = context.latest_price
else:
    # Handle degraded data quality
    logger.warning("Using cached/partial data due to API issues")
```

### Configuration Management

```python
# Use environment-based configuration for different environments
config = TradingAgentsConfig.from_env()

# Override specific settings for testing
test_config = config.copy()
test_config.online_tools = False  # Use cached data only
test_config.max_debate_rounds = 1  # Speed up tests
```