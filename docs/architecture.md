# TradingAgents Architecture Documentation

## Multi-Agent Trading Framework

TradingAgents implements a sophisticated multi-agent system that mirrors real-world trading firms with specialized roles and structured workflows.

## Core Architecture Components

### 1. **Agent Teams** (Sequential Workflow)

```
Analyst Team → Research Team → Trading Team → Risk Management Team
```

**Analyst Team** (`tradingagents/agents/analysts/`)
- **Market Analyst**: Technical analysis using Yahoo Finance and StockStats
- **Fundamentals Analyst**: Financial statements and company fundamentals via SimFin/Finnhub
- **News Analyst**: News sentiment analysis and world affairs impact
- **Social Media Analyst**: Reddit and social platform sentiment analysis

**Research Team** (`tradingagents/agents/researchers/`)
- **Bull Researcher**: Advocates for investment opportunities and growth potential
- **Bear Researcher**: Highlights risks and argues against investments
- **Research Manager**: Synthesizes debates and creates investment recommendations

**Trading Team** (`tradingagents/agents/trader/`)
- **Trader**: Converts investment plans into specific trading decisions

**Risk Management Team** (`tradingagents/agents/risk_mgmt/`)
- **Aggressive/Conservative/Neutral Debators**: Different risk perspectives
- **Risk Manager**: Final decision maker balancing risk and reward

### 2. **Domain-Driven Architecture** (`tradingagents/domains/`)

**Domain-Driven Design (DDD) Architecture** (Current):
The system has been restructured using Domain-Driven Design principles with three main bounded contexts:

**Domain Boundaries & Bounded Contexts:**
- **Financial Data Domain** (`tradingagents/domains/marketdata/`): Market prices, technical indicators, fundamentals, insider data
- **News Domain** (`tradingagents/domains/news/`): News articles, sentiment analysis, content aggregation  
- **Social Media Domain** (`tradingagents/domains/socialmedia/`): Social media posts, engagement metrics, sentiment analysis

**DDD Tactical Patterns per Domain:**
- **Domain Services**: Business logic encapsulated in domain-specific services (`MarketDataService`, `NewsService`, `SocialMediaService`)
- **Value Objects**: Immutable data structures (`SentimentScore`, `TechnicalIndicatorData`, `PostMetadata`)
- **Entities**: Objects with identity and lifecycle (`NewsArticle`, `PostData`)
- **Repository Pattern**: Domain-specific data access with smart caching, deduplication, and gap detection
- **Context Objects**: Structured domain data containers (`MarketDataContext`, `NewsContext`, `SocialContext`)

**Domain Infrastructure per Bounded Context:**
```
marketdata/
├── clients/          # YFinanceClient, FinnhubClient (domain-specific)
├── repos/           # MarketDataRepository, FundamentalRepository
├── services/        # MarketDataService, FundamentalDataService, InsiderDataService  
└── models/          # Domain Value Objects and Entities

news/
├── clients/         # GoogleNewsClient (domain-specific)
├── repositories/    # NewsRepository with article deduplication
├── services/        # NewsService with sentiment analysis
└── models/          # NewsArticle, SentimentScore

socialmedia/
├── clients/         # RedditClient (domain-specific) 
├── repositories/    # SocialMediaRepository with engagement tracking
├── services/        # SocialMediaService with sentiment analysis
└── models/          # PostData, EngagementMetrics
```

**Agent Integration Strategy - Anti-Corruption Layer (ACL):**
- **AgentToolkit as ACL**: Mediates between agents (string-based, procedural) and domains (object-oriented, rich models)
- **Data Translation**: Converts rich Pydantic domain models to structured JSON strings for LLM consumption
- **Parameter Adaptation**: Handles interface mismatches (single date → date ranges, etc.)
- **Backward Compatibility**: Preserves existing agent tool interface while providing domain service benefits

### 3. **Graph Orchestration** (`tradingagents/graph/`)

LangGraph-based workflow management:

- **TradingAgentsGraph**: Main orchestrator class
- **State Management**: `AgentState`, `InvestDebateState`, `RiskDebateState` track workflow progress
- **Conditional Logic**: Dynamic routing based on tool usage and debate completion
- **Memory System**: ChromaDB-based vector memory for learning from past decisions

### 4. **Configuration System**
- **TradingAgentsConfig**: Centralized configuration with environment variable support
- **Multi-LLM Support**: OpenAI, Anthropic, Google, Ollama, OpenRouter
- **Data Modes**: Online (live APIs) vs offline (cached data)

## Architecture Design Patterns and Principles

### Core Design Principles

1. **Separation of Concerns**: Each domain has clear boundaries and responsibilities
2. **Single Responsibility Principle**: Each class and module has one reason to change
3. **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations
4. **Open/Closed Principle**: Modules are open for extension but closed for modification
5. **Interface Segregation**: Clients only depend on methods they actually use

### Key Architectural Patterns

1. **Domain-Driven Design (DDD)**:
   - Bounded contexts ensure clear separation between domains
   - Ubiquitous language ensures consistent terminology across code and business
   - Entities, Value Objects, and Aggregates provide rich domain models
   - Repositories abstract data persistence concerns

2. **Anti-Corruption Layer (ACL)**:
   - AgentToolkit protects domain models from agent implementation details
   - Translation layer ensures clean integration between procedural agents and object-oriented domains
   - Backward compatibility maintained while improving architecture

3. **Repository Pattern**:
   - Smart caching reduces API calls and improves performance
   - Data deduplication ensures consistency
   - Gap detection identifies missing data ranges
   - Local storage provides offline capability

4. **Service Layer Pattern**:
   - Business logic encapsulated in domain services
   - Thin client implementations for API integrations
   - Rich context objects provide structured data for agents

### Data Flow Architecture

1. **Request Processing**:
   - Agents request data through AgentToolkit
   - Toolkit translates requests to domain service calls
   - Services retrieve data from repositories or clients
   - Results are formatted as structured context objects

2. **Data Updates**:
   - Services can update repositories with fresh data
   - Clients fetch data from external APIs
   - Data is cached with metadata for quality assessment

3. **Memory Integration**:
   - ChromaDB stores vector embeddings of past decisions
   - Similar situations are retrieved for context-aware decision making
   - Learning from historical performance improves future decisions

## Key Design Patterns

1. **Debate-Driven Decision Making**: Critical decisions emerge from structured agent debates
2. **Memory-Augmented Learning**: Agents learn from past similar situations using vector similarity
3. **Repository-First Data Strategy**: Services always read from repositories with separate update operations
4. **Structured JSON Contexts**: Replace error-prone string parsing with rich Pydantic models
5. **Factory Pattern**: Agent creation via factory functions for flexible configuration
6. **Signal Processing**: Final trading decisions processed into clean BUY/SELL/HOLD signals
7. **Quality-Aware Data**: All contexts include quality metadata to help agents make better decisions

## Working with Agents

**Current Approach** (AgentToolkit as Anti-Corruption Layer):
- Use `AgentToolkit` from `tradingagents.agents.libs.agent_toolkit`
- Toolkit injects all domain services via dependency injection
- Provides LangChain `@tool` decorated methods for agent consumption
- Returns rich Pydantic domain models directly to agents
- Handles parameter validation, date calculations, and error handling

**Agent Integration Pattern**:
```python
from tradingagents.agents.libs.agent_toolkit import AgentToolkit

# AgentToolkit acts as Anti-Corruption Layer
toolkit = AgentToolkit(
    news_service=news_service,
    marketdata_service=marketdata_service,
    fundamentaldata_service=fundamentaldata_service,
    socialmedia_service=socialmedia_service,
    insiderdata_service=insiderdata_service
)

# Agents use toolkit tools that return rich domain contexts
@tool
def analyze_stock(symbol: str, date: str):
    # Get structured contexts from domain services via toolkit
    market_data = toolkit.get_market_data(symbol, start_date, end_date)
    social_data = toolkit.get_socialmedia_stock_info(symbol, date)
    news_data = toolkit.get_news(symbol, start_date, end_date)
    
    # Work with rich Pydantic models
    price = market_data.latest_price
    sentiment = social_data.sentiment_summary.score
    article_count = news_data.article_count
```

## Working with Data Sources

**Current Domain Service Approach**:
- **Repository-First**: Services always read data from repositories (local storage)
- **Separate Update Operations**: Use dedicated update methods to fetch fresh data from APIs and store in repositories
- **Clear Separation**: Reading data vs updating data are separate concerns
- **Structured Contexts**: Services return rich Pydantic models with metadata
- **Quality Awareness**: All contexts include data quality and source information

**Service Usage Pattern**:
```python
# Services use dependency injection
service = MarketDataService(
    yfin_client=YFinanceClient(),
    repo=MarketDataRepository("cache_dir")
)

# Always read from repository
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")

# Separate update operation to refresh repository data
service.update_market_data("AAPL", "2024-01-01", "2024-01-31")
```

## Performance Optimization Guidelines

The TradingAgents framework is designed for efficiency in financial analysis workflows. Here are key optimization strategies:

### 1. Caching and Data Management

**Repository Pattern Caching:**
- All domain services use repository-first data access pattern
- Data is cached locally to minimize API calls
- Smart caching with automatic invalidation based on data freshness
- Deduplication and gap detection in stored data

**Best Practices:**
```python
# Efficient data access pattern
service = MarketDataService.build(config)

# Always read from repository first (cached data)
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")

# Only update when fresh data is needed
service.update_market_data("AAPL", "2024-01-01", "2024-01-31")
```

### 2. LLM Cost and Performance Optimization

**Model Selection Strategy:**
- Use `quick_think_llm` for simple data retrieval and formatting tasks
- Reserve `deep_think_llm` for complex analysis and decision-making
- Configure appropriate models based on your cost/performance requirements

**API Call Minimization:**
- Batch similar requests when possible
- Cache LLM responses for identical queries
- Use structured outputs to reduce need for follow-up clarifications

### 3. Memory Management

**Vector Memory Optimization:**
- ChromaDB-based memory system automatically manages vector storage
- Configure memory retention policies to balance performance and storage
- Use memory efficiently by storing only key decision points and learnings

### 4. Parallel Processing

**Graph Execution Optimization:**
- LangGraph workflows can execute independent nodes in parallel
- Configure appropriate concurrency limits to avoid API rate limiting
- Monitor and optimize critical paths in the workflow graph

**Configuration for Performance:**
```python
# Optimize for cost-sensitive environments
config = TradingAgentsConfig(
    deep_think_llm="gpt-4.1-mini",  # Lower cost model
    quick_think_llm="gpt-4.1-mini",  # Lower cost model
    max_debate_rounds=1,  # Reduce debate rounds
    online_tools=False,  # Use cached data when possible
    default_lookback_days=30  # Limit data range
)
```

## Data Directory Structure

The TradingAgents framework expects a specific directory structure for data storage:

```
project_dir/
├── results/                 # Analysis results (configurable via results_dir)
├── dataflows/               
│   └── data_cache/          # Cached data (automatically managed)
├── tradingagents/            # Core framework code
└── cli/                     # Command-line interface
```

For custom data directories, ensure the path exists and is writable. The framework will automatically create necessary subdirectories for caching.

## File Structure Context
- **`cli/`**: Interactive command-line interface
- **`tradingagents/agents/`**: All agent implementations
  - **`libs/agent_toolkit.py`**: AgentToolkit Anti-Corruption Layer with LangChain @tool decorators
  - **`libs/context_helpers.py`**: Helper functions for parsing structured JSON data
  - **`libs/agent_utils.py`**: Legacy Toolkit (being phased out)
- **`tradingagents/domains/`**: Domain-Driven Design bounded contexts
  - **`marketdata/`**: Financial data domain (prices, indicators, fundamentals, insider data)
  - **`news/`**: News domain (articles, sentiment analysis)
  - **`socialmedia/`**: Social media domain (posts, engagement, sentiment)
- **`tradingagents/dataflows/`**: Legacy data source integrations (being phased out)
- **`tradingagents/graph/`**: LangGraph workflow orchestration
- **`tradingagents/config.py`**: Configuration management
- **`main.py`**: Direct Python usage example
- **`docs/agent-development.md`**: Detailed agent documentation

## Trading Strategy Implementation

The strategy is implemented through a graph-based workflow using LangGraph:

1. **Sequential Processing**: Analyst teams process data in sequence
2. **Debate-Driven Decision Making**: Researchers engage in structured debates
3. **Risk Assessment**: Risk managers evaluate potential downside scenarios
4. **Signal Generation**: Final trading decisions are processed into clean signals