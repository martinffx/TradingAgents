# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

This project uses [mise](https://mise.jdx.dev/) for tool and task management. All development tasks are managed through mise.

### Initial Setup
- **First-time setup**: `mise run setup` - Install tools and dependencies
- **Install tools only**: `mise install` - Install Python, uv, ruff, pyright
- **Install dependencies**: `mise run install` - Install project dependencies with uv

### Development Workflow
- **CLI Application**: `mise run dev` - Interactive CLI for running trading analysis
- **Direct Python Usage**: `mise run run` - Run main.py programmatically
- **Format code**: `mise run format` - Auto-format with ruff
- **Lint code**: `mise run lint` - Check code quality with ruff
- **Type checking**: `mise run typecheck` - Run pyright type checker
- **Fix lint issues**: `mise run fix` - Auto-fix linting issues
- **Run all checks**: `mise run all` - Format, lint, and typecheck
- **Clean artifacts**: `mise run clean` - Remove cache and build files

### Testing

#### Running Tests
- **Run all tests**: `mise run test` - Run tests with pytest
- **Run specific test file**: `uv run pytest test_social_media_service.py` - Run individual test file
- **Verbose output**: `uv run pytest -v` - Run tests with detailed output
- **Run with output**: `uv run pytest -s` - Show print statements and debug output
- **Test coverage**: `uv run pytest --cov=tradingagents` - Run tests with coverage report

#### Test Development (TDD Approach)
This project follows **Test-Driven Development (TDD)** for service layer development:

1. **Write test first**: Create `test_{service_name}_service.py` with comprehensive test cases
2. **Run test (should fail)**: Verify test fails with appropriate error messages
3. **Implement minimum code**: Write just enough code to make the test pass
4. **Refactor**: Improve code while keeping tests passing
5. **Repeat**: Add more test cases and implement additional functionality

#### Test Structure and Conventions
- **Test files**: Named `test_{component}_service.py` and placed next to source code (not in separate tests/ directory)
- **Test functions**: Named `test_{functionality}()` and should not return values (use `assert` statements)
- **Mock clients**: Create mock implementations of BaseClient for testing services
- **Real repositories**: Use actual repository implementations (don't mock the repository layer)
- **Test data**: Use realistic mock data that matches expected API responses
- **Date handling**: Use fixed dates (e.g., `datetime(2024, 1, 2)`) in mocks for predictable filtering

#### Service Testing Pattern
Example test structure for services:
```python
def test_online_mode_with_mock_client():
    """Test service in online mode with mock client."""
    mock_client = MockServiceClient()
    real_repo = ServiceRepository("test_data")
    
    service = ServiceClass(
        client=mock_client,
        repository=real_repo,
        online_mode=True
    )
    
    context = service.get_context("TEST", "2024-01-01", "2024-01-05")
    
    # Validate structure
    assert isinstance(context, ContextModel)
    assert context.symbol == "TEST"
    assert len(context.data) > 0
    
    # Test JSON serialization
    json_output = context.model_dump_json()
    assert len(json_output) > 0
```

#### Mock Client Guidelines
- **Extend BaseClient**: All mock clients must implement the abstract `get_data()` method
- **Realistic data**: Return data structures that match actual API responses
- **Date consistency**: Use fixed dates that work with test date ranges
- **Error simulation**: Create broken clients for testing error handling paths
- **Multiple scenarios**: Provide different data for different test cases

### Configuration
- **Environment Variables**: Create `.env` file with API keys (see `.env.example`)
- **Config Class**: `TradingAgentsConfig` in `tradingagents/config.py` handles all configuration
- **Tool Configuration**: `.mise.toml` manages Python 3.13, uv, ruff, pyright
- **Code Quality**: `pyproject.toml` contains ruff and pyright configurations

#### Required Environment Variables

##### Core LLM APIs (Choose One)
```bash
# For OpenAI (default)
export OPENAI_API_KEY="your_openai_api_key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# For Google Gemini
export GOOGLE_API_KEY="your_google_api_key"
```

##### Data Sources (Optional)
```bash
# For financial data
export FINNHUB_API_KEY="your_finnhub_api_key"

# For Reddit data
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="your_app_name"
```

## High-Level Architecture

### Multi-Agent Trading Framework
TradingAgents implements a sophisticated multi-agent system that mirrors real-world trading firms with specialized roles and structured workflows.

### Core Architecture Components

#### 1. **Agent Teams** (Sequential Workflow)
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

#### 2. **Data Layer** (`tradingagents/services/` + `tradingagents/dataflows/`)
**New Service-Based Architecture** (Current):
- **Service Layer**: `MarketDataService`, `NewsService`, `SocialMediaService`, `FundamentalDataService`, `InsiderDataService`, `OpenAIDataService`
  - Orchestrate between clients and repositories with local-first data strategy
  - Provide structured JSON contexts via Pydantic models
  - Support `force_refresh=False` parameter for explicit data refreshing
  - Automatically check local cache first, fetch from APIs only when data is missing
- **Client Layer**: Live API integrations - `YFinanceClient`, `FinnhubClient`, `RedditClient`, `GoogleNewsClient`, `SimFinClient`
- **Repository Layer**: Smart cached data storage with gap detection - `MarketDataRepository`, `NewsRepository`, `SocialRepository`
- **Context Models**: Pydantic models for structured JSON data - `MarketDataContext`, `NewsContext`, `SocialContext`, `FundamentalContext`
- **Toolkit Integration**: `ServiceToolkit` provides 100% backward-compatible interface returning JSON contexts

**Legacy Data Integration System** (Being phased out):
- **Yahoo Finance** (`yfin_utils.py`): Stock prices, financials, analyst recommendations
- **Finnhub** (`finnhub_utils.py`): News, insider trading, SEC filings
- **Reddit** (`reddit_utils.py`): Social sentiment from curated subreddits
- **Google News** (`googlenews_utils.py`): Web-scraped news with retry logic
- **SimFin**: Balance sheets, cash flow, income statements
- **StockStats** (`stockstats_utils.py`): Technical indicators (MACD, RSI, etc.)
- **Interface Layer** (`interface.py`): Standardized agent-facing APIs with markdown formatting

#### 3. **Graph Orchestration** (`tradingagents/graph/`)
LangGraph-based workflow management:

- **TradingAgentsGraph**: Main orchestrator class
- **State Management**: `AgentState`, `InvestDebateState`, `RiskDebateState` track workflow progress
- **Conditional Logic**: Dynamic routing based on tool usage and debate completion
- **Memory System**: ChromaDB-based vector memory for learning from past decisions

#### 4. **Configuration System**
- **TradingAgentsConfig**: Centralized configuration with environment variable support
- **Multi-LLM Support**: OpenAI, Anthropic, Google, Ollama, OpenRouter
- **Data Modes**: Online (live APIs) vs offline (cached data)

### Key Design Patterns

1. **Debate-Driven Decision Making**: Critical decisions emerge from structured agent debates
2. **Memory-Augmented Learning**: Agents learn from past similar situations using vector similarity
3. **Local-First Data Strategy**: Automatically check local cache first, fetch from APIs only when needed
4. **Structured JSON Contexts**: Replace error-prone string parsing with rich Pydantic models
5. **Factory Pattern**: Agent creation via factory functions for flexible configuration
6. **Signal Processing**: Final trading decisions processed into clean BUY/SELL/HOLD signals
7. **Quality-Aware Data**: All contexts include quality metadata to help agents make better decisions

### Code Style Guidelines

#### General Style
- **Functions**: Snake_case naming (e.g., `fundamentals_analyst_node`, `create_fundamentals_analyst`)
- **Classes**: PascalCase (e.g., `TradingAgentsGraph`, `MessageBuffer`)
- **Variables**: Snake_case (e.g., `current_date`, `company_of_interest`)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_CONFIG`)
- **Imports**: Standard library first, third-party, then local imports (langchain, tradingagents modules)

#### Ruff Formatting & Linting Rules
**Formatting** (`mise run format`):
- **Line length**: 88 characters maximum
- **Quote style**: Double quotes (`"string"`)
- **Indentation**: 4 spaces (no tabs)
- **Trailing commas**: Preserved for multi-line structures
- **Line endings**: Auto-detected based on platform

**Linting** (`mise run lint`):
- **Selected rules**:
  - `E`, `W`: pycodestyle errors and warnings
  - `F`: pyflakes (undefined names, unused imports)
  - `I`: isort (import sorting)
  - `B`: flake8-bugbear (common bugs)
  - `C4`: flake8-comprehensions (list/dict comprehensions)
  - `UP`: pyupgrade (Python syntax modernization)
  - `ARG`: flake8-unused-arguments
  - `SIM`: flake8-simplify (code simplification)
  - `TCH`: flake8-type-checking (type annotation imports)

- **Ignored rules**:
  - `E501`: Line too long (handled by formatter)
  - `B008`: Function calls in argument defaults (allowed for LangChain)
  - `C901`: Complex functions (legacy code tolerance)
  - `ARG001`, `ARG002`: Unused arguments (common in callbacks)

- **Import sorting**: `tradingagents` and `cli` treated as first-party modules

#### Pyright Type Checking Rules
**Configuration** (`mise run typecheck`):
- **Tool**: pyright 1.1.390+ with standard type checking mode
- **Python version**: 3.10+ (configured for compatibility with modern syntax)
- **Coverage**: Includes `tradingagents/`, `cli/`, and `main.py`
- **Exclusions**: `__pycache__`, `node_modules`, `.venv`, `venv`, `build`, `dist`

**Configured Type Checking Rules**:
- `reportMissingImports = true` - Catch undefined module imports
- `reportMissingTypeStubs = false` - Allow libraries without type stubs  
- `reportGeneralTypeIssues = true` - General type inconsistencies
- `reportOptionalMemberAccess = true` - Unsafe access to optional values
- `reportOptionalCall = true` - Calling potentially None values
- `reportOptionalIterable = true` - Iterating over potentially None values
- `reportOptionalContextManager = true` - Using None in with statements
- `reportOptionalOperand = true` - Operations on potentially None values
- `reportTypedDictNotRequiredAccess = false` - Flexible TypedDict access
- `reportPrivateImportUsage = false` - Allow importing private modules
- `reportUnknownParameterType = false` - Allow untyped parameters
- `reportUnknownArgumentType = false` - Allow untyped arguments
- `reportUnknownLambdaType = false` - Allow untyped lambdas
- `reportUnknownVariableType = false` - Allow untyped variables
- `reportUnknownMemberType = false` - Allow untyped attributes

**Type Annotation Guidelines**:
- Use modern Python 3.10+ union syntax: `str | None` instead of `Optional[str]`
- Use built-in generics: `list[str]` instead of `List[str]`
- Use `dict[str, Any]` for flexible dictionaries
- Import `from typing import Any` for untyped data structures
- Prefer explicit return types on public functions
- Use `# type: ignore` sparingly with explanatory comments

### Development Guidelines

#### Working with Agents

**Current Approach** (JSON Contexts):
- Import `ServiceToolkit` from `tradingagents.agents.utils.service_toolkit` 
- Use `context_helpers` for parsing structured JSON data (MarketDataParser, NewsParser, etc.)
- All toolkit methods return structured JSON instead of markdown strings
- Check data quality with `is_high_quality_data()` before analysis
- Use `extract_latest_price()`, `extract_sentiment_score()` for quick data extraction

**Legacy Approach** (Being phased out):
- Agents use the unified `Toolkit` for markdown-formatted data access

#### Working with Data Sources

**Current Service-Based Approach**:
- **Local-First Strategy**: Services automatically check local cache first, only fetch from APIs when data is missing
- **Force Refresh**: Use `force_refresh=True` parameter to bypass local data and get fresh API data
- **Structured Contexts**: Services return Pydantic models (`MarketDataContext`, `NewsContext`, etc.) with rich metadata
- **Quality Awareness**: All contexts include data_quality (HIGH/MEDIUM/LOW) and data_source information
- **JSON Serialization**: All context models support `.model_dump_json()` for agent consumption
- **Smart Caching**: Repositories detect data gaps and fetch only missing periods
- **Cost Efficient**: Minimizes expensive API calls through intelligent local-first logic

**Service Configuration**:
```python
# Services use dependency injection
service = MarketDataService(
    client=YFinanceClient(),
    repository=MarketDataRepository("cache_dir"),
    online_mode=True  # Enable API fetching when local data insufficient
)

# Get data with local-first strategy
context = service.get_context("AAPL", "2024-01-01", "2024-01-31")

# Force fresh data when needed
fresh_context = service.get_context("AAPL", "2024-01-01", "2024-01-31", force_refresh=True)
```

**Legacy Approach** (Being phased out):
- Interface functions return markdown-formatted strings for LLM consumption
- All data utilities follow consistent date range patterns: `curr_date + look_back_days`
- Check `online_tools` config flag to determine live vs cached data usage
- Data caching happens in `data_cache_dir` for online mode

#### Configuration Management
- Use `TradingAgentsConfig.from_env()` for environment-based configuration
- Key settings: `max_debate_rounds`, `llm_provider`, `online_tools`
- Results are saved to `results_dir/{ticker}/{date}/` with structured reports

#### CLI Development
- CLI uses Rich for terminal UI with live updating displays
- Agent progress tracking through `MessageBuffer` class
- Questionnaire-driven configuration collection
- Real-time streaming of analysis results

### File Structure Context
- **`cli/`**: Interactive command-line interface
- **`tradingagents/agents/`**: All agent implementations
  - **`utils/service_toolkit.py`**: New ServiceToolkit with JSON contexts (100% backward compatible)
  - **`utils/context_helpers.py`**: Helper functions for parsing structured JSON data
  - **`utils/agent_utils.py`**: Legacy Toolkit (being phased out)
- **`tradingagents/services/`**: Service layer with local-first data strategy
- **`tradingagents/dataflows/`**: Legacy data source integrations (being phased out)
- **`tradingagents/graph/`**: LangGraph workflow orchestration
- **`tradingagents/config.py`**: Configuration management
- **`main.py`**: Direct Python usage example
- **`AGENTS.md`**: Detailed agent documentation
- **`examples/agent_json_migration.py`**: Migration guide from markdown to JSON contexts