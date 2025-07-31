# AGENTS.md - TradingAgents Development Guide

## What TradingAgents Does

**TradingAgents** is a multi-agent LLM financial trading framework that simulates a real-world trading firm using specialized AI agents. The system analyzes stocks through collaborative decision-making, mirroring professional trading teams.

### Core Architecture
- Built on **LangGraph** with state-based workflows
- **TradingAgentsGraph** orchestrates the entire process
- Agents work sequentially and in parallel to analyze market conditions

### Agent Teams & Workflow
1. **Analyst Team**: Market, Social Media, News, and Fundamentals analysts gather data
2. **Research Team**: Bull/Bear researchers debate, Research Manager decides
3. **Trading Team**: Trader develops detailed trading plans
4. **Risk Management**: Risk analysts debate, Risk Manager makes final decision

### Data Sources
- Yahoo Finance, FinnHub API, Reddit, Google News, StockStats
- Supports both real-time online data and cached offline data for backtesting

### Decision Process
Sequential analysis → Structured debate → Managerial oversight → Risk assessment → Memory & learning

## Build/Test Commands

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
- **Run tests**: `mise run test` - Run tests with pytest (when available)

### Configuration
- **Environment Variables**: Create `.env` file with API keys (see `.env.example`)
- **Tool Configuration**: `.mise.toml` manages Python 3.13, uv, ruff, pyright
- **Code Quality**: `pyproject.toml` contains ruff and pyright configurations

## Configuration System

### Environment Variables
Create `.env` file with API keys (see `.env.example`):

#### Core LLM APIs (Choose One)
```bash
# For OpenAI (default)
export OPENAI_API_KEY="your_openai_api_key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# For Google Gemini
export GOOGLE_API_KEY="your_google_api_key"
```

#### Data Sources (Optional)
```bash
# For financial data
export FINNHUB_API_KEY="your_finnhub_api_key"

# For Reddit data
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="your_app_name"
```

### Configuration Management
- **Config Class**: `TradingAgentsConfig` in `tradingagents/config.py` handles all configuration
- Use `TradingAgentsConfig.from_env()` for environment-based configuration
- Key settings: `max_debate_rounds`, `llm_provider`, `online_tools`
- Results are saved to `results_dir/{ticker}/{date}/` with structured reports

### Configuration Examples

#### Anthropic Setup
```python
config = TradingAgentsConfig.from_env()
config.llm_provider = "anthropic"
config.deep_think_llm = "claude-3-5-sonnet-20241022"
config.quick_think_llm = "claude-3-5-haiku-20241022"
```

#### Google Gemini Setup
```python
config.llm_provider = "google"
config.deep_think_llm = "gemini-2.0-flash"
config.quick_think_llm = "gemini-2.0-flash"
```

#### Data Mode Configuration
- `config.online_tools = True` - Real-time data (requires API keys)
- `config.online_tools = False` - Cached data (faster, historical only)

## Code Style Guidelines

- **Imports**: Standard library first, third-party, then local imports (langchain, tradingagents modules)
- **Formatting**: Auto-formatted with ruff (`mise run format`)
- **Linting**: Code quality checked with ruff (`mise run lint`)
- **Type Checking**: Static analysis with pyright (`mise run typecheck`)
- **Functions**: Snake_case naming (e.g., `fundamentals_analyst_node`, `create_fundamentals_analyst`)
- **Classes**: PascalCase (e.g., `TradingAgentsGraph`, `MessageBuffer`)
- **Variables**: Snake_case (e.g., `current_date`, `company_of_interest`)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_CONFIG`)

## Project Structure

- **Main entry**: `main.py` for package usage, `cli/main.py` for CLI
- **Core logic**: `tradingagents/` package with agents, dataflows, graph modules
- **Configuration**: `tradingagents/config.py` for LLM and system settings
- **CLI interface**: `cli/` directory with rich-based terminal UI
- **Tool Management**: `.mise.toml` for development tool configuration
- **Dependencies**: `pyproject.toml` for project dependencies and tool settings

## Key Patterns

- **Agent creation**: Factory functions that return node functions (e.g., `create_fundamentals_analyst`)
- **State management**: Dictionary-based state passed between graph nodes
- **Tool integration**: LangChain tools bound to LLMs via `llm.bind_tools(tools)`
- **Configuration**: Use `TradingAgentsConfig.from_env()` for environment-based configuration
- **Debate-Driven Decision Making**: Critical decisions emerge from structured agent debates
- **Memory-Augmented Learning**: Agents learn from past similar situations using vector similarity
- **Dual-Mode Data Access**: Support for both live API calls and pre-processed cached data
- **Factory Pattern**: Agent creation via factory functions for flexible configuration
- **Signal Processing**: Final trading decisions processed into clean BUY/SELL/HOLD signals

## Development Guidelines

### Working with Agents
- Each agent has its own memory instance in `FinancialSituationMemory`
- Agents use the unified `Toolkit` for data access
- Agent state is passed sequentially through the workflow
- Configuration affects debate rounds, LLM selection, and data sources

### Working with Data Sources
- All data utilities follow consistent date range patterns: `curr_date + look_back_days`
- Interface functions return markdown-formatted strings for LLM consumption
- Check `online_tools` config flag to determine live vs cached data usage
- Data caching happens in `data_cache_dir` for online mode

### CLI Development
- CLI uses Rich for terminal UI with live updating displays
- Agent progress tracking through `MessageBuffer` class
- Questionnaire-driven configuration collection
- Real-time streaming of analysis results

### File Structure Context
- **`cli/`**: Interactive command-line interface
- **`tradingagents/agents/`**: All agent implementations
- **`tradingagents/dataflows/`**: Data source integrations
- **`tradingagents/graph/`**: LangGraph workflow orchestration
- **`tradingagents/config.py`**: Configuration management
- **`main.py`**: Direct Python usage example
- **`CLAUDE.md`**: Guidance for Claude Code development