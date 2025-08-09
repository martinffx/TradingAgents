# TradingAgents Troubleshooting Guide

This guide covers common issues and solutions when working with the TradingAgents framework.

## Common Issues

### Installation and Setup Issues

#### Missing Dependencies
**Problem:** `ModuleNotFoundError` when importing TradingAgents components.

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Or use mise for development setup
mise run setup
```

#### Python Version Compatibility
**Problem:** Type annotation errors or syntax issues.

**Solution:** TradingAgents requires Python 3.10+. Check your version:
```bash
python --version  # Should be 3.10 or higher

# If using conda
conda create -n tradingagents python=3.13
conda activate tradingagents
```

### Configuration Issues

#### API Key Not Found
**Problem:** `API key not found` errors when running analysis.

**Solutions:**
```bash
# Set required API keys
export OPENAI_API_KEY="your_openai_api_key"
export FINNHUB_API_KEY="your_finnhub_api_key"

# For other providers
export ANTHROPIC_API_KEY="your_anthropic_api_key" 
export GOOGLE_API_KEY="your_google_api_key"

# Verify keys are set
echo $OPENAI_API_KEY
```

#### Invalid LLM Provider
**Problem:** `Invalid LLM_PROVIDER` error.

**Solution:** Check valid providers:
```python
# Valid options: openai, anthropic, google, ollama, openrouter
export LLM_PROVIDER="openai"  # or anthropic, google, etc.
```

#### Directory Permission Issues
**Problem:** `Permission denied` when accessing data directories.

**Solutions:**
```bash
# Check directory permissions
ls -la ./results ./dataflows

# Create directories with proper permissions
mkdir -p results dataflows/data_cache
chmod 755 results dataflows/data_cache

# Or use environment variables for custom paths
export TRADINGAGENTS_RESULTS_DIR="./my_results"
export TRADINGAGENTS_DATA_DIR="./my_data"
```

### Data and API Issues

#### Rate Limiting
**Problem:** API rate limit exceeded errors.

**Solutions:**
```python
# Use cached data to reduce API calls
config = TradingAgentsConfig(online_tools=False)

# Reduce debate rounds to minimize LLM API calls
config = TradingAgentsConfig(
    max_debate_rounds=1,
    max_risk_discuss_rounds=1
)

# Use smaller/cheaper models
config = TradingAgentsConfig(
    deep_think_llm="gpt-4o-mini",
    quick_think_llm="gpt-4o-mini"
)
```

#### Empty or Invalid Data
**Problem:** Services return empty contexts or "LOW" data quality.

**Solutions:**
```python
# Check data quality in context metadata
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")
if context.metadata.get("data_quality") == "LOW":
    # Try updating data first
    service.update_market_data("AAPL", "2024-01-01", "2024-01-31")
    context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")

# Check for weekend/holiday dates
from datetime import datetime
date_obj = datetime.strptime("2024-01-15", "%Y-%m-%d")
if date_obj.weekday() >= 5:  # Saturday=5, Sunday=6
    print("Markets are closed on weekends")
```

#### Network/Connectivity Issues
**Problem:** Network timeouts or connection errors.

**Solutions:**
```python
# Increase timeout settings (implementation dependent)
# Use offline mode during network issues
config = TradingAgentsConfig(online_tools=False)

# Check network connectivity
import requests
try:
    requests.get("https://api.openai.com/v1/models", timeout=10)
    print("Network connectivity OK")
except requests.RequestException as e:
    print(f"Network issue: {e}")
```

### Performance Issues

#### Slow Analysis Performance
**Problem:** Trading analysis takes too long to complete.

**Optimizations:**
```python
# Use faster models for non-critical operations
config = TradingAgentsConfig(
    quick_think_llm="gpt-4o-mini",  # Fastest model for simple tasks
    deep_think_llm="gpt-4o",        # Balance of speed and quality
)

# Reduce debate complexity
config = TradingAgentsConfig(
    max_debate_rounds=1,
    max_risk_discuss_rounds=1
)

# Use cached data when possible
config = TradingAgentsConfig(online_tools=False)

# Reduce data range
config = TradingAgentsConfig(
    default_lookback_days=14,
    default_ta_lookback_days=14
)
```

#### High Memory Usage
**Problem:** Memory usage grows during analysis.

**Solutions:**
```python
# Clear memory between analyses
import gc
gc.collect()

# Limit vector memory size (if using memory features)
# Configure ChromaDB retention policies in graph initialization

# Process stocks one at a time instead of in batches
for symbol in ["AAPL", "GOOGL", "MSFT"]:
    result, decision = ta.propagate(symbol, "2024-01-15")
    # Process result immediately
    gc.collect()  # Clear memory between iterations
```

#### High API Costs
**Problem:** LLM API costs are higher than expected.

**Cost Optimization:**
```python
# Use smaller models
config = TradingAgentsConfig(
    deep_think_llm="gpt-4o-mini",   # Much cheaper than gpt-4
    quick_think_llm="gpt-4o-mini"
)

# Reduce agent interactions
config = TradingAgentsConfig(
    max_debate_rounds=1,  # Fewer back-and-forth discussions
    max_risk_discuss_rounds=1
)

# Use cached data to reduce context size
config = TradingAgentsConfig(online_tools=False)

# Monitor token usage
ta = TradingAgentsGraph(debug=True, config=config)  # Enable debug for token tracking
```

### Code Development Issues

#### Type Checking Errors
**Problem:** `pyright` or type checker complaints.

**Solutions:**
```bash
# Run type checker to see specific issues
mise run typecheck

# Common fixes:
# 1. Add type annotations
def get_data(symbol: str, date: str) -> dict[str, Any]:
    return {}

# 2. Use proper imports
from typing import Any, Optional
from datetime import datetime

# 3. Handle optional types
def process_data(data: dict[str, Any] | None) -> str:
    if data is None:
        return "No data"
    return str(data)
```

#### Import Errors
**Problem:** Cannot import TradingAgents modules.

**Solutions:**
```python
# Ensure you're in the correct directory
import os
print(os.getcwd())  # Should be /path/to/TradingAgents

# Add project root to Python path if needed
import sys
sys.path.append("/path/to/TradingAgents")

# Use absolute imports
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.config import TradingAgentsConfig
```

#### Test Failures
**Problem:** Tests fail when running `mise run test`.

**Solutions:**
```bash
# Run tests with verbose output
mise run test -v

# Run specific test file
uv run pytest tradingagents/domains/marketdata/market_data_service_test.py -v

# Run tests with debug output
uv run pytest -s  # Shows print statements

# Check test dependencies
pip install pytest pytest-cov responses  # Common test dependencies
```

### Data Quality Issues

#### Inconsistent Results
**Problem:** Same analysis produces different results.

**Causes and Solutions:**
```python
# 1. Non-deterministic LLM responses
config = TradingAgentsConfig(
    # Use temperature settings if your LLM provider supports it
    # This varies by provider
)

# 2. Real-time data changes
config = TradingAgentsConfig(online_tools=False)  # Use cached data for consistent results

# 3. Date/time sensitivity
# Always use consistent date formats and timezones
from datetime import datetime
date_str = datetime.now().strftime("%Y-%m-%d")
```

#### Missing Historical Data
**Problem:** No data available for specific dates or symbols.

**Solutions:**
```python
# Check if symbol exists and markets were open
def is_valid_trading_day(date_str: str) -> bool:
    from datetime import datetime
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Skip weekends
    if date_obj.weekday() >= 5:
        return False
    # Add holiday checking logic here
    return True

# Use broader date ranges
service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")  # Month range
# Instead of single day which might be missing
```

## Debugging Strategies

### Enable Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug in TradingAgentsGraph
ta = TradingAgentsGraph(debug=True, config=config)
```

### Check Configuration
```python
# Print current configuration
config = TradingAgentsConfig.from_env()
print("Current config:")
for key, value in config.to_dict().items():
    print(f"  {key}: {value}")
```

### Validate Data Quality
```python
# Always check context metadata
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")
print(f"Data quality: {context.metadata.get('data_quality', 'UNKNOWN')}")
print(f"Data source: {context.metadata.get('source', 'UNKNOWN')}")
print(f"Last updated: {context.metadata.get('last_updated', 'UNKNOWN')}")
```

### Test Individual Components
```python
# Test services individually
from tradingagents.domains.marketdata.market_data_service import MarketDataService

service = MarketDataService.build(config)
context = service.get_market_data_context("AAPL", "2024-01-01", "2024-01-31")
print(f"Service working: {len(context.price_data) > 0}")
```

### Monitor API Usage
```python
# Track API calls in debug mode
# Many providers show token usage in debug output
ta = TradingAgentsGraph(debug=True, config=config)
result, decision = ta.propagate("AAPL", "2024-01-15")
# Check debug output for token usage statistics
```

## Performance Monitoring

### Measure Execution Time
```python
import time
from tradingagents.graph.trading_graph import TradingAgentsGraph

start_time = time.time()
ta = TradingAgentsGraph(debug=True, config=config)
result, decision = ta.propagate("AAPL", "2024-01-15")
end_time = time.time()

print(f"Analysis completed in {end_time - start_time:.2f} seconds")
```

### Memory Monitoring
```python
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory before: {get_memory_usage():.1f} MB")
result, decision = ta.propagate("AAPL", "2024-01-15")
print(f"Memory after: {get_memory_usage():.1f} MB")
```

### Cost Tracking
```python
# Track API costs (varies by provider)
# Most LLM providers show token usage in their API responses
# Monitor your provider's dashboard for detailed cost tracking

# For OpenAI, costs can be estimated:
# gpt-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
# gpt-4: ~$30 per 1M input tokens, ~$60 per 1M output tokens
```

## Getting Help

### Check Logs
```bash
# Enable Python logging
export PYTHONPATH="."
export LOGLEVEL="DEBUG"
python -m cli.main
```

### Community Resources
- **GitHub Issues**: [TauricResearch/TradingAgents/issues](https://github.com/TauricResearch/TradingAgents/issues)
- **Discord**: [TradingResearch Community](https://discord.com/invite/hk9PGKShPK)
- **Documentation**: [API Reference](./API_REFERENCE.md) and [CLAUDE.md](./CLAUDE.md)

### Reporting Issues
When reporting issues, include:
1. Python version (`python --version`)
2. Operating system
3. Full error traceback
4. Configuration used (remove API keys)
5. Steps to reproduce
6. Expected vs actual behavior

### Emergency Recovery
If the system becomes completely unusable:
```bash
# Reset to clean state
rm -rf dataflows/data_cache/  # Clear cached data
rm -rf results/              # Clear result files

# Reinstall dependencies
pip uninstall tradingagents -y
pip install -r requirements.txt

# Use minimal configuration
export ONLINE_TOOLS=false
export MAX_DEBATE_ROUNDS=1
export DEEP_THINK_LLM="gpt-4o-mini"
export QUICK_THINK_LLM="gpt-4o-mini"
```