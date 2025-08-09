# Agent Development Guide

This guide covers how to develop, modify, and extend agents in the TradingAgents framework.

## Agent Architecture Overview

The TradingAgents framework uses a multi-agent system where each agent has specific responsibilities in the trading decision workflow.

### Agent Categories

1. **Analyst Team** (`tradingagents/agents/analysts/`): Data analysis and market intelligence
2. **Research Team** (`tradingagents/agents/researchers/`): Investment debate and recommendation synthesis
3. **Trading Team** (`tradingagents/agents/trader/`): Trading decision execution
4. **Risk Management** (`tradingagents/agents/risk_mgmt/`): Risk assessment and portfolio management

### Agent Implementation Pattern

All agents follow a consistent implementation pattern:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from tradingagents.agents.libs.agent_toolkit import AgentToolkit

def create_market_analyst(toolkit: AgentToolkit, config: dict) -> dict:
    """Factory function to create a market analyst agent."""
    
    # Define agent's role and responsibilities
    system_prompt = """You are a Market Analyst specializing in technical analysis.
    
    Your responsibilities:
    - Analyze price trends and trading patterns
    - Calculate and interpret technical indicators
    - Identify support and resistance levels
    - Assess market momentum and volatility
    
    Use the available tools to gather market data and provide actionable insights."""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", "{input}")
    ])
    
    # Create the agent with tools
    agent = create_agent(
        llm=get_llm(config.get("quick_think_llm", "gpt-4o-mini")),
        tools=[
            toolkit.get_market_data,
            toolkit.get_ta_report,
        ],
        prompt=prompt
    )
    
    return agent
```

## Adding New Agents

### Step 1: Define Agent Role

Create a new file for your agent (e.g., `custom_analyst.py`):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from tradingagents.agents.libs.agent_toolkit import AgentToolkit
from tradingagents.agents.libs.agent_base import create_agent, get_llm

def create_custom_analyst(toolkit: AgentToolkit, config: dict) -> dict:
    """Create a custom analyst with specific expertise."""
    
    system_prompt = """You are a Custom Market Analyst specializing in [specific domain].
    
    Your responsibilities:
    - [List specific responsibilities]
    - [Define analysis focus]
    - [Specify output format]
    
    Always provide:
    1. Clear analysis summary
    2. Key findings with evidence
    3. Confidence level in your assessment
    4. Risk factors to consider
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", "Analyze {symbol} for date {date}. Context: {context}")
    ])
    
    # Select appropriate tools for this agent
    agent_tools = [
        toolkit.get_market_data,
        toolkit.get_news,
        toolkit.get_socialmedia_stock_info,
        # Add other relevant tools
    ]
    
    agent = create_agent(
        llm=get_llm(config.get("deep_think_llm", "o4-mini")),
        tools=agent_tools,
        prompt=prompt
    )
    
    return agent
```

### Step 2: Create Agent Node Function

Create a node function for LangGraph integration:

```python
from langchain_core.messages import HumanMessage
from tradingagents.graph.state_models import AgentState

def custom_analyst_node(state: AgentState, config: dict) -> dict:
    """Node function for custom analyst in the trading workflow."""
    
    # Extract relevant information from state
    symbol = state.get("symbol", "")
    date = state.get("date", "")
    context = state.get("context", {})
    
    # Get the agent instance
    toolkit = state.get("toolkit")
    agent = create_custom_analyst(toolkit, config)
    
    # Prepare input message
    input_msg = f"""Analyze {symbol} for {date}.
    
    Previous analysis context:
    {context}
    
    Provide detailed analysis focusing on [specific aspects].
    """
    
    # Execute agent
    response = agent.invoke({
        "input": input_msg,
        "symbol": symbol,
        "date": date,
        "context": context
    })
    
    # Extract analysis from response
    analysis = response.get("output", "")
    
    # Update state with results
    return {
        "custom_analysis": analysis,
        "agents_completed": state.get("agents_completed", []) + ["custom_analyst"]
    }
```

### Step 3: Integrate into Workflow

Add your agent to the trading graph:

```python
# In tradingagents/graph/trading_graph.py

from tradingagents.agents.analysts.custom_analyst import custom_analyst_node

class TradingAgentsGraph:
    def _build_graph(self):
        # ... existing code ...
        
        # Add your custom analyst node
        graph.add_node("custom_analyst", custom_analyst_node)
        
        # Define connections in the workflow
        graph.add_edge("market_analyst", "custom_analyst")
        graph.add_edge("custom_analyst", "research_team")
        
        # ... rest of graph construction ...
```

## Extending Existing Agents

### Modifying Agent Prompts

To customize an existing agent's behavior, modify its system prompt:

```python
def create_enhanced_fundamentals_analyst(toolkit: AgentToolkit, config: dict) -> dict:
    """Enhanced version of fundamentals analyst with additional capabilities."""
    
    enhanced_prompt = """You are a Senior Fundamentals Analyst with expertise in financial modeling.
    
    Your enhanced responsibilities:
    - Analyze financial statements with deep ratio analysis
    - Build DCF models when appropriate
    - Compare metrics against industry benchmarks
    - Assess management quality and corporate governance
    - Evaluate ESG factors and sustainability metrics
    
    Analysis Framework:
    1. Financial Health Assessment (40%)
    2. Valuation Analysis (30%)
    3. Growth Potential (20%)
    4. Risk Assessment (10%)
    
    Always provide quantitative metrics with qualitative insights.
    """
    
    # ... rest of implementation
```

### Adding New Tools

Create custom tools for specialized data sources:

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def get_industry_comparison(
    symbol: Annotated[str, "Stock symbol to analyze"],
    metrics: Annotated[list[str], "List of metrics to compare"]
) -> str:
    """Compare stock metrics against industry averages."""
    
    # Implementation to fetch industry data
    # This could integrate with additional APIs or databases
    
    industry_data = fetch_industry_metrics(symbol, metrics)
    stock_data = fetch_stock_metrics(symbol, metrics)
    
    comparison = compare_metrics(stock_data, industry_data)
    
    return format_comparison_report(comparison)

# Add to agent toolkit
def create_enhanced_toolkit(config: dict) -> AgentToolkit:
    """Create toolkit with additional custom tools."""
    
    toolkit = AgentToolkit.build(config)
    
    # Add custom tools
    toolkit.add_tool(get_industry_comparison)
    
    return toolkit
```

## Agent Communication Patterns

### Structured Information Passing

Agents communicate through structured state objects:

```python
@dataclass
class AnalysisState:
    symbol: str
    date: str
    market_analysis: str | None = None
    fundamental_analysis: str | None = None
    sentiment_analysis: str | None = None
    risk_assessment: str | None = None
    final_recommendation: str | None = None
    confidence_score: float | None = None

def analyst_coordination_node(state: AnalysisState, config: dict) -> dict:
    """Coordinate multiple analysts and synthesize results."""
    
    analyses = {
        "market": state.market_analysis,
        "fundamental": state.fundamental_analysis,
        "sentiment": state.sentiment_analysis
    }
    
    # Synthesis logic
    synthesized_analysis = synthesize_analyses(analyses)
    
    return {
        "synthesized_analysis": synthesized_analysis,
        "confidence_score": calculate_confidence(analyses)
    }
```

### Debate Mechanisms

Implement structured debates between agents:

```python
def investment_debate_node(state: AgentState, config: dict) -> dict:
    """Facilitate debate between bull and bear researchers."""
    
    max_rounds = config.get("max_debate_rounds", 3)
    current_round = state.get("debate_round", 0)
    
    if current_round >= max_rounds:
        # End debate and synthesize
        return finalize_debate(state)
    
    if current_round % 2 == 0:
        # Bull researcher's turn
        response = bull_researcher.invoke(state)
        return {
            "bull_arguments": state.get("bull_arguments", []) + [response],
            "debate_round": current_round + 1,
            "current_speaker": "bear"
        }
    else:
        # Bear researcher's turn
        response = bear_researcher.invoke(state)
        return {
            "bear_arguments": state.get("bear_arguments", []) + [response],
            "debate_round": current_round + 1,
            "current_speaker": "bull"
        }
```

## Agent Testing

### Unit Testing Agents

Create comprehensive tests for agent behavior:

```python
import pytest
from unittest.mock import Mock, patch
from tradingagents.agents.analysts.custom_analyst import create_custom_analyst

def test_custom_analyst_creation():
    """Test agent creation with mock toolkit."""
    mock_toolkit = Mock()
    config = {"deep_think_llm": "gpt-4o-mini"}
    
    agent = create_custom_analyst(mock_toolkit, config)
    
    assert agent is not None
    assert hasattr(agent, 'invoke')

def test_custom_analyst_analysis():
    """Test agent analysis with mock data."""
    mock_toolkit = Mock()
    
    # Configure mock responses
    mock_toolkit.get_market_data.return_value = "Mock market data"
    mock_toolkit.get_news.return_value = "Mock news data"
    
    agent = create_custom_analyst(mock_toolkit, {})
    
    with patch('tradingagents.agents.libs.agent_base.get_llm') as mock_llm:
        mock_llm.return_value.invoke.return_value = {"output": "Test analysis"}
        
        result = agent.invoke({
            "input": "Analyze AAPL",
            "symbol": "AAPL",
            "date": "2024-01-15"
        })
        
        assert "output" in result
        assert result["output"] == "Test analysis"

def test_agent_node_integration():
    """Test agent node function."""
    state = {
        "symbol": "AAPL",
        "date": "2024-01-15",
        "toolkit": Mock(),
        "agents_completed": []
    }
    
    result = custom_analyst_node(state, {})
    
    assert "custom_analysis" in result
    assert "custom_analyst" in result["agents_completed"]
```

### Integration Testing

Test agent interactions within the workflow:

```python
def test_agent_workflow_integration():
    """Test agent integration in trading workflow."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.config import TradingAgentsConfig
    
    config = TradingAgentsConfig(
        online_tools=False,  # Use cached data for testing
        max_debate_rounds=1
    )
    
    graph = TradingAgentsGraph(debug=True, config=config)
    
    # Test with known stock and date
    result, decision = graph.propagate("AAPL", "2024-01-15")
    
    assert result is not None
    assert decision in ["BUY", "SELL", "HOLD"]
    assert "custom_analysis" in result  # If your agent was integrated
```

## Agent Performance Optimization

### Model Selection Strategy

Choose appropriate models for different agent types:

```python
def get_optimal_model(agent_type: str, config: dict) -> str:
    """Select optimal model based on agent requirements."""
    
    model_mapping = {
        # Fast models for data retrieval and formatting
        "data_fetchers": config.get("quick_think_llm", "gpt-4o-mini"),
        
        # Powerful models for complex analysis
        "analysts": config.get("deep_think_llm", "o4-mini"),
        
        # Balanced models for decision making
        "traders": config.get("deep_think_llm", "o4-mini"),
        
        # Conservative models for risk assessment
        "risk_managers": config.get("deep_think_llm", "o4-mini")
    }
    
    return model_mapping.get(agent_type, config.get("quick_think_llm"))
```

### Caching Agent Responses

Implement caching for expensive agent operations:

```python
from functools import lru_cache
import hashlib

def cache_key(symbol: str, date: str, analysis_type: str) -> str:
    """Generate cache key for agent analysis."""
    return hashlib.md5(f"{symbol}_{date}_{analysis_type}".encode()).hexdigest()

@lru_cache(maxsize=100)
def cached_analysis(cache_key: str, symbol: str, date: str) -> str:
    """Cache expensive analysis operations."""
    # This would be called by agents that perform expensive operations
    pass

def create_cached_agent(base_agent, cache_size: int = 100):
    """Wrap agent with caching functionality."""
    
    @lru_cache(maxsize=cache_size)
    def cached_invoke(input_hash: str, **kwargs):
        return base_agent.invoke(kwargs)
    
    def invoke(inputs: dict):
        # Create hash of inputs for cache key
        input_str = str(sorted(inputs.items()))
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        return cached_invoke(input_hash, **inputs)
    
    base_agent.invoke = invoke
    return base_agent
```

### Parallel Agent Execution

Implement parallel execution for independent agents:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_analysis_node(state: AgentState, config: dict) -> dict:
    """Execute multiple analysts in parallel."""
    
    symbol = state.get("symbol")
    date = state.get("date")
    toolkit = state.get("toolkit")
    
    # Create analysts
    market_analyst = create_market_analyst(toolkit, config)
    fundamentals_analyst = create_fundamentals_analyst(toolkit, config)
    sentiment_analyst = create_sentiment_analyst(toolkit, config)
    
    # Define analysis tasks
    async def run_analyst(analyst, input_data):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, analyst.invoke, input_data)
    
    # Execute in parallel
    tasks = [
        run_analyst(market_analyst, {"symbol": symbol, "date": date}),
        run_analyst(fundamentals_analyst, {"symbol": symbol, "date": date}),
        run_analyst(sentiment_analyst, {"symbol": symbol, "date": date})
    ]
    
    results = await asyncio.gather(*tasks)
    
    return {
        "market_analysis": results[0]["output"],
        "fundamental_analysis": results[1]["output"],
        "sentiment_analysis": results[2]["output"]
    }
```

## Best Practices

### Agent Design Principles

1. **Single Responsibility**: Each agent should have one clear purpose
2. **Clear Communication**: Use structured inputs/outputs
3. **Error Handling**: Gracefully handle missing data or API failures
4. **Testability**: Design agents to be easily testable with mocks
5. **Performance**: Consider token usage and API costs

### Code Organization

```
tradingagents/agents/
├── analysts/
│   ├── __init__.py
│   ├── market_analyst.py
│   ├── fundamentals_analyst.py
│   ├── custom_analyst.py
│   └── analyst_base.py       # Common analyst functionality
├── researchers/
│   ├── __init__.py
│   ├── bull_researcher.py
│   ├── bear_researcher.py
│   └── researcher_base.py
├── libs/
│   ├── agent_toolkit.py      # Anti-corruption layer
│   ├── agent_base.py        # Common agent utilities
│   └── context_helpers.py   # Helper functions
└── tests/
    ├── test_analysts.py
    ├── test_researchers.py
    └── test_integration.py
```

### Error Handling

Implement robust error handling in agents:

```python
def safe_agent_invoke(agent, inputs: dict, default_response: str = "Analysis unavailable") -> str:
    """Safely invoke agent with error handling."""
    
    try:
        result = agent.invoke(inputs)
        return result.get("output", default_response)
    
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return f"{default_response}. Error: {str(e)}"

def create_resilient_analyst(toolkit: AgentToolkit, config: dict) -> dict:
    """Create analyst with built-in resilience."""
    
    base_agent = create_custom_analyst(toolkit, config)
    
    def resilient_invoke(inputs: dict):
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return base_agent.invoke(inputs)
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    return {
                        "output": "Analysis failed after multiple attempts",
                        "error": str(e),
                        "confidence": 0.0
                    }
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
        
    base_agent.invoke = resilient_invoke
    return base_agent
```

This guide provides a comprehensive foundation for developing and extending agents in the TradingAgents framework. Follow these patterns and best practices to create robust, testable, and performant agents.