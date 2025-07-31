from tradingagents.config import TradingAgentsConfig
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Create a custom config using Anthropic
config = TradingAgentsConfig(
    llm_provider="anthropic",
    deep_think_llm="claude-3-5-sonnet-20241022",
    quick_think_llm="claude-3-5-haiku-20241022",
    max_debate_rounds=1,
    online_tools=True,
)

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
