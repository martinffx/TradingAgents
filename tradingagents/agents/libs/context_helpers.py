"""
Helper functions for agents to work with JSON contexts from the new ServiceToolkit.
Provides utilities to parse and extract data from structured contexts.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextParser:
    """Helper class to parse and extract data from JSON contexts."""

    @staticmethod
    def parse_context(context_json: str) -> dict[str, Any]:
        """
        Parse JSON context string into dictionary.

        Args:
            context_json: JSON string from toolkit method

        Returns:
            Dictionary representation of the context
        """
        try:
            return json.loads(context_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON context: {e}")
            return {}

    @staticmethod
    def get_data_quality(context: dict[str, Any]) -> str:
        """Extract data quality from context metadata."""
        return context.get("metadata", {}).get("data_quality", "UNKNOWN")

    @staticmethod
    def get_data_source(context: dict[str, Any]) -> str:
        """Extract data source from context metadata."""
        return context.get("metadata", {}).get("data_source", "unknown")

    @staticmethod
    def is_high_quality(context: dict[str, Any]) -> bool:
        """Check if context has high quality data."""
        return ContextParser.get_data_quality(context) == "HIGH"

    @staticmethod
    def is_fresh_data(context: dict[str, Any]) -> bool:
        """Check if context contains fresh (non-cached) data."""
        source = ContextParser.get_data_source(context)
        return source in ["live_api", "live_api_refresh"]


class MarketDataParser(ContextParser):
    """Parser for MarketDataContext objects."""

    @staticmethod
    def get_price_data(context: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract price data from market context."""
        return context.get("price_data", [])

    @staticmethod
    def get_latest_price(context: dict[str, Any]) -> float | None:
        """Get the most recent closing price."""
        price_data = MarketDataParser.get_price_data(context)
        if price_data:
            return price_data[-1].get("Close")
        return None

    @staticmethod
    def get_technical_indicators(context: dict[str, Any]) -> dict[str, Any]:
        """Extract technical indicators from context."""
        return context.get("technical_indicators", {})

    @staticmethod
    def get_indicator_value(context: dict[str, Any], indicator: str) -> float | None:
        """Get the latest value for a specific technical indicator."""
        indicators = MarketDataParser.get_technical_indicators(context)
        indicator_data = indicators.get(indicator, {})

        if isinstance(indicator_data, dict) and "values" in indicator_data:
            values = indicator_data["values"]
            if values:
                return values[-1]  # Get latest value
        return None

    @staticmethod
    def format_price_summary(context: dict[str, Any]) -> str:
        """Create a formatted summary of price data for agents."""
        symbol = context.get("symbol", "UNKNOWN")
        period = context.get("period", {})
        price_data = MarketDataParser.get_price_data(context)

        if not price_data:
            return f"No price data available for {symbol}"

        latest = price_data[-1]
        first = price_data[0]

        latest_price = latest.get("Close", 0)
        start_price = first.get("Close", 0)
        change = latest_price - start_price
        change_pct = (change / start_price * 100) if start_price else 0

        summary = f"""
Market Data Summary for {symbol}:
- Period: {period.get("start")} to {period.get("end")}
- Latest Price: ${latest_price:.2f}
- Period Change: ${change:.2f} ({change_pct:+.2f}%)
- Data Points: {len(price_data)}
- Data Quality: {MarketDataParser.get_data_quality(context)}
- Data Source: {MarketDataParser.get_data_source(context)}
        """.strip()

        return summary


class NewsParser(ContextParser):
    """Parser for NewsContext objects."""

    @staticmethod
    def get_articles(context: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract articles from news context."""
        return context.get("articles", [])

    @staticmethod
    def get_sentiment_summary(context: dict[str, Any]) -> dict[str, Any]:
        """Extract overall sentiment summary."""
        return context.get("sentiment_summary", {})

    @staticmethod
    def get_sentiment_score(context: dict[str, Any]) -> float:
        """Get overall sentiment score (-1 to 1)."""
        sentiment = NewsParser.get_sentiment_summary(context)
        return sentiment.get("score", 0.0)

    @staticmethod
    def get_sentiment_label(context: dict[str, Any]) -> str:
        """Get sentiment label (positive/negative/neutral)."""
        sentiment = NewsParser.get_sentiment_summary(context)
        return sentiment.get("label", "neutral")

    @staticmethod
    def format_news_summary(context: dict[str, Any]) -> str:
        """Create a formatted summary of news data for agents."""
        symbol = context.get("symbol", "GLOBAL")
        period = context.get("period", {})
        articles = NewsParser.get_articles(context)
        sentiment = NewsParser.get_sentiment_summary(context)

        summary = f"""
News Analysis for {symbol}:
- Period: {period.get("start")} to {period.get("end")}
- Articles: {len(articles)}
- Overall Sentiment: {sentiment.get("label", "neutral").upper()} (score: {sentiment.get("score", 0):.2f})
- Confidence: {sentiment.get("confidence", 0):.2f}
- Data Quality: {NewsParser.get_data_quality(context)}
- Sources: {", ".join(context.get("sources", []))}
        """.strip()

        return summary

    @staticmethod
    def get_recent_headlines(context: dict[str, Any], limit: int = 5) -> list[str]:
        """Get recent headlines for quick overview."""
        articles = NewsParser.get_articles(context)
        return [article.get("headline", "") for article in articles[:limit]]


class SocialParser(ContextParser):
    """Parser for SocialContext objects."""

    @staticmethod
    def get_posts(context: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract posts from social context."""
        return context.get("posts", [])

    @staticmethod
    def get_engagement_metrics(context: dict[str, Any]) -> dict[str, float]:
        """Extract engagement metrics."""
        return context.get("engagement_metrics", {})

    @staticmethod
    def format_social_summary(context: dict[str, Any]) -> str:
        """Create a formatted summary of social media data for agents."""
        symbol = context.get("symbol", "GLOBAL")
        period = context.get("period", {})
        posts = SocialParser.get_posts(context)
        engagement = SocialParser.get_engagement_metrics(context)

        summary = f"""
Social Media Analysis for {symbol}:
- Period: {period.get("start")} to {period.get("end")}
- Posts: {len(posts)}
- Total Engagement: {engagement.get("total_engagement", 0)}
- Average Engagement: {engagement.get("average_engagement", 0):.1f}
- Data Quality: {SocialParser.get_data_quality(context)}
- Platforms: {", ".join(context.get("platforms", []))}
        """.strip()

        return summary


class FundamentalParser(ContextParser):
    """Parser for FundamentalContext objects."""

    @staticmethod
    def get_key_ratios(context: dict[str, Any]) -> dict[str, float]:
        """Extract key financial ratios."""
        return context.get("key_ratios", {})

    @staticmethod
    def get_balance_sheet(context: dict[str, Any]) -> dict[str, Any] | None:
        """Extract balance sheet data."""
        return context.get("balance_sheet")

    @staticmethod
    def get_income_statement(context: dict[str, Any]) -> dict[str, Any] | None:
        """Extract income statement data."""
        return context.get("income_statement")

    @staticmethod
    def format_fundamental_summary(context: dict[str, Any]) -> str:
        """Create a formatted summary of fundamental data for agents."""
        symbol = context.get("symbol", "UNKNOWN")
        ratios = FundamentalParser.get_key_ratios(context)

        key_metrics = []
        if "current_ratio" in ratios:
            key_metrics.append(f"Current Ratio: {ratios['current_ratio']:.2f}")
        if "debt_to_equity" in ratios:
            key_metrics.append(f"D/E Ratio: {ratios['debt_to_equity']:.2f}")
        if "roe" in ratios:
            key_metrics.append(f"ROE: {ratios['roe']:.2%}")

        summary = f"""
Fundamental Analysis for {symbol}:
- Key Ratios: {len(ratios)} available
- {chr(10).join(["- " + metric for metric in key_metrics[:5]])}
- Data Quality: {FundamentalParser.get_data_quality(context)}
        """.strip()

        return summary


def create_context_summary(context_json: str, context_type: str = "auto") -> str:
    """
    Create a human-readable summary of any context.

    Args:
        context_json: JSON string from toolkit
        context_type: Type of context (auto-detect if not specified)

    Returns:
        Formatted summary string
    """
    try:
        context = ContextParser.parse_context(context_json)

        if context_type == "auto":
            # Auto-detect context type based on fields
            if "price_data" in context:
                context_type = "market"
            elif "articles" in context:
                context_type = "news"
            elif "posts" in context:
                context_type = "social"
            elif "key_ratios" in context:
                context_type = "fundamental"

        # Generate appropriate summary
        if context_type == "market":
            return MarketDataParser.format_price_summary(context)
        elif context_type == "news":
            return NewsParser.format_news_summary(context)
        elif context_type == "social":
            return SocialParser.format_social_summary(context)
        elif context_type == "fundamental":
            return FundamentalParser.format_fundamental_summary(context)
        else:
            # Generic summary
            symbol = context.get("symbol", "N/A")
            data_quality = ContextParser.get_data_quality(context)
            data_source = ContextParser.get_data_source(context)

            return (
                f"Context for {symbol} - Quality: {data_quality}, Source: {data_source}"
            )

    except Exception as e:
        logger.error(f"Error creating context summary: {e}")
        return f"Error parsing context: {e}"


# Convenience functions for common operations
def extract_latest_price(market_context_json: str) -> float | None:
    """Quick extraction of latest price from market context."""
    context = ContextParser.parse_context(market_context_json)
    return MarketDataParser.get_latest_price(context)


def extract_sentiment_score(news_context_json: str) -> float:
    """Quick extraction of sentiment score from news context."""
    context = ContextParser.parse_context(news_context_json)
    return NewsParser.get_sentiment_score(context)


def is_high_quality_data(context_json: str) -> bool:
    """Quick check if context contains high quality data."""
    context = ContextParser.parse_context(context_json)
    return ContextParser.is_high_quality(context)


def create_msg_delete():
    """
    Create a message deletion node function for LangGraph workflows.

    This function returns a node that clears all messages from the state,
    which is useful for preventing context pollution between different
    phases of multi-agent workflows.

    Returns:
        Callable: A function that can be used as a LangGraph node
    """
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    def delete_messages(state):
        """Delete all messages from the current state."""
        del state  # Acknowledge the parameter
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    return delete_messages
