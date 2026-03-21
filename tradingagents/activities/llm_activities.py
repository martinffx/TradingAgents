"""LLM Activities for Temporal workflow - thin shell over LLM client."""

from temporalio import activity

from tradingagents.lib.llm_client import LLMClient


@activity.defn
class LLMActivities:
    """Activity class for LLM operations with injected client.

    This is an imperative shell - thin wrapper over LLMClient.
    All business logic lives in domain entities.

    Temporal handles:
    - Retry on 429 (rate limit) with exponential backoff
    - Timeout management
    - Activity heartbeats
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize with LLM client dependency.

        Args:
            llm_client: Injected LLM client for sentiment and embeddings.
        """
        self._llm_client = llm_client

    @activity.defn
    async def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using LLM.

        Temporal retry policy handles 429 rate limits automatically.

        Args:
            text: Text content to analyze.

        Returns:
            Dict with sentiment label, confidence, and reasoning.
        """
        result = await self._llm_client.analyze_sentiment(text)
        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    @activity.defn
    async def create_embedding(self, text: str) -> list[float]:
        """Create vector embedding for text.

        Temporal retry policy handles 429 rate limits automatically.

        Args:
            text: Text content to embed.

        Returns:
            List of float values representing embedding vector.
        """
        return await self._llm_client.create_embedding(text)
