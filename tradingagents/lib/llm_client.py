"""LLM client for sentiment analysis and embeddings using OpenAI SDK with LangChain text splitting."""

import json
import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from tradingagents.config import TradingAgentsConfig

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_EMBEDDING_DIMENSIONS = 1536


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class SentimentAnalysisError(LLMError):
    """Raised when sentiment analysis fails."""

    pass


class EmbeddingGenerationError(LLMError):
    """Raised when embedding generation fails."""

    pass


class InsufficientTextError(LLMError):
    """Raised when text is too short for analysis."""

    pass


@dataclass(frozen=True)
class SentimentResult:
    """Structured sentiment analysis result."""

    sentiment: str
    confidence: float
    reasoning: str | None = None


class LLMClient:
    """LLM client for sentiment analysis and embeddings.

    Uses OpenAI SDK directly for API calls and LangChain text splitter for chunking.
    All methods raise exceptions on failure (no silent defaults).
    Temporal handles retry with exponential backoff on these exceptions.
    """

    def __init__(self, config: TradingAgentsConfig):
        if not config.openrouter_api_key:
            raise ValueError("openrouter_api_key is required in config")

        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.backend_url,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        self.embedding_dimensions = getattr(
            config, "embedding_dimensions", DEFAULT_EMBEDDING_DIMENSIONS
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of news article content.

        Raises:
            InsufficientTextError: If text is too short (< 50 chars).
            SentimentAnalysisError: If LLM call fails or response is invalid.
        """
        if not text or len(text.strip()) < 50:
            raise InsufficientTextError(
                f"Text must be at least 50 characters, got {len(text.strip())}"
            )

        truncated_text = self._prepare_text_for_analysis(text)
        prompt = self._create_sentiment_prompt(truncated_text)

        try:
            response = await self.client.chat.completions.create(
                model=self.config.news_sentiment_llm,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news sentiment analyst. Always respond with valid JSON in the specified format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            raise SentimentAnalysisError(f"LLM call failed: {e}") from e

        content = response.choices[0].message.content
        if not content:
            raise SentimentAnalysisError("Empty response from LLM")

        return self._parse_sentiment_response(content)

    def _prepare_text_for_analysis(self, text: str) -> str:
        """Prepare text for analysis with truncation and chunking."""
        if len(text) > 1000:
            chunks = self.text_splitter.split_text(text)
            if len(chunks) > 1:
                truncated_text = chunks[0]
                if len(truncated_text) < 800 and len(chunks) > 1:
                    additional_content = chunks[1][: min(200, len(chunks[1]))]
                    truncated_text += f"\n\n[Continued] {additional_content}"
                return truncated_text
            return chunks[0] if chunks else text[:1000]
        return text

    def _parse_sentiment_response(self, content: str) -> SentimentResult:
        """Parse LLM response into SentimentResult.

        Raises:
            SentimentAnalysisError: If response cannot be parsed.
        """
        try:
            sentiment_data = json.loads(content)

            sentiment = sentiment_data.get("sentiment", "neutral")
            if sentiment not in ("positive", "negative", "neutral"):
                raise SentimentAnalysisError(f"Invalid sentiment: {sentiment}")

            confidence = float(sentiment_data.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                raise SentimentAnalysisError(
                    f"Confidence must be 0.0-1.0, got {confidence}"
                )

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                reasoning=sentiment_data.get("reasoning"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            raise SentimentAnalysisError(f"Failed to parse response: {e}") from e

    async def create_embedding(self, text: str) -> list[float]:
        """Create vector embedding for text.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for embedding")

        try:
            if len(text) > 2000:
                return await self._create_chunked_embedding(text)

            response = await self.client.embeddings.create(
                model=self.config.news_embedding_llm,
                input=text,
            )
            embedding = list(response.data[0].embedding)

            if len(embedding) != self.embedding_dimensions:
                raise EmbeddingGenerationError(
                    f"Embedding dimension mismatch: got {len(embedding)}, "
                    f"expected {self.embedding_dimensions}"
                )

            return embedding

        except EmbeddingGenerationError:
            raise
        except Exception as e:
            logger.error(f"LLM embedding generation failed: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}") from e

    async def _create_chunked_embedding(self, text: str) -> list[float]:
        """Create embedding for long text using chunking and averaging."""
        chunks = self.text_splitter.split_text(text)

        chunk_embeddings = []
        for chunk in chunks:
            if chunk.strip():
                response = await self.client.embeddings.create(
                    model=self.config.news_embedding_llm,
                    input=chunk,
                )
                embedding = list(response.data[0].embedding)
                chunk_embeddings.append(embedding)

        if not chunk_embeddings:
            raise EmbeddingGenerationError(
                "No valid chunks to embed (all chunks were empty)"
            )

        dimension = len(chunk_embeddings[0])
        if dimension != self.embedding_dimensions:
            raise EmbeddingGenerationError(
                f"Embedding dimension mismatch in chunks: got {dimension}, "
                f"expected {self.embedding_dimensions}"
            )

        averaged_embedding = [
            sum(embedding[i] for embedding in chunk_embeddings) / len(chunk_embeddings)
            for i in range(dimension)
        ]

        return averaged_embedding

    def _create_sentiment_prompt(self, text: str) -> str:
        """Create structured prompt for sentiment analysis with delimiter protection."""
        safe_text = self._sanitize_text_for_prompt(text)

        return f"""Analyze the sentiment of this financial news article.

---
CONTENT START
{safe_text}
---
CONTENT END

Respond with JSON in this exact format:
{{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}

Focus on:
- Overall market/stock sentiment impact
- Financial performance indicators
- Risk factors mentioned
- Business outlook expressed"""

    def _sanitize_text_for_prompt(self, text: str) -> str:
        """Sanitize text to prevent prompt injection and ensure proper parsing."""
        max_chars = 5000
        truncated = text[:max_chars] if len(text) > max_chars else text

        truncated = truncated.replace("---", "")
        truncated = truncated.replace('{"', "").replace('"}', "")
        truncated = truncated.replace("CONTENT START", "").replace("CONTENT END", "")

        return truncated
