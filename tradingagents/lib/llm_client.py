"""LLM client for sentiment analysis and embeddings with timeout and exception handling."""

import json
import logging
import os
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from tradingagents.config import TradingAgentsConfig

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90
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
    """LLM client for sentiment analysis and embeddings using LangChain.

    All methods raise exceptions on failure (no silent defaults).
    Temporal handles retry with exponential backoff on these exceptions.
    """

    def __init__(self, config: TradingAgentsConfig):
        self.config = config
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.timeout = getattr(config, "llm_timeout", DEFAULT_TIMEOUT_SECONDS)
        self.request_timeout = getattr(
            config, "llm_request_timeout", DEFAULT_REQUEST_TIMEOUT_SECONDS
        )
        self.embedding_dimensions = getattr(
            config, "embedding_dimensions", DEFAULT_EMBEDDING_DIMENSIONS
        )

        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(self.api_key),
            model=config.news_sentiment_llm,
            temperature=0.1,
            timeout=self.timeout,
        )

        self.embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(self.api_key),
            model=config.news_embedding_llm,
            timeout=self.timeout,
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

        messages = [
            SystemMessage(
                content="You are a financial news sentiment analyst. Always respond with valid JSON in the specified format."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            raise SentimentAnalysisError(f"LLM call failed: {e}") from e

        return self._parse_sentiment_response(response)

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

    def _parse_sentiment_response(self, response) -> SentimentResult:
        """Parse LLM response into SentimentResult.

        Raises:
            SentimentAnalysisError: If response cannot be parsed.
        """
        try:
            if isinstance(response.content, str):
                sentiment_data = json.loads(response.content)
            elif isinstance(response.content, dict):
                sentiment_data = response.content
            else:
                raise SentimentAnalysisError(
                    f"Unexpected response type: {type(response.content)}"
                )

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
            else:
                embedding = await self.embeddings.aembed_query(text)

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
                embedding = await self.embeddings.aembed_query(chunk)
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
        """Sanitize text to prevent prompt injection and ensure proper parsing.

        Uses delimiter-based protection and truncation.
        """
        max_chars = 5000
        truncated = text[:max_chars] if len(text) > max_chars else text

        truncated = truncated.replace("---", "")
        truncated = truncated.replace('{"', "").replace('"}', "")
        truncated = truncated.replace("CONTENT START", "").replace("CONTENT END", "")

        return truncated
