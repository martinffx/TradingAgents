"""
OpenRouter client for LLM-powered sentiment analysis and embeddings.
"""

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


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""

    sentiment: str
    confidence: float
    reasoning: str | None = None


class OpenRouterClient:
    """OpenRouter client for sentiment analysis and embeddings using LangChain."""

    def __init__(self, config: TradingAgentsConfig):
        self.config = config
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Initialize LangChain OpenAI client for OpenRouter chat models
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(self.api_key),
            model=config.news_sentiment_llm,
            temperature=0.1,  # Low temperature for consistent results
        )

        # Initialize LangChain OpenAI embeddings client for OpenRouter
        self.embeddings = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(self.api_key),
            model=config.news_embedding_llm,
        )

        # Initialize RecursiveCharacterTextSplitter with production defaults
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 400-600 tokens sweet spot, ~500 characters
            chunk_overlap=75,  # 75-100 token overlap (15-20%)
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],  # Respect paragraph/sentence boundaries
            keep_separator=True,
        )

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of news article content using OpenRouter LLM via LangChain.

        Args:
            text: News article content to analyze

        Returns:
            SentimentResult with structured sentiment data
        """
        if not text or len(text.strip()) < 50:
            return SentimentResult(
                sentiment="neutral",
                confidence=0.0,
                reasoning="Insufficient text for analysis",
            )

        try:
            # Use chunking for articles that might be too long for effective analysis
            # Reasonable threshold: ~1000 characters for when to apply intelligent chunking
            if len(text) > 1000:
                # Split into chunks and analyze key parts
                chunks = self.text_splitter.split_text(text)
                # For sentiment analysis, use first chunk plus a bit of context from subsequent chunks
                if len(chunks) > 1:
                    # Combine first chunk with beginning of second for better context
                    truncated_text = chunks[0]
                    if len(chunks) > 1 and len(truncated_text) < 800:
                        # Add some content from the second chunk if we have space
                        additional_content = chunks[1][: min(200, len(chunks[1]))]
                        truncated_text += f"\n\n[Continued] {additional_content}"
                else:
                    truncated_text = chunks[0]
            else:
                truncated_text = text  # No truncation needed for short texts

            # Create sentiment analysis prompt
            prompt = self._create_sentiment_prompt(truncated_text)

            # Create messages for LangChain
            messages = [
                SystemMessage(
                    content="You are a financial news sentiment analyst. Always respond with valid JSON in the specified format."
                ),
                HumanMessage(content=prompt),
            ]

            # Use LangChain to call the LLM
            response = await self.llm.ainvoke(messages)

            # Parse the response

            # Try to parse as JSON
            if isinstance(response.content, str):
                sentiment_data = json.loads(response.content)
                return SentimentResult(
                    sentiment=sentiment_data.get("sentiment", "neutral"),
                    confidence=sentiment_data.get("confidence", 0.0),
                    reasoning=sentiment_data.get("reasoning", "LLM analysis complete"),
                )
            else:
                # Handle case where response.content might be a list or dict
                sentiment_data = response.content
                if isinstance(sentiment_data, dict):
                    return SentimentResult(
                        sentiment=sentiment_data.get("sentiment", "neutral"),
                        confidence=sentiment_data.get("confidence", 0.0),
                        reasoning=sentiment_data.get(
                            "reasoning", "LLM analysis complete"
                        ),
                    )
                else:
                    # Fallback to neutral sentiment
                    return SentimentResult(
                        sentiment="neutral",
                        confidence=0.0,
                        reasoning="Failed to parse LLM response",
                    )
        except Exception as e:
            logger.error(f"OpenRouter sentiment analysis failed: {e}")
            return SentimentResult(
                sentiment="neutral",
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
            )

    async def create_embedding(self, text: str) -> list[float]:
        """
        Create vector embedding for text using OpenRouter embeddings API via LangChain.

        Args:
            text: Text to embed (processed with chunking for long texts)

        Returns:
            List of float values representing the embedding vector
        """
        if not text:
            raise ValueError("Text cannot be empty for embedding")

        try:
            # For texts that might exceed token limits, use chunking and combine embeddings
            # Reasonable threshold: ~2000 characters for when to apply chunking
            if len(text) > 2000:
                # Split into chunks using RecursiveCharacterTextSplitter
                chunks = self.text_splitter.split_text(text)

                # Generate embeddings for each chunk
                chunk_embeddings = []
                for chunk in chunks:
                    if chunk.strip():  # Skip empty chunks
                        embedding = await self.embeddings.aembed_query(chunk)
                        chunk_embeddings.append(embedding)

                # Average the embeddings to get a single representation
                if chunk_embeddings:
                    averaged_embedding = []
                    for i in range(len(chunk_embeddings[0])):
                        avg_value = sum(
                            embedding[i] for embedding in chunk_embeddings
                        ) / len(chunk_embeddings)
                        averaged_embedding.append(avg_value)
                    return averaged_embedding
                else:
                    # Fallback to zero vector if no valid chunks
                    return [0.0] * 1536
            else:
                # For shorter texts, use directly
                embedding = await self.embeddings.aembed_query(text)

                if len(embedding) != 1536:
                    logger.error(
                        f"Unexpected embedding dimension: {len(embedding)}, expected 1536"
                    )

                return embedding

        except Exception as e:
            logger.error(f"OpenRouter embedding generation failed: {e}")
            raise

    def _create_sentiment_prompt(self, text: str) -> str:
        """Create structured prompt for sentiment analysis."""
        return f"""Analyze the sentiment of this financial news article. Respond with JSON in this exact format:
        {{
        "sentiment": "positive|negative|neutral",
        "confidence": 0.0-1.0,
        "reasoning": "Brief explanation"
        }}

        Article:
        {text}

        Focus on:
        - Overall market/stock sentiment impact
        - Financial performance indicators
        - Risk factors mentioned
        - Business outlook expressed

        Consider financial context and avoid overreacting to minor fluctuations. Be objective and data-driven."""
