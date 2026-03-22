"""
Tests for LLM client using mocked OpenAI SDK.
Unit tests for sentiment analysis and embedding logic.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tradingagents.config import TradingAgentsConfig
from tradingagents.lib.llm_client import (
    EmbeddingGenerationError,
    InsufficientTextError,
    LLMClient,
    SentimentAnalysisError,
    SentimentResult,
)


@pytest.fixture
def mock_config():
    """Mock TradingAgentsConfig for testing."""
    config = Mock(spec=TradingAgentsConfig)
    config.openrouter_api_key = "test-key"
    config.news_sentiment_llm = "test-model"
    config.news_embedding_llm = "test-embedding-model"
    config.backend_url = "https://openrouter.ai/api/v1"
    config.embedding_dimensions = 1536
    return config


@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client."""
    return AsyncMock()


@pytest.fixture
def client(mock_config, mock_openai_client):
    """LLMClient with mocked OpenAI client."""
    with patch(
        "tradingagents.lib.llm_client.AsyncOpenAI", return_value=mock_openai_client
    ):
        llm_client = LLMClient(mock_config)
    return llm_client


class TestLLMClientInitialization:
    """Test LLMClient initialization."""

    def test_initialization_with_api_key(self, mock_config):
        """Test client initializes correctly with API key."""
        with patch("tradingagents.lib.llm_client.AsyncOpenAI") as mock_class:
            mock_class.return_value = AsyncMock()
            client = LLMClient(mock_config)

            assert client.config == mock_config
            assert hasattr(client, "client")
            assert hasattr(client, "text_splitter")

    def test_initialization_without_api_key(self, mock_config):
        """Test client raises ValueError without API key."""
        mock_config.openrouter_api_key = ""

        with pytest.raises(
            ValueError, match="openrouter_api_key is required in config"
        ):
            LLMClient(mock_config)

    def test_sentiment_result_dataclass(self):
        """Test SentimentResult dataclass."""
        result = SentimentResult(
            sentiment="positive",
            confidence=0.85,
            reasoning="Strong financial performance indicators",
        )

        assert result.sentiment == "positive"
        assert result.confidence == 0.85
        assert result.reasoning == "Strong financial performance indicators"


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, client, mock_openai_client):
        """Test successful sentiment analysis."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "positive",
                "confidence": 0.85,
                "reasoning": "Strong earnings beat",
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await client.analyze_sentiment(
            "Apple stock rises 5% on strong earnings report beating expectations."
        )

        assert isinstance(result, SentimentResult)
        assert result.sentiment == "positive"
        assert result.confidence == 0.85
        assert result.reasoning == "Strong earnings beat"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_sentiment_insufficient_text(self, client):
        """Test sentiment analysis with insufficient text raises error."""
        with pytest.raises(
            InsufficientTextError, match="Text must be at least 50 characters"
        ):
            await client.analyze_sentiment("Too short")

        with pytest.raises(InsufficientTextError):
            await client.analyze_sentiment("")

        with pytest.raises(InsufficientTextError):
            await client.analyze_sentiment("   ")

    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative_sentiment(
        self, client, mock_openai_client
    ):
        """Test sentiment analysis with negative response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "negative",
                "confidence": 0.72,
                "reasoning": "Earnings missed expectations",
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await client.analyze_sentiment(
            "Company reports declining revenue and missed earnings targets."
        )

        assert result.sentiment == "negative"
        assert result.confidence == 0.72

    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral_sentiment(
        self, client, mock_openai_client
    ):
        """Test sentiment analysis with neutral response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "neutral",
                "confidence": 0.55,
                "reasoning": "Mixed signals in report",
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await client.analyze_sentiment(
            "Company releases quarterly report with mixed metrics."
        )

        assert result.sentiment == "neutral"
        assert result.confidence == 0.55

    @pytest.mark.asyncio
    async def test_analyze_sentiment_api_failure(self, client, mock_openai_client):
        """Test sentiment analysis handles API failure."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")

        with pytest.raises(SentimentAnalysisError, match="LLM call failed"):
            await client.analyze_sentiment(
                "Apple stock rises 5% on strong earnings report today."
            )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_invalid_json(self, client, mock_openai_client):
        """Test sentiment analysis handles invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(SentimentAnalysisError, match="Failed to parse response"):
            await client.analyze_sentiment(
                "Apple stock rises 5% on strong earnings report today."
            )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_invalid_sentiment_value(
        self, client, mock_openai_client
    ):
        """Test sentiment analysis handles invalid sentiment value."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "very_positive",
                "confidence": 0.85,
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(SentimentAnalysisError, match="Invalid sentiment"):
            await client.analyze_sentiment(
                "Apple stock rises 5% on strong earnings report today."
            )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_invalid_confidence(
        self, client, mock_openai_client
    ):
        """Test sentiment analysis handles invalid confidence value."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "positive",
                "confidence": 1.5,
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(SentimentAnalysisError, match="Confidence must be 0.0-1.0"):
            await client.analyze_sentiment(
                "Apple stock rises 5% on strong earnings report today."
            )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_empty_response(self, client, mock_openai_client):
        """Test sentiment analysis handles empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(SentimentAnalysisError, match="Empty response"):
            await client.analyze_sentiment(
                "Apple stock rises 5% on strong earnings report today."
            )

    @pytest.mark.asyncio
    async def test_analyze_sentiment_long_text_uses_chunking(
        self, client, mock_openai_client
    ):
        """Test sentiment analysis with long text uses text preparation."""
        long_text = "A" * 1500

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sentiment": "neutral",
                "confidence": 0.5,
            }
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await client.analyze_sentiment(long_text)

        assert isinstance(result, SentimentResult)
        mock_openai_client.chat.completions.create.assert_called_once()


class TestEmbeddingCreation:
    """Test embedding creation functionality."""

    @pytest.mark.asyncio
    async def test_create_embedding_success(self, client, mock_openai_client):
        """Test successful embedding creation."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 1536
        mock_openai_client.embeddings.create.return_value = mock_response

        result = await client.create_embedding("Financial news about market trends.")

        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(v, float) for v in result)
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_embedding_empty_text(self, client):
        """Test embedding creation with empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await client.create_embedding("")

        with pytest.raises(ValueError):
            await client.create_embedding("   ")

    @pytest.mark.asyncio
    async def test_create_embedding_api_failure(self, client, mock_openai_client):
        """Test embedding creation handles API failure."""
        mock_openai_client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(
            EmbeddingGenerationError, match="Embedding generation failed"
        ):
            await client.create_embedding("Financial news about market trends.")

    @pytest.mark.asyncio
    async def test_create_embedding_dimension_mismatch(
        self, client, mock_openai_client
    ):
        """Test embedding creation handles dimension mismatch."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 512
        mock_openai_client.embeddings.create.return_value = mock_response

        with pytest.raises(
            EmbeddingGenerationError, match="Embedding dimension mismatch"
        ):
            await client.create_embedding("Financial news about market trends.")

    @pytest.mark.asyncio
    async def test_create_embedding_long_text_uses_chunking(
        self, client, mock_openai_client
    ):
        """Test embedding creation with long text uses chunking and averaging."""
        long_text = "A" * 2500

        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1] * 1536
        mock_openai_client.embeddings.create.return_value = mock_response

        result = await client.create_embedding(long_text)

        assert isinstance(result, list)
        assert len(result) == 1536
        assert mock_openai_client.embeddings.create.call_count > 1


class TestPromptCreation:
    """Test prompt creation functionality."""

    def test_create_sentiment_prompt_structure(self, client):
        """Test sentiment prompt has correct structure."""
        text = "Apple reports strong quarterly earnings"

        prompt = client._create_sentiment_prompt(text)

        assert "Analyze the sentiment of this financial news article" in prompt
        assert '"sentiment": "positive|negative|neutral"' in prompt
        assert '"confidence": 0.0-1.0' in prompt
        assert '"reasoning": "Brief explanation"' in prompt
        assert text in prompt
        assert "CONTENT START" in prompt
        assert "CONTENT END" in prompt
        assert "Overall market/stock sentiment impact" in prompt
        assert "Financial performance indicators" in prompt
        assert "Risk factors mentioned" in prompt
        assert "Business outlook expressed" in prompt

    def test_sanitize_text_removes_injection_markers(self, client):
        """Test text sanitization removes potential injection markers."""
        unsafe_text = 'Hello --- world {"key": "value"} CONTENT START end'

        safe = client._sanitize_text_for_prompt(unsafe_text)

        assert "---" not in safe
        assert '{"' not in safe
        assert '"}' not in safe
        assert "CONTENT START" not in safe
        assert "CONTENT END" not in safe

    def test_sanitize_text_truncates_long_text(self, client):
        """Test text sanitization truncates very long text."""
        long_text = "A" * 10000

        safe = client._sanitize_text_for_prompt(long_text)

        assert len(safe) == 5000


class TestTextSplitter:
    """Test text splitter configuration and behavior."""

    def test_text_splitter_configuration(self, client):
        """Test text splitter has correct configuration."""
        splitter = client.text_splitter

        test_text = "A" * 600
        chunks = splitter.split_text(test_text)

        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_text_splitter_respects_separators(self, client):
        """Test text splitter splits on natural boundaries."""
        text = "A" * 300 + ".\n\n" + "B" * 300 + ".\n\n" + "C" * 300 + "."

        chunks = client.text_splitter.split_text(text)

        assert len(chunks) >= 2
