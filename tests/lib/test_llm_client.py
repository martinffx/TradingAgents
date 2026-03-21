"""
Tests for OpenRouter LLM client using pytest-vcr for HTTP interactions.
Integration tests only - no mocking of client behavior.
"""

import pytest

from tradingagents.config import TradingAgentsConfig
from tradingagents.lib.llm_client import LLMClient, SentimentResult

# VCR configuration
vcr = pytest.mark.vcr(
    cassette_library_dir="tests/fixtures/vcr_cassettes/llm",
    record_mode="once",  # Record once, then replay
    match_on=["uri", "method", "body"],
    filter_headers=["authorization", "cookie", "user-agent", "x-stainless-id"],
    decode_compressed_response=True,
)


@pytest.fixture
def client():
    """LLMClient instance for testing."""
    from unittest.mock import Mock, patch

    mock_config = Mock(spec=TradingAgentsConfig)
    mock_config.news_sentiment_llm = "anthropic/claude-3.5-sonnet"
    mock_config.news_embedding_llm = "openai/text-embedding-3-small"

    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return LLMClient(mock_config)


class TestLLMClient:
    """Test LLMClient with VCR integration tests."""

    def test_initialization_with_api_key(self):
        """Test client initializes correctly with API key."""
        from unittest.mock import Mock, patch

        mock_config = Mock(spec=TradingAgentsConfig)
        mock_config.news_sentiment_llm = "anthropic/claude-3.5-sonnet"
        mock_config.news_embedding_llm = "openai/text-embedding-3-small"

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            client = LLMClient(mock_config)

            assert client.config == mock_config
            assert client.api_key == "test-key"
            # Verify the client was initialized without errors
            assert hasattr(client, "llm")
            assert hasattr(client, "embeddings")
            assert hasattr(client, "text_splitter")

    def test_initialization_without_api_key(self):
        """Test client raises ValueError without API key."""
        from unittest.mock import Mock, patch

        mock_config = Mock(spec=TradingAgentsConfig)

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                ValueError, match="OPENROUTER_API_KEY environment variable is required"
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

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_insufficient_text(self, client):
        """Test sentiment analysis with insufficient text returns neutral."""
        # Empty text
        result = await client.analyze_sentiment("")
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.reasoning == "Insufficient text for analysis"

        # Very short text
        result = await client.analyze_sentiment("Short")
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.reasoning == "Insufficient text for analysis"

        # Whitespace only
        result = await client.analyze_sentiment("   ")
        assert result.sentiment == "neutral"
        assert result.confidence == 0.0
        assert result.reasoning == "Insufficient text for analysis"

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_short_text_no_chunking(self, client):
        """Test sentiment analysis with short text doesn't use chunking."""
        text = "Apple stock rises 5% on strong earnings report beating expectations."

        result = await client.analyze_sentiment(text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_long_text_uses_chunking(self, client):
        """Test sentiment analysis with long text uses chunking."""
        # Create text longer than 1000 characters
        long_text = "Apple stock performance " * 50  # ~1000+ characters

        result = await client.analyze_sentiment(long_text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_chunking_with_multiple_chunks(self, client):
        """Test sentiment analysis chunking combines multiple chunks."""
        long_text = "A" * 1200  # Long text that will be split

        result = await client.analyze_sentiment(long_text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive_news(self, client):
        """Test sentiment analysis with positive financial news."""
        text = """
        Apple Inc. reported strong quarterly earnings that beat analyst expectations,
        sending the stock higher in after-hours trading. The tech giant posted revenue
        of $89.5 billion, up 8% from the same quarter last year, with earnings per share
        of $1.26, exceeding the expected $1.20. iPhone sales remained robust, and the
        company's services segment showed significant growth. Management raised its
        guidance for the upcoming quarter, citing strong demand across all product lines.
        """

        result = await client.analyze_sentiment(text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative_news(self, client):
        """Test sentiment analysis with negative financial news."""
        text = """
        Wells Fargo & Co. reported disappointing quarterly results as rising
        interest rates impacted its lending business. The bank's net income
        fell 23% to $3.5 billion, missing analyst expectations. provisions for
        credit losses increased significantly, reflecting concerns about
        potential loan defaults in a higher-rate environment. The stock
        declined sharply on the news.
        """

        result = await client.analyze_sentiment(text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_analyze_sentiment_neutral_news(self, client):
        """Test sentiment analysis with neutral/mixed financial news."""
        text = """
        The Federal Reserve held interest rates steady at its latest meeting,
        maintaining the current federal funds rate range of 5.25% to 5.50%.
        Officials indicated they would remain data-dependent regarding future
        rate adjustments. Markets showed muted reaction to the announcement,
        with investors awaiting clearer signals on monetary policy direction.
        """

        result = await client.analyze_sentiment(text)

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)

    @vcr
    @pytest.mark.asyncio
    async def test_create_embedding_empty_text_error(self, client):
        """Test embedding creation raises error for empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty for embedding"):
            await client.create_embedding("")

    @vcr
    @pytest.mark.asyncio
    async def test_create_embedding_short_text_no_chunking(self, client):
        """Test embedding creation with short text doesn't use chunking."""
        text = "Brief financial news summary"

        result = await client.create_embedding(text)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have embedding dimensions
        assert all(isinstance(value, float) for value in result)

    @vcr
    @pytest.mark.asyncio
    async def test_create_embedding_long_text_uses_chunking(self, client):
        """Test embedding creation with long text uses chunking and averaging."""
        # Create text longer than 2000 characters
        long_text = "Financial market analysis " * 100  # ~3000+ characters

        result = await client.create_embedding(long_text)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have embedding dimensions
        assert all(isinstance(value, float) for value in result)

    @vcr
    @pytest.mark.asyncio
    async def test_create_embedding_typical_news_text(self, client):
        """Test embedding creation with typical news text."""
        text = """
        Tesla Motors announced record vehicle deliveries for the quarter,
        surpassing analyst estimates. The electric vehicle manufacturer delivered
        422,875 vehicles, a 36% increase year-over-year, driven by strong demand
        for Model 3 and Model Y vehicles. Stock prices surged on the news.
        """

        result = await client.create_embedding(text)

        assert isinstance(result, list)
        assert len(result) > 0  # Should have embedding dimensions
        assert all(isinstance(value, float) for value in result)

    @vcr
    @pytest.mark.asyncio
    async def test_create_embedding_long_text_chunking_real(self, client):
        """Test embedding creation with long text requiring chunking."""
        # Create a very long financial news article
        long_text = (
            """
        Microsoft Corporation reported exceptional financial results for the fiscal quarter,
        demonstrating robust growth across its cloud computing, artificial intelligence, and
        productivity software divisions. The technology giant exceeded analyst expectations
        with revenue reaching $56.1 billion, representing a 13% increase year-over-year.
        Azure cloud services continued their impressive expansion, growing 29% and now
        contributing significantly to the company's overall revenue stream. The integration
        of OpenAI's technologies into Microsoft's product ecosystem has yielded substantial
        benefits, with AI-powered features driving increased adoption of Office 365 and
        other productivity solutions. The company's gaming division also performed well,
        with Xbox sales and Game Pass subscriptions showing steady growth. Microsoft's
        strategic investments in data centers and AI infrastructure appear to be paying
        dividends, positioning the company favorably in the competitive cloud services market.
        Management expressed optimism about future prospects, citing strong demand for
        digital transformation services and AI capabilities across various industries.
        The company's stock has responded positively to these results, reflecting investor
        confidence in Microsoft's ability to capitalize on emerging technology trends and
        maintain its competitive advantage in the rapidly evolving tech landscape.
        """
            * 3
        )  # Repeat to make it very long

        result = await client.create_embedding(long_text)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(value, float) for value in result)

    def test_create_sentiment_prompt_structure(self, client):
        """Test sentiment prompt creation has correct structure."""
        text = "Apple reports strong quarterly earnings"

        prompt = client._create_sentiment_prompt(text)

        assert "Analyze the sentiment of this financial news article" in prompt
        assert '"sentiment": "positive|negative|neutral"' in prompt
        assert '"confidence": 0.0-1.0' in prompt
        assert '"reasoning": "Brief explanation"' in prompt
        assert text in prompt
        assert "Overall market/stock sentiment impact" in prompt
        assert "Financial performance indicators" in prompt
        assert "Risk factors mentioned" in prompt
        assert "Business outlook expressed" in prompt

    def test_text_splitter_configuration(self, client):
        """Test text splitter has correct configuration for financial news."""
        splitter = client.text_splitter

        # Verify splitter was initialized with expected behavior
        # Test that it can split text as expected for our use case
        test_text = "A" * 600  # Longer than chunk size
        chunks = splitter.split_text(test_text)

        # Should create multiple chunks due to chunk_size=500
        assert len(chunks) > 1
        # Each chunk should be reasonable size
        assert all(len(chunk) > 0 for chunk in chunks)
