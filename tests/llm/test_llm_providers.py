"""Tests for LLM providers."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

from intuit.config.audio_config import LLMConfig
from intuit.llm.llm_ollama import OllamaProvider
from intuit.llm.llm_openai import OpenAIProvider


class TestOllamaProvider:
    """Tests for Ollama LLM provider."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LLMConfig(
            provider="ollama",
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2000,
            streaming=True
        )
    
    def test_initialization(self, config):
        """Test provider initialization."""
        provider = OllamaProvider(config)
        assert provider.config == config
        assert provider.model_name == "llama3.2:3b"
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        config = LLMConfig(
            provider="ollama",
            model="mistral:7b",
            base_url="http://custom:11434",
            temperature=0.5,
            max_tokens=1000,
            streaming=False
        )
        provider = OllamaProvider(config)
        assert provider.model_name == "mistral:7b"
        assert provider.config.base_url == "http://custom:11434"
    
    @pytest.mark.asyncio
    async def test_generate_response(self, config):
        """Test response generation."""
        provider = OllamaProvider(config)
        
        # Mock the model's ainvoke method
        mock_response = AIMessage(content="Test response")
        provider._model.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await provider.generate("Test prompt")
        assert response == "Test response"
        provider._model.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_history(self, config):
        """Test generation with conversation history."""
        provider = OllamaProvider(config)
        
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        mock_response = AIMessage(content="How can I help?")
        provider._model.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await provider.generate("What can you do?", history=history)
        assert response == "How can I help?"
        
        # Verify history was included in the call
        call_args = provider._model.ainvoke.call_args[0][0]
        assert len(call_args) == 3  # history + new message
    
    @pytest.mark.asyncio
    async def test_stream_response(self, config):
        """Test streaming response generation."""
        provider = OllamaProvider(config)
        
        # Mock streaming chunks
        async def mock_stream(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.content = chunk
                yield mock_chunk
        
        provider._model.astream = mock_stream
        
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " ", "world", "!"]
        assert "".join(chunks) == "Hello world!"
    
    def test_get_model_info(self, config):
        """Test getting model information."""
        provider = OllamaProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "ollama"
        assert info["model"] == "llama3.2:3b"
        assert info["base_url"] == "http://localhost:11434"
        assert info["temperature"] == 0.7
        assert info["max_tokens"] == 2000
        assert info["streaming"] is True


class TestOpenAIProvider:
    """Tests for OpenAI LLM provider."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            streaming=True
        )
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization(self, config):
        """Test provider initialization."""
        provider = OpenAIProvider(config)
        assert provider.config == config
        assert provider.model_name == "gpt-4"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_initialization_without_api_key_in_config(self):
        """Test initialization uses environment variable."""
        config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo"
        )
        provider = OpenAIProvider(config)
        assert provider.model_name == "gpt-3.5-turbo"
    
    def test_initialization_missing_api_key(self, config):
        """Test initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider(config)
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @pytest.mark.asyncio
    async def test_generate_response(self, config):
        """Test response generation."""
        provider = OpenAIProvider(config)
        
        # Mock the model's ainvoke method
        mock_response = AIMessage(content="Test response")
        provider._model.ainvoke = AsyncMock(return_value=mock_response)
        
        response = await provider.generate("Test prompt")
        assert response == "Test response"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @pytest.mark.asyncio
    async def test_stream_response(self, config):
        """Test streaming response generation."""
        provider = OpenAIProvider(config)
        
        # Mock streaming chunks
        async def mock_stream(*args, **kwargs):
            chunks = ["Hello", " ", "world"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.content = chunk
                yield mock_chunk
        
        provider._model.astream = mock_stream
        
        chunks = []
        async for chunk in provider.stream("Test prompt"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " ", "world"]
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_get_model_info(self, config):
        """Test getting model information."""
        provider = OpenAIProvider(config)
        info = provider.get_model_info()
        
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4"
        assert info["temperature"] == 0.7
        assert info["max_tokens"] == 2000


class TestProviderComparison:
    """Tests comparing different providers."""
    
    @pytest.mark.asyncio
    async def test_both_providers_same_interface(self):
        """Test that both providers implement the same interface."""
        ollama_config = LLMConfig(provider="ollama", model="llama3.2:3b")
        ollama_provider = OllamaProvider(ollama_config)
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            openai_config = LLMConfig(provider="openai", model="gpt-4")
            openai_provider = OpenAIProvider(openai_config)
        
        # Both should have the same methods
        assert hasattr(ollama_provider, 'generate')
        assert hasattr(openai_provider, 'generate')
        assert hasattr(ollama_provider, 'stream')
        assert hasattr(openai_provider, 'stream')
        assert hasattr(ollama_provider, 'get_model_info')
        assert hasattr(openai_provider, 'get_model_info')
    
    def test_model_info_structure(self):
        """Test that model info has consistent structure."""
        ollama_config = LLMConfig(provider="ollama", model="llama3.2:3b")
        ollama_provider = OllamaProvider(ollama_config)
        ollama_info = ollama_provider.get_model_info()
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            openai_config = LLMConfig(provider="openai", model="gpt-4")
            openai_provider = OpenAIProvider(openai_config)
            openai_info = openai_provider.get_model_info()
        
        # Both should have the same keys
        assert set(ollama_info.keys()) == set(openai_info.keys())
        assert "provider" in ollama_info
        assert "model" in ollama_info
        assert "temperature" in ollama_info