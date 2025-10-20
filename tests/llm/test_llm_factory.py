"""Tests for LLM factory."""
import pytest
from unittest.mock import patch, Mock

from intuit.config.audio_config import LLMConfig
from intuit.llm.llm_factory import get_llm_provider
from intuit.llm.llm_ollama import OllamaProvider
from intuit.llm.llm_openai import OpenAIProvider


class TestLLMFactory:
    """Tests for LLM factory function."""
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.2:3b"
        )
        
        provider = get_llm_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider.config == config
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4"
        )
        
        provider = get_llm_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.config == config
    
    def test_invalid_provider(self):
        """Test handling of invalid provider."""
        config = LLMConfig(
            provider="ollama",  # Valid in config
            model="test"
        )
        # Manually set to invalid value to test error handling
        config.provider = "invalid"
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_provider(config)
    
    def test_fallback_to_openai_on_ollama_failure(self):
        """Test automatic fallback to OpenAI when Ollama fails."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.2:3b"
        )
        
        # Mock Ollama to fail
        with patch('intuit.llm.llm_factory.OllamaProvider') as mock_ollama:
            mock_ollama.side_effect = Exception("Ollama not available")
            
            # Mock OpenAI to succeed
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                with patch('intuit.llm.llm_factory.OpenAIProvider') as mock_openai:
                    mock_openai.return_value = Mock(spec=OpenAIProvider)
                    
                    provider = get_llm_provider(config, fallback_to_openai=True)
                    
                    # Should have tried Ollama first
                    mock_ollama.assert_called_once()
                    # Then fallen back to OpenAI
                    mock_openai.assert_called_once()
    
    def test_no_fallback_when_disabled(self):
        """Test that fallback doesn't occur when disabled."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.2:3b"
        )
        
        # Mock Ollama to fail
        with patch('intuit.llm.llm_factory.OllamaProvider') as mock_ollama:
            mock_ollama.side_effect = Exception("Ollama not available")
            
            with pytest.raises(Exception, match="Ollama not available"):
                get_llm_provider(config, fallback_to_openai=False)
    
    def test_fallback_fails_without_api_key(self):
        """Test fallback fails gracefully without OpenAI API key."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.2:3b"
        )
        
        # Mock Ollama to fail
        with patch('intuit.llm.llm_factory.OllamaProvider') as mock_ollama:
            mock_ollama.side_effect = Exception("Ollama not available")
            
            # Ensure no API key in environment
            with patch.dict('os.environ', {}, clear=True):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    get_llm_provider(config, fallback_to_openai=True)
    
    def test_default_config_from_env(self):
        """Test creating provider with config from environment."""
        with patch.dict('os.environ', {
            'LLM_PROVIDER': 'ollama',
            'LLM_MODEL': 'mistral:7b',
            'OLLAMA_BASE_URL': 'http://custom:11434'
        }):
            config = LLMConfig.from_env()
            provider = get_llm_provider(config)
            
            assert isinstance(provider, OllamaProvider)
            assert provider.config.model == "mistral:7b"
            assert provider.config.base_url == "http://custom:11434"
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_provider_switching(self):
        """Test switching between providers."""
        # Create Ollama provider
        ollama_config = LLMConfig(provider="ollama", model="llama3.2:3b")
        ollama_provider = get_llm_provider(ollama_config)
        assert isinstance(ollama_provider, OllamaProvider)
        
        # Create OpenAI provider
        openai_config = LLMConfig(provider="openai", model="gpt-4")
        openai_provider = get_llm_provider(openai_config)
        assert isinstance(openai_provider, OpenAIProvider)
        
        # Verify they're different instances
        assert ollama_provider is not openai_provider
    
    def test_config_validation(self):
        """Test that invalid configurations are caught."""
        # Test with empty model
        with pytest.raises(Exception):
            config = LLMConfig(provider="ollama", model="")
            get_llm_provider(config)
    
    def test_custom_parameters_preserved(self):
        """Test that custom parameters are preserved in provider."""
        config = LLMConfig(
            provider="ollama",
            model="llama3.2:3b",
            temperature=0.9,
            max_tokens=4000,
            streaming=False
        )
        
        provider = get_llm_provider(config)
        info = provider.get_model_info()
        
        assert info["temperature"] == 0.9
        assert info["max_tokens"] == 4000
        assert info["streaming"] is False


class TestFactoryIntegration:
    """Integration tests for factory with real configurations."""
    
    def test_factory_with_pipeline_config(self):
        """Test factory works with AudioPipelineConfig."""
        from intuit.config.audio_config import AudioPipelineConfig
        
        pipeline_config = AudioPipelineConfig(
            llm=LLMConfig(
                provider="ollama",
                model="llama3.2:3b"
            )
        )
        
        provider = get_llm_provider(pipeline_config.llm)
        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "llama3.2:3b"
    
    @patch.dict('os.environ', {
        'LLM_PROVIDER': 'ollama',
        'LLM_MODEL': 'llama3.2:3b',
        'LLM_TEMPERATURE': '0.8',
        'LLM_MAX_TOKENS': '3000'
    })
    def test_factory_with_env_config(self):
        """Test factory with configuration from environment."""
        from intuit.config.audio_config import AudioPipelineConfig
        
        config = AudioPipelineConfig.from_env()
        provider = get_llm_provider(config.llm)
        
        assert isinstance(provider, OllamaProvider)
        info = provider.get_model_info()
        assert info["model"] == "llama3.2:3b"
        assert info["temperature"] == 0.8
        assert info["max_tokens"] == 3000