"""
Tests for TTS providers.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import os

from intuit.audio.tts_base import TTSProvider
from intuit.audio.tts_coqui import CoquiTTSProvider
from intuit.audio.tts_gtts import GTTSProvider
from intuit.audio.tts_factory import TTSFactory
from intuit.config.audio_config import TTSConfig


class MockTTSProvider(TTSProvider):
    """Mock TTS provider for testing."""
    
    def __init__(self):
        self.synthesize_called = False
        self.last_text = None
        self.cleanup_called = False
    
    async def synthesize(self, text: str, output_path=None) -> np.ndarray:
        """Mock synthesize method."""
        self.synthesize_called = True
        self.last_text = text
        # Return dummy audio data
        return np.array([0, 100, -100, 0], dtype=np.int16)
    
    async def cleanup(self):
        """Mock cleanup method."""
        self.cleanup_called = True


@pytest.fixture
def mock_tts_config():
    """Create a mock TTS configuration."""
    return TTSConfig(
        provider="coqui",
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        device="cpu"
    )


@pytest.fixture
def gtts_config():
    """Create gTTS configuration."""
    return TTSConfig(
        provider="gtts",
        language="en"
    )


class TestTTSFactory:
    """Tests for TTS factory."""
    
    def test_create_coqui_provider(self, mock_tts_config):
        """Test creating Coqui TTS provider."""
        with patch('intuit.audio.tts_coqui.TTS'):
            provider = TTSFactory.create(mock_tts_config)
            assert isinstance(provider, CoquiTTSProvider)
            assert provider.config == mock_tts_config
    
    def test_create_gtts_provider(self, gtts_config):
        """Test creating gTTS provider."""
        provider = TTSFactory.create(gtts_config)
        assert isinstance(provider, GTTSProvider)
        assert provider.config == gtts_config
    
    def test_unsupported_provider(self):
        """Test error on unsupported provider."""
        config = TTSConfig(provider="unsupported")
        with pytest.raises(ValueError, match="Unsupported TTS provider"):
            TTSFactory.create(config)
    
    def test_register_custom_provider(self):
        """Test registering custom provider."""
        TTSFactory.register_provider("mock", MockTTSProvider)
        assert "mock" in TTSFactory.get_available_providers()
        
        # Clean up
        del TTSFactory._providers["mock"]
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = TTSFactory.get_available_providers()
        assert "coqui" in providers
        assert "gtts" in providers


class TestGTTSProvider:
    """Tests for gTTS provider."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, gtts_config):
        """Test gTTS provider initialization."""
        provider = GTTSProvider(gtts_config)
        assert provider.config == gtts_config
    
    @pytest.mark.asyncio
    async def test_synthesize(self, gtts_config):
        """Test speech synthesis with gTTS."""
        provider = GTTSProvider(gtts_config)
        
        with patch('intuit.audio.tts_gtts.gTTS') as mock_gtts, \
             patch('intuit.audio.tts_gtts.sf.read') as mock_read:
            # Mock gTTS
            mock_tts_instance = MagicMock()
            mock_gtts.return_value = mock_tts_instance
            
            # Mock audio file reading
            mock_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_read.return_value = (mock_audio, 22050)
            
            # Synthesize
            audio = await provider.synthesize("Hello world")
            
            # Verify
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.int16
            mock_gtts.assert_called_once_with(text="Hello world", lang="en")
            mock_tts_instance.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_with_output_path(self, gtts_config):
        """Test synthesis with specified output path."""
        provider = GTTSProvider(gtts_config)
        
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp3'
        ) as tmp:
            output_path = tmp.name
        
        try:
            with patch('intuit.audio.tts_gtts.gTTS') as mock_gtts, \
                 patch('intuit.audio.tts_gtts.sf.read') as mock_read:
                mock_tts_instance = MagicMock()
                mock_gtts.return_value = mock_tts_instance
                
                mock_audio = np.array([0.1], dtype=np.float32)
                mock_read.return_value = (mock_audio, 22050)
                
                await provider.synthesize("Test", output_path=output_path)
                
                # Verify save was called with our path
                mock_tts_instance.save.assert_called_once_with(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, gtts_config):
        """Test cleanup (should be no-op for gTTS)."""
        provider = GTTSProvider(gtts_config)
        await provider.cleanup()  # Should not raise


class TestCoquiTTSProvider:
    """Tests for Coqui TTS provider."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_tts_config):
        """Test Coqui TTS provider initialization."""
        with patch('intuit.audio.tts_coqui.torch'):
            provider = CoquiTTSProvider(mock_tts_config)
            assert provider.config == mock_tts_config
            assert not provider._model_loaded
    
    @pytest.mark.asyncio
    async def test_device_selection_cuda(self, mock_tts_config):
        """Test CUDA device selection."""
        with patch('intuit.audio.tts_coqui.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            provider = CoquiTTSProvider(mock_tts_config)
            assert provider.device == "cuda"
    
    @pytest.mark.asyncio
    async def test_device_selection_mps(self, mock_tts_config):
        """Test MPS device selection."""
        with patch('intuit.audio.tts_coqui.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            
            provider = CoquiTTSProvider(mock_tts_config)
            assert provider.device == "mps"
    
    @pytest.mark.asyncio
    async def test_device_selection_cpu(self, mock_tts_config):
        """Test CPU fallback."""
        with patch('intuit.audio.tts_coqui.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            
            provider = CoquiTTSProvider(mock_tts_config)
            assert provider.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_synthesize(self, mock_tts_config):
        """Test speech synthesis with Coqui TTS."""
        with patch('intuit.audio.tts_coqui.torch'), \
             patch('intuit.audio.tts_coqui.TTS') as mock_tts_class, \
             patch('intuit.audio.tts_coqui.sf.read') as mock_read:
            
            # Mock TTS model
            mock_model = MagicMock()
            mock_model.tts_to_file = MagicMock()
            mock_tts_class.return_value = mock_model
            
            # Mock audio file reading
            mock_audio = np.array([0.1, 0.2], dtype=np.float32)
            mock_read.return_value = (mock_audio, 22050)
            
            provider = CoquiTTSProvider(mock_tts_config)
            audio = await provider.synthesize("Hello world")
            
            # Verify
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.int16
            assert provider._model_loaded
            mock_model.tts_to_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_tts_config):
        """Test cleanup releases resources."""
        with patch('intuit.audio.tts_coqui.torch') as mock_torch, \
             patch('intuit.audio.tts_coqui.TTS') as mock_tts_class:
            
            mock_model = MagicMock()
            mock_tts_class.return_value = mock_model
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache = MagicMock()
            
            provider = CoquiTTSProvider(mock_tts_config)
            provider.model = mock_model
            provider._model_loaded = True
            
            await provider.cleanup()
            
            assert provider.model is None
            assert not provider._model_loaded
            mock_torch.cuda.empty_cache.assert_called_once()


class TestTTSIntegration:
    """Integration tests for TTS system."""
    
    @pytest.mark.asyncio
    async def test_factory_creates_working_provider(self, gtts_config):
        """Test that factory creates a working provider."""
        provider = TTSFactory.create(gtts_config)
        
        with patch('intuit.audio.tts_gtts.gTTS') as mock_gtts, \
             patch('intuit.audio.tts_gtts.sf.read') as mock_read:
            mock_tts_instance = MagicMock()
            mock_gtts.return_value = mock_tts_instance
            
            mock_audio = np.array([0.1], dtype=np.float32)
            mock_read.return_value = (mock_audio, 22050)
            
            audio = await provider.synthesize("Test")
            assert isinstance(audio, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """Test switching between providers."""
        # Create gTTS provider
        gtts_provider = TTSFactory.create(
            TTSConfig(provider="gtts", language="en")
        )
        assert isinstance(gtts_provider, GTTSProvider)
        
        # Create Coqui provider
        with patch('intuit.audio.tts_coqui.torch'):
            coqui_provider = TTSFactory.create(
                TTSConfig(
                    provider="coqui",
                    model_name="tts_models/en/ljspeech/tacotron2-DDC"
                )
            )
            assert isinstance(coqui_provider, CoquiTTSProvider)