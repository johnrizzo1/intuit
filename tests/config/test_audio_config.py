"""Unit tests for audio configuration."""
import os
from unittest.mock import patch
from intuit.config.audio_config import (
    AudioPipelineConfig,
    STTConfig,
    TTSConfig,
    LLMConfig,
)


class TestSTTConfig:
    """Test STT configuration."""
    
    def test_default_values(self):
        """Test default STT configuration values."""
        config = STTConfig()
        
        assert config.provider == "whisper"
        assert config.model_size == "base"
        assert config.device is None
        assert config.language == "en"
        assert config.use_fp16 is True
        assert config.compute_type == "float16"
    
    def test_custom_values(self):
        """Test custom STT configuration values."""
        config = STTConfig(
            provider="google",
            model_size="small",
            device="cuda",
            language="es",
            use_fp16=False,
            compute_type="float32",
        )
        
        assert config.provider == "google"
        assert config.model_size == "small"
        assert config.device == "cuda"
        assert config.language == "es"
        assert config.use_fp16 is False
        assert config.compute_type == "float32"


class TestTTSConfig:
    """Test TTS configuration."""
    
    def test_default_values(self):
        """Test default TTS configuration values."""
        config = TTSConfig()
        
        assert config.provider == "coqui"
        assert config.model == "tts_models/en/ljspeech/tacotron2-DDC"
        assert config.device is None
        assert config.use_gpu is True
        assert config.vocoder is None
        assert config.speaker is None
    
    def test_custom_values(self):
        """Test custom TTS configuration values."""
        config = TTSConfig(
            provider="gtts",
            model="custom_model",
            device="mps",
            use_gpu=False,
            vocoder="custom_vocoder",
            speaker="speaker_1",
        )
        
        assert config.provider == "gtts"
        assert config.model == "custom_model"
        assert config.device == "mps"
        assert config.use_gpu is False
        assert config.vocoder == "custom_vocoder"
        assert config.speaker == "speaker_1"


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_values(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        
        assert config.provider == "ollama"
        assert config.model == "llama3.2:3b"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.streaming is True
    
    def test_custom_values(self):
        """Test custom LLM configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com",
            temperature=0.5,
            max_tokens=1000,
            streaming=False,
        )
        
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.base_url == "https://api.openai.com"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.streaming is False


class TestAudioPipelineConfig:
    """Test complete audio pipeline configuration."""
    
    def test_default_values(self):
        """Test default pipeline configuration values."""
        config = AudioPipelineConfig()
        
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.llm, LLMConfig)
        assert config.auto_detect_hardware is True
    
    def test_custom_values(self):
        """Test custom pipeline configuration values."""
        config = AudioPipelineConfig(
            stt=STTConfig(provider="google"),
            tts=TTSConfig(provider="gtts"),
            llm=LLMConfig(provider="openai"),
            auto_detect_hardware=False,
        )
        
        assert config.stt.provider == "google"
        assert config.tts.provider == "gtts"
        assert config.llm.provider == "openai"
        assert config.auto_detect_hardware is False
    
    @patch.dict(os.environ, {
        "STT_PROVIDER": "google",
        "STT_MODEL_SIZE": "small",
        "STT_DEVICE": "cuda",
        "STT_LANGUAGE": "es",
        "STT_USE_FP16": "false",
        "STT_COMPUTE_TYPE": "float32",
        "TTS_PROVIDER": "gtts",
        "TTS_MODEL": "custom_model",
        "TTS_DEVICE": "mps",
        "TTS_USE_GPU": "false",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4",
        "OLLAMA_BASE_URL": "http://custom:11434",
        "LLM_TEMPERATURE": "0.5",
        "LLM_MAX_TOKENS": "1000",
        "LLM_STREAMING": "false",
        "AUTO_DETECT_HARDWARE": "false",
    })
    def test_from_env(self):
        """Test loading configuration from environment variables."""
        config = AudioPipelineConfig.from_env()
        
        # Check STT config
        assert config.stt.provider == "google"
        assert config.stt.model_size == "small"
        assert config.stt.device == "cuda"
        assert config.stt.language == "es"
        assert config.stt.use_fp16 is False
        assert config.stt.compute_type == "float32"
        
        # Check TTS config
        assert config.tts.provider == "gtts"
        assert config.tts.model == "custom_model"
        assert config.tts.device == "mps"
        assert config.tts.use_gpu is False
        
        # Check LLM config
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"
        assert config.llm.base_url == "http://custom:11434"
        assert config.llm.temperature == 0.5
        assert config.llm.max_tokens == 1000
        assert config.llm.streaming is False
        
        # Check auto detect
        assert config.auto_detect_hardware is False
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = AudioPipelineConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "stt" in config_dict
        assert "tts" in config_dict
        assert "llm" in config_dict
        assert "auto_detect_hardware" in config_dict
    
    def test_validate_config_no_warnings(self):
        """Test configuration validation with no warnings."""
        config = AudioPipelineConfig(
            stt=STTConfig(device="cuda"),
            tts=TTSConfig(use_gpu=True),
            llm=LLMConfig(base_url="http://custom:11434"),
        )
        
        messages = config.validate_config()
        assert len(messages) == 0
    
    def test_validate_config_with_warnings(self):
        """Test configuration validation with warnings."""
        config = AudioPipelineConfig(
            stt=STTConfig(provider="whisper", device="cpu"),
            tts=TTSConfig(provider="coqui", use_gpu=False),
            llm=LLMConfig(provider="ollama"),
        )
        
        messages = config.validate_config()
        assert len(messages) == 3
        assert any("Whisper on CPU" in msg for msg in messages)
        assert any("Coqui TTS without GPU" in msg for msg in messages)
        assert any("default Ollama URL" in msg for msg in messages)