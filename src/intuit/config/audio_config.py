"""Audio pipeline configuration."""
import os
from pydantic import BaseModel, Field
from typing import Literal, Optional


class STTConfig(BaseModel):
    """Speech-to-Text configuration."""
    provider: Literal["whisper", "google"] = Field(
        default="whisper",
        description="STT provider to use"
    )
    model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        description=(
            "Whisper model size "
            "(tiny=39M, base=74M, small=244M, medium=769M, large=1550M)"
        )
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use (cuda/mps/cpu), auto-detect if None"
    )
    language: str = Field(
        default="en",
        description="Language code for transcription"
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for faster inference (requires GPU)"
    )
    compute_type: Literal["float16", "float32", "int8"] = Field(
        default="float16",
        description="Compute precision type"
    )


class TTSConfig(BaseModel):
    """Text-to-Speech configuration."""
    provider: Literal["coqui", "piper", "gtts"] = Field(
        default="coqui",
        description="TTS provider to use"
    )
    model: str = Field(
        default="tts_models/en/ljspeech/tacotron2-DDC",
        description="TTS model identifier"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device to use (cuda/mps/cpu), auto-detect if None"
    )
    use_gpu: bool = Field(
        default=True,
        description="Use GPU acceleration if available"
    )
    language: Optional[str] = Field(
        default="en",
        description="Language code for TTS (e.g., 'en', 'es', 'fr')"
    )
    vocoder: Optional[str] = Field(
        default=None,
        description="Vocoder model (optional, uses default if None)"
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker ID for multi-speaker models"
    )


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="LLM provider to use"
    )
    model: str = Field(
        default="llama3.2:3b",
        description="Model identifier"
    )
    base_url: Optional[str] = Field(
        default="http://localhost:11434",
        description="Ollama server URL (optional for OpenAI)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens to generate"
    )
    streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )


class AudioPipelineConfig(BaseModel):
    """Complete audio pipeline configuration."""
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    auto_detect_hardware: bool = Field(
        default=True,
        description="Automatically detect and use best available hardware"
    )
    
    @classmethod
    def from_env(cls) -> "AudioPipelineConfig":
        """Load configuration from environment variables."""
        return cls(
            stt=STTConfig(
                provider=os.getenv("STT_PROVIDER", "whisper"),
                model_size=os.getenv("STT_MODEL_SIZE", "base"),
                device=os.getenv("STT_DEVICE"),
                language=os.getenv("STT_LANGUAGE", "en"),
                use_fp16=os.getenv("STT_USE_FP16", "true").lower() == "true",
                compute_type=os.getenv("STT_COMPUTE_TYPE", "float16"),
            ),
            tts=TTSConfig(
                provider=os.getenv("TTS_PROVIDER", "coqui"),
                model=os.getenv(
                    "TTS_MODEL",
                    "tts_models/en/ljspeech/tacotron2-DDC"
                ),
                device=os.getenv("TTS_DEVICE"),
                use_gpu=os.getenv("TTS_USE_GPU", "true").lower() == "true",
                vocoder=os.getenv("TTS_VOCODER"),
                speaker=os.getenv("TTS_SPEAKER"),
            ),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "ollama"),
                model=os.getenv("LLM_MODEL", "llama3.2:3b"),
                base_url=os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434"
                ),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
                streaming=(
                    os.getenv("LLM_STREAMING", "true").lower() == "true"
                ),
            ),
            auto_detect_hardware=(
                os.getenv("AUTO_DETECT_HARDWARE", "true").lower() == "true"
            ),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def validate_config(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages (empty if all valid)
        """
        messages = []
        
        # Check STT configuration
        if self.stt.provider == "whisper" and self.stt.device == "cpu":
            messages.append("Warning: Whisper on CPU will be slower than GPU")
        
        # Check TTS configuration
        if self.tts.provider == "coqui" and not self.tts.use_gpu:
            messages.append("Warning: Coqui TTS without GPU will be slower")
        
        # Check LLM configuration
        if (self.llm.provider == "ollama" and
                self.llm.base_url == "http://localhost:11434"):
            messages.append(
                "Info: Using default Ollama URL (ensure Ollama is running)"
            )
        
        return messages


# Alias for backward compatibility
AudioConfig = AudioPipelineConfig