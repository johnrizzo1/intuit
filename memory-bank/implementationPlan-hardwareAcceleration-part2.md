
# Implementation Plan: Hardware-Accelerated Audio Pipeline (Part 2)

## Continued from Part 1...

### 2. Configuration System (Continued)

```python
# src/intuit/config/audio_config.py (continued)
"""Audio pipeline configuration."""

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
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
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
            ),
            tts=TTSConfig(
                provider=os.getenv("TTS_PROVIDER", "coqui"),
                model=os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"),
                device=os.getenv("TTS_DEVICE"),
            ),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "ollama"),
                model=os.getenv("LLM_MODEL", "llama3.2:3b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
        )
```

### 3. STT Provider Implementation

```python
# src/intuit/audio/stt_whisper.py
"""Whisper-based STT implementation with hardware acceleration."""
import whisper
import torch
import numpy as np
from typing import Optional, Dict, Any
import logging
import asyncio
from .stt_base import STTProvider
from ..hardware.detector import HardwareDetector

logger = logging.getLogger(__name__)

class WhisperSTT(STTProvider):
    """Hardware-accelerated Whisper STT."""
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: str = "en",
        use_fp16: bool = True,
        compute_type: str = "float16",
    ):
        """Initialize Whisper STT with hardware acceleration."""
        self.device = device or HardwareDetector.get_device_string()
        self.language = language
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.compute_type = compute_type if self.device != "cpu" else "float32"
        
        logger.info(f"Loading Whisper model '{model_size}' on {self.device}")
        logger.info(f"Using compute type: {self.compute_type}, FP16: {self.use_fp16}")
        
        # Load model with appropriate settings
        self.model = whisper.load_model(model_size, device=self.device)
        
        # Warm up the model
        self._warmup()
        
        logger.info("Whisper model loaded and warmed up successfully")
    
    def _warmup(self) -> None:
        """Warm up the model with a dummy input."""
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            self.model.transcribe(dummy_audio, language=self.language, fp16=self.use_fp16)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def transcribe(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        **kwargs
    ) -> Optional[str]:
        """Transcribe audio using Whisper with GPU acceleration."""
        try:
            # Preprocess audio
            audio_data = self._preprocess_audio(audio_data, sample_rate)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )
            
            text = result["text"].strip()
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"Transcribed: {text} (confidence: {confidence:.2f})")
            return text
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            return None
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription (runs in thread pool)."""
        return self.model.transcribe(
            audio_data,
            language=self.language,
            fp16=self.use_fp16,
            verbose=False,
        )
    
    def _preprocess_audio(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """Preprocess audio for Whisper."""
        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Resample to 16kHz if needed (Whisper requirement)
        if sample_rate != 16000:
            audio_data = self._resample(audio_data, sample_rate, 16000)
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        return audio_data
    
    def _resample(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio using torchaudio."""
        import torchaudio
        
        audio_tensor = torch.from_numpy(audio).to(self.device)
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        ).to(self.device)
        
        resampled = resampler(audio_tensor)
        return resampled.cpu().numpy()
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return list(whisper.tokenizer.LANGUAGES.keys())
    
    def cleanup(self) -> None:
        """Cleanup Whisper resources."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper resources cleaned up")
```

### 4. TTS Provider Implementation

```python
# src/intuit/audio/tts_coqui.py
"""Coqui TTS implementation with hardware acceleration."""
from TTS.api import TTS
import numpy as np
import torch
from typing import Optional
import logging
import asyncio
from .tts_base import TTSProvider
from ..hardware.detector import HardwareDetector

logger = logging.getLogger(__name__)

class CoquiTTS(TTSProvider):
    """Hardware-accelerated Coqui TTS."""
    
    def __init__(
        self,
        model: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: Optional[str] = None,
        use_gpu: bool = True,
        vocoder: Optional[str] = None,
    ):
        """Initialize Coqui TTS with hardware acceleration."""
        self.device = device or HardwareDetector.get_device_string()
        self.use_gpu = use_gpu and self.device != "cpu"
        
        logger.info(f"Loading Coqui TTS model '{model}' on {self.device}")
        
        # Initialize TTS
        self.tts = TTS(model, gpu=self.use_gpu)
        
        if vocoder:
            logger.info(f"Loading vocoder: {vocoder}")
            self.tts.load_vocoder(vocoder, gpu=self.use_gpu)
        
        # Get sample rate from model
        self.sample_rate = self.tts.synthesizer.output_sample_rate
        
        # Warm up the model
        self._warmup()
        
        logger.info(f"Coqui TTS loaded (sample rate: {self.sample_rate}Hz)")
    
    def _warmup(self) -> None:
        """Warm up the model with a dummy input."""
        try:
            self.tts.tts("Hello")
            logger.info("TTS model warmup completed")
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")
    
    async def synthesize(self, text: str, **kwargs) -> np.ndarray:
        """Synthesize text to audio with GPU acceleration."""
        try:
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text
            )
            
            # Convert to numpy array
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            audio = np.array(wav, dtype=np.float32)
            logger.info(f"Synthesized {len(audio)/self.sample_rate:.2f}s of audio")
            
            return audio
            
        except Exception as e:
            logger.error(f"Coqui TTS synthesis error: {e}", exc_info=True)
            raise
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis (runs in thread pool)."""
        return self.tts.tts(text)
    
    def get_sample_rate(self) -> int:
        """Get audio sample rate."""
        return self.sample_rate
    
    def list_models(self) -> list:
        """List available TTS models."""
        return TTS().list_models()
    
    def cleanup(self) -> None:
        """Cleanup TTS resources."""
        del self.tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Coqui TTS resources cleaned up")
```

### 5. Cross-Platform Audio Player

```python
# src/intuit/audio/audio_player.py
"""Cross-platform audio playback using sounddevice."""
import sounddevice as sd
import numpy as np
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

class AudioPlayer:
    """Cross-platform audio player."""
    
    @staticmethod
    async def play(
        audio_data: np.ndarray, 
        sample_rate: int,
        blocking: bool = True
    ) -> None:
        """
        Play audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            blocking: Wait for playback to complete
        """
        try:
            logger.debug(f"Playing audio: {len(audio_data)} samples at {sample_rate}Hz")
            
            # Ensure audio is in correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            if blocking:
                # Blocking playback
                sd.play(audio_data, sample_rate)
                sd.wait()
            else:
                # Non-blocking playback
                sd.play(audio_data, sample_rate)
                # Give it a moment to start
                await asyncio.sleep(0.1)
            
            logger.debug("Audio playback completed")
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}", exc_info=True)
            raise
    
    @staticmethod
    def stop() -> None:
        """Stop current playback."""
        sd.stop()
    
    @staticmethod
    def get_devices() -> list:
        """Get list of available audio devices."""
        return sd.query_devices()
    
    @staticmethod
    def set_default_device(device_id: int) -> None:
        """Set default audio output device."""
        sd.default.device = device_id
```

---

## Testing Strategy

### Unit Tests

```python
# tests/audio/test_stt_whisper.py
"""Unit tests for Whisper STT."""
import pytest
import numpy as np
from intuit.audio.stt_whisper import WhisperSTT

@pytest.fixture
def whisper_stt():
    """Create Whisper STT instance."""
    return WhisperSTT(model_size="tiny", device="cpu")  # Use tiny model for tests

@pytest.mark.asyncio
async def test_transcribe_silence(whisper_stt):
    """Test transcription of silence."""
    audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    result = await whisper_stt.transcribe(audio, 16000)
    assert result is not None

@pytest.mark.asyncio
async def test_transcribe_with_resampling(whisper_stt):
    """Test transcription with resampling."""
    audio = np.random.randn(44100).astype(np.float32)  # 1