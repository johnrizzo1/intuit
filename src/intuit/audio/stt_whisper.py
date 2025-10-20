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
        """
        Initialize Whisper STT with hardware acceleration.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cuda/mps/cpu), auto-detect if None
            language: Language code for transcription
            use_fp16: Use FP16 precision for faster inference
            compute_type: Compute precision type (float16/float32/int8)
        """
        self.model_size = model_size
        detected_device = device or HardwareDetector.get_device_string()
        
        # Force CPU on MPS due to sparse tensor limitations in Whisper
        if detected_device == "mps":
            logger.warning(
                "MPS detected but OpenAI Whisper has sparse tensor "
                "limitations. Using CPU instead. For GPU-accelerated STT "
                "on Apple Silicon, consider using faster-whisper library."
            )
            self.device = "cpu"
        else:
            self.device = detected_device
        
        self.language = language
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.compute_type = (
            compute_type if self.device != "cpu" else "float32"
        )
        
        logger.info(
            f"Loading Whisper model '{model_size}' on {self.device}"
        )
        logger.info(
            f"Using compute type: {self.compute_type}, FP16: {self.use_fp16}"
        )
        
        # Load model with appropriate settings
        self.model = whisper.load_model(model_size, device=self.device)
        
        # Warm up the model
        self._warmup()
        
        logger.info("Whisper model loaded and warmed up successfully")
    
    def _warmup(self) -> None:
        """Warm up the model with a dummy input."""
        try:
            # 1 second of silence
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.model.transcribe(
                dummy_audio,
                language=self.language,
                fp16=self.use_fp16
            )
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Optional[str]:
        """
        Transcribe audio using Whisper with GPU acceleration.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            **kwargs: Additional Whisper parameters
            
        Returns:
            Transcribed text or None if transcription failed
        """
        try:
            # Preprocess audio
            audio_data = self._preprocess_audio(audio_data, sample_rate)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data,
                kwargs
            )
            
            text = result["text"].strip()
            confidence = result.get("confidence", 0.0)
            
            logger.info(
                f"Transcribed: {text} (confidence: {confidence:.2f})"
            )
            return text
            
        except Exception as e:
            logger.error(
                f"Whisper transcription error: {e}",
                exc_info=True
            )
            return None
    
    def _transcribe_sync(
        self,
        audio_data: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronous transcription (runs in thread pool)."""
        return self.model.transcribe(
            audio_data,
            language=self.language,
            fp16=self.use_fp16,
            verbose=False,
            **kwargs
        )
    
    def _preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Preprocess audio for Whisper.
        
        Args:
            audio_data: Raw audio data
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio data
        """
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
        """
        Resample audio using torchaudio.
        
        Args:
            audio: Audio data to resample
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio data
        """
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "provider": "WhisperSTT",
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "use_fp16": self.use_fp16,
            "compute_type": self.compute_type,
            "supported_languages": len(self.get_supported_languages()),
        }
    
    def cleanup(self) -> None:
        """Cleanup Whisper resources."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper resources cleaned up")