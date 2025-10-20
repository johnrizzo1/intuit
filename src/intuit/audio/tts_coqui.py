"""
Coqui TTS provider with GPU acceleration.

This module implements the TTSProvider interface using Coqui TTS,
providing high-quality, GPU-accelerated text-to-speech synthesis.
"""
import asyncio
import logging
from typing import Optional
import numpy as np
import torch
import tempfile
import os

from .tts_base import TTSProvider
from ..config.audio_config import TTSConfig

logger = logging.getLogger(__name__)


class CoquiTTSProvider(TTSProvider):
    """Coqui TTS provider with GPU acceleration.
    
    Features:
    - GPU acceleration (CUDA/MPS)
    - Multiple model support
    - High-quality synthesis
    - Voice cloning capability
    - Fast inference (5-10x realtime on GPU)
    """
    
    def __init__(self, config: TTSConfig):
        """Initialize Coqui TTS provider.
        
        Args:
            config: TTS configuration
        """
        self.config = config
        self.model = None
        self.device = self._get_device()
        
        logger.info(
            f"Initializing Coqui TTS provider with model: "
            f"{config.model} on device: {self.device}"
        )
        
        # Lazy load model on first use
        self._model_loaded = False
    
    def _get_device(self) -> str:
        """Determine the best device for TTS.
        
        Returns:
            Device string: "cuda" or "cpu"
            
        Note:
            MPS is not supported due to sparse tensor limitations.
            PyTorch's MPS backend doesn't support sparse tensors,
            which are used by some TTS models. We force CPU mode on
            Apple Silicon to avoid crashes.
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and \
                    torch.backends.mps.is_available():
                # Force CPU on Apple Silicon - MPS sparse tensor issue
                logger.warning(
                    "MPS detected but not supported by Coqui TTS due to "
                    "sparse tensor limitations. Using CPU instead. For "
                    "GPU-accelerated TTS on Apple Silicon, consider using "
                    "gTTS or other providers."
                )
                return "cpu"
            else:
                return "cpu"
        elif self.config.device == "mps":
            # User explicitly requested MPS, warn and fall back to CPU
            logger.warning(
                "MPS explicitly requested but not supported by Coqui "
                "TTS. Falling back to CPU. Use TTS_DEVICE=cpu in .env "
                "to suppress this warning."
            )
            return "cpu"
        return self.config.device
    
    async def _load_model(self):
        """Load the TTS model (lazy loading)."""
        if self._model_loaded:
            return
        
        try:
            # Import TTS here to avoid loading if not needed
            from TTS.api import TTS
            
            logger.info(f"Loading Coqui TTS model: {self.config.model}")
            
            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Force CPU mode if device is not CUDA
            # MPS is not supported due to sparse tensor limitations
            use_gpu = (self.device == "cuda")
            
            # Save current default device
            original_device = None
            if hasattr(torch, 'get_default_device'):
                try:
                    original_device = torch.get_default_device()
                except (AttributeError, RuntimeError):
                    pass
            
            # Temporarily set default device to CPU for MPS systems
            if not use_gpu and hasattr(torch.backends, "mps") and \
                    torch.backends.mps.is_available():
                torch.set_default_device("cpu")
                logger.info(
                    "Temporarily forcing CPU mode for Coqui TTS model "
                    "loading to avoid MPS sparse tensor limitations"
                )
            
            try:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: TTS(
                        model_name=self.config.model,
                        progress_bar=False,
                        gpu=use_gpu
                    )
                )
                
                # Only move to device if using CUDA
                if use_gpu:
                    self.model.to(self.device)
            finally:
                # Restore original default device
                if original_device is not None:
                    try:
                        torch.set_default_device(original_device)
                        logger.debug(
                            f"Restored default device to {original_device}"
                        )
                    except (AttributeError, RuntimeError):
                        pass
            
            self._model_loaded = True
            logger.info(
                f"Coqui TTS model loaded successfully on {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS model: {e}")
            raise
    
    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text using Coqui TTS.
        
        Args:
            text: The text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as numpy array (int16, mono, 22050 Hz)
            
        Raises:
            Exception: If synthesis fails
        """
        # Ensure model is loaded
        await self._load_model()
        
        try:
            logger.debug(f"Synthesizing text: {text[:50]}...")
            
            # Create temp file if no output path specified
            temp_file = None
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.wav'
                )
                output_path = temp_file.name
                temp_file.close()
            
            # Synthesize in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Prepare synthesis kwargs
            synthesis_kwargs = {
                "text": text,
                "file_path": output_path
            }
            
            # Add speaker_wav if provided (for voice cloning)
            if self.config.speaker_wav:
                synthesis_kwargs["speaker_wav"] = self.config.speaker_wav
            
            # Add language if specified
            if self.config.language:
                synthesis_kwargs["language"] = self.config.language
            
            # Run synthesis
            await loop.run_in_executor(
                None,
                lambda: self.model.tts_to_file(**synthesis_kwargs)
            )
            
            # Load audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(output_path)
            
            # Convert to int16 mono
            if audio_data.ndim > 1:
                # Convert stereo to mono
                audio_data = audio_data.mean(axis=1)
            
            # Normalize and convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Clean up temp file if created
            if temp_file is not None:
                try:
                    os.unlink(output_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
            
            logger.debug(
                f"Synthesis complete: {len(audio_data)} samples "
                f"at {sample_rate} Hz"
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up TTS resources."""
        if self.model is not None:
            logger.info("Cleaning up Coqui TTS resources")
            # Move model to CPU to free GPU memory
            if self.device != "cpu":
                self.model.to("cpu")
            # Delete model
            del self.model
            self.model = None
            self._model_loaded = False
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()