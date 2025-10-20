"""
Google Text-to-Speech (gTTS) provider.

This module implements the TTSProvider interface using gTTS,
providing a cloud-based fallback option for text-to-speech synthesis.
"""
import asyncio
import logging
from typing import Optional
import numpy as np
import tempfile
import os

from .tts_base import TTSProvider
from ..config.audio_config import TTSConfig

logger = logging.getLogger(__name__)


class GTTSProvider(TTSProvider):
    """Google TTS provider (cloud-based).
    
    Features:
    - Cloud-based synthesis
    - No GPU required
    - Multiple language support
    - Reliable fallback option
    - Free tier available
    
    Note: This provider requires internet connectivity.
    """
    
    def __init__(self, config: TTSConfig):
        """Initialize gTTS provider.
        
        Args:
            config: TTS configuration
        """
        self.config = config
        logger.info(
            f"Initializing gTTS provider with language: "
            f"{config.language or 'en'}"
        )
    
    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text using gTTS.
        
        Args:
            text: The text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as numpy array (int16, mono)
            
        Raises:
            Exception: If synthesis fails
        """
        try:
            from gtts import gTTS
            import soundfile as sf
            
            logger.debug(f"Synthesizing text with gTTS: {text[:50]}...")
            
            # Create temp file if no output path specified
            temp_file = None
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.mp3'
                )
                output_path = temp_file.name
                temp_file.close()
            
            # Create gTTS object
            language = self.config.language or 'en'
            tts = gTTS(text=text, lang=language)
            
            # Save to file (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: tts.save(output_path)
            )
            
            # Load audio file
            audio_data, sample_rate = sf.read(output_path)
            
            # Convert to int16 mono
            if audio_data.ndim > 1:
                # Convert stereo to mono
                audio_data = audio_data.mean(axis=1)
            
            # Normalize and convert to int16
            if audio_data.dtype in (np.float32, np.float64):
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Clean up temp file if created
            if temp_file is not None:
                try:
                    os.unlink(output_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
            
            logger.debug(
                f"gTTS synthesis complete: {len(audio_data)} samples "
                f"at {sample_rate} Hz"
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up TTS resources.
        
        gTTS doesn't require cleanup as it's stateless.
        """
        logger.debug("gTTS cleanup (no-op)")
        pass