"""Google Speech Recognition STT implementation."""
import speech_recognition as sr
import numpy as np
from typing import Optional, Dict, Any
import logging
from .stt_base import STTProvider


logger = logging.getLogger(__name__)


class GoogleSTT(STTProvider):
    """Google Speech Recognition STT (cloud-based)."""
    
    def __init__(self, language: str = "en-US"):
        """
        Initialize Google STT.
        
        Args:
            language: Language code for transcription (e.g., 'en-US')
        """
        self.language = language
        self.recognizer = sr.Recognizer()
        
        # Configure recognizer for better performance
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
        logger.info(f"Google STT initialized with language: {language}")
    
    async def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Optional[str]:
        """
        Transcribe audio using Google Speech Recognition.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Transcribed text or None if transcription failed
        """
        try:
            # Convert numpy array to AudioData format
            audio = self._numpy_to_audio_data(audio_data, sample_rate)
            
            # Transcribe using Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio,
                language=self.language
            )
            
            logger.info(f"Transcribed: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Google STT transcription error: {e}",
                exc_info=True
            )
            return None
    
    def _numpy_to_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> sr.AudioData:
        """
        Convert numpy array to AudioData format.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            AudioData object for speech_recognition
        """
        # Convert to int16 if needed
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        
        # Create AudioData object
        return sr.AudioData(
            audio_data.tobytes(),
            sample_rate=sample_rate,
            sample_width=2  # 16-bit audio
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        return {
            "provider": "GoogleSTT",
            "language": self.language,
            "cloud_based": True,
            "requires_internet": True,
        }
    
    def cleanup(self) -> None:
        """Cleanup resources (no-op for Google STT)."""
        logger.info("Google STT cleanup (no resources to free)")