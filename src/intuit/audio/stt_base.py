"""Base class for STT implementations."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class STTProvider(ABC):
    """Base class for Speech-to-Text providers."""
    
    @abstractmethod
    async def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> Optional[str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Transcribed text or None if transcription failed
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources used by the provider."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.__class__.__name__,
            "version": "1.0.0",
        }