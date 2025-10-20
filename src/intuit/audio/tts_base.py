"""
Base class for Text-to-Speech (TTS) providers.

This module defines the abstract interface that all TTS providers must
implement, enabling a flexible provider pattern for text-to-speech synthesis.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class TTSProvider(ABC):
    """Abstract base class for TTS providers.
    
    All TTS providers must implement this interface to ensure consistent
    behavior across different TTS engines (Coqui, gTTS, etc.).
    """
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: The text to convert to speech
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as numpy array (int16, mono)
            
        Raises:
            Exception: If synthesis fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources used by the provider.
        
        This method should release any GPU memory, close files,
        or perform other cleanup operations.
        """
        pass