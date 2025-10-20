"""Factory for creating STT providers."""
from typing import Optional
import logging
from .stt_base import STTProvider
from .stt_google import GoogleSTT
from ..config.audio_config import STTConfig


logger = logging.getLogger(__name__)


class STTFactory:
    """Factory for creating STT providers."""
    
    @staticmethod
    def create(config: Optional[STTConfig] = None) -> STTProvider:
        """
        Create STT provider based on configuration.
        
        Args:
            config: STT configuration. If None, uses defaults.
            
        Returns:
            Configured STT provider instance
            
        Raises:
            ValueError: If provider is unknown
        """
        if config is None:
            config = STTConfig()
        
        logger.info(f"Creating STT provider: {config.provider}")
        
        if config.provider == "whisper":
            # Lazy import to avoid loading whisper if not needed
            from .stt_whisper import WhisperSTT
            return WhisperSTT(
                model_size=config.model_size,
                device=config.device,
                language=config.language,
                use_fp16=config.use_fp16,
                compute_type=config.compute_type,
            )
        elif config.provider == "google":
            # Convert language code format (en -> en-US)
            language = config.language
            if "-" not in language:
                language = f"{language}-US"
            return GoogleSTT(language=language)
        else:
            raise ValueError(f"Unknown STT provider: {config.provider}")
    
    @staticmethod
    def list_providers() -> list[str]:
        """
        Get list of available STT providers.
        
        Returns:
            List of provider names
        """
        return ["whisper", "google"]
    
    @staticmethod
    def get_provider_info(provider: str) -> dict:
        """
        Get information about a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with provider information
            
        Raises:
            ValueError: If provider is unknown
        """
        if provider == "whisper":
            return {
                "name": "Whisper",
                "type": "local",
                "gpu_accelerated": True,
                "requires_internet": False,
                "models": ["tiny", "base", "small", "medium", "large"],
                "languages": "99+",
            }
        elif provider == "google":
            return {
                "name": "Google Speech Recognition",
                "type": "cloud",
                "gpu_accelerated": False,
                "requires_internet": True,
                "models": ["default"],
                "languages": "120+",
            }
        else:
            raise ValueError(f"Unknown STT provider: {provider}")