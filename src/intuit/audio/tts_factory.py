"""
Factory for creating TTS provider instances.

This module provides a factory pattern for creating TTS providers
based on configuration, enabling easy switching between different
TTS engines (Coqui, gTTS, etc.).
"""
import logging
from typing import Dict, Type

from .tts_base import TTSProvider
from .tts_gtts import GTTSProvider
from ..config.audio_config import TTSConfig

logger = logging.getLogger(__name__)


class TTSFactory:
    """Factory for creating TTS provider instances.
    
    This factory manages the creation of TTS providers based on
    configuration, providing a centralized way to instantiate
    the appropriate provider.
    """
    
    @staticmethod
    def _get_provider_class(provider_name: str) -> Type[TTSProvider]:
        """Get provider class with lazy loading.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider class
        """
        if provider_name == "gtts":
            return GTTSProvider
        elif provider_name == "coqui":
            # Lazy import to avoid loading TTS if not needed
            from .tts_coqui import CoquiTTSProvider
            return CoquiTTSProvider
        else:
            raise ValueError(f"Unknown TTS provider: {provider_name}")
    
    @classmethod
    def create(cls, config: TTSConfig) -> TTSProvider:
        """Create a TTS provider based on configuration.
        
        Args:
            config: TTS configuration specifying provider and settings
            
        Returns:
            Initialized TTS provider instance
            
        Raises:
            ValueError: If provider is not supported
            
        Example:
            >>> from intuit.config.audio_config import TTSConfig
            >>> config = TTSConfig(provider="coqui", device="cuda")
            >>> provider = TTSFactory.create(config)
            >>> audio = await provider.synthesize("Hello world")
        """
        provider_name = config.provider.lower()
        logger.info(f"Creating TTS provider: {provider_name}")
        
        try:
            provider_class = cls._get_provider_class(provider_name)
            provider = provider_class(config)
            return provider
        except ValueError as e:
            logger.error(f"Unsupported TTS provider: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create TTS provider {provider_name}: {e}")
            raise
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[TTSProvider]
    ):
        """Register a custom TTS provider.
        
        This allows extending the factory with custom TTS providers
        without modifying the factory code.
        
        Args:
            name: Provider name (e.g., "custom_tts")
            provider_class: Provider class implementing TTSProvider
            
        Example:
            >>> class CustomTTS(TTSProvider):
            ...     async def synthesize(self, text): ...
            ...     async def cleanup(self): ...
            >>> TTSFactory.register_provider("custom", CustomTTS)
        """
        logger.info(f"Registering custom TTS provider: {name}")
        cls._providers[name] = provider_class
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available TTS provider names.
        
        Returns:
            List of provider names
            
        Example:
            >>> TTSFactory.get_available_providers()
            ['coqui', 'gtts']
        """
        return list(cls._providers.keys())