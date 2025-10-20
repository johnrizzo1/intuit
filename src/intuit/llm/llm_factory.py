"""
LLM provider factory.

This module provides a factory for creating LLM providers based on
configuration, with automatic fallback handling.
"""
import logging
from typing import Optional

from .llm_base import LLMProvider
from .llm_ollama import OllamaProvider
from .llm_openai import OpenAIProvider
from intuit.config.audio_config import LLMConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM providers.
    
    Features:
    - Provider selection based on configuration
    - Automatic fallback to cloud provider
    - Lazy initialization
    - Error handling and logging
    """
    
    _instance: Optional[LLMProvider] = None
    _config: Optional[LLMConfig] = None
    
    @classmethod
    def create_provider(
        cls,
        config: Optional[LLMConfig] = None,
        force_recreate: bool = False
    ) -> LLMProvider:
        """Create or get cached LLM provider.
        
        Args:
            config: LLM configuration (uses default if None)
            force_recreate: Force creation of new provider
            
        Returns:
            LLM provider instance
            
        Raises:
            RuntimeError: If provider creation fails
        """
        # Use cached instance if available and config matches
        if (
            not force_recreate
            and cls._instance is not None
            and cls._config == config
        ):
            logger.debug("Using cached LLM provider")
            return cls._instance
        
        # Get or create config
        if config is None:
            config = LLMConfig()
        
        logger.info(
            f"Creating LLM provider: {config.provider} "
            f"(model: {config.model})"
        )
        
        # Try to create requested provider
        provider = None
        try:
            if config.provider == "ollama":
                provider = cls._create_ollama(config)
            elif config.provider == "openai":
                provider = cls._create_openai(config)
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
                
        except Exception as e:
            logger.warning(
                f"Failed to create {config.provider} provider: {e}"
            )
            
            # Try fallback if enabled
            if config.fallback_to_cloud and config.provider != "openai":
                logger.info("Attempting fallback to OpenAI provider")
                try:
                    # Create fallback config
                    fallback_config = LLMConfig(
                        provider="openai",
                        model_name=config.model,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        base_url=config.base_url,
                        fallback_to_cloud=False,  # Prevent infinite loop
                    )
                    provider = cls._create_openai(fallback_config)
                    config = fallback_config
                    logger.info("Successfully fell back to OpenAI provider")
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback to OpenAI failed: {fallback_error}"
                    )
                    raise RuntimeError(
                        f"Failed to create LLM provider: {e}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Failed to create LLM provider: {e}"
                ) from e
        
        if provider is None:
            raise RuntimeError("Failed to create LLM provider")
        
        # Cache the provider
        cls._instance = provider
        cls._config = config
        
        logger.info(
            f"LLM provider created successfully: {provider.model_name}"
        )
        return provider
    
    @classmethod
    def _create_ollama(cls, config: LLMConfig) -> OllamaProvider:
        """Create Ollama provider.
        
        Args:
            config: LLM configuration
            
        Returns:
            Ollama provider instance
            
        Raises:
            Exception: If provider creation fails
        """
        logger.debug("Creating Ollama provider")
        return OllamaProvider(config)
    
    @classmethod
    def _create_openai(cls, config: LLMConfig) -> OpenAIProvider:
        """Create OpenAI provider.
        
        Args:
            config: LLM configuration
            
        Returns:
            OpenAI provider instance
            
        Raises:
            Exception: If provider creation fails
        """
        logger.debug("Creating OpenAI provider")
        return OpenAIProvider(config)
    
    @classmethod
    async def cleanup(cls):
        """Clean up cached provider resources."""
        if cls._instance is not None:
            logger.debug("Cleaning up cached LLM provider")
            await cls._instance.cleanup()
            cls._instance = None
            cls._config = None
    
    @classmethod
    def reset(cls):
        """Reset factory state (for testing)."""
        cls._instance = None
        cls._config = None


def get_llm_provider(
    config: Optional[LLMConfig] = None,
    force_recreate: bool = False
) -> LLMProvider:
    """Convenience function to get LLM provider.
    
    Args:
        config: LLM configuration (uses default if None)
        force_recreate: Force creation of new provider
        
    Returns:
        LLM provider instance
    """
    return LLMFactory.create_provider(config, force_recreate)