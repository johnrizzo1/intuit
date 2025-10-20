"""
OpenAI LLM provider (cloud-based).

This module implements the LLMProvider interface using OpenAI's API,
providing a cloud-based fallback option for language model inference.
"""
import logging
from typing import AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from .llm_base import LLMProvider
from intuit.config.audio_config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (cloud-based).
    
    Features:
    - Cloud-based inference
    - High-quality responses
    - Function calling support
    - Streaming support
    - Multiple model options (GPT-3.5, GPT-4, etc.)
    - Reliable fallback option
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        
        logger.info(
            f"Initializing OpenAI provider with model: "
            f"{config.model}"
        )
        
        # Initialize OpenAI chat model
        openai_kwargs = {
            "model_name": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "streaming": True,
        }
        
        # Add base URL if specified (for OpenAI-compatible APIs)
        if config.base_url:
            openai_kwargs["openai_api_base"] = config.base_url
        
        self._model = ChatOpenAI(**openai_kwargs)
        
        logger.info("OpenAI provider initialized successfully")
    
    def get_model(self) -> BaseChatModel:
        """Get the underlying LangChain chat model.
        
        Returns:
            ChatOpenAI instance for use with LangChain agents
        """
        return self._model
    
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate a response from OpenAI.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            logger.debug(f"Generating response for prompt: {prompt[:50]}...")
            
            # Use LangChain's invoke method
            response = await self._model.ainvoke(prompt)
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug(f"Generated response: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from OpenAI.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
        """
        try:
            logger.debug(f"Streaming response for prompt: {prompt[:50]}...")
            
            # Use LangChain's astream method
            async for chunk in self._model.astream(prompt):
                # Extract content from chunk
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = str(chunk)
                if content:
                    yield content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up OpenAI resources.
        
        OpenAI manages its own resources, so this is a no-op.
        """
        logger.debug("OpenAI cleanup (no-op)")
        pass
    
    @property
    def model_name(self) -> str:
        """Get the model name.
        
        Returns:
            Name of the OpenAI model being used
        """
        return self.config.model
    
    @property
    def supports_functions(self) -> bool:
        """Check if the model supports function calling.
        
        Returns:
            True if function calling is supported
            
        Note:
            Most OpenAI models support function calling,
            including GPT-3.5-turbo and GPT-4.
        """
        # OpenAI models that support function calling
        function_capable_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
        ]
        
        model_lower = self.config.model.lower()
        return any(model in model_lower for model in function_capable_models)