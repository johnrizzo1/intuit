"""
Ollama LLM provider with local GPU acceleration.

This module implements the LLMProvider interface using Ollama,
providing local, GPU-accelerated language model inference.
"""
import logging
from typing import AsyncIterator
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

from .llm_base import LLMProvider
from intuit.config.audio_config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider with local GPU acceleration.
    
    Features:
    - Local GPU acceleration
    - Multiple model support (Llama 2, Mistral, etc.)
    - Streaming support
    - Function calling (model-dependent)
    - Privacy-first (100% local)
    - Zero API costs
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        
        logger.info(
            f"Initializing Ollama provider with model: "
            f"{config.model} at {config.base_url}"
        )
        
        # Initialize Ollama chat model
        self._model = ChatOllama(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )
        
        logger.info("Ollama provider initialized successfully")
    
    def get_model(self) -> BaseChatModel:
        """Get the underlying LangChain chat model.
        
        Returns:
            ChatOllama instance for use with LangChain agents
        """
        return self._model
    
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate a response from Ollama.
        
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
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from Ollama.
        
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
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up Ollama resources.
        
        Ollama manages its own resources, so this is a no-op.
        """
        logger.debug("Ollama cleanup (no-op)")
        pass
    
    @property
    def model_name(self) -> str:
        """Get the model name.
        
        Returns:
            Name of the Ollama model being used
        """
        return self.config.model
    
    @property
    def supports_functions(self) -> bool:
        """Check if the model supports function calling.
        
        Returns:
            True if function calling is supported
            
        Note:
            Function calling support depends on the specific Ollama model.
            Models like Llama 2 and Mistral have varying levels of support.
        """
        # Check if model supports function calling
        # This is model-dependent in Ollama
        function_capable_models = [
            "llama2",
            "mistral",
            "mixtral",
            "codellama",
        ]
        
        model_lower = self.config.model.lower()
        return any(model in model_lower for model in function_capable_models)