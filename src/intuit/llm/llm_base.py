"""
Base class for Language Model (LLM) providers.

This module defines the abstract interface that all LLM providers must
implement, enabling a flexible provider pattern for language models.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator
from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers must implement this interface to ensure consistent
    behavior across different LLM engines (Ollama, OpenAI, etc.).
    
    This class wraps LangChain's BaseChatModel to provide a consistent
    interface while maintaining compatibility with LangChain agents.
    """
    
    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """Get the underlying LangChain chat model.
        
        Returns:
            BaseChatModel instance for use with LangChain agents
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of generated text
        """
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources used by the provider.
        
        This method should release any connections, close files,
        or perform other cleanup operations.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name.
        
        Returns:
            Name of the model being used
        """
        pass
    
    @property
    @abstractmethod
    def supports_functions(self) -> bool:
        """Check if the model supports function calling.
        
        Returns:
            True if function calling is supported
        """
        pass