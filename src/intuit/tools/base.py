"""
Base tool interface for Intuit.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool as LangChainBaseTool

class BaseTool(LangChainBaseTool):
    """Base class for all Intuit tools."""
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")

    @abstractmethod
    async def _arun(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool's functionality asynchronously.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            Dict containing the tool's output
        """
        pass

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool's functionality synchronously.
        This is not implemented as we use async tools.
        """
        raise NotImplementedError("This tool only supports async execution")

    def __str__(self) -> str:
        return f"{self.name}: {self.description}" 