"""
Mock memory tools for testing.
"""
from typing import Dict, Any, List, Optional, Type
import asyncio
import logging
from unittest.mock import MagicMock
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MemoryAddInput(BaseModel):
    """Input for adding a memory."""
    content: str = Field(..., description="The content to remember")
    importance: int = Field(default=5, description="Importance level (1-10)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the memory")

class MemorySearchInput(BaseModel):
    """Input for searching memories."""
    query: str = Field(..., description="The search query")
    limit: int = Field(default=5, description="Maximum number of results to return")

class MemoryGetInput(BaseModel):
    """Input for retrieving a specific memory."""
    memory_id: str = Field(..., description="The ID of the memory to retrieve")

class MemoryDeleteInput(BaseModel):
    """Input for deleting a specific memory."""
    memory_id: str = Field(..., description="The ID of the memory to delete")

class MemoryAddTool(BaseTool):
    """Tool for adding a memory."""
    name: str = "add_memory"
    description: str = "Add an important piece of information to your long-term memory"
    args_schema: type[MemoryAddInput] = MemoryAddInput
    memory_store: Any  # Allow any type for testing with MagicMock
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, memory_store):
        """Initialize the tool with a memory store."""
        super().__init__(memory_store=memory_store)
    
    async def _arun(self, content: str, importance: int = 5, tags: Optional[List[str]] = None) -> str:
        """Run the tool asynchronously."""
        tags = tags or []
        try:
            logger.info(f"Adding memory: {content} (importance: {importance}, tags: {tags})")
            memory_id = await self.memory_store.add_memory(
                content=content,
                metadata={"importance": importance, "tags": tags}
            )
            return f"I've remembered that {content} (Memory ID: {memory_id})"
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return f"I couldn't remember that information: {str(e)}"
    
    def _run(self, content: str, importance: int = 5, tags: Optional[List[str]] = None) -> str:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(content, importance, tags))

class MemorySearchTool(BaseTool):
    """Tool for searching memories."""
    name: str = "search_memory"
    description: str = "Search your long-term memory for relevant information"
    args_schema: type[MemorySearchInput] = MemorySearchInput
    memory_store: Any  # Allow any type for testing with MagicMock
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, memory_store):
        """Initialize the tool with a memory store."""
        super().__init__(memory_store=memory_store)
    
    async def _arun(self, query: str, limit: int = 5) -> str:
        """Run the tool asynchronously."""
        try:
            logger.info(f"Searching memories with query: {query} (limit: {limit})")
            memories = await self.memory_store.search_memories(query, limit)
            
            if not memories:
                return "I don't have any memories related to that."
            
            result = "Here's what I remember:\n\n"
            for i, memory in enumerate(memories):
                result += f"{i+1}. {memory['content']}\n"
            return result
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return f"I couldn't search my memories: {str(e)}"
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(query, limit))

class MemoryGetTool(BaseTool):
    """Tool for retrieving a specific memory."""
    name: str = "get_memory"
    description: str = "Retrieve a specific memory by ID"
    args_schema: type[MemoryGetInput] = MemoryGetInput
    memory_store: Any  # Allow any type for testing with MagicMock
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, memory_store):
        """Initialize the tool with a memory store."""
        super().__init__(memory_store=memory_store)
    
    async def _arun(self, memory_id: str) -> str:
        """Run the tool asynchronously."""
        try:
            logger.info(f"Getting memory with ID: {memory_id}")
            memory = await self.memory_store.get_memory(memory_id)
            
            if not memory:
                return f"I couldn't find a memory with ID {memory_id}."
            
            return f"Memory {memory_id}: {memory['content']}"
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return f"I couldn't retrieve that memory: {str(e)}"
    
    def _run(self, memory_id: str) -> str:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(memory_id))

class MemoryDeleteTool(BaseTool):
    """Tool for deleting a specific memory."""
    name: str = "delete_memory"
    description: str = "Delete a specific memory by ID"
    args_schema: type[MemoryDeleteInput] = MemoryDeleteInput
    memory_store: Any  # Allow any type for testing with MagicMock
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, memory_store):
        """Initialize the tool with a memory store."""
        super().__init__(memory_store=memory_store)
    
    async def _arun(self, memory_id: str) -> str:
        """Run the tool asynchronously."""
        try:
            logger.info(f"Deleting memory with ID: {memory_id}")
            success = await self.memory_store.delete_memory(memory_id)
            
            if not success:
                return f"I couldn't delete the memory with ID {memory_id}."
            
            return f"I've forgotten the memory with ID {memory_id}."
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return f"I couldn't delete that memory: {str(e)}"
    
    def _run(self, memory_id: str) -> str:
        """Run the tool synchronously."""
        return asyncio.run(self._arun(memory_id))

def get_memory_tools(memory_store) -> List[BaseTool]:
    """
    Get custom memory tools.
    
    Args:
        memory_store: The memory store to use
        
    Returns:
        List of memory tools
    """
    return [
        MemoryAddTool(memory_store),
        MemorySearchTool(memory_store),
        MemoryGetTool(memory_store),
        MemoryDeleteTool(memory_store),
    ]

def get_langmem_tools(store) -> List[BaseTool]:
    """
    Get LangMem's built-in memory tools.
    
    Args:
        store: The storage backend to use
        
    Returns:
        List of LangMem tools
    """
    logger.info("Creating mock LangMem tools")
    # Create mock tools
    manage_memory = MagicMock(spec=BaseTool)
    manage_memory.name = "manage_memory"
    manage_memory.description = "Manage your long-term memory"
    
    search_memory = MagicMock(spec=BaseTool)
    search_memory.name = "search_memory"
    search_memory.description = "Search your long-term memory"
    
    return [manage_memory, search_memory]