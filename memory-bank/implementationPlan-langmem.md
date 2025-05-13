# LangMem Integration Plan

## Overview

This document outlines the plan to integrate LangMem into the Intuit project to provide the agent with persistent memory capabilities. LangMem will help our agent learn and adapt from interactions over time, extract important information from conversations, optimize behavior through prompt refinement, and maintain long-term memory across sessions.

## Current Memory Architecture

Currently, Intuit has the following memory-related components:

1. **Memory Bank Files**: A set of markdown files that document the project's context, progress, and technical details. These files are manually updated and serve as documentation rather than dynamic memory.

2. **Vector Store**: A ChromaDB-based implementation for indexing and searching filesystem content. This provides RAG capabilities but is focused on external document retrieval rather than conversation memory.

3. **Chat History**: The agent maintains a chat history list for the current session only, with no persistence between sessions.

4. **Tool State**: Tools like Calendar, Notes, and Reminders maintain their own state in JSON files, but this is tool-specific data rather than agent memory.

## LangMem Integration Benefits

Integrating LangMem will provide the following benefits:

1. **Persistent Memory**: The agent will remember important information across sessions.
2. **Adaptive Learning**: The agent will improve over time based on interactions.
3. **Personalization**: The agent will remember user preferences and adapt accordingly.
4. **Knowledge Consolidation**: Important information will be automatically extracted and organized.
5. **Improved Context Awareness**: The agent will maintain context across conversations.

## Technical Changes Required

### 1. Dependencies

Add LangMem and related dependencies to the project:

```python
# In requirements.txt or pyproject.toml
langmem>=0.1.0
langgraph>=0.1.0
```

### 2. Memory Store Integration

Create a new module for memory management that integrates with LangMem:

```
src/intuit/memory/
├── __init__.py
├── store.py          # Memory store implementation
├── tools.py          # Memory-related tools
└── manager.py        # Background memory manager
```

### 3. Agent Architecture Updates

Update the agent architecture to incorporate LangMem:

1. Modify the `Agent` class to use LangMem's memory store
2. Add memory tools to the agent's toolset
3. Implement background memory processing

### 4. MCP Server Extensions

Extend the MCP server to expose memory-related functionality:

1. Add memory-related tools to the MCP server
2. Create endpoints for memory management

## Detailed Implementation Plan

### Phase 1: Core Memory Store Setup

1. Create a memory store implementation using LangMem and LangGraph's storage layer:

```python
# src/intuit/memory/store.py
from typing import Optional, Dict, Any, List
import os
from langmem import MemoryStore
from langgraph.store.memory import InMemoryStore, RedisStore

class IntuitMemoryStore:
    """Memory store implementation for Intuit using LangMem."""
    
    def __init__(self, persist_directory: str = ".memory"):
        """Initialize the memory store."""
        # Use Redis if configured, otherwise use InMemoryStore
        if os.getenv("REDIS_URL"):
            self.store = RedisStore(
                url=os.getenv("REDIS_URL"),
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
        else:
            self.store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )
        
        # Initialize LangMem's MemoryStore with our storage backend
        self.memory = MemoryStore(store=self.store, namespace=("memories",))
    
    async def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory to the store."""
        return await self.memory.add(content=content, metadata=metadata or {})
    
    async def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity."""
        return await self.memory.search(query=query, limit=limit)
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        return await self.memory.get(memory_id)
    
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory by ID."""
        await self.memory.delete(memory_id)
    
    async def clear_memories(self) -> None:
        """Clear all memories."""
        await self.memory.clear()
```

### Phase 2: Memory Tools Implementation

2. Create memory tools for the agent to use:

```python
# src/intuit/memory/tools.py
from typing import Dict, Any, List, Optional
from langmem import create_manage_memory_tool, create_search_memory_tool
from .store import IntuitMemoryStore

def get_memory_tools(store: IntuitMemoryStore):
    """Get memory tools for the agent."""
    manage_memory = create_manage_memory_tool(namespace=("memories",))
    search_memory = create_search_memory_tool(namespace=("memories",))
    
    return [manage_memory, search_memory]
```

### Phase 3: Background Memory Manager

3. Implement a background memory manager:

```python
# src/intuit/memory/manager.py
import asyncio
import logging
from typing import List, Dict, Any, Optional
from langmem import MemoryManager
from .store import IntuitMemoryStore

logger = logging.getLogger(__name__)

class IntuitMemoryManager:
    """Background memory manager for Intuit."""
    
    def __init__(self, store: IntuitMemoryStore):
        """Initialize the memory manager."""
        self.store = store
        self.manager = MemoryManager(store=store.store, namespace=("memories",))
        self.running = False
        self.task = None
    
    async def process_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """Process a conversation to extract and consolidate memories."""
        await self.manager.process_conversation(messages)
    
    async def run_background_tasks(self, interval: int = 3600) -> None:
        """Run background tasks periodically."""
        self.running = True
        while self.running:
            try:
                logger.info("Running memory consolidation...")
                await self.manager.consolidate_memories()
                logger.info("Memory consolidation complete.")
            except Exception as e:
                logger.error(f"Error in memory consolidation: {e}")
            
            await asyncio.sleep(interval)
    
    def start(self, interval: int = 3600) -> None:
        """Start the background memory manager."""
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run_background_tasks(interval))
            logger.info("Background memory manager started.")
    
    def stop(self) -> None:
        """Stop the background memory manager."""
        if self.task and not self.task.done():
            self.running = False
            self.task.cancel()
            logger.info("Background memory manager stopped.")
```

### Phase 4: Agent Integration

4. Update the Agent class to use LangMem:

```python
# Modifications to src/intuit/agent.py

# Add imports
from intuit.memory.store import IntuitMemoryStore
from intuit.memory.tools import get_memory_tools
from intuit.memory.manager import IntuitMemoryManager

class Agent:
    """
    Core agent implementation that handles conversation and tool execution.
    """
    def __init__(
        self,
        tools: List[BaseTool],
        config: Optional[AgentConfig] = None,
        reminder_service: Optional[ReminderService] = None,
    ):
        # Existing initialization code...
        
        # Initialize memory store
        self.memory_store = IntuitMemoryStore()
        
        # Get memory tools
        memory_tools = get_memory_tools(self.memory_store)
        
        # Add memory tools to the agent's tools
        self.tools.extend(memory_tools)
        
        # Initialize memory manager
        self.memory_manager = IntuitMemoryManager(self.memory_store)
        
        # Start background memory processing
        self.memory_manager.start()
        
        # Initialize agent executor with all tools
        self.agent_executor = self._create_agent_executor()
    
    async def process_input(self, user_input: str) -> str:
        # Existing code...
        
        # Add the user message to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Process the conversation with the memory manager
        await self.memory_manager.process_conversation(self.chat_history)
        
        # Continue with existing processing...
    
    async def shutdown(self):
        """Shutdown the agent and its components."""
        # Stop the memory manager
        if hasattr(self, 'memory_manager'):
            self.memory_manager.stop()
        
        # Shutdown MCP clients
        await self.shutdown_mcp_clients()
```

### Phase 5: MCP Server Extensions

5. Add memory-related tools to the MCP server:

```python
# Additions to src/intuit/mcp_server.py

# Import memory components
from intuit.memory.store import IntuitMemoryStore

# Initialize memory store for MCP server
try:
    memory_store = IntuitMemoryStore()
    logger.info("Memory store initialized for MCP server")
except Exception as e:
    logger.error(f"Failed to initialize memory store: {e}")
    memory_store = None

# Memory tools
@mcp_server.tool()
async def memory_add(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Add a memory to the store.
    
    Args:
        content: The content of the memory
        metadata: Optional metadata about the memory
        
    Returns:
        Memory ID
    """
    logger.info(f"MCP: Adding memory: {content}")
    if memory_store:
        memory_id = await memory_store.add_memory(content, metadata)
        return f"Memory added with ID: {memory_id}"
    return "Memory store not available"

@mcp_server.tool()
async def memory_search(query: str, limit: int = 5) -> str:
    """
    Search memories by semantic similarity.
    
    Args:
        query: The search query
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        Search results
    """
    logger.info(f"MCP: Searching memories for: {query}")
    if memory_store:
        memories = await memory_store.search_memories(query, limit)
        if not memories:
            return "No memories found matching your query."
        
        result = "Found memories:\n\n"
        for i, memory in enumerate(memories):
            result += f"{i+1}. {memory['content']}\n"
        return result
    return "Memory store not available"

@mcp_server.tool()
async def memory_get(memory_id: str) -> str:
    """
    Get a specific memory by ID.
    
    Args:
        memory_id: The ID of the memory to retrieve
        
    Returns:
        Memory content
    """
    logger.info(f"MCP: Getting memory with ID: {memory_id}")
    if memory_store:
        memory = await memory_store.get_memory(memory_id)
        if memory:
            return f"Memory: {memory['content']}"
        return f"Memory with ID {memory_id} not found."
    return "Memory store not available"

@mcp_server.tool()
async def memory_delete(memory_id: str) -> str:
    """
    Delete a specific memory by ID.
    
    Args:
        memory_id: The ID of the memory to delete
        
    Returns:
        Confirmation message
    """
    logger.info(f"MCP: Deleting memory with ID: {memory_id}")
    if memory_store:
        await memory_store.delete_memory(memory_id)
        return f"Memory with ID {memory_id} deleted."
    return "Memory store not available"

@mcp_server.tool()
async def memory_clear() -> str:
    """
    Clear all memories.
    
    Returns:
        Confirmation message
    """
    logger.info("MCP: Clearing all memories")
    if memory_store:
        await memory_store.clear_memories()
        return "All memories cleared."
    return "Memory store not available"
```

### Phase 6: CLI Integration

6. Add memory-related commands to the CLI:

```python
# Additions to src/intuit/cli.py or appropriate CLI module

@cli_app.command()
def memory_list():
    """List all memories."""
    agent = get_agent()
    result = agent.memory_store.search_memories("", limit=100)
    if not result:
        print("No memories found.")
        return
    
    print("Memories:")
    for i, memory in enumerate(result):
        print(f"{i+1}. {memory['content']}")

@cli_app.command()
def memory_clear():
    """Clear all memories."""
    agent = get_agent()
    agent.memory_store.clear_memories()
    print("All memories cleared.")
```

## Testing Plan

1. **Unit Tests**: Create unit tests for the memory store, tools, and manager components.
2. **Integration Tests**: Test the integration of memory components with the agent.
3. **End-to-End Tests**: Test the complete memory functionality in a real conversation.

## Deployment Considerations

1. **Storage Backend**: Consider using Redis for production deployments to ensure persistence.
2. **Memory Limits**: Monitor memory usage and implement limits to prevent excessive growth.
3. **Privacy**: Implement mechanisms to allow users to control what information is stored.
4. **Backup**: Implement regular backups of the memory store.

## Timeline

1. **Phase 1 (Core Memory Store)**: 2 days
2. **Phase 2 (Memory Tools)**: 1 day
3. **Phase 3 (Background Manager)**: 2 days
4. **Phase 4 (Agent Integration)**: 2 days
5. **Phase 5 (MCP Server Extensions)**: 1 day
6. **Phase 6 (CLI Integration)**: 1 day
7. **Testing and Refinement**: 3 days

**Total Estimated Time**: 12 days

## Conclusion

Integrating LangMem into Intuit will significantly enhance the agent's capabilities by providing persistent memory across sessions, enabling it to learn from interactions, and personalizing responses based on user preferences. This implementation plan provides a structured approach to incorporating LangMem while maintaining compatibility with the existing architecture.