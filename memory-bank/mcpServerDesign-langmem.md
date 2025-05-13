# LangMem Integration Technical Design

## Architecture Overview

This document provides a detailed technical design for integrating LangMem into the Intuit project. The integration will enhance Intuit's memory capabilities, allowing the agent to maintain persistent memory across sessions, learn from interactions, and personalize responses.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Intuit Agent                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐  │
│  │  Agent Core   │   │  Tool System  │   │  Memory System    │  │
│  │               │◄──┤               │◄──┤                   │  │
│  │  - LLM        │   │  - Calendar   │   │  - LangMem Store  │  │
│  │  - Executor   │   │  - Notes      │   │  - Memory Tools   │  │
│  │  - Router     │   │  - Weather    │   │  - Memory Manager │  │
│  └───────┬───────┘   └───────────────┘   └───────────────────┘  │
│          │                                                      │
└──────────┼──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Server                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐  │
│  │  Tool APIs    │   │  Resources    │   │  Memory APIs      │  │
│  │               │   │               │   │                   │  │
│  │  - Calendar   │   │  - Files      │   │  - Add Memory     │  │
│  │  - Notes      │   │  - Images     │   │  - Search Memory  │  │
│  │  - Weather    │   │  - Data       │   │  - Get Memory     │  │
│  └───────────────┘   └───────────────┘   └───────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Memory Store

The Memory Store is the foundation of the LangMem integration. It provides persistent storage for memories and supports semantic search.

#### Key Features:
- Persistent storage using LangGraph's storage layer
- Vector embeddings for semantic search
- Flexible backend options (InMemory for development, Redis for production)
- Metadata support for rich memory context
#### Implementation Details:

```python
# src/intuit/memory/store.py
from typing import Optional, Dict, Any, List
import os
import logging
from langmem import MemoryStore
from langgraph.store.memory import InMemoryStore, RedisStore

logger = logging.getLogger(__name__)

class IntuitMemoryStore:
    """Memory store implementation for Intuit using LangMem."""
    
    def __init__(self, persist_directory: str = ".memory", namespace: str = "memories"):
        """
        Initialize the memory store.
        
        Args:
            persist_directory: Directory for persistent storage
            namespace: Namespace for memories in the store
        """
        self.persist_directory = persist_directory
        self.namespace = namespace
        
        # Configure storage backend
        if os.getenv("REDIS_URL"):
            logger.info(f"Using Redis storage backend at {os.getenv('REDIS_URL')}")
            self.store = RedisStore(
                url=os.getenv("REDIS_URL"),
                index={
                    "dims": 1536,
                    "embed": os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small"),
                }
            )
        else:
            logger.info(f"Using in-memory storage backend with persist directory: {persist_directory}")
            self.store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small"),
                }
            )
        
        # Initialize LangMem's MemoryStore with our storage backend
        self.memory = MemoryStore(store=self.store, namespace=(self.namespace,))
        logger.info("Memory store initialized successfully")
    
    async def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory to the store.
        
        Args:
            content: The content of the memory
            metadata: Optional metadata about the memory
            
        Returns:
            Memory ID
        """
        try:
            memory_id = await self.memory.add(content=content, metadata=metadata or {})
            logger.info(f"Added memory with ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    async def search_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of memories matching the query
        """
        try:
            # Handle empty query to list all memories
            if not query:
                logger.info("Empty query, listing all memories")
                memories = await self.memory.list(limit=limit)
            else:
                logger.info(f"Searching memories with query: {query}")
                memories = await self.memory.search(query=query, limit=limit)
            
            logger.info(f"Found {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found
        """
        try:
            memory = await self.memory.get(memory_id)
            if memory:
                logger.info(f"Retrieved memory with ID: {memory_id}")
            else:
                logger.info(f"Memory with ID {memory_id} not found")
            return memory
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.memory.delete(memory_id)
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def clear_memories(self) -> bool:
        """
        Clear all memories.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.memory.clear()
            logger.info("Cleared all memories")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False
    
    async def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The ID of the memory to update
            content: The new content
            metadata: The new metadata (or None to keep existing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing memory
            existing = await self.get_memory(memory_id)
            if not existing:
                logger.error(f"Cannot update memory {memory_id}: not found")
                return False
            
            # Update memory
            new_metadata = metadata if metadata is not None else existing.get("metadata", {})
            await self.memory.update(memory_id, content=content, metadata=new_metadata)
            logger.info(f"Updated memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False
```

### 2. Memory Tools

Memory Tools provide the agent with the ability to manage its own memory during conversations.

#### Key Features:
- Store important information from conversations
- Search for relevant memories
- Manage memory lifecycle (create, retrieve, update, delete)
#### Implementation Details:

```python
# src/intuit/memory/tools.py
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .store import IntuitMemoryStore

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
    name = "add_memory"
    description = "Add an important piece of information to your long-term memory"
    args_schema = MemoryAddInput
    
    def __init__(self, memory_store: IntuitMemoryStore):
        super().__init__()
        self.memory_store = memory_store
    
    async def _arun(self, content: str, importance: int = 5, tags: List[str] = None) -> str:
        """Run the tool asynchronously."""
        tags = tags or []
        memory_id = await self.memory_store.add_memory(
            content=content,
            metadata={"importance": importance, "tags": tags}
        )
        return f"I've remembered that {content} (Memory ID: {memory_id})"
    
    def _run(self, content: str, importance: int = 5, tags: List[str] = None) -> str:
        """Run the tool synchronously."""
        import asyncio
        return asyncio.run(self._arun(content, importance, tags))

class MemorySearchTool(BaseTool):
    """Tool for searching memories."""
    name = "search_memory"
    description = "Search your long-term memory for relevant information"
    args_schema = MemorySearchInput
    
    def __init__(self, memory_store: IntuitMemoryStore):
        super().__init__()
        self.memory_store = memory_store
    
    async def _arun(self, query: str, limit: int = 5) -> str:
        """Run the tool asynchronously."""
        memories = await self.memory_store.search_memories(query, limit)
        if not memories:
            return "I don't have any memories related to that."
        
        result = "Here's what I remember:\n\n"
        for i, memory in enumerate(memories):
            result += f"{i+1}. {memory['content']}\n"
        return result
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Run the tool synchronously."""
        import asyncio
        return asyncio.run(self._arun(query, limit))

class MemoryGetTool(BaseTool):
    """Tool for retrieving a specific memory."""
    name = "get_memory"
    description = "Retrieve a specific memory by ID"
    args_schema = MemoryGetInput
    
    def __init__(self, memory_store: IntuitMemoryStore):
        super().__init__()
        self.memory_store = memory_store
    
    async def _arun(self, memory_id: str) -> str:
        """Run the tool asynchronously."""
        memory = await self.memory_store.get_memory(memory_id)
        if not memory:
            return f"I couldn't find a memory with ID {memory_id}."
        return f"Memory {memory_id}: {memory['content']}"
    
    def _run(self, memory_id: str) -> str:
        """Run the tool synchronously."""
        import asyncio
        return asyncio.run(self._arun(memory_id))

class MemoryDeleteTool(BaseTool):
    """Tool for deleting a specific memory."""
    name = "delete_memory"
    description = "Delete a specific memory by ID"
    args_schema = MemoryDeleteInput
    
    def __init__(self, memory_store: IntuitMemoryStore):
        super().__init__()
        self.memory_store = memory_store
    
    async def _arun(self, memory_id: str) -> str:
        """Run the tool asynchronously."""
        success = await self.memory_store.delete_memory(memory_id)
        if not success:
            return f"I couldn't delete the memory with ID {memory_id}."
        return f"I've forgotten the memory with ID {memory_id}."
    
    def _run(self, memory_id: str) -> str:
        """Run the tool synchronously."""
        import asyncio
        return asyncio.run(self._arun(memory_id))

def get_memory_tools(memory_store: IntuitMemoryStore) -> List[BaseTool]:
    """Get all memory tools."""
    return [
        MemoryAddTool(memory_store),
        MemorySearchTool(memory_store),
        MemoryGetTool(memory_store),
        MemoryDeleteTool(memory_store),
    ]

def get_langmem_tools(store) -> List[BaseTool]:
    """Get LangMem's built-in memory tools."""
    from langmem import create_manage_memory_tool, create_search_memory_tool
    
    manage_memory = create_manage_memory_tool(namespace=("memories",))
    search_memory = create_search_memory_tool(namespace=("memories",))
    
    return [manage_memory, search_memory]
```

### 3. Memory Manager

The Memory Manager handles background processing of conversations to extract and consolidate memories.

#### Key Features:
- Automatic memory extraction from conversations
- Memory consolidation to reduce redundancy
- Periodic background processing
- Conversation summarization
#### Implementation Details:

```python
# src/intuit/memory/manager.py
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from langmem import MemoryManager as LangMemManager
from .store import IntuitMemoryStore

logger = logging.getLogger(__name__)

class IntuitMemoryManager:
    """Background memory manager for Intuit."""
    
    def __init__(self, store: IntuitMemoryStore):
        """
        Initialize the memory manager.
        
        Args:
            store: The memory store to use
        """
        self.store = store
        self.manager = LangMemManager(store=store.store, namespace=(store.namespace,))
        self.running = False
        self.task = None
        logger.info("Memory manager initialized")
    
    async def process_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """
        Process a conversation to extract and consolidate memories.
        
        Args:
            messages: List of conversation messages
        """
        try:
            logger.info(f"Processing conversation with {len(messages)} messages")
            # Convert messages to the format expected by LangMem
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Process the conversation
            await self.manager.process_conversation(formatted_messages)
            logger.info("Conversation processing complete")
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
    
    async def consolidate_memories(self) -> None:
        """Consolidate memories to reduce redundancy."""
        try:
            logger.info("Starting memory consolidation")
            start_time = time.time()
            await self.manager.consolidate_memories()
            duration = time.time() - start_time
            logger.info(f"Memory consolidation complete (took {duration:.2f}s)")
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
    
    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of a conversation.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Summary of the conversation
        """
        try:
            logger.info(f"Summarizing conversation with {len(messages)} messages")
            # Convert messages to the format expected by LangMem
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
            
            # Generate summary
            summary = await self.manager.summarize_conversation(formatted_messages)
            logger.info("Conversation summarization complete")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            return "Failed to summarize conversation."
    
    async def run_background_tasks(self, interval: int = 3600) -> None:
        """
        Run background tasks periodically.
        
        Args:
            interval: Interval between consolidation runs (in seconds)
        """
        self.running = True
        while self.running:
            try:
                logger.info("Running memory consolidation...")
                await self.consolidate_memories()
            except Exception as e:
                logger.error(f"Error in background task: {e}")
            
            # Sleep for the specified interval
            for _ in range(interval):
                if not self.running:
                    break
                await asyncio.sleep(1)
    
    def start(self, interval: int = 3600) -> None:
        """
        Start the background memory manager.
        
        Args:
            interval: Interval between consolidation runs (in seconds)
        """
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run_background_tasks(interval))
            logger.info(f"Background memory manager started with interval {interval}s")
    
    def stop(self) -> None:
        """Stop the background memory manager."""
        if self.task and not self.task.done():
            self.running = False
            self.task.cancel()
            logger.info("Background memory manager stopped")
```

### 4. Agent Integration

The Agent Integration connects the memory components to the Intuit agent.

#### Key Features:
- Memory tools available to the agent
- Automatic conversation processing
- Background memory management
- Enhanced prompt with memory context
#### Implementation Details:

```python
# Modifications to src/intuit/agent.py

# Add imports
from intuit.memory.store import IntuitMemoryStore
from intuit.memory.tools import get_memory_tools, get_langmem_tools
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
        
        # Get memory tools - choose either custom tools or LangMem's built-in tools
        # memory_tools = get_memory_tools(self.memory_store)  # Custom tools
        memory_tools = get_langmem_tools(self.memory_store.store)  # LangMem's built-in tools
        
        # Add memory tools to the agent's tools
        self.tools.extend(memory_tools)
        
        # Initialize memory manager
        self.memory_manager = IntuitMemoryManager(self.memory_store)
        
        # Start background memory processing
        self.memory_manager.start()
        
        # Initialize agent executor with all tools
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self, tools_override: Optional[List[BaseTool]] = None) -> AgentExecutor:
        """Create the agent executor with tools and prompt template."""
        
        # Existing code...
        
        # Update the prompt template to include memory context
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are Intuit, a helpful personal assistant with memory capabilities.
You have access to various tools to help the user:

1. Web search for online information
2. Filesystem operations for searching, reading, and managing files
3. Gmail integration for email management (when enabled)
4. Weather information (when enabled)
5. Calendar management for adding, listing, searching, and deleting events
6. Notes management for adding, listing, searching, and deleting notes
7. Reminders management for adding, listing, searching, and deleting reminders
8. Memory management for storing and retrieving important information

You can use your memory tools to:
- Store important information with add_memory
- Search your memories with search_memory
- Retrieve specific memories with get_memory
- Delete memories with delete_memory

Use your memory to provide personalized assistance and remember important details about the user.
When appropriate, search your memory for relevant information before responding.

[Existing prompt content...]
'''),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Continue with existing code...
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and generate a response."""
        
        # Add the user message to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Retrieve relevant memories for context
        memories = await self.memory_store.search_memories(user_input, limit=3)
        memory_context = ""
        if memories:
            memory_context = "Relevant memories:\n"
            for memory in memories:
                memory_context += f"- {memory['content']}\n"
        
        # Process the conversation with the memory manager
        await self.memory_manager.process_conversation(self.chat_history)
        
        # Continue with existing processing...
        # When invoking the agent, include memory context
        response = await self.agent_executor.ainvoke({
            "input": user_input,
            "chat_history": self.chat_history,
            "memory_context": memory_context
        })
        
        # Add the agent's response to chat history
        self.chat_history.append({"role": "assistant", "content": response["output"]})
        
        # Return the response
        return response["output"]
    
    async def shutdown(self):
        """Shutdown the agent and its components."""
        # Stop the memory manager
        if hasattr(self, 'memory_manager'):
            self.memory_manager.stop()
        
        # Shutdown MCP clients
        await self.shutdown_mcp_clients()
```

### 5. MCP Server Integration

The MCP Server Integration exposes memory functionality to external clients.

#### Key Features:
- Memory-related tools exposed via MCP
- Memory resource access
- Consistent API for memory operations
#### Implementation Details:

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
async def memory_add(content: str, importance: int = 5, tags: Optional[List[str]] = None) -> str:
    """
    Add a memory to the store.
    
    Args:
        content: The content of the memory
        importance: Importance level (1-10)
        tags: Optional tags for categorizing the memory
        
    Returns:
        Memory ID
    """
    logger.info(f"MCP: Adding memory: {content}")
    if memory_store:
        tags = tags or []
        memory_id = await memory_store.add_memory(
            content=content,
            metadata={"importance": importance, "tags": tags}
        )
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
        success = await memory_store.delete_memory(memory_id)
        if success:
            return f"Memory with ID {memory_id} deleted."
        return f"Failed to delete memory with ID {memory_id}."
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
        success = await memory_store.clear_memories()
        if success:
            return "All memories cleared."
        return "Failed to clear memories."
    return "Memory store not available"

@mcp_server.tool()
async def memory_update(memory_id: str, content: str, importance: Optional[int] = None, tags: Optional[List[str]] = None) -> str:
    """
    Update an existing memory.
    
    Args:
        memory_id: The ID of the memory to update
        content: The new content
        importance: New importance level (1-10)
        tags: New tags for categorizing the memory
        
    Returns:
        Confirmation message
    """
    logger.info(f"MCP: Updating memory with ID: {memory_id}")
    if memory_store:
        # Get existing memory
        existing = await memory_store.get_memory(memory_id)
        if not existing:
            return f"Memory with ID {memory_id} not found."
        
        # Prepare metadata
        metadata = existing.get("metadata", {})
        if importance is not None:
            metadata["importance"] = importance
        if tags is not None:
            metadata["tags"] = tags
        
        # Update memory
        success = await memory_store.update_memory(memory_id, content, metadata)
        if success:
            return f"Memory with ID {memory_id} updated."
        return f"Failed to update memory with ID {memory_id}."
    return "Memory store not available"
```

### 6. CLI Integration

The CLI Integration provides command-line access to memory functionality.

#### Key Features:
- List memories
- Search memories
- Add memories
- Delete memories
- Clear all memories

#### Implementation Details:

```python
# Additions to src/intuit/cli.py or appropriate CLI module

@cli_app.command()
def memory_list(limit: int = 10):
    """
    List all memories.
    
    Args:
        limit: Maximum number of memories to list
    """
    import asyncio
    
    async def _list_memories():
        agent = get_agent()
        memories = await agent.memory_store.search_memories("", limit=limit)
        if not memories:
            print("No memories found.")
            return
        
        print(f"Found {len(memories)} memories:")
        for i, memory in enumerate(memories):
            print(f"{i+1}. [ID: {memory['id']}] {memory['content']}")
    
    asyncio.run(_list_memories())

@cli_app.command()
def memory_search(query: str, limit: int = 5):
    """
    Search memories.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
    """
    import asyncio
    
    async def _search_memories():
        agent = get_agent()
        memories = await agent.memory_store.search_memories(query, limit=limit)
        if not memories:
            print("No memories found matching your query.")
            return
        
        print(f"Found {len(memories)} memories matching '{query}':")
        for i, memory in enumerate(memories):
            print(f"{i+1}. [ID: {memory['id']}] {memory['content']}")
    
    asyncio.run(_search_memories())

@cli_app.command()
def memory_add(content: str, importance: int = 5):
    """
    Add a memory.
    
    Args:
        content: Memory content
        importance: Importance level (1-10)
    """
    import asyncio
    
    async def _add_memory():
        agent = get_agent()
        memory_id = await agent.memory_store.add_memory(
            content=content,
            metadata={"importance": importance}
        )
        print(f"Memory added with ID: {memory_id}")
    
    asyncio.run(_add_memory())

@cli_app.command()
def memory_delete(memory_id: str):
    """
    Delete a memory.
    
    Args:
        memory_id: ID of the memory to delete
    """
    import asyncio
    
    async def _delete_memory():
        agent = get_agent()
        success = await agent.memory_store.delete_memory(memory_id)
        if success:
            print(f"Memory with ID {memory_id} deleted.")
        else:
            print(f"Failed to delete memory with ID {memory_id}.")
    
    asyncio.run(_delete_memory())

@cli_app.command()
def memory_clear():
    """Clear all memories."""
    import asyncio
    
    async def _clear_memories():
        agent = get_agent()
        success = await agent.memory_store.clear_memories()
        if success:
            print("All memories cleared.")
        else:
            print("Failed to clear memories.")
    
    asyncio.run(_clear_memories())
```

## Testing Strategy

### Unit Tests

```python
# tests/memory/test_store.py
import pytest
import asyncio
from intuit.memory.store import IntuitMemoryStore

@pytest.fixture
async def memory_store():
    """Create a memory store for testing."""
    store = IntuitMemoryStore(persist_directory=".test_memory")
    yield store
    # Clean up
    await store.clear_memories()

@pytest.mark.asyncio
async def test_add_memory(memory_store):
    """Test adding a memory."""
    memory_id = await memory_store.add_memory("Test memory", {"importance": 5})
    assert memory_id is not None
    assert isinstance(memory_id, str)

@pytest.mark.asyncio
async def test_search_memories(memory_store):
    """Test searching memories."""
    # Add some test memories
    await memory_store.add_memory("Python is a programming language", {"importance": 5})
    await memory_store.add_memory("JavaScript is used for web development", {"importance": 4})
    
    # Search for memories
    memories = await memory_store.search_memories("programming")
    assert len(memories) > 0
    assert "Python" in memories[0]["content"]

@pytest.mark.asyncio
async def test_get_memory(memory_store):
    """Test getting a memory by ID."""
    # Add a memory
    memory_id = await memory_store.add_memory("Remember to buy milk", {"importance": 3})
    
    # Get the memory
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == "Remember to buy milk"
    assert memory["metadata"]["importance"] == 3

@pytest.mark.asyncio
async def test_delete_memory(memory_store):
    """Test deleting a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Delete me", {"importance": 1})
    
    # Delete the memory
    success = await memory_store.delete_memory(memory_id)
    assert success is True
    
    # Verify it's gone
    memory = await memory_store.get_memory(memory_id)
    assert memory is None

@pytest.mark.asyncio
async def test_clear_memories(memory_store):
    """Test clearing all memories."""
    # Add some memories
    await memory_store.add_memory("Memory 1", {"importance": 1})
    await memory_store.add_memory("Memory 2", {"importance": 2})
    
    # Clear memories
    success = await memory_store.clear_memories()
    assert success is True
    
    # Verify they're gone
    memories = await memory_store.search_memories("")
    assert len(memories) == 0
```

## Deployment Considerations

1. **Storage Backend**:
   - Use Redis for production deployments to ensure persistence
   - Configure appropriate TTL (Time To Live) for memories based on importance
   - Implement backup and recovery procedures

2. **Memory Management**:
   - Monitor memory usage to prevent excessive growth
   - Implement automatic pruning of low-importance memories
   - Consider privacy implications and provide user controls

3. **Performance**:
   - Optimize embedding generation for large conversations
   - Use batching for memory operations when possible
   - Consider caching frequently accessed memories

4. **Security**:
   - Encrypt sensitive memories at rest
   - Implement access controls for multi-user environments
   - Provide mechanisms for users to delete their data

## Conclusion

The LangMem integration provides Intuit with powerful memory capabilities that enhance the agent's ability to provide personalized assistance. By leveraging LangMem's core memory API, memory management tools, and background memory manager, Intuit can maintain consistent knowledge across sessions and continuously improve based on user interactions.

The implementation follows a modular design that integrates seamlessly with the existing architecture while providing new capabilities. The memory components are exposed through the MCP server, making them available to external clients and ensuring a consistent API for memory operations.