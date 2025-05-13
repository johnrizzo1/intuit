"""
Memory store implementation for Intuit using LangMem.
"""
from typing import Optional, Dict, Any, List
import os
import logging
from langmem import create_memory_store_manager
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

class IntuitMemoryStore:
    """Memory store implementation for Intuit using LangMem."""
    
    def __init__(self, persist_directory: str = ".memory", namespace: str = "memories", model: str = "gpt-4o-mini"):
        """
        Initialize the memory store.
        
        Args:
            persist_directory: Directory for persistent storage
            namespace: Namespace for memories in the store
            model: The model to use for memory operations
        """
        self.model = model
        self.persist_directory = persist_directory
        self.namespace = namespace
        
        # Configure storage backend
        # Redis is not currently supported in langgraph.store.memory
        if os.getenv("REDIS_URL"):
            logger.warning(f"Redis URL found ({os.getenv('REDIS_URL')}), but Redis is not currently supported. Using in-memory storage instead.")
            self.store = InMemoryStore(
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
        
        # Initialize LangMem's memory store manager with our storage backend
        self.memory = create_memory_store_manager(model, namespace=(self.namespace,), store=self.store)
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
            # Generate a unique ID for the memory
            import uuid
            memory_id = str(uuid.uuid4())
            
            # Create the memory object
            memory_data = {
                "content": content,
                "metadata": metadata or {}
            }
            
            # Store the memory
            await self.memory.aput(memory_id, memory_data)
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
            # Use the asearch method to search for memories
            logger.info(f"Searching memories with query: {query}")
            search_results = await self.memory.asearch(query=query, limit=limit)
            
            # Convert search results to a list of dictionaries
            memories = []
            for item in search_results:
                memory = {
                    "id": item.key,
                    "content": item.value.get("content", ""),
                    "metadata": item.value.get("metadata", {}),
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                memories.append(memory)
            
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
            item = await self.memory.aget(memory_id)
            if item:
                logger.info(f"Retrieved memory with ID: {memory_id}")
                memory = {
                    "id": item.key,
                    "content": item.value.get("content", ""),
                    "metadata": item.value.get("metadata", {}),
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                return memory
            else:
                logger.info(f"Memory with ID {memory_id} not found")
                return None
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
            await self.memory.adelete(memory_id)
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
            # There's no direct clear method, so we'll need to get all memories and delete them
            search_results = await self.memory.asearch(query="", limit=1000)
            for item in search_results:
                await self.memory.adelete(item.key)
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
            item = await self.memory.aget(memory_id)
            if not item:
                logger.error(f"Cannot update memory {memory_id}: not found")
                return False
            
            # Update memory
            existing_metadata = item.value.get("metadata", {})
            new_metadata = metadata if metadata is not None else existing_metadata
            
            # Create updated memory data
            memory_data = {
                "content": content,
                "metadata": new_metadata
            }
            
            # Store the updated memory
            await self.memory.aput(memory_id, memory_data)
            logger.info(f"Updated memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False