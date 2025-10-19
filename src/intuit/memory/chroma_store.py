"""
ChromaDB-backed memory store implementation for Intuit.
"""
from typing import Optional, Dict, Any, List
import os
import logging
import json
import uuid
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..utils.spinner import ThinkingSpinner, with_spinner

logger = logging.getLogger(__name__)

# Suppress ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

class ChromaMemoryStore:
    """Memory store implementation for Intuit using ChromaDB for persistence."""
    
    def __init__(self, persist_directory: str = ".memory", collection_name: str = "memories", model: str = "openai:text-embedding-3-small"):
        """
        Initialize the memory store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            model: The embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model = model
        
        # Extract provider and model name
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider = "openai"  # Default provider
            model_name = model
            
        logger.info(f"Initializing ChromaDB with directory: {persist_directory}")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Use the default embedding function that doesn't require an API key
        # This is just for testing purposes
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        logger.info("Using default embedding function")
        self.embedding_function = DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"ChromaDB memory store initialized with collection: {collection_name}")
    
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
            logger.info(f"[MEMORY STORE] add_memory called with content='{content[:50]}...', metadata={metadata}")
            
            # Generate a unique ID for the memory
            memory_id = str(uuid.uuid4())
            logger.debug(f"[MEMORY STORE] Generated memory ID: {memory_id}")
            
            # Prepare metadata with timestamps
            now = datetime.now().isoformat()
            memory_metadata = metadata or {}
            
            # Convert any list values to strings since ChromaDB doesn't support lists
            for key, value in memory_metadata.items():
                if isinstance(value, list):
                    memory_metadata[key] = ",".join(str(item) for item in value)
            
            memory_metadata.update({
                "created_at": now,
                "updated_at": now
            })
            logger.debug(f"[MEMORY STORE] Prepared metadata: {memory_metadata}")
            
            # Add the memory to the collection
            logger.debug(f"[MEMORY STORE] Adding to ChromaDB collection...")
            self.collection.add(
                documents=[content],
                metadatas=[memory_metadata],
                ids=[memory_id]
            )
            
            logger.info(f"[MEMORY STORE] Successfully added memory with ID: {memory_id}")
            logger.debug(f"[MEMORY STORE] Collection now has {self.collection.count()} memories")
            return memory_id
        except Exception as e:
            logger.error(f"[MEMORY STORE] Error adding memory: {e}", exc_info=True)
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
        with ThinkingSpinner(text=f"Searching memories for '{query}'",
                            spinner=None,
                            color="magenta") as spinner:
            try:
                logger.info(f"[MEMORY STORE] search_memories called with query='{query}', limit={limit}")
                logger.debug(f"[MEMORY STORE] Collection has {self.collection.count()} total memories")
                
                # Handle empty query to list all memories
                if not query:
                    logger.debug(f"[MEMORY STORE] Empty query - listing all memories")
                    results = self.collection.get(
                        limit=limit
                    )
                    logger.debug(f"[MEMORY STORE] Got {len(results.get('documents', []))} results")
                    
                    # Convert results to a list of dictionaries
                    memories = []
                    for i in range(len(results['documents'])):
                        memory = {
                            "id": results['ids'][i],
                            "content": results['documents'][i],
                            "metadata": results['metadatas'][i],
                            "created_at": results['metadatas'][i].get("created_at"),
                            "updated_at": results['metadatas'][i].get("updated_at")
                        }
                        memories.append(memory)
                else:
                    # Use query_texts for semantic search
                    logger.debug(f"[MEMORY STORE] Performing semantic search with query: {query}")
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=limit
                    )
                    logger.debug(f"[MEMORY STORE] Query returned {len(results.get('documents', [[]])[0])} results")
                    
                    # Extract results from the first query
                    if results and 'documents' in results and results['documents'] and results['documents'][0]:
                        documents = results['documents'][0]
                        metadatas = results['metadatas'][0]
                        ids = results['ids'][0]
                        distances = results.get('distances', [[]])[0] if 'distances' in results else []
                        
                        # Convert results to a list of dictionaries
                        memories = []
                        for i in range(len(documents)):
                            memory = {
                                "id": ids[i],
                                "content": documents[i],
                                "metadata": metadatas[i],
                                "created_at": metadatas[i].get("created_at"),
                                "updated_at": metadatas[i].get("updated_at")
                            }
                            # Add distance if available
                            if distances and i < len(distances):
                                memory["distance"] = distances[i]
                            memories.append(memory)
                    else:
                        memories = []
                
                # No need to sort since we're using string matching
                
                logger.info(f"Found {len(memories)} memories")
                spinner.ok()
                return memories
            except Exception as e:
                logger.error(f"Error searching memories: {e}")
                spinner.fail()
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
            # Get the memory from the collection
            result = self.collection.get(
                ids=[memory_id]
            )
            
            # Check if memory was found
            if result and result['ids'] and len(result['ids']) > 0:
                logger.info(f"Retrieved memory with ID: {memory_id}")
                memory = {
                    "id": result['ids'][0],
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0],
                    "created_at": result['metadatas'][0].get("created_at"),
                    "updated_at": result['metadatas'][0].get("updated_at")
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
            # Delete the memory from the collection
            self.collection.delete(
                ids=[memory_id]
            )
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
            # Delete all memories from the collection
            self.collection.delete(
                where={}
            )
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
            
            # Update metadata
            existing_metadata = existing.get("metadata", {})
            new_metadata = metadata if metadata is not None else existing_metadata
            
            # Convert any list values to strings since ChromaDB doesn't support lists
            for key, value in new_metadata.items():
                if isinstance(value, list):
                    new_metadata[key] = ",".join(str(item) for item in value)
            
            new_metadata["updated_at"] = datetime.now().isoformat()
            
            # Update the memory in the collection
            self.collection.update(
                ids=[memory_id],
                documents=[content],
                metadatas=[new_metadata]
            )
            
            logger.info(f"Updated memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False