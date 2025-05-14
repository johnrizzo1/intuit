#!/usr/bin/env python3
"""
Demo script showing the progress indicator in action with the memory system.
"""
import asyncio
import os
import logging
from intuit.memory.chroma_store import ChromaMemoryStore
from intuit.memory.manager import IntuitMemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def demo_memory_system():
    """Demonstrate the memory system with progress indicators."""
    print("\n=== Memory System with Progress Indicators ===")
    
    # Create a temporary directory for the memory store
    temp_dir = ".temp_memory"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize the memory store and manager
    print("\nInitializing memory store...")
    store = ChromaMemoryStore(persist_directory=temp_dir)
    manager = IntuitMemoryManager(store=store)
    
    # Add some memories
    print("\nAdding memories...")
    await store.add_memory(
        content="The sky is blue and the clouds are white.",
        metadata={"importance": 3, "tags": ["observation", "weather"]}
    )
    await store.add_memory(
        content="Python is a versatile programming language used for web development, data analysis, AI, and more.",
        metadata={"importance": 7, "tags": ["programming", "python"]}
    )
    await store.add_memory(
        content="The tqdm library provides a fast, extensible progress bar for Python and CLI.",
        metadata={"importance": 5, "tags": ["programming", "python", "library"]}
    )
    
    # Search memories with progress indicator
    print("\nSearching memories (with progress indicator)...")
    results = await store.search_memories("python")
    print(f"Found {len(results)} memories about Python:")
    for i, memory in enumerate(results):
        print(f"{i+1}. {memory['content']}")
    
    # Process a conversation with progress indicator
    print("\nProcessing conversation (with progress indicator)...")
    conversation = [
        {"role": "user", "content": "Tell me about progress bars in Python."},
        {"role": "assistant", "content": "Progress bars in Python can be implemented using libraries like tqdm, which is popular for its ease of use and features."},
        {"role": "user", "content": "How do I install tqdm?"},
        {"role": "assistant", "content": "You can install tqdm using pip: `pip install tqdm`"}
    ]
    await manager.process_conversation(conversation)
    
    # Summarize conversation with progress indicator
    print("\nSummarizing conversation (with progress indicator)...")
    summary = await manager.summarize_conversation(conversation)
    print(f"Conversation summary: {summary}")
    
    # Clean up
    print("\nCleaning up...")
    await store.clear_memories()
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    asyncio.run(demo_memory_system())