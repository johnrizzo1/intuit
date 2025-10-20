import asyncio
from intuit.memory.chroma_store import ChromaMemoryStore

async def main():
    # Initialize memory store
    memory_store = ChromaMemoryStore(model="ollama:nomic-embed-text")
    print("Memory store initialized")
    
    # Add a memory
    memory_id = await memory_store.add_memory("This is a test memory", {"importance": 8, "tags": ["test", "memory"]})
    print(f"Added memory with ID: {memory_id}")
    
    # Search for the memory
    memories = await memory_store.search_memories("test")
    print(f"Found {len(memories)} memories matching 'test':")
    for memory in memories:
        print(f"- ID: {memory.get('id')}")
        print(f"  Content: {memory.get('content')}")
        print(f"  Metadata: {memory.get('metadata')}")
    
    # Get the memory by ID
    memory = await memory_store.get_memory(memory_id)
    print(f"\nGot memory by ID {memory_id}:")
    if memory:
        print(f"- Content: {memory.get('content')}")
        print(f"  Metadata: {memory.get('metadata')}")
    else:
        print("Memory not found")

if __name__ == "__main__":
    asyncio.run(main())