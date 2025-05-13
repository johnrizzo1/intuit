"""
Tests for the IntuitMemoryStore class.
"""
import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# We'll need to import the IntuitMemoryStore class once it's implemented
# from intuit.memory.store import IntuitMemoryStore

# Mock the create_memory_store_manager function
@pytest.fixture
def mock_create_memory_store_manager():
    """Mock the create_memory_store_manager function."""
    with patch('intuit.memory.store.create_memory_store_manager') as mock_create:
        mock_memory = MagicMock()
        mock_create.return_value = mock_memory
        yield mock_create, mock_memory

# Test constants
TEST_PERSIST_DIR = ".test_memory"
TEST_NAMESPACE = "test_memories"
TEST_MEMORY_CONTENT = "This is a test memory"
TEST_MEMORY_METADATA = {"importance": 5, "tags": ["test", "memory"]}


@pytest.fixture
async def memory_store(mock_create_memory_store_manager):
    """Create a memory store for testing."""
    # Import here to avoid circular imports during testing
    from intuit.memory.store import IntuitMemoryStore
    
    # Unpack the mock
    mock_create, mock_memory = mock_create_memory_store_manager
    
    # Create a test store with a unique directory
    store = IntuitMemoryStore(persist_directory=TEST_PERSIST_DIR, namespace=TEST_NAMESPACE)
    
    # Configure the mock memory
    mock_memory.add = AsyncMock(return_value="test-memory-id")
    mock_memory.search = AsyncMock(return_value=[
        {"id": "memory-1", "content": "Test memory 1", "metadata": {"importance": 5}},
        {"id": "memory-2", "content": "Test memory 2", "metadata": {"importance": 3}}
    ])
    mock_memory.list = AsyncMock(return_value=[
        {"id": "memory-1", "content": "Test memory 1", "metadata": {"importance": 5}},
        {"id": "memory-2", "content": "Test memory 2", "metadata": {"importance": 3}}
    ])
    mock_memory.get = AsyncMock(return_value={"id": "test-memory-id", "content": "Test memory", "metadata": {"importance": 5}})
    mock_memory.delete = AsyncMock()
    mock_memory.clear = AsyncMock()
    mock_memory.update = AsyncMock()
    
    yield store


@pytest.mark.asyncio
async def test_init():
    """Test initialization of the memory store."""
    # Import here to avoid circular imports during testing
    from intuit.memory.store import IntuitMemoryStore
    
    # Test with default parameters
    store = IntuitMemoryStore()
    assert store.persist_directory == ".memory"
    assert store.namespace == "memories"
    assert store.store is not None
    assert store.memory is not None
    
    # Test with custom parameters
    store = IntuitMemoryStore(persist_directory=TEST_PERSIST_DIR, namespace=TEST_NAMESPACE)
    assert store.persist_directory == TEST_PERSIST_DIR
    assert store.namespace == TEST_NAMESPACE


@pytest.mark.asyncio
async def test_add_memory(memory_store):
    """Test adding a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory(TEST_MEMORY_CONTENT, TEST_MEMORY_METADATA)
    
    # Verify it was added
    assert memory_id is not None
    assert isinstance(memory_id, str)
    
    # Verify we can retrieve it
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == TEST_MEMORY_CONTENT
    assert memory["metadata"]["importance"] == TEST_MEMORY_METADATA["importance"]
    assert memory["metadata"]["tags"] == TEST_MEMORY_METADATA["tags"]


@pytest.mark.asyncio
async def test_add_memory_error(memory_store):
    """Test error handling when adding a memory."""
    # Mock the memory.add method to raise an exception
    with patch.object(memory_store.memory, 'add', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await memory_store.add_memory(TEST_MEMORY_CONTENT, TEST_MEMORY_METADATA)


@pytest.mark.asyncio
async def test_search_memories(memory_store):
    """Test searching memories."""
    # Add some test memories
    await memory_store.add_memory("Python is a programming language", {"tags": ["programming"]})
    await memory_store.add_memory("JavaScript is used for web development", {"tags": ["programming", "web"]})
    
    # Search for memories
    memories = await memory_store.search_memories("programming")
    assert len(memories) > 0
    assert any("Python" in memory["content"] for memory in memories)
    
    # Test empty query (should list all memories)
    all_memories = await memory_store.search_memories("")
    assert len(all_memories) >= 2


@pytest.mark.asyncio
async def test_search_memories_error(memory_store):
    """Test error handling when searching memories."""
    # Mock the memory.search method to raise an exception
    with patch.object(memory_store.memory, 'search', side_effect=Exception("Test error")):
        # Should return empty list on error
        memories = await memory_store.search_memories("programming")
        assert memories == []


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
    
    # Test getting a non-existent memory
    non_existent = await memory_store.get_memory("non-existent-id")
    assert non_existent is None


@pytest.mark.asyncio
async def test_get_memory_error(memory_store):
    """Test error handling when getting a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})
    
    # Mock the memory.get method to raise an exception
    with patch.object(memory_store.memory, 'get', side_effect=Exception("Test error")):
        # Should return None on error
        memory = await memory_store.get_memory(memory_id)
        assert memory is None


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
    
    # Test deleting a non-existent memory
    success = await memory_store.delete_memory("non-existent-id")
    assert success is False


@pytest.mark.asyncio
async def test_delete_memory_error(memory_store):
    """Test error handling when deleting a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})
    
    # Mock the memory.delete method to raise an exception
    with patch.object(memory_store.memory, 'delete', side_effect=Exception("Test error")):
        # Should return False on error
        success = await memory_store.delete_memory(memory_id)
        assert success is False


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


@pytest.mark.asyncio
async def test_clear_memories_error(memory_store):
    """Test error handling when clearing memories."""
    # Add some memories
    await memory_store.add_memory("Memory 1", {"importance": 1})
    
    # Mock the memory.clear method to raise an exception
    with patch.object(memory_store.memory, 'clear', side_effect=Exception("Test error")):
        # Should return False on error
        success = await memory_store.clear_memories()
        assert success is False


@pytest.mark.asyncio
async def test_update_memory(memory_store):
    """Test updating a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Original content", {"importance": 1})
    
    # Update the memory
    success = await memory_store.update_memory(
        memory_id, 
        "Updated content", 
        {"importance": 2, "tags": ["updated"]}
    )
    assert success is True
    
    # Verify it was updated
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == "Updated content"
    assert memory["metadata"]["importance"] == 2
    assert memory["metadata"]["tags"] == ["updated"]
    
    # Test updating a non-existent memory
    success = await memory_store.update_memory(
        "non-existent-id", 
        "Content", 
        {}
    )
    assert success is False


@pytest.mark.asyncio
async def test_update_memory_error(memory_store):
    """Test error handling when updating a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})
    
    # Mock the memory.update method to raise an exception
    with patch.object(memory_store.memory, 'update', side_effect=Exception("Test error")):
        # Should return False on error
        success = await memory_store.update_memory(memory_id, "Updated content", {})
        assert success is False


@pytest.mark.asyncio
async def test_redis_backend():
    """Test initialization with Redis backend."""
    # Import here to avoid circular imports during testing
    from intuit.memory.store import IntuitMemoryStore
    
    # Mock environment variable
    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
        # Mock RedisStore to avoid actual connection
        with patch('intuit.memory.store.RedisStore') as mock_redis:
            # Create a mock instance for RedisStore
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance
            
            # Initialize with Redis backend
            store = IntuitMemoryStore()
            
            # Verify Redis was used
            mock_redis.assert_called_once()
            assert store.store == mock_redis_instance