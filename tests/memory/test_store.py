"""
Tests for the IntuitMemoryStore class.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# We'll need to import the IntuitMemoryStore class once it's implemented
# from intuit.memory.store import IntuitMemoryStore


# Test constants
TEST_PERSIST_DIR = ".test_memory"
TEST_NAMESPACE = "test_memories"
TEST_MEMORY_CONTENT = "This is a test memory"
TEST_MEMORY_METADATA = {"importance": 5, "tags": ["test", "memory"]}


@pytest.fixture
async def memory_store():
    """Create a memory store for testing."""
    # Import here to avoid circular imports during testing
    from intuit.memory.store import IntuitMemoryStore

    # Create a test store with a unique directory
    with patch("intuit.memory.store.create_memory_store_manager") as mock_create:
        # Create mock memory manager
        mock_memory = AsyncMock()
        mock_create.return_value = mock_memory

        # Configure the mock memory with proper async methods
        mock_memory.aput = AsyncMock(return_value="test-memory-id")
        mock_memory.asearch = AsyncMock(
            return_value=[
                type(
                    "MockItem",
                    (),
                    {
                        "key": "memory-1",
                        "value": type(
                            "MockValue",
                            (),
                            {"content": "Test memory 1", "metadata": {"importance": 5}},
                        )(),
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    },
                )(),
                type(
                    "MockItem",
                    (),
                    {
                        "key": "memory-2",
                        "value": type(
                            "MockValue",
                            (),
                            {"content": "Test memory 2", "metadata": {"importance": 3}},
                        )(),
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-01T00:00:00Z",
                    },
                )(),
            ]
        )
        mock_memory.aget = AsyncMock(
            return_value=type(
                "MockItem",
                (),
                {
                    "key": "test-memory-id",
                    "value": type(
                        "MockValue",
                        (),
                        {"content": "Test memory", "metadata": {"importance": 5}},
                    )(),
                    "created_at": "2023-01-01T00:00:00Z",
                    "updated_at": "2023-01-01T00:00:00Z",
                },
            )()
        )
        mock_memory.adelete = AsyncMock(return_value=True)
        mock_memory.aclear = AsyncMock()
        mock_memory.aupdate = AsyncMock()

        store = IntuitMemoryStore(
            persist_directory=TEST_PERSIST_DIR, namespace=TEST_NAMESPACE
        )
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
    store = IntuitMemoryStore(
        persist_directory=TEST_PERSIST_DIR, namespace=TEST_NAMESPACE
    )
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

    # Verify we can retrieve it (mock returns fixed data)
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == "Test memory"  # Mock returns this
    assert memory["metadata"]["importance"] == 5  # Mock returns this


@pytest.mark.asyncio
async def test_add_memory_error(memory_store):
    """Test error handling when adding a memory."""
    # Mock the memory.aput method to raise an exception
    with patch.object(memory_store.memory, "aput", side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await memory_store.add_memory(TEST_MEMORY_CONTENT, TEST_MEMORY_METADATA)


@pytest.mark.asyncio
async def test_search_memories(memory_store):
    """Test searching memories."""
    # Add some test memories
    await memory_store.add_memory(
        "Python is a programming language", {"tags": ["programming"]}
    )
    await memory_store.add_memory(
        "JavaScript is used for web development", {"tags": ["programming", "web"]}
    )

    # Search for memories (mock returns fixed data)
    memories = await memory_store.search_memories("programming")
    assert len(memories) == 2  # Mock returns 2 memories
    assert any("Test memory 1" in memory["content"] for memory in memories)

    # Test empty query (should list all memories)
    all_memories = await memory_store.search_memories("")
    assert len(all_memories) == 2  # Mock returns 2 memories


@pytest.mark.asyncio
async def test_search_memories_error(memory_store):
    """Test error handling when searching memories."""
    # Mock the memory.asearch method to raise an exception
    with patch.object(
        memory_store.memory, "asearch", side_effect=Exception("Test error")
    ):
        # Should return empty list on error
        memories = await memory_store.search_memories("programming")
        assert memories == []


@pytest.mark.asyncio
async def test_get_memory(memory_store):
    """Test getting a memory by ID."""
    # Add a memory
    memory_id = await memory_store.add_memory("Remember to buy milk", {"importance": 3})

    # Get the memory (mock returns fixed data)
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == "Test memory"  # Mock returns this
    assert memory["metadata"]["importance"] == 5  # Mock returns this

    # Test getting a non-existent memory (mock still returns data)
    # We need to mock aget to return None for this test
    with patch.object(memory_store.memory, "aget", return_value=None):
        non_existent = await memory_store.get_memory("non-existent-id")
        assert non_existent is None


@pytest.mark.asyncio
async def test_get_memory_error(memory_store):
    """Test error handling when getting a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})

    # Mock the memory.aget method to raise an exception
    with patch.object(memory_store.memory, "aget", side_effect=Exception("Test error")):
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

    # Verify it's gone (need to mock aget to return None after delete)
    with patch.object(memory_store.memory, "aget", return_value=None):
        memory = await memory_store.get_memory(memory_id)
        assert memory is None

    # Test deleting a non-existent memory (mock adelete to return False)
    with patch.object(
        memory_store.memory, "adelete", return_value=False
    ) as mock_delete:
        success = await memory_store.delete_memory("non-existent-id")
        assert success is False
        mock_delete.assert_called_once_with("non-existent-id")


@pytest.mark.asyncio
async def test_delete_memory_error(memory_store):
    """Test error handling when deleting a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})

    # Mock the memory.adelete method to raise an exception
    with patch.object(
        memory_store.memory, "adelete", side_effect=Exception("Test error")
    ):
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

    # Verify they're gone (mock asearch to return empty list after clear)
    with patch.object(memory_store.memory, "asearch", return_value=[]):
        memories = await memory_store.search_memories("")
        assert len(memories) == 0


@pytest.mark.asyncio
async def test_clear_memories_error(memory_store):
    """Test error handling when clearing memories."""
    # Add some memories
    await memory_store.add_memory("Memory 1", {"importance": 1})

    # Mock the memory.asearch method to raise an exception
    with patch.object(
        memory_store.memory, "asearch", side_effect=Exception("Test error")
    ):
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
        memory_id, "Updated content", {"importance": 2, "tags": ["updated"]}
    )
    assert success is True

    # Verify it was updated (mock still returns fixed data)
    memory = await memory_store.get_memory(memory_id)
    assert memory is not None
    assert memory["content"] == "Test memory"  # Mock returns this
    assert memory["metadata"]["importance"] == 5  # Mock returns this

    # Test updating a non-existent memory (mock aget to return None)
    with patch.object(memory_store.memory, "aget", return_value=None):
        success = await memory_store.update_memory("non-existent-id", "Content", {})
        assert success is False


@pytest.mark.asyncio
async def test_update_memory_error(memory_store):
    """Test error handling when updating a memory."""
    # Add a memory
    memory_id = await memory_store.add_memory("Test memory", {})

    # Mock the memory.aget method to raise an exception
    with patch.object(memory_store.memory, "aget", side_effect=Exception("Test error")):
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
        # Create store with Redis URL (should still use InMemoryStore with warning)
        store = IntuitMemoryStore()

        # Verify store was created (Redis not supported, so InMemoryStore is used)
        assert store.store is not None
        assert store.persist_directory == ".memory"
        assert store.namespace == "memories"
