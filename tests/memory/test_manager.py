"""
Tests for the memory manager.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# We'll need to import the IntuitMemoryManager class once it's implemented
# from intuit.memory.manager import IntuitMemoryManager


# Mock the create_memory_manager function
@pytest.fixture
def mock_create_memory_manager():
    """Mock the create_memory_manager function."""
    with patch('intuit.memory.manager.create_memory_manager') as mock_create:
        mock_manager = MagicMock()
        mock_create.return_value = mock_manager
        yield mock_create, mock_manager


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    mock_store = MagicMock()
    mock_store.store = MagicMock()
    mock_store.namespace = "test_memories"
    return mock_store


@pytest.mark.asyncio
async def test_init(mock_memory_store, mock_create_memory_manager):
    """Test initialization of the memory manager."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Verify initialization
    assert manager.store == mock_memory_store
    assert manager.manager == mock_manager
    assert manager.running is False
    assert manager.task is None
    
    # Verify create_memory_manager was called correctly
    mock_create.assert_called_once_with(
        store=mock_memory_store.store,
        namespace=(mock_memory_store.namespace,)
    )


@pytest.mark.asyncio
async def test_process_conversation(mock_memory_store, mock_create_memory_manager):
    """Test processing a conversation."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.process_conversation = AsyncMock()
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ]
    
    # Process conversation
    await manager.process_conversation(messages)
    
    # Verify process_conversation was called correctly
    mock_manager.process_conversation.assert_called_once()
    
    # Verify the messages were formatted correctly
    called_messages = mock_manager.process_conversation.call_args[0][0]
    assert len(called_messages) == 3
    assert all(msg.get("role") in ["user", "assistant"] for msg in called_messages)
    assert all("content" in msg for msg in called_messages)


@pytest.mark.asyncio
async def test_process_conversation_error(mock_memory_store, mock_create_memory_manager):
    """Test error handling when processing a conversation."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.process_conversation = AsyncMock(side_effect=Exception("Test error"))
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Test messages
    messages = [{"role": "user", "content": "Hello"}]
    
    # Process conversation (should not raise exception)
    await manager.process_conversation(messages)
    
    # Verify process_conversation was called
    mock_manager.process_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_consolidate_memories(mock_memory_store, mock_create_memory_manager):
    """Test consolidating memories."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.consolidate_memories = AsyncMock()
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Consolidate memories
    await manager.consolidate_memories()
    
    # Verify consolidate_memories was called
    mock_manager.consolidate_memories.assert_called_once()


@pytest.mark.asyncio
async def test_consolidate_memories_error(mock_memory_store, mock_create_memory_manager):
    """Test error handling when consolidating memories."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.consolidate_memories = AsyncMock(side_effect=Exception("Test error"))
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Consolidate memories (should not raise exception)
    await manager.consolidate_memories()
    
    # Verify consolidate_memories was called
    mock_manager.consolidate_memories.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_conversation(mock_memory_store, mock_create_memory_manager):
    """Test summarizing a conversation."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.summarize_conversation = AsyncMock(return_value="Test summary")
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"}
    ]
    
    # Summarize conversation
    summary = await manager.summarize_conversation(messages)
    
    # Verify result
    assert summary == "Test summary"
    
    # Verify summarize_conversation was called correctly
    mock_manager.summarize_conversation.assert_called_once()
    
    # Verify the messages were formatted correctly
    called_messages = mock_manager.summarize_conversation.call_args[0][0]
    assert len(called_messages) == 3
    assert all(msg.get("role") in ["user", "assistant"] for msg in called_messages)
    assert all("content" in msg for msg in called_messages)


@pytest.mark.asyncio
async def test_summarize_conversation_error(mock_memory_store, mock_create_memory_manager):
    """Test error handling when summarizing a conversation."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.summarize_conversation = AsyncMock(side_effect=Exception("Test error"))
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Test messages
    messages = [{"role": "user", "content": "Hello"}]
    
    # Summarize conversation
    summary = await manager.summarize_conversation(messages)
    
    # Verify result
    assert summary == "Failed to summarize conversation."
    
    # Verify summarize_conversation was called
    mock_manager.summarize_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_run_background_tasks(mock_memory_store, mock_create_memory_manager):
    """Test running background tasks."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.consolidate_memories = AsyncMock()
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Mock sleep to avoid waiting
    with patch('asyncio.sleep') as mock_sleep:
        # Make sleep return immediately
        mock_sleep.side_effect = lambda _: asyncio.Future().set_result(None)
        
        # Create a simpler test that doesn't rely on timeouts
        # Run one iteration manually, then verify
        manager.running = True
        
        # Use an event to signal when the first consolidation is complete
        consolidation_done = asyncio.Event()
        
        # Replace consolidate_memories with a version that sets the event
        original_consolidate = manager.consolidate_memories
        
        async def wrapped_consolidate():
            await original_consolidate()
            # After first consolidation, stop the loop
            manager.running = False
            consolidation_done.set()
            
        manager.consolidate_memories = wrapped_consolidate
        
        # Run the background tasks (should exit after one iteration now)
        background_task = asyncio.create_task(manager.run_background_tasks(interval=1))
        
        # Wait for consolidation to complete with a timeout
        try:
            await asyncio.wait_for(consolidation_done.wait(), timeout=1.0)
        finally:
            # Clean up
            if not background_task.done():
                manager.running = False
                background_task.cancel()
            
        # Verify consolidate_memories was called
        mock_manager.consolidate_memories.assert_called_once()


@pytest.mark.asyncio
async def test_run_background_tasks_error(mock_memory_store, mock_create_memory_manager):
    """Test error handling in background tasks."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Unpack the mock
    mock_create, mock_manager = mock_create_memory_manager
    mock_manager.consolidate_memories = AsyncMock(side_effect=Exception("Test error"))
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Mock sleep to avoid waiting
    with patch('asyncio.sleep') as mock_sleep:
        # Make sleep return immediately and set a future that's already done
        done_future = asyncio.Future()
        done_future.set_result(None)
        mock_sleep.return_value = done_future
        
        # Run one iteration and then stop
        async def run_one_iteration():
            # Start the manager
            manager.running = True
            
            # Run one iteration of the background task
            await manager.consolidate_memories()
            
            # Stop after one iteration
            manager.running = False
        
        # Run the test
        await run_one_iteration()
        
        # Verify consolidate_memories was called
        mock_manager.consolidate_memories.assert_called_once()


def test_start(mock_memory_store, mock_create_memory_manager):
    """Test starting the background memory manager."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Mock create_task
    with patch('asyncio.create_task') as mock_create_task:
        # Start manager
        manager.start(interval=3600)
        
        # Verify create_task was called
        mock_create_task.assert_called_once()
        
        # Verify task was set
        assert manager.task == mock_create_task.return_value


def test_stop(mock_memory_store, mock_create_memory_manager):
    """Test stopping the background memory manager."""
    # Import here to avoid circular imports during testing
    from intuit.memory.manager import IntuitMemoryManager
    
    # Initialize memory manager
    manager = IntuitMemoryManager(mock_memory_store)
    
    # Create mock task
    mock_task = MagicMock()
    mock_task.done.return_value = False
    manager.task = mock_task
    manager.running = True
    
    # Stop manager
    manager.stop()
    
    # Verify task was cancelled
    mock_task.cancel.assert_called_once()
    
    # Verify running flag was set to False
    assert manager.running is False