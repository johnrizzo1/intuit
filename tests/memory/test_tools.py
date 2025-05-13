"""
Tests for memory tools.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain.tools import BaseTool


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    mock_store = MagicMock()
    
    # Set up async methods as AsyncMock
    mock_store.add_memory = AsyncMock()
    mock_store.search_memories = AsyncMock()
    mock_store.get_memory = AsyncMock()
    mock_store.delete_memory = AsyncMock()
    
    # Configure return values
    mock_store.add_memory.return_value = "test-memory-id"
    mock_store.search_memories.return_value = [
        {"id": "memory-1", "content": "Test memory 1", "metadata": {"importance": 5}},
        {"id": "memory-2", "content": "Test memory 2", "metadata": {"importance": 3}}
    ]
    mock_store.get_memory.return_value = {"id": "test-memory-id", "content": "Test memory", "metadata": {"importance": 5}}
    mock_store.delete_memory.return_value = True
    
    return mock_store


def test_get_memory_tools(mock_memory_store):
    """Test getting memory tools."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import get_memory_tools
    
    # Get tools
    tools = get_memory_tools(mock_memory_store)
    
    # Verify tools
    assert len(tools) == 4
    assert all(isinstance(tool, BaseTool) for tool in tools)
    
    # Verify tool names
    tool_names = [tool.name for tool in tools]
    assert "add_memory" in tool_names
    assert "search_memory" in tool_names
    assert "get_memory" in tool_names
    assert "delete_memory" in tool_names


def test_add_memory_tool(mock_memory_store):
    """Test the add_memory tool."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import MemoryAddTool
    
    # Create tool
    tool = MemoryAddTool(mock_memory_store)
    
    # Test sync run
    result = tool._run(
        content="Test memory",
        importance=5,
        tags=["test", "memory"]
    )
    
    # Verify result
    assert "I've remembered that Test memory" in result
    assert "test-memory-id" in result
    
    # Verify store was called correctly
    mock_memory_store.add_memory.assert_called_once_with(
        content="Test memory",
        metadata={"importance": 5, "tags": ["test", "memory"]}
    )
    
    # Test sync run
    with patch('asyncio.run') as mock_run:
        mock_run.return_value = "Mocked result"
        result = tool._run(
            content="Test memory",
            importance=5,
            tags=["test", "memory"]
        )
        assert result == "Mocked result"
        mock_run.assert_called_once()


def test_search_memory_tool(mock_memory_store):
    """Test the search_memory tool."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import MemorySearchTool
    
    # Create tool
    tool = MemorySearchTool(mock_memory_store)
    
    # Test sync run
    result = tool._run(query="test", limit=2)
    
    # Verify result
    assert "Here's what I remember:" in result
    assert "Test memory 1" in result
    assert "Test memory 2" in result
    
    # Verify store was called correctly
    mock_memory_store.search_memories.assert_called_once_with("test", 2)
    
    # Test with no results
    mock_memory_store.search_memories.return_value = []
    result = tool._run(query="nonexistent", limit=2)
    assert "I don't have any memories related to that" in result


def test_get_memory_tool(mock_memory_store):
    """Test the get_memory tool."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import MemoryGetTool
    
    # Create tool
    tool = MemoryGetTool(mock_memory_store)
    
    # Test sync run
    result = tool._run(memory_id="test-memory-id")
    
    # Verify result
    assert "Memory test-memory-id: Test memory" in result
    
    # Verify store was called correctly
    mock_memory_store.get_memory.assert_called_once_with("test-memory-id")
    
    # Test with non-existent memory
    mock_memory_store.get_memory.return_value = None
    result = tool._run(memory_id="nonexistent-id")
    assert "I couldn't find a memory with ID nonexistent-id" in result


def test_delete_memory_tool(mock_memory_store):
    """Test the delete_memory tool."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import MemoryDeleteTool
    
    # Create tool
    tool = MemoryDeleteTool(mock_memory_store)
    
    # Test sync run
    result = tool._run(memory_id="test-memory-id")
    
    # Verify result
    assert "I've forgotten the memory with ID test-memory-id" in result
    
    # Verify store was called correctly
    mock_memory_store.delete_memory.assert_called_once_with("test-memory-id")
    
    # Test with failure
    mock_memory_store.delete_memory.return_value = False
    result = tool._run(memory_id="nonexistent-id")
    assert "I couldn't delete the memory with ID nonexistent-id" in result


def test_get_langmem_tools():
    """Test getting LangMem's built-in tools."""
    # Import from mock_tools to avoid dependency on langmem
    from .mock_tools import get_langmem_tools
    
    # Get tools
    mock_store = MagicMock()
    tools = get_langmem_tools(mock_store)
    
    # Verify tools
    assert len(tools) == 2
    
    # Verify tool names
    assert tools[0].name == "manage_memory"
    assert tools[1].name == "search_memory"