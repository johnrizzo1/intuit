"""
Tests for the Filesystem tool.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import os

from intuit.tools.filesystem import FilesystemTool

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.add_document = AsyncMock()
    store.search = AsyncMock(return_value=[
        MagicMock(metadata={'path': 'test1.txt'}),
        MagicMock(metadata={'path': 'test2.txt'})
    ])
    return store

@pytest.fixture
def filesystem_tool(mock_vector_store):
    """Create a filesystem tool with a mock vector store."""
    return FilesystemTool(vector_store=mock_vector_store)

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create test files
    (data_dir / "test1.txt").write_text("Test content 1")
    (data_dir / "test2.txt").write_text("Test content 2")
    (data_dir / "test3.md").write_text("Test content 3")
    
    # Create subdirectory
    subdir = data_dir / "subdir"
    subdir.mkdir()
    (subdir / "test4.txt").write_text("Test content 4")
    
    return data_dir

@pytest.mark.asyncio
async def test_filesystem_tool_initialization(filesystem_tool, mock_vector_store):
    """Test that the filesystem tool initializes correctly."""
    assert filesystem_tool.name == "filesystem"
    assert "filesystem" in filesystem_tool.description.lower()
    assert filesystem_tool._vector_store == mock_vector_store

@pytest.mark.asyncio
async def test_get_file_info(filesystem_tool, test_data_dir):
    """Test the _get_file_info method."""
    test_file = test_data_dir / "test1.txt"
    
    # Test getting file info
    info = await filesystem_tool._get_file_info(str(test_file))
    
    # Verify the info
    assert info["name"] == "test1.txt"
    assert info["path"] == str(test_file.absolute())
    assert info["size"] > 0
    assert "created" in info
    assert "modified" in info
    assert info["type"] == "text/plain"
    assert info["is_file"] is True
    assert info["is_dir"] is False

@pytest.mark.asyncio
async def test_list_directory(filesystem_tool, test_data_dir):
    """Test the _list_directory method."""
    # Test listing directory
    files = await filesystem_tool._list_directory(str(test_data_dir))
    
    # Verify the results
    assert len(files) == 4
    file_names = {f["name"] for f in files}
    assert "test1.txt" in file_names
    assert "test2.txt" in file_names
    assert "test3.md" in file_names
    assert "subdir" in file_names

@pytest.mark.asyncio
async def test_list_directory_recursive(filesystem_tool, test_data_dir):
    """Test the _list_directory method with recursive=True."""
    # Test recursive listing
    files = await filesystem_tool._list_directory(str(test_data_dir), recursive=True)
    
    # Verify the results
    assert len(files) == 5  # 3 files in root + 1 subdir + 1 file in subdir
    file_names = {f["name"] for f in files}
    assert "test1.txt" in file_names
    assert "test2.txt" in file_names
    assert "test3.md" in file_names
    assert "test4.txt" in file_names
    assert "subdir" in file_names

@pytest.mark.asyncio
async def test_read_file(filesystem_tool, test_data_dir):
    """Test the _read_file method."""
    test_file = test_data_dir / "test1.txt"
    
    # Test reading file
    content = await filesystem_tool._read_file(str(test_file))
    
    # Verify the content
    assert content == "Test content 1"

@pytest.mark.asyncio
async def test_write_file(filesystem_tool, test_data_dir):
    """Test the _write_file method."""
    test_file = test_data_dir / "test5.txt"
    content = "Test content 5"
    
    # Test writing file
    info = await filesystem_tool._write_file(str(test_file), content)
    
    # Verify the file was written
    assert test_file.exists()
    assert test_file.read_text() == content
    assert info["name"] == "test5.txt"
    assert info["size"] > 0

@pytest.mark.asyncio
async def test_search_files(filesystem_tool, mock_vector_store, test_data_dir):
    """Test the _search_files method."""
    # Update mock to return paths from test_data_dir
    mock_vector_store.search.return_value = [
        MagicMock(metadata={'path': str(test_data_dir / "test1.txt")}),
        MagicMock(metadata={'path': str(test_data_dir / "test2.txt")})
    ]
    
    # Test searching files
    results = await filesystem_tool._search_files("test query")
    
    # Verify the results
    assert len(results) == 2
    assert results[0]["name"] == "test1.txt"
    assert results[1]["name"] == "test2.txt"

@pytest.mark.asyncio
async def test_run_list(filesystem_tool, test_data_dir):
    """Test the run method with list action."""
    result = await filesystem_tool.run("list", path=str(test_data_dir))
    
    # Verify the result
    assert result["status"] == "success"
    assert len(result["contents"]) == 4
    file_names = {f["name"] for f in result["contents"]}
    assert "test1.txt" in file_names
    assert "test2.txt" in file_names
    assert "test3.md" in file_names

@pytest.mark.asyncio
async def test_run_read(filesystem_tool, test_data_dir):
    """Test the run method with read action."""
    result = await filesystem_tool.run("read", path=str(test_data_dir / "test1.txt"))
    
    # Verify the result
    assert result["status"] == "success"
    assert result["content"] == "Test content 1"

@pytest.mark.asyncio
async def test_run_write(filesystem_tool, test_data_dir):
    """Test the run method with write action."""
    test_file = test_data_dir / "test6.txt"
    content = "Test content 6"
    
    result = await filesystem_tool.run("write", path=str(test_file), content=content)
    
    # Verify the result
    assert result["status"] == "success"
    assert "Successfully wrote to" in result["message"]
    assert result["file"]["name"] == "test6.txt"

@pytest.mark.asyncio
async def test_run_info(filesystem_tool, test_data_dir):
    """Test the run method with info action."""
    result = await filesystem_tool.run("info", path=str(test_data_dir / "test1.txt"))
    
    # Verify the result
    assert result["status"] == "success"
    assert result["file"]["name"] == "test1.txt"
    assert result["file"]["type"] == "text/plain"

@pytest.mark.asyncio
async def test_run_search(filesystem_tool, mock_vector_store, test_data_dir):
    """Test the run method with search action."""
    # Update mock to return paths from test_data_dir
    mock_vector_store.search.return_value = [
        MagicMock(metadata={'path': str(test_data_dir / "test1.txt")}),
        MagicMock(metadata={'path': str(test_data_dir / "test2.txt")})
    ]
    
    result = await filesystem_tool.run("search", path=str(test_data_dir), query="test query")
    
    # Verify the result
    assert result["status"] == "success"
    assert len(result["results"]) == 2
    assert result["results"][0]["name"] == "test1.txt"

@pytest.mark.asyncio
async def test_run_error_handling(filesystem_tool):
    """Test error handling in the run method."""
    # Test with invalid action
    result = await filesystem_tool.run("invalid_action", path="test_path")
    
    # Verify error handling
    assert result["status"] == "error"
    assert "Invalid action" in result["message"] 