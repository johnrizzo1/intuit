"""
Tests for the vector store indexer.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
import os

from intuit.vector_store.indexer import VectorStore
from intuit.vector_store.document import Document

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    with patch('chromadb.PersistentClient') as mock_client, \
         patch('chromadb.utils.embedding_functions.OpenAIEmbeddingFunction') as mock_embedding:
        # Set up mock collection
        mock_collection = MagicMock()
        mock_collection.add = AsyncMock()
        mock_collection.query = AsyncMock(return_value={
            'documents': [['Test result 1', 'Test result 2']],
            'metadatas': [[{'source': 'test1.txt'}, {'source': 'test2.txt'}]],
            'distances': [0.1, 0.2],
            'ids': [['id1', 'id2']]
        })
        mock_collection.delete = AsyncMock()
        
        # Set up mock client
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Create vector store with mocked dependencies
        store = VectorStore()
        store._chunk_text = MagicMock(return_value=['Test chunk'])
        return store

@pytest.mark.asyncio
async def test_vector_store_initialization(mock_vector_store):
    """Test that the vector store initializes correctly."""
    assert mock_vector_store.persist_directory == ".chroma"
    assert mock_vector_store.collection_name == "filesystem"
    assert mock_vector_store.chunk_size == 1000
    assert mock_vector_store.chunk_overlap == 200

@pytest.mark.asyncio
async def test_index_file(mock_vector_store, test_data_dir):
    """Test indexing a single file."""
    test_file = test_data_dir / "test1.txt"
    
    # Create test file
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("Test content")
    
    # Test indexing the file
    await mock_vector_store.index_file(test_file)
    
    # Verify the collection's add method was called
    mock_vector_store.collection.add.assert_called_once()

@pytest.mark.asyncio
async def test_index_directory(mock_vector_store, test_data_dir):
    """Test indexing a directory of files."""
    # Create test files
    test_data_dir.mkdir(parents=True, exist_ok=True)
    (test_data_dir / "test1.txt").write_text("Test content 1")
    (test_data_dir / "test2.txt").write_text("Test content 2")
    
    # Test indexing the directory
    await mock_vector_store.index_directory(test_data_dir)
    
    # Verify the collection's add method was called for each file
    assert mock_vector_store.collection.add.call_count > 0

@pytest.mark.asyncio
async def test_search(mock_vector_store):
    """Test searching the vector store."""
    # Test searching
    results = await mock_vector_store.search("test query", n_results=2)
    
    # Verify the results
    assert len(results) == 2
    assert results[0].content == "Test result 1"
    assert results[1].content == "Test result 2"

@pytest.mark.asyncio
async def test_delete_document(mock_vector_store):
    """Test deleting a document from the vector store."""
    # Test deleting a document
    await mock_vector_store.delete_document("test1.txt")
    
    # Verify the collection's delete method was called
    mock_vector_store.collection.delete.assert_called_once_with(ids=["test1.txt"])

@pytest.mark.asyncio
async def test_clear(mock_vector_store):
    """Test clearing the vector store."""
    # Test clearing the store
    await mock_vector_store.clear()
    
    # Verify the collection's delete method was called
    mock_vector_store.collection.delete.assert_called_once_with(where={})

@pytest.mark.asyncio
async def test_error_handling(mock_vector_store, test_data_dir):
    """Test error handling in the vector store."""
    # Create test file
    test_file = test_data_dir / "test1.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("Test content")
    
    # Mock an error
    mock_vector_store.collection.add.side_effect = Exception("Indexing error")
    
    # Test indexing with error
    with pytest.raises(Exception) as exc_info:
        await mock_vector_store.index_file(test_file)
    
    # Verify error message
    assert "Indexing error" in str(exc_info.value) 