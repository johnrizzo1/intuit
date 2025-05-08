"""
Vector store implementation for indexing filesystem content.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import tiktoken
from pydantic import BaseModel, Field

from .document import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store implementation using ChromaDB for indexing and searching filesystem content.
    """
    def __init__(
        self,
        persist_directory: str = ".chroma",
        collection_name: str = "filesystem",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        logger.info(f"Initializing VectorStore with directory: {persist_directory}")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info("VectorStore initialized successfully")
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size tokens.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    async def index_file(self, file_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Index a file's content in the vector store.
        
        Args:
            file_path: Path to the file to index
            metadata: Additional metadata about the file
        """
        logger.info(f"Indexing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Skipping binary file: {file_path}")
            return
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return
        
        # Create base metadata
        base_metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix,
            "last_modified": os.path.getmtime(file_path),
        }
        
        # Update with provided metadata
        if metadata:
            base_metadata.update(metadata)
        
        # Chunk the content
        chunks = self._chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks for {file_path}")
        
        # Add chunks to collection
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path}_{i}"
            try:
                # Use synchronous add method
                self.collection.add(
                    documents=[chunk],
                    metadatas=[base_metadata],
                    ids=[chunk_id]
                )
            except Exception as e:
                logger.error(f"Error adding chunk {i} from {file_path}: {e}")

    async def index_directory(self, directory: Path, exclude_patterns: Optional[List[str]] = None) -> None:
        """
        Index all files in a directory recursively.
        
        Args:
            directory: Directory to index
            exclude_patterns: List of glob patterns to exclude
        """
        logger.info(f"Indexing directory: {directory}")
        exclude_patterns = exclude_patterns or []
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    # Skip excluded files
                    if any(file_path.match(pattern) for pattern in exclude_patterns):
                        logger.debug(f"Skipping excluded file: {file_path}")
                        continue
                        
                    # Skip hidden files and directories
                    if any(part.startswith('.') for part in file_path.parts):
                        logger.debug(f"Skipping hidden file: {file_path}")
                        continue
                        
                    await self.index_file(file_path)
            logger.info(f"Finished indexing directory: {directory}")
        except Exception as e:
            logger.error(f"Error indexing directory {directory}: {e}")

    async def search(self, query: str, n_results: int = 5) -> List[Document]:
        """
        Search the vector store for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Searching for: {query}")
        try:
            # Use synchronous query method
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                doc = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    id=results['ids'][0][i]
                )
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} results")
            return documents
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    async def delete_document(self, document_id: str) -> None:
        """Delete a document from the collection."""
        logger.info(f"Deleting document: {document_id}")
        try:
            # Use synchronous delete method
            self.collection.delete(ids=[document_id])
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")

    async def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.info("Clearing all documents")
        try:
            # Use synchronous delete method
            self.collection.delete(where={})
        except Exception as e:
            logger.error(f"Error clearing documents: {e}") 