"""
Filesystem tool implementation for Intuit.
"""
import os
import mimetypes
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import Field, PrivateAttr

from intuit.tools.basetool import BaseTool

# Set up logging
logger = logging.getLogger(__name__)

class FilesystemTool(BaseTool):
    """Tool for interacting with the filesystem."""
    
    name: str = Field(default="filesystem")
    description: str = Field(
        default=(
            "Tool for filesystem operations like listing directories, reading files, and searching content. "
            "Use the 'search' action to find files by their content using semantic search. "
            "Available actions:\n"
            "- search: Find files by content (requires: path, query)\n"
            "- list: List directory contents (requires: path)\n"
            "- read: Read file contents (requires: path)\n"
            "- write: Write content to file (requires: path, content)\n"
            "- info: Get file information (requires: path)"
        )
    )
    
    _vector_store: Optional[Any] = PrivateAttr(default=None)
    
    def __init__(self, vector_store=None, **data):
        super().__init__(**data)
        self._vector_store = vector_store
        logger.info("FilesystemTool initialized with vector store: %s", bool(vector_store))
    
    async def _get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            'name': file_path.name,
            'path': str(file_path.absolute()),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
            'type': mime_type or 'application/octet-stream'
        }
    
    async def _list_directory(self, path: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """List contents of a directory."""
        dir_path = Path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        contents = []
        if recursive:
            for item in dir_path.rglob('*'):
                contents.append(await self._get_file_info(str(item)))
        else:
            for item in dir_path.iterdir():
                contents.append(await self._get_file_info(str(item)))
        
        return contents
    
    async def _read_file(self, path: str) -> str:
        """Read contents of a file."""
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        
        return file_path.read_text()
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        file_path = Path(path)
        file_path.write_text(content)
        
        # Update vector store if available
        if self._vector_store:
            await self._vector_store.add_document(str(file_path), content)
        
        return await self._get_file_info(path)
    
    async def _search_files(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for files using vector store or basic file system search."""
        logger.info("Searching files with query: %s", query)
        
        if self._vector_store:
            # Use semantic search if vector store is available
            try:
                semantic_results = await self._vector_store.search(query, limit=limit)
                logger.info("Found %d results using semantic search", len(semantic_results))
                semantic_file_info: List[Dict[str, Any]] = []
                for result in semantic_results:
                    info: Dict[str, Any] = await self._get_file_info(result.metadata['path'])
                    semantic_file_info.append(info)
                return semantic_file_info
            except Exception as e:
                logger.error("Error in semantic search: %s", str(e))
                # Fall back to basic search if semantic search fails
                logger.info("Falling back to basic search")
        
        # Basic file system search
        try:
            # Search in current directory and subdirectories
            path = Path(".")
            basic_results: List[Dict[str, Any]] = []
            
            # Handle empty query or wildcard query to list all files
            if not query or query == '*':
                # List all files in the directory
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_info: Dict[str, Any] = await self._get_file_info(str(file_path))
                            basic_results.append(file_info)
                            
                            if len(basic_results) >= limit:
                                break
                        except Exception as e:
                            logger.warning("Error processing file %s: %s", file_path, str(e))
                            continue
            else:
                # Convert query to lowercase for case-insensitive search
                query_lower = query.lower()
                
                # Search through all files
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            # Check if query is in filename
                            if query_lower in file_path.name.lower():
                                filename_info: Dict[str, Any] = await self._get_file_info(str(file_path))
                                basic_results.append(filename_info)
                                continue
                                
                            # Check if query is in file content
                            try:
                                content = file_path.read_text().lower()
                                if query_lower in content:
                                    content_info: Dict[str, Any] = await self._get_file_info(str(file_path))
                                    basic_results.append(content_info)
                            except UnicodeDecodeError:
                                # Skip binary files
                                continue
                                
                            if len(basic_results) >= limit:
                                break
                        except Exception as e:
                            logger.warning("Error processing file %s: %s", file_path, str(e))
                            continue
            
            logger.info("Found %d results using basic search", len(basic_results))
            return basic_results
        except Exception as e:
            logger.error("Error in basic search: %s", str(e))
            raise
    
    async def run(
        self,
        action: str,
        path: str = ".",  # Default to current directory
        content: Optional[str] = None,
        recursive: bool = False,
        query: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Execute filesystem operations.
        
        Args:
            action: Operation to perform (list, read, write, info, search)
            path: File or directory path (defaults to current directory)
            content: Content to write (for write action)
            recursive: List directory recursively
            query: Search query (for search action, use empty string or '*' to list all indexed files)
            limit: Maximum number of search results
            
        Returns:
            Dict containing operation results
        """
        logger.info(f"FilesystemTool.run called with action={action}, path={path}, query={query}, limit={limit}")
        logger.info("Running filesystem operation: %s", action)
        try:
            if action == "list":
                contents = await self._list_directory(path, recursive)
                return {
                    "status": "success",
                    "message": f"Listed {len(contents)} items in {path}",
                    "contents": contents
                }
            elif action == "read":
                content = await self._read_file(path)
                return {
                    "status": "success",
                    "message": f"Successfully read {path}",
                    "content": content
                }
            elif action == "write":
                if not content:
                    raise ValueError("Content required for write action")
                file_info = await self._write_file(path, content)
                return {
                    "status": "success",
                    "message": f"Successfully wrote to {path}",
                    "file": file_info
                }
            elif action == "info":
                file_info = await self._get_file_info(path)
                return {
                    "status": "success",
                    "message": f"Got info for {path}",
                    "file": file_info
                }
            elif action == "search":
                # Allow empty query to list all indexed files
                query_str = query if query else ""
                logger.info(f"FilesystemTool.run: Executing search with query='{query_str}', limit={limit}")
                
                # Check if vector store is available
                if self._vector_store:
                    logger.info("FilesystemTool.run: Vector store is available for search")
                else:
                    logger.warning("FilesystemTool.run: Vector store is NOT available for search")
                
                results = await self._search_files(query_str, limit)
                logger.info(f"FilesystemTool.run: Search returned {len(results)} results")
                
                # Customize message based on query
                if not query or query == '*':
                    message = f"Found {len(results)} indexed files"
                else:
                    message = f"Found {len(results)} results for '{query}'"
                
                # Log the results for debugging
                for i, result in enumerate(results):
                    logger.info(f"FilesystemTool.run: Result {i+1}: {result.get('name', 'Unknown')} - {result.get('path', 'Unknown path')}")
                    
                return {
                    "status": "success",
                    "message": message,
                    "results": results
                }
            else:
                raise ValueError(f"Invalid action: {action}")
        except Exception as e:
            logger.error("Error in filesystem operation: %s", str(e))
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _arun(self, **kwargs) -> Any:
        """Run the tool asynchronously."""
        return await self.run(**kwargs) 