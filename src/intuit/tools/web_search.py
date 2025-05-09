"""
Web search tool implementation for Intuit.
"""
import os
from typing import Dict, Any, List, Optional
import aiohttp
from pydantic import Field, PrivateAttr

from .base import BaseTool

class WebSearchTool(BaseTool):
    """Tool for searching the web."""
    
    name: str = Field(default="web_search")
    description: str = Field(
        default="Search the web for information. Input should be a search query. Use this tool when you need to find up-to-date information, news, or facts that aren't in your training data."
    )
    
    _api_key: str = PrivateAttr(default=None)
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://google.serper.dev/search")
    
    def __init__(self, **data):
        super().__init__(**data)
        self._api_key = os.getenv("SERPER_API_KEY")
        if not self._api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")
    
    async def _ensure_client(self):
        """Ensure the aiohttp client session exists."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
    
    async def _search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform a web search using Serper.dev API."""
        await self._ensure_client()
        
        headers = {
            'X-API-KEY': self._api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': max_results,
            'gl': 'us',  # Search in US
            'hl': 'en'   # Results in English
        }
        
        async with self._client.post(self._base_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            
            results = []
            
            # Process organic results
            if 'organic' in data:
                for result in data['organic'][:max_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('link', ''),
                        'snippet': result.get('snippet', '')
                    })
            
            # Process knowledge graph if available
            if 'knowledgeGraph' in data:
                kg = data['knowledgeGraph']
                results.insert(0, {
                    'title': kg.get('title', ''),
                    'url': kg.get('link', ''),
                    'snippet': kg.get('description', '')
                })
            
            return results
    
    async def _arun(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search the web.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results
        """
        try:
            results = await self._search(query, max_results)
            
            # Close the client session after we're done
            if self._client and not self._client.closed:
                await self._client.close()
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results)
            }
        except Exception as e:
            # Ensure client is closed even if there's an error
            if self._client and not self._client.closed:
                await self._client.close()
            return {'error': str(e)}
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client and not self._client.closed:
            await self._client.close() 