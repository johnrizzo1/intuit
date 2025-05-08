"""
Web search tool implementation for Intuit.
"""
import os
from typing import Dict, Any, List, Optional, Annotated
import aiohttp
from bs4 import BeautifulSoup
from pydantic import Field, PrivateAttr

from .base import BaseTool

class WebSearchTool(BaseTool):
    """Tool for searching the web."""
    
    name: str = Field(default="web_search")
    description: str = Field(
        default="Search the web for information. Input should be a search query."
    )
    
    _client: Optional[Any] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://www.google.com/search")
    
    async def _search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform a web search."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {'q': query, 'num': max_results}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self._base_url, params=params, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                results = []
                
                for result in soup.select('.g'):
                    title_elem = result.select_one('.r h3')
                    link_elem = result.select_one('.r a')
                    snippet_elem = result.select_one('.s .st')
                    
                    if title_elem and link_elem and snippet_elem:
                        results.append({
                            'title': title_elem.get_text(),
                            'url': link_elem['href'],
                            'snippet': snippet_elem.get_text()
                        })
                
                return results[:max_results]
    
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
            return {
                'query': query,
                'results': results,
                'total_results': len(results)
            }
        except Exception as e:
            return {'error': str(e)} 