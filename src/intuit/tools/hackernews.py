"""
Hacker News tool implementation for Intuit.
"""
import aiohttp
from typing import Dict, Any, List, Optional
from pydantic import Field, PrivateAttr

from .basetool import BaseTool

class HackerNewsTool(BaseTool):
    """Tool for fetching content from Hacker News."""
    
    name: str = Field(default="hackernews")
    description: str = Field(
        default="Fetch content from Hacker News. You can get top stories, new stories, or details about a specific story."
    )
    
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="https://hacker-news.firebaseio.com/v0")
    
    async def _ensure_client(self):
        """Ensure the aiohttp client session exists."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
    
    async def _get_story_ids(self, story_type: str = "top", limit: int = 10) -> List[int]:
        """Get story IDs based on type (top, new, best)."""
        await self._ensure_client()
        
        endpoint = f"{self._base_url}/{story_type}stories.json"
        async with self._client.get(endpoint) as response:
            response.raise_for_status()
            story_ids = await response.json()
            return story_ids[:limit]  # Return only the requested number of stories
    
    async def _get_item(self, item_id: int) -> Dict[str, Any]:
        """Get details for a specific item (story, comment, etc.)."""
        await self._ensure_client()
        
        endpoint = f"{self._base_url}/item/{item_id}.json"
        async with self._client.get(endpoint) as response:
            response.raise_for_status()
            return await response.json()
    
    async def _get_stories(self, story_type: str = "top", limit: int = 10) -> List[Dict[str, Any]]:
        """Get a list of stories with details."""
        story_ids = await self._get_story_ids(story_type, limit)
        stories = []
        
        for story_id in story_ids:
            story = await self._get_item(story_id)
            if story and story.get('type') == 'story':  # Ensure it's a story
                stories.append({
                    'id': story.get('id'),
                    'title': story.get('title', 'No title'),
                    'url': story.get('url', f"https://news.ycombinator.com/item?id={story.get('id')}"),
                    'score': story.get('score', 0),
                    'by': story.get('by', 'unknown'),
                    'time': story.get('time', 0),
                    'descendants': story.get('descendants', 0)  # Number of comments
                })
        
        return stories
    
    async def _arun(
        self,
        action: str = "top",
        limit: int = 10,
        item_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch content from Hacker News.
        
        Args:
            action: The type of stories to fetch ("top", "new", "best") or "item" for a specific story
            limit: Maximum number of stories to return
            item_id: ID of a specific item to fetch (only used when action is "item")
            
        Returns:
            Dict containing the requested Hacker News content
        """
        try:
            if action == "item" and item_id:
                # Get a specific item
                item = await self._get_item(item_id)
                result = {
                    'action': 'item',
                    'item': item
                }
            else:
                # Get stories based on type
                story_type = action if action in ["top", "new", "best"] else "top"
                stories = await self._get_stories(story_type, limit)
                result = {
                    'action': story_type,
                    'stories': stories,
                    'count': len(stories)
                }
            
            # Close the client session after we're done
            if self._client and not self._client.closed:
                await self._client.close()
            
            return result
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

    # Convenience methods for direct use
    async def get_top_stories(self, limit: int = 10) -> Dict[str, Any]:
        """Get top stories from Hacker News."""
        return await self._arun(action="top", limit=limit)
    
    async def get_new_stories(self, limit: int = 10) -> Dict[str, Any]:
        """Get new stories from Hacker News."""
        return await self._arun(action="new", limit=limit)
    
    async def get_best_stories(self, limit: int = 10) -> Dict[str, Any]:
        """Get best stories from Hacker News."""
        return await self._arun(action="best", limit=limit)
    
    async def get_story(self, item_id: int) -> Dict[str, Any]:
        """Get a specific story by ID."""
        return await self._arun(action="item", item_id=item_id)