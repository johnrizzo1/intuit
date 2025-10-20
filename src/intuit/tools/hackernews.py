"""
Hacker News tool implementation for Intuit.
"""
import aiohttp
import webbrowser
from typing import Dict, Any, List, Optional
from pydantic import Field, PrivateAttr

from .basetool import BaseTool


class HackerNewsTool(BaseTool):
    """Tool for fetching content from Hacker News."""
    
    name: str = Field(default="hackernews")
    description: str = Field(
        default=(
            "Fetch content from Hacker News. You can get top stories, "
            "new stories, details about a specific story, or open a "
            "story in your web browser."
        )
    )
    
    _client: Optional[aiohttp.ClientSession] = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(
        default="https://hacker-news.firebaseio.com/v0"
    )
    _stories_cache: List[Dict[str, Any]] = PrivateAttr(
        default_factory=list
    )
    
    async def _ensure_client(self):
        """Ensure the aiohttp client session exists."""
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
    
    async def _get_story_ids(
        self, story_type: str = "top", limit: int = 10
    ) -> List[int]:
        """Get story IDs based on type (top, new, best)."""
        await self._ensure_client()
        
        endpoint = f"{self._base_url}/{story_type}stories.json"
        async with self._client.get(endpoint) as response:
            response.raise_for_status()
            story_ids = await response.json()
            # Return only the requested number of stories
            return story_ids[:limit]
    
    async def _get_item(self, item_id: int) -> Dict[str, Any]:
        """Get details for a specific item (story, comment, etc.)."""
        await self._ensure_client()
        
        endpoint = f"{self._base_url}/item/{item_id}.json"
        async with self._client.get(endpoint) as response:
            response.raise_for_status()
            return await response.json()
    
    async def _get_stories(
        self, story_type: str = "top", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get a list of stories with details."""
        story_ids = await self._get_story_ids(story_type, limit)
        stories = []
        
        for story_id in story_ids:
            story = await self._get_item(story_id)
            if story and story.get('type') == 'story':  # Ensure it's a story
                story_url = story.get(
                    'url',
                    f"https://news.ycombinator.com/item?id={story.get('id')}"
                )
                stories.append({
                    'id': story.get('id'),
                    'title': story.get('title', 'No title'),
                    'url': story_url,
                    'score': story.get('score', 0),
                    'by': story.get('by', 'unknown'),
                    'time': story.get('time', 0),
                    'descendants': story.get('descendants', 0)
                })
        
        # Cache the stories for potential browser opening
        self._stories_cache = stories
        return stories
    
    def _format_for_voice(self, stories: List[Dict[str, Any]]) -> str:
        """
        Format stories for voice output - titles only, no URLs or metadata.
        
        Args:
            stories: List of story dictionaries
            
        Returns:
            Formatted string with just story titles
        """
        if not stories:
            return "No stories found."
        
        # Create a simple numbered list of titles
        formatted = []
        for i, story in enumerate(stories, 1):
            title = story.get('title', 'No title')
            formatted.append(f"{i}. {title}")
        
        return "\n".join(formatted)
    
    def _open_story_in_browser(self, story_number: int) -> str:
        """
        Open a story URL in the default web browser.
        
        Args:
            story_number: The story number (1-based index) from the
                cached stories
            
        Returns:
            Success or error message
        """
        if not self._stories_cache:
            return (
                "No stories available. Please fetch stories first "
                "using 'top', 'new', or 'best' action."
            )
        
        if story_number < 1 or story_number > len(self._stories_cache):
            max_num = len(self._stories_cache)
            return (
                f"Invalid story number. Please choose a number "
                f"between 1 and {max_num}."
            )
        
        story = self._stories_cache[story_number - 1]
        url = story.get('url')
        title = story.get('title', 'No title')
        
        try:
            webbrowser.open(url)
            return (
                f"Opening story #{story_number}: '{title}' "
                f"in your browser."
            )
        except Exception as e:
            return f"Error opening browser: {str(e)}"
    
    async def _arun(
        self,
        action: str = "top",
        limit: int = 10,
        item_id: Optional[int] = None,
        story_number: Optional[int] = None
    ) -> str:
        """
        Fetch content from Hacker News or open a story in browser.
        
        Args:
            action: The type of stories to fetch ("top", "new", "best"),
                "item" for a specific story, or "open" to open a story
                in browser
            limit: Maximum number of stories to return
            item_id: ID of a specific item to fetch (only used when
                action is "item")
            story_number: Story number to open in browser (1-based index,
                only used when action is "open")
            
        Returns:
            Formatted string with story titles (voice-friendly) or
            browser open confirmation
        """
        try:
            if action == "open":
                # Open a story in the browser
                if story_number is None:
                    return (
                        "Please provide a story number to open "
                        "(e.g., story_number=1)."
                    )
                return self._open_story_in_browser(story_number)
            elif action == "item" and item_id:
                # Get a specific item
                item = await self._get_item(item_id)
                title = item.get('title', 'No title')
                return f"Story: {title}"
            else:
                # Get stories based on type
                valid_types = ["top", "new", "best"]
                story_type = action if action in valid_types else "top"
                stories = await self._get_stories(story_type, limit)
                
                # Format for voice output
                header = (
                    f"Here are the {story_type} {len(stories)} "
                    f"stories:\n\n"
                )
                formatted_stories = self._format_for_voice(stories)
                result = header + formatted_stories
            
            # Close the client session after we're done
            if self._client and not self._client.closed:
                await self._client.close()
            
            return result
        except Exception as e:
            # Ensure client is closed even if there's an error
            if self._client and not self._client.closed:
                await self._client.close()
            return f"Error fetching Hacker News: {str(e)}"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client and not self._client.closed:
            await self._client.close()

    # Convenience methods for direct use
    async def get_top_stories(self, limit: int = 10) -> str:
        """Get top stories from Hacker News (voice-friendly format)."""
        return await self._arun(action="top", limit=limit)
    
    async def get_new_stories(self, limit: int = 10) -> str:
        """Get new stories from Hacker News (voice-friendly format)."""
        return await self._arun(action="new", limit=limit)
    
    async def get_best_stories(self, limit: int = 10) -> str:
        """Get best stories from Hacker News (voice-friendly format)."""
        return await self._arun(action="best", limit=limit)
    
    async def get_story(self, item_id: int) -> str:
        """Get a specific story by ID (voice-friendly format)."""
        return await self._arun(action="item", item_id=item_id)
    
    async def open_story(self, story_number: int) -> str:
        """
        Open a story in the default web browser.
        
        Args:
            story_number: The story number (1-based index) from the
                most recently fetched stories
            
        Returns:
            Confirmation message
        """
        return await self._arun(action="open", story_number=story_number)