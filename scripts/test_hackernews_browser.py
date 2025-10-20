#!/usr/bin/env python3
"""
Test script for the Hacker News tool with browser opening functionality.
"""
import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intuit.tools.hackernews import HackerNewsTool


async def main() -> None:
    """Test the Hacker News tool with browser opening."""
    print("Testing Hacker News tool with browser opening...")
    
    # Create the tool
    hn_tool = HackerNewsTool()
    
    # Test getting top stories
    print("\n=== Getting top 5 stories ===")
    top_stories = await hn_tool.get_top_stories(limit=5)
    print(top_stories)
    
    # Test opening a story in the browser
    print("\n=== Opening story #1 in browser ===")
    result = await hn_tool.open_story(story_number=1)
    print(result)
    
    # Wait a moment for the browser to open
    await asyncio.sleep(2)
    
    # Test opening another story
    print("\n=== Opening story #3 in browser ===")
    result = await hn_tool.open_story(story_number=3)
    print(result)
    
    # Test error handling - invalid story number
    print("\n=== Testing invalid story number ===")
    result = await hn_tool.open_story(story_number=99)
    print(result)
    
    # Test error handling - no stories cached
    print("\n=== Testing with no cached stories ===")
    hn_tool2 = HackerNewsTool()
    result = await hn_tool2.open_story(story_number=1)
    print(result)
    
    print("\n=== Test complete ===")


if __name__ == "__main__":
    asyncio.run(main())