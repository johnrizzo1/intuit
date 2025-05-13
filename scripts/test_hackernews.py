#!/usr/bin/env python3
"""
Test script for the Hacker News tool.
"""
import asyncio
from src.intuit.tools.hackernews import HackerNewsTool

async def main() -> None:
    """Test the Hacker News tool."""
    print("Testing Hacker News tool...")
    
    # Create the tool
    hn_tool = HackerNewsTool()
    
    # Test getting top stories
    print("\nGetting top stories...")
    top_stories = await hn_tool.get_top_stories(limit=5)
    print(f"Top stories: {top_stories}")
    
    # Test getting new stories
    print("\nGetting new stories...")
    new_stories = await hn_tool.get_new_stories(limit=5)
    print(f"New stories: {new_stories}")
    
    # Test getting best stories
    print("\nGetting best stories...")
    best_stories = await hn_tool.get_best_stories(limit=5)
    print(f"Best stories: {best_stories}")
    
    # If we have a story ID from the top stories, get its details
    if top_stories and 'stories' in top_stories and top_stories['stories']:
        story_id = top_stories['stories'][0]['id']
        print(f"\nGetting details for story {story_id}...")
        story = await hn_tool.get_story(story_id)
        print(f"Story details: {story}")

if __name__ == "__main__":
    asyncio.run(main())