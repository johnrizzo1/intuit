#!/usr/bin/env python3
"""
Command-line interface for the Hacker News tool.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intuit.tools.hackernews import HackerNewsTool

async def get_top_stories(limit=10):
    """Get top stories from Hacker News."""
    hn_tool = HackerNewsTool()
    result = await hn_tool.get_top_stories(limit=limit)
    return result

async def get_new_stories(limit=10):
    """Get new stories from Hacker News."""
    hn_tool = HackerNewsTool()
    result = await hn_tool.get_new_stories(limit=limit)
    return result

async def get_best_stories(limit=10):
    """Get best stories from Hacker News."""
    hn_tool = HackerNewsTool()
    result = await hn_tool.get_best_stories(limit=limit)
    return result

async def get_story(item_id):
    """Get details of a specific story."""
    hn_tool = HackerNewsTool()
    result = await hn_tool.get_story(item_id)
    return result


async def open_story_in_browser(story_number):
    """Open a story in the default web browser."""
    hn_tool = HackerNewsTool()
    # First, fetch stories to populate the cache
    await hn_tool.get_top_stories(limit=10)
    # Then open the requested story
    result = await hn_tool.open_story(story_number)
    return result

def format_stories(result):
    """Format stories for display."""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    if 'stories' not in result or not result['stories']:
        return "No stories found."
    
    output = []
    for story in result['stories']:
        output.append(f"Title: {story['title']}")
        output.append(f"URL: {story['url']}")
        output.append(f"Score: {story['score']} | Comments: {story['descendants']}")
        output.append(f"Posted by: {story['by']}")
        output.append("")
    
    return "\n".join(output)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hacker News CLI with browser opening support"
    )
    parser.add_argument(
        "action",
        choices=["top", "new", "best", "story", "open"],
        help="Action to perform"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of stories to fetch"
    )
    parser.add_argument(
        "--id",
        type=int,
        help="Story ID (required for 'story' action)"
    )
    parser.add_argument(
        "--number",
        type=int,
        help="Story number to open in browser (required for 'open' action)"
    )
    args = parser.parse_args()
    
    if args.action == "story" and args.id is None:
        parser.error("--id is required for 'story' action")
    
    if args.action == "open" and args.number is None:
        parser.error("--number is required for 'open' action")
    
    if args.action == "top":
        result = asyncio.run(get_top_stories(args.limit))
        print(result)
    elif args.action == "new":
        result = asyncio.run(get_new_stories(args.limit))
        print(result)
    elif args.action == "best":
        result = asyncio.run(get_best_stories(args.limit))
        print(result)
    elif args.action == "story":
        result = asyncio.run(get_story(args.id))
        print(result)
    elif args.action == "open":
        result = asyncio.run(open_story_in_browser(args.number))
        print(result)

if __name__ == "__main__":
    main()