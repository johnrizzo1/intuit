# HackerNews Browser Opening Feature

## Overview

The HackerNews tool now supports opening stories directly in your default web browser. This feature allows you to quickly view any story from the fetched list without manually copying URLs.

## How It Works

1. **Fetch Stories**: First, fetch a list of stories using one of the available methods:
   - `get_top_stories(limit=10)`
   - `get_new_stories(limit=10)`
   - `get_best_stories(limit=10)`

2. **Open a Story**: Once stories are fetched, they are cached internally. You can then open any story by its number (1-based index) using:
   - `open_story(story_number=N)`

## Usage Examples

### Python API

```python
import asyncio
from intuit.tools.hackernews import HackerNewsTool

async def main():
    hn_tool = HackerNewsTool()
    
    # Fetch top 5 stories
    stories = await hn_tool.get_top_stories(limit=5)
    print(stories)
    
    # Open the first story in browser
    result = await hn_tool.open_story(story_number=1)
    print(result)  # "Opening story #1: 'Story Title' in your browser."
    
    # Open the third story
    result = await hn_tool.open_story(story_number=3)
    print(result)

asyncio.run(main())
```

### Command Line Interface

```bash
# Fetch and display top stories
devenv shell -- python scripts/hackernews_cli.py top --limit 5

# Open story #2 in browser (after fetching stories)
devenv shell -- python scripts/hackernews_cli.py open --number 2
```

### Using the _arun Method Directly

```python
import asyncio
from intuit.tools.hackernews import HackerNewsTool

async def main():
    hn_tool = HackerNewsTool()
    
    # Fetch stories
    result = await hn_tool._arun(action="top", limit=5)
    print(result)
    
    # Open a story
    result = await hn_tool._arun(action="open", story_number=1)
    print(result)

asyncio.run(main())
```

## Features

### Story Caching
- Stories are automatically cached when fetched
- The cache persists for the lifetime of the HackerNewsTool instance
- Each new fetch operation updates the cache

### Error Handling
- **No stories cached**: If you try to open a story before fetching any, you'll get a helpful error message
- **Invalid story number**: If you provide a number outside the valid range (1 to N), you'll get an error with the valid range
- **Browser errors**: Any errors opening the browser are caught and reported

### Browser Behavior
- Opens stories in your system's default web browser
- Uses the Python `webbrowser` module for cross-platform compatibility
- Works on Linux, macOS, and Windows

## Implementation Details

### New Methods

1. **`_open_story_in_browser(story_number: int) -> str`**
   - Internal method that handles browser opening
   - Validates story number
   - Opens URL using `webbrowser.open()`

2. **`open_story(story_number: int) -> str`**
   - Public convenience method
   - Calls `_arun(action="open", story_number=story_number)`

### Updated Methods

1. **`_arun()`**
   - Added `story_number` parameter
   - Added "open" action support
   - Maintains backward compatibility

2. **`_get_stories()`**
   - Now caches fetched stories in `_stories_cache`

### New Attributes

- **`_stories_cache: List[Dict[str, Any]]`**
  - Private attribute storing the most recently fetched stories
  - Initialized as empty list
  - Updated on each story fetch operation

## Voice-Friendly Output

The tool maintains its voice-friendly output format:
- Story lists show only numbered titles
- Browser opening confirmations are concise and clear
- Error messages are descriptive but brief

## Integration with Agent

The HackerNews tool can be used by the agent to:
1. Fetch and read story titles to the user
2. Open specific stories when requested
3. Handle natural language requests like "open the third story"

Example agent interaction:
```
User: "Show me the top Hacker News stories"
Agent: [Fetches and reads story titles]

User: "Open the second one"
Agent: [Opens story #2 in browser]
```

## Testing

To test the browser opening functionality:

```bash
# Run the test script
devenv shell -- python scripts/test_hackernews_browser.py
```

The test script demonstrates:
- Fetching stories
- Opening stories by number
- Error handling for invalid inputs
- Error handling when no stories are cached

## Notes

- The browser opens in a new tab/window depending on your browser settings
- The tool does not wait for the browser to finish loading
- Multiple stories can be opened in succession
- Each HackerNewsTool instance maintains its own story cache