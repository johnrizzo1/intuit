# HackerNews Tool Auto-Execution Fix

## Problem

When asking the agent "what tools do you have access to?", it would automatically fetch and display the latest Hacker News stories along with the tool list. This was unexpected behavior.

## Root Cause

In [`src/intuit/mcp_server.py`](../src/intuit/mcp_server.py), there was a problematic tool registration:

```python
@mcp_server.tool()
def hackernews_tool() -> HackerNewsTool:
    """Hacker News information tool"""
    return HackerNewsTool()
```

This tool had several issues:

1. **Vague Description**: The description "Hacker News information tool" was too generic and might have been triggering the LLM to call it when listing tools
2. **Returns Object**: It returned a `HackerNewsTool` instance rather than useful data
3. **Unnecessary**: The tool served no practical purpose since specific tools like `hackernews_top`, `hackernews_new`, etc. already existed

## Solution

### 1. Removed Problematic Tool

Removed the `hackernews_tool()` function entirely from the MCP server registration.

### 2. Added Browser Opening Tool

Added a new MCP tool for opening stories in the browser:

```python
@mcp_server.tool()
def hackernews_open(story_number: int) -> str:
    """
    Open a Hacker News story in the default web browser.
    
    Args:
        story_number: Story number (1-based index) from the most
            recently fetched stories
        
    Returns:
        Confirmation message
    """
    logger.info(
        f"MCP: Opening Hacker News story #{story_number} in browser"
    )
    hackernews_tool = HackerNewsTool()
    return str(hackernews_tool.open_story(story_number))
```

## Available HackerNews MCP Tools

After the fix, these are the available HackerNews tools in the MCP server:

1. **`hackernews_top(limit: int = 10)`** - Get top stories
2. **`hackernews_new(limit: int = 10)`** - Get new stories  
3. **`hackernews_best(limit: int = 10)`** - Get best stories
4. **`hackernews_story(item_id: int)`** - Get specific story details
5. **`hackernews_open(story_number: int)`** - Open story in browser (NEW)

## Testing

To verify the fix:

1. Ask the agent: "What tools do you have access to?"
2. The agent should list tools WITHOUT automatically fetching Hacker News stories
3. To test browser opening: 
   - "Get top 5 Hacker News stories"
   - "Open story number 2"

## Related Files

- [`src/intuit/tools/hackernews.py`](../src/intuit/tools/hackernews.py) - Core tool implementation
- [`src/intuit/mcp_server.py`](../src/intuit/mcp_server.py) - MCP server tool registrations
- [`docs/hackernews_browser_feature.md`](hackernews_browser_feature.md) - Browser feature documentation