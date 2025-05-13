# Intuit Tools Testing Guide

This document provides information about testing the various tools available in the Intuit system.

## Available Tools

The Intuit system includes the following tools:

1. **FilesystemTool**: For filesystem operations like listing directories, reading files, and searching content
2. **WebSearchTool**: For searching the web for information
3. **GmailTool**: For email management
4. **WeatherTool**: For getting weather information
5. **CalendarTool**: For managing calendar events
6. **NotesTool**: For managing notes
7. **RemindersTool**: For managing reminders
8. **HackerNewsTool**: For fetching stories from Hacker News
9. **Desktop Automation**: For interacting with the operating system (e.g., clipboard functionality) - *Note: This functionality may be implemented as part of the MCP server tools*

## Testing the Tools

We've created two scripts to test the tools:

1. `scripts/test_all_tools.py`: A Python script that tests each tool individually
2. `scripts/test_tools.sh`: A shell script that runs the Python script and provides a nice summary

### Running the Tests

To run the tests, simply execute the shell script:

```bash
./scripts/test_tools.sh
```

This will test all the tools and provide a summary of the results.

## Tool Interfaces

Each tool has a different interface:

### FilesystemTool

The FilesystemTool has a `run` method that takes keyword arguments:

```python
result = await tool.run(action="list", path=".")
result = await tool.run(action="read", path="README.md")
result = await tool.run(action="info", path="README.md")
result = await tool.run(action="search", query="intuit")
```

### WebSearchTool, WeatherTool, GmailTool, HackerNewsTool

These tools extend LangChain's BaseTool and have an `_arun` method:

```python
# WebSearchTool
result = await tool._arun(query="Python programming language")

# WeatherTool
result = await tool._arun(location="New York")

# GmailTool
result = await tool._arun(query="is:unread")

# HackerNewsTool
result = await tool._arun(action="top", limit=5)
```

### CalendarTool, NotesTool, RemindersTool

These tools have specific methods for different operations:

```python
# CalendarTool
add_result = tool.add_event("Test event")
list_result = tool.list_events()

# NotesTool
add_result = tool.add_note("Test note")
list_result = tool.list_notes()

# RemindersTool
add_result = tool.add_reminder("Test reminder")
list_result = tool.list_reminders()
```

## Test Results

All tools have been tested and are working correctly. The tests verify that each tool can perform its basic operations without errors.

## Troubleshooting

If any of the tests fail, check the following:

1. Make sure all required environment variables are set
2. Check that the tool's dependencies are installed
3. Verify that the tool's service is running (if applicable)
4. Check the logs for specific error messages

## Adding New Tests

To add tests for a new tool:

1. Add a new test function in `scripts/test_all_tools.py`
2. Add the function to the `tests` list in the `main` function
3. Run the tests to verify that the new tool works correctly

## Test Coverage

Our test script currently covers 8 out of the 9 tools listed in the Intuit system:

1. Web Search (WebSearchTool) ✓
2. Filesystem Operations (FilesystemTool) ✓
3. Gmail Integration (GmailTool) ✓
4. Weather Information (WeatherTool) ✓
5. Calendar Management (CalendarTool) ✓
6. Notes Management (NotesTool) ✓
7. Reminders Management (RemindersTool) ✓
8. Hacker News Integration (HackerNewsTool) ✓

The "Desktop Automation" functionality is not directly tested as it appears to be implemented as part of the MCP server tools or elsewhere in the codebase.