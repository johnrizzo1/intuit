"""
MCP Server implementation for Intuit.
"""
import io
import time
import asyncio
import threading
import logging
import typer
import multiprocessing
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image as MCPImage # Renamed to avoid conflict
import pyautogui

from intuit.tools.calendar import CalendarTool
from intuit.tools.notes import NotesTool
from intuit.tools.reminders import RemindersTool
from intuit.tools.weather import WeatherTool
from intuit.tools.web_search import WebSearchTool
from intuit.tools.filesystem import FilesystemTool
from intuit.tools.hackernews import HackerNewsTool
from intuit.vector_store.indexer import VectorStore
from intuit.memory.chroma_store import ChromaMemoryStore
from intuit.memory.manager import IntuitMemoryManager
import os
from intuit.memory.tools import get_memory_tools

logger = logging.getLogger(__name__)

# Create server instance with log_level set to ERROR to reduce verbosity
mcp_server = FastMCP("Intuit Tools", dependencies=["pyautogui", "Pillow", "pydantic", "chromadb"], log_level="ERROR")

# Server configuration
DEFAULT_SERVER_HOST = "localhost"
DEFAULT_SERVER_PORT = 8000

# Initialize vector store for RAG pipeline
try:
    vector_store = VectorStore()
    logger.info("Vector store initialized for MCP server")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None

# Initialize memory store and manager for MCP server
try:
    # Use a default model for the memory store
    default_model = os.getenv("OPENAI_MODEL_NAME", "llama3.2:3b")
    memory_store = ChromaMemoryStore(model=default_model)
    logger.info("Memory store initialized for MCP server")
    # Initialize the memory manager with a default model
    default_model = os.getenv("OPENAI_MODEL_NAME", "llama3.2:3b")
    memory_manager = IntuitMemoryManager(store=memory_store, model=default_model)
    logger.info(f"Memory manager initialized for MCP server with model: {default_model}")
except Exception as e:
    logger.error(f"Failed to initialize memory store: {e}")
    memory_store = None
    memory_manager = None

# --- Tool Definitions ---

# Register all tools with the MCP server

# Calendar tools
@mcp_server.tool()
def calendar_tool() -> CalendarTool:
    """Calendar management tool"""
    return CalendarTool()

@mcp_server.tool()
def calendar_add(event: str) -> str:
    """
    Add a new calendar event.

    Args:
        event: Details of the calendar event to add

    Returns:
        Confirmation message with the event ID
    """
    logger.info(f"MCP: Adding calendar event: {event}")
    calendar_tool = CalendarTool()
    return str(calendar_tool.add_event(event))

@mcp_server.tool()
def calendar_list() -> str:
    """Lists all calendar events."""
    logger.info("MCP: Listing calendar events")
    calendar_tool = CalendarTool()
    return str(calendar_tool.list_events())

@mcp_server.tool()
def calendar_search(keyword: str) -> str:
    """Searches calendar events for a keyword."""
    logger.info(f"MCP: Searching calendar events for: {keyword}")
    calendar_tool = CalendarTool()
    return str(calendar_tool.search_events(keyword))

@mcp_server.tool()
def calendar_delete(filename: str) -> str:
    """Deletes a calendar event by filename."""
    logger.info(f"MCP: Deleting calendar event: {filename}")
    calendar_tool = CalendarTool()
    return str(calendar_tool.delete_event(filename))

# Notes tools
@mcp_server.tool()
def notes_tool() -> NotesTool:
    """Notes management tool"""
    return NotesTool()

@mcp_server.tool()
def notes_add(content: str) -> str:
    """Adds a new note."""
    logger.info(f"MCP: Adding note: {content}")
    notes_tool = NotesTool()
    return str(notes_tool.add_note(content))

@mcp_server.tool()
def notes_list() -> str:
    """Lists all notes."""
    logger.info("MCP: Listing notes")
    notes_tool = NotesTool()
    return str(notes_tool.list_notes())

@mcp_server.tool()
def notes_search(keyword: str) -> str:
    """Searches notes for a keyword."""
    logger.info(f"MCP: Searching notes for: {keyword}")
    notes_tool = NotesTool()
    return str(notes_tool.search_notes(keyword))

@mcp_server.tool()
def notes_delete(id: str) -> str:
    """Deletes a note by ID."""
    logger.info(f"MCP: Deleting note: {id}")
    notes_tool = NotesTool()
    return str(notes_tool.delete_note(id))

# Reminders tools
@mcp_server.tool()
def reminders_tool() -> RemindersTool:
    """Reminders management tool"""
    return RemindersTool()

@mcp_server.tool()
def reminders_add(content: str, reminder_time: Optional[str] = None) -> str:
    """
    Adds a new reminder.
    
    Args:
        content: Content of the reminder
        reminder_time: Optional reminder time in ISO 8601 format (e.g., '2025-12-31T23:59:59')
        
    Returns:
        Confirmation message with the reminder ID
    """
    logger.info(f"MCP: Adding reminder: {content} at {reminder_time}")
    reminders_tool = RemindersTool()
    return str(reminders_tool.add_reminder(content, reminder_time))

@mcp_server.tool()
def reminders_list() -> str:
    """Lists all reminders."""
    logger.info("MCP: Listing reminders")
    reminders_tool = RemindersTool()
    return str(reminders_tool.list_reminders())

@mcp_server.tool()
def reminders_search(keyword: str) -> str:
    """Searches reminders for a keyword."""
    logger.info(f"MCP: Searching reminders for: {keyword}")
    reminders_tool = RemindersTool()
    return str(reminders_tool.search_reminders(keyword))

@mcp_server.tool()
def reminders_delete(id: str) -> str:
    """Deletes a reminder by ID."""
    logger.info(f"MCP: Deleting reminder: {id}")
    reminders_tool = RemindersTool()
    return str(reminders_tool.delete_reminder(id))

# Weather tool
@mcp_server.tool()
def weather_tool() -> WeatherTool:
    """Weather information tool"""
    return WeatherTool()

@mcp_server.tool()
def weather_get(location: str) -> str:
    """
    Get weather information for a location.
    
    Args:
        location: The location to get weather for (e.g., 'London, UK' or 'New York, NY')
        
    Returns:
        Weather information for the location
    """
    logger.info(f"MCP: Getting weather for: {location}")
    weather_tool = WeatherTool()
    
    # Use asyncio to run the async method
    import asyncio
    result = asyncio.run(weather_tool.get_weather_async(location))
    return result

# Web search tool
@mcp_server.tool()
def web_search_tool() -> WebSearchTool:
    """Web search tool"""
    return WebSearchTool()

@mcp_server.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Search results
    """
    logger.info(f"MCP: Searching web for: {query}")
    web_search_tool = WebSearchTool()
    result = web_search_tool.search(query, max_results)
    return str(result)  # Convert result to string to match return type

# Filesystem tool
@mcp_server.tool()
def filesystem_tool() -> FilesystemTool:
    """Filesystem access tool"""
    # Use the vector store if available
    return FilesystemTool(vector_store=vector_store)

@mcp_server.tool()
def filesystem_list(path: str) -> str:
    """
    List directory contents.
    
    Args:
        path: The path to list
        
    Returns:
        Directory contents
    """
    logger.info(f"MCP: Listing directory: {path}")
    filesystem_tool = FilesystemTool(vector_store=vector_store)
    return str(filesystem_tool.list_directory(path))

@mcp_server.tool()
def filesystem_read(path: str) -> str:
    """
    Read file contents.
    
    Args:
        path: The path to read
        
    Returns:
        File contents
    """
    logger.info(f"MCP: Reading file: {path}")
    filesystem_tool = FilesystemTool(vector_store=vector_store)
    return str(filesystem_tool.read_file(path))

@mcp_server.tool()
def filesystem_write(path: str, content: str) -> str:
    """
    Write content to file.
    
    Args:
        path: The path to write to
        content: The content to write
        
    Returns:
        Confirmation message
    """
    logger.info(f"MCP: Writing to file: {path}")
    filesystem_tool = FilesystemTool(vector_store=vector_store)
    return str(filesystem_tool.write_file(path, content))

@mcp_server.tool()
def filesystem_search(path: str, query: str) -> str:
    """
    Search for files by content.
    
    Args:
        path: The path to search in
        query: The search query
        
    Returns:
        Search results
    """
    logger.info(f"MCP: Searching for: {query} in {path}")
    filesystem_tool = FilesystemTool(vector_store=vector_store)
    return str(filesystem_tool.search(path, query))

@mcp_server.tool()
def take_screenshot() -> MCPImage:
    """
    Take a screenshot of the user's screen and return it as an image.
    Use this tool anytime the user wants you to look at something they're doing.
    """
    logger.info("Taking screenshot...")
    try:
        screenshot = pyautogui.screenshot()
        buffer = io.BytesIO()
        # Convert to RGB before saving as JPEG to avoid issues with alpha channel
        screenshot.convert("RGB").save(buffer, format="JPEG", quality=85)
        logger.info("Screenshot taken successfully.")
        return MCPImage(data=buffer.getvalue(), format="jpeg")
    except Exception as e:
        # Log at INFO level instead of ERROR so it only shows with -v flag
        logger.info(f"Error taking screenshot: {e}")
        # Consider returning an error object or raising an MCP-specific exception
        raise

# Hacker News tool
@mcp_server.tool()
def hackernews_tool() -> HackerNewsTool:
    """Hacker News information tool"""
    return HackerNewsTool()

@mcp_server.tool()
def hackernews_top(limit: int = 10) -> str:
    """
    Get top stories from Hacker News.
    
    Args:
        limit: Maximum number of stories to return (default: 10)
        
    Returns:
        Top stories from Hacker News
    """
    logger.info(f"MCP: Getting top stories from Hacker News (limit: {limit})")
    hackernews_tool = HackerNewsTool()
    return str(hackernews_tool.get_top_stories(limit))

@mcp_server.tool()
def hackernews_new(limit: int = 10) -> str:
    """
    Get new stories from Hacker News.
    
    Args:
        limit: Maximum number of stories to return (default: 10)
        
    Returns:
        New stories from Hacker News
    """
    logger.info(f"MCP: Getting new stories from Hacker News (limit: {limit})")
    hackernews_tool = HackerNewsTool()
    return str(hackernews_tool.get_new_stories(limit))

@mcp_server.tool()
def hackernews_best(limit: int = 10) -> str:
    """
    Get best stories from Hacker News.
    
    Args:
        limit: Maximum number of stories to return (default: 10)
        
    Returns:
        Best stories from Hacker News
    """
    logger.info(f"MCP: Getting best stories from Hacker News (limit: {limit})")
    hackernews_tool = HackerNewsTool()
    return str(hackernews_tool.get_best_stories(limit))

@mcp_server.tool()
def hackernews_story(item_id: int) -> str:
    """
    Get details of a specific Hacker News story.
    
    Args:
        item_id: ID of the story to retrieve
        
    Returns:
        Details of the specified Hacker News story
    """
    logger.info(f"MCP: Getting Hacker News story with ID: {item_id}")
    hackernews_tool = HackerNewsTool()
    return str(hackernews_tool.get_story(item_id))

# Memory tools
@mcp_server.tool()
def memory_add(content: str, importance: int = 5, tags: list[str] = None) -> str:
    """
    Add a memory to the store.
    
    Args:
        content: The content of the memory
        importance: Importance level (1-10)
        tags: Optional tags for categorizing the memory
        
    Returns:
        Memory ID
    """
    logger.info(f"MCP: Adding memory: {content}")
    if memory_store:
        try:
            tags = tags or []
            asyncio.run(memory_store.add_memory(
                content=content,
                metadata={"importance": importance, "tags": tags}
            ))
            return "Memory saved"
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return f"Error adding memory: {str(e)}"
    return "Memory store not available"

@mcp_server.tool()
def memory_search(query: str, limit: int = 5) -> str:
    """
    Search memories by semantic similarity.
    
    Args:
        query: The search query
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        Search results
    """
    logger.info(f"MCP: Searching memories for: {query}")
    if memory_store:
        try:
            memories = asyncio.run(memory_store.search_memories(query, limit))
            if not memories:
                return "No memories found matching your query."
            
            result = "Found memories:\n\n"
            for i, memory in enumerate(memories):
                result += f"{i+1}. {memory['content']}\n"
            return result
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return f"Error searching memories: {str(e)}"
    return "Memory store not available"

@mcp_server.tool()
def memory_get(memory_id: str) -> str:
    """
    Get a specific memory by ID.
    
    Args:
        memory_id: The ID of the memory to retrieve
        
    Returns:
        Memory content
    """
    logger.info(f"MCP: Getting memory with ID: {memory_id}")
    if memory_store:
        try:
            memory = asyncio.run(memory_store.get_memory(memory_id))
            if memory:
                return f"Memory: {memory['content']}"
            return f"Memory with ID {memory_id} not found."
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return f"Error getting memory: {str(e)}"
    return "Memory store not available"

@mcp_server.tool()
def memory_delete(memory_id: str) -> str:
    """
    Delete a specific memory by ID.
    
    Args:
        memory_id: The ID of the memory to delete
        
    Returns:
        Confirmation message
    """
    logger.info(f"MCP: Deleting memory with ID: {memory_id}")
    if memory_store:
        try:
            success = asyncio.run(memory_store.delete_memory(memory_id))
            if success:
                return "Memory deleted"
            return "Failed to delete memory"
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return f"Error deleting memory: {str(e)}"
    return "Memory store not available"

@mcp_server.tool()
def memory_clear() -> str:
    """
    Clear all memories.
    
    Returns:
        Confirmation message
    """
    logger.info("MCP: Clearing all memories")
    if memory_store:
        try:
            success = asyncio.run(memory_store.clear_memories())
            if success:
                return "Memories cleared"
            return "Failed to clear memories"
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return f"Error clearing memories: {str(e)}"
    return "Memory store not available"

# --- Server Management ---

# Top-level function to run the server in a separate process
# This avoids pickling the MCPServerManager instance or the FastMCP instance directly.
def _run_mcp_server_process(host: str, port: int) -> None:
    """Target function for the MCP server process."""
    # Access the globally defined and configured mcp_server instance.
    # In 'spawn' context, the module is re-imported, re-running decorators
    # and recreating the mcp_server instance with registered tools.
    logger.info(f"MCP server process started. Running server on {host}:{port}")
    try:
        # Use the global mcp_server instance which has tools registered.
        # Use SSE transport with host/port parameters
        # Note: FastMCP.run() doesn't accept log_level parameter
        mcp_server.run("sse", mount_path=f"/{host}:{port}")
    except Exception as e:
        # Log at INFO level instead of ERROR so it only shows with -v flag
        logger.info(f"Error running MCP server in child process: {e}", exc_info=True)

class MCPServerManager:
    """Manager for the MCP server instance."""

    def __init__(self, host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT) -> None:
        self.host = host
        self.port = port
        # No longer need to store server_instance here for starting
        self.is_running = False
        self.server_process: multiprocessing.Process | None = None

    # _run_server method removed, replaced by _run_mcp_server_process top-level function

    def start(self) -> str:
        """Start the MCP server in a separate process."""
        if not self.is_running:
            logger.info(f"Starting MCP server on http://{self.host}:{self.port} (process mode)")
            # Use the top-level function as the target
            self.server_process = multiprocessing.Process(
                target=_run_mcp_server_process,
                args=(self.host, self.port), # Pass config via args
                daemon=True
            )
            self.server_process.start()
            self.is_running = True
            time.sleep(2)  # Increased sleep slightly for server startup
            if self.server_process and self.server_process.is_alive():
                logger.info(f"MCP server process started successfully (PID: {self.server_process.pid}) on http://{self.host}:{self.port}")
                return f"MCP server started at http://{self.host}:{self.port} (process mode)"
            else:
                # Log at INFO level instead of ERROR so it only shows with -v flag
                logger.info("MCP server process failed to start or terminated unexpectedly.")
                self.is_running = False
                # Attempt to get exit code if possible
                exitcode = getattr(self.server_process, 'exitcode', 'N/A')
                # Log at INFO level instead of ERROR so it only shows with -v flag
                logger.info(f"Server process exit code: {exitcode}")
                return "MCP server failed to start. Check logs."
        logger.info("MCP server is already running.")
        return "MCP server is already running."

    def stop(self) -> str:
        """Stop the MCP server process."""
        if self.is_running and self.server_process:
            logger.info("Stopping MCP server process...")
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            self.is_running = False
            logger.info("MCP server process stopped.")
            return "MCP server process stopped."
        logger.info("MCP server is not running.")
        return "MCP server is not running."

    def get_tools(self) -> list[dict[str, object]] | str:
        """Returns a list of available tools on the server."""
        # FastMCP keeps a registry of tools, typically as a dict: {name: Tool}
        # We'll extract name, description, and parameters/schema if available
        # Access the global mcp_server instance for tool definitions
        if mcp_server and hasattr(mcp_server, 'tools'):
            tools = []
            for name, tool in mcp_server.tools.items(): # Use global mcp_server
                # Try to get description and parameters/schema if available
                desc = getattr(tool, 'description', tool.__doc__ or "")
                # Try to get input schema if available (FastMCP tools may have 'input_schema' or similar)
                schema = getattr(tool, 'input_schema', None)
                tools.append({
                    'name': name,
                    'description': desc,
                    'parameters': schema,
                })
            return tools
        return "MCP Server not initialized or no tools available."

# --- CLI for Server Management (Optional, can be integrated into main.py) ---
cli_app = typer.Typer(name="mcp-server", help="Manage the Intuit MCP Server", invoke_without_command=True)
_manager_instance = None # To hold a global manager instance for CLI

def get_manager() -> MCPServerManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MCPServerManager()
    return _manager_instance

def get_registered_tools() -> list[dict[str, object]]:
    """Return registered tools from the global mcp_server instance (for CLI listing)."""
    import asyncio
    tools = []
    
    # First try to get tools from the server directly
    try:
        logger.info("Getting tools from MCP server directly")
        # Get tools directly from the mcp_server instance
        if hasattr(mcp_server, 'tools'):
            for name, tool in mcp_server.tools.items():
                logger.info(f"Found tool: {name}")
                # Try to get description and parameters/schema if available
                desc = getattr(tool, 'description', None) or tool.__doc__ or ""
                # Try to get input schema if available
                schema = getattr(tool, 'input_schema', None)
                tools.append({
                    'name': name,
                    'description': desc,
                    'parameters': schema,
                })
            logger.info(f"Found {len(tools)} tools directly from MCP server")
            return tools
    except Exception as e:
        # Log at INFO level instead of ERROR so it only shows with -v flag
        logger.info(f"Error getting tools directly from MCP server: {e}")
    
    # If that fails, try to get tools using the list_tools method
    if not tools:
        try:
            logger.info("Getting tools using list_tools method")
            tool_objs = asyncio.run(mcp_server.list_tools())
            for tool in tool_objs:
                logger.info(f"Found tool: {getattr(tool, 'name', 'unknown')}")
                tools.append({
                    'name': getattr(tool, 'name', None),
                    'description': getattr(tool, 'description', None),
                    'parameters': getattr(tool, 'inputSchema', None),
                })
            logger.info(f"Found {len(tools)} tools using list_tools method")
        except Exception as e:
            # Log at INFO level instead of ERROR so it only shows with -v flag
            logger.info(f"Error getting tools using list_tools method: {e}")
    
    # If we still don't have any tools, create a hardcoded list
    if not tools:
        logger.info("Using hardcoded list of tools")
        tools = [
            {
                'name': 'calendar_add',
                'description': 'Add a new calendar event.\n\nArgs:\nevent: Details of the calendar event to add\nReturns:\nConfirmation message with the event ID',
                'parameters': {'type': 'object', 'properties': {'event': {'type': 'string'}}, 'required': ['event']}
            },
            {
                'name': 'calendar_list',
                'description': 'Lists all calendar events.',
                'parameters': {'type': 'object', 'properties': {}}
            },
            {
                'name': 'calendar_search',
                'description': 'Searches calendar events for a keyword.',
                'parameters': {'type': 'object', 'properties': {'keyword': {'type': 'string'}}, 'required': ['keyword']}
            },
            {
                'name': 'calendar_delete',
                'description': 'Deletes a calendar event by filename.',
                'parameters': {'type': 'object', 'properties': {'filename': {'type': 'string'}}, 'required': ['filename']}
            },
            {
                'name': 'notes_add',
                'description': 'Adds a new note.',
                'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']}
            },
            {
                'name': 'notes_list',
                'description': 'Lists all notes.',
                'parameters': {'type': 'object', 'properties': {}}
            },
            {
                'name': 'notes_search',
                'description': 'Searches notes for a keyword.',
                'parameters': {'type': 'object', 'properties': {'keyword': {'type': 'string'}}, 'required': ['keyword']}
            },
            {
                'name': 'notes_delete',
                'description': 'Deletes a note by ID.',
                'parameters': {'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']}
            },
            {
                'name': 'take_screenshot',
                'description': 'Take a screenshot of the user\'s screen and return it as an image.\nUse this tool anytime the user wants you to look at something they\'re doing.',
                'parameters': {'type': 'object', 'properties': {}}
            },
            {
                'name': 'hackernews_top',
                'description': 'Get top stories from Hacker News.',
                'parameters': {'type': 'object', 'properties': {'limit': {'type': 'integer', 'default': 10}}}
            },
            {
                'name': 'hackernews_new',
                'description': 'Get new stories from Hacker News.',
                'parameters': {'type': 'object', 'properties': {'limit': {'type': 'integer', 'default': 10}}}
            },
            {
                'name': 'hackernews_best',
                'description': 'Get best stories from Hacker News.',
                'parameters': {'type': 'object', 'properties': {'limit': {'type': 'integer', 'default': 10}}}
            },
            {
                'name': 'hackernews_story',
                'description': 'Get details of a specific Hacker News story.',
                'parameters': {'type': 'object', 'properties': {'item_id': {'type': 'integer'}}, 'required': ['item_id']}
            }
        ]
        logger.info(f"Created hardcoded list of {len(tools)} tools")
    
    return tools

@cli_app.callback()
def main(ctx: typer.Context) -> None:
    """MCP Server CLI."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

@cli_app.command()
def start(
    host: str = typer.Option(DEFAULT_SERVER_HOST, help="Server host"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, help="Server port")
) -> None:
    """Start the MCP server."""
    manager = MCPServerManager(host=host, port=port)
    print(manager.start())
    if manager.is_running:
        print(f"MCP Server is running on http://{host}:{port}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping MCP server...")
            # manager.stop() # Stop is not fully effective for daemon thread
            print("MCP server shut down.")

@cli_app.command()
def status() -> None:
    """Check the MCP server status."""
    manager = get_manager()
    if manager.is_running:
        print(f"MCP server is RUNNING on http://{manager.host}:{manager.port}")
        print("Available tools:")
        tools = manager.get_tools()
        if isinstance(tools, list): # get_tools_json returns a list
            for tool in tools:
                print(f"  - {tool.get('name')}: {tool.get('description')}")
        else:
            print(tools) # Print error message if not a list
    else:
        print("MCP server is STOPPED.")

@cli_app.command()
def list_tools() -> None:
    """List available tools on the MCP server in a human-readable format."""
    manager = get_manager()
    # Attempt to get tools regardless of running state, assuming get_tools can handle it
    # or that the user expects to see the defined tools even if the server isn't active.
    tools_data = manager.get_tools()

    if not isinstance(tools_data, list):
        print(f"Could not retrieve tools: {tools_data}")
        return

    if not tools_data:
        print("No MCP tools found.")
        return

    print("\nAvailable MCP Tools:")
    print("--------------------")

    # Group tools by prefix (e.g., 'calendar', 'notes')
    grouped_tools: dict[str, list[dict[str, object]]] = {}
    for tool in tools_data:
        name = str(tool.get('name', 'unknown'))
        # Use the part before the first underscore as the group key
        prefix = name.split('_')[0] if '_' in name else 'general'
        if prefix not in grouped_tools:
            grouped_tools[prefix] = []
        grouped_tools[prefix].append(tool)

    # Sort groups alphabetically and tools within groups alphabetically
    for prefix in sorted(grouped_tools.keys()):
        # Skip empty groups if any somehow occur
        if not grouped_tools[prefix]:
            continue

        print(f"\n{prefix.capitalize()} Tools:")
        for tool in sorted(grouped_tools[prefix], key=lambda x: str(x.get('name', ''))):
            name = str(tool.get('name', 'N/A'))
            # Clean up description, handle potential None and ensure string type
            description = str(tool.get('description') or 'No description available.').strip()
            params_schema = tool.get('parameters') # This is the schema dict

            # Format parameters simply - just list names if available
            param_str = " (No parameters)"
            if isinstance(params_schema, dict) and 'properties' in params_schema:
                param_names = list(params_schema['properties'].keys())
                if param_names:
                    param_str = f" (Parameters: {', '.join(param_names)})"
                elif 'properties' in params_schema and not params_schema['properties']:
                     # Explicitly handle case with parameters object but no properties
                     param_str = " (No parameters)"


            print(f"  - {name}{param_str}")
            # Indent description for readability, handle multi-line descriptions
            desc_lines = description.split('\n')
            # Print first line indented
            print(f"    > {desc_lines[0].strip()}")
            # Print subsequent lines further indented
            for line in desc_lines[1:]:
                print(f"      {line.strip()}")

    print("\n--------------------")
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)  # Set to ERROR to reduce verbosity
    cli_app()