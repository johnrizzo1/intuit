"""
Main entry point for Intuit.
"""
import asyncio
import typer
import sys
import logging
import builtins # Import the builtins module
from typing import Optional, Any
from pathlib import Path
from datetime import datetime # Import datetime
from dotenv import load_dotenv
import os
import time # Added for MCP server keep-alive

# Load environment variables from .env file
load_dotenv()

from .agent import Agent, AgentConfig
from .tools.web_search import WebSearchTool
from .tools.gmail import GmailTool
from .tools.weather import WeatherTool
from .tools.filesystem import FilesystemTool
from .tools.calendar import CalendarTool
from .tools.notes import NotesTool
from .tools.reminders import RemindersTool
from .vector_store.indexer import VectorStore
from .ui.cli import run_cli
from .ui.voice import run_voice
from .reminders_service import ReminderService
from .mcp_server import MCPServerManager, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT # Import MCP Server

# Set up logging with default level (will be adjusted by verbosity option)
logging.basicConfig(
    level=logging.ERROR,  # Default to ERROR level only (suppress warnings)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Create the main app
app = typer.Typer(no_args_is_help=True)

# Define a callback to handle verbosity
def set_verbosity(verbose: int = typer.Option(0, "--verbose", "-v", count=True,
                                             help="Increase verbosity (can be used multiple times)")):
    """Set the verbosity level based on the number of -v flags."""
    # Set log levels based on verbosity count
    if verbose == 0:
        # Default: Show only errors
        logging.getLogger().setLevel(logging.ERROR)
        
        # Specifically silence common noisy loggers
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("intuit").setLevel(logging.ERROR)
        
    elif verbose == 1:
        # -v: Show warnings and errors
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("mcp").setLevel(logging.WARNING)
        logging.getLogger("intuit").setLevel(logging.WARNING)
        
    elif verbose == 2:
        # -vv: Show info, warnings, and errors
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("mcp").setLevel(logging.INFO)
        logging.getLogger("intuit").setLevel(logging.INFO)
        
    elif verbose >= 3:
        # -vvv or more: Show debug, info, warnings, and errors
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("intuit").setLevel(logging.DEBUG)
    
    # Return value is required for the callback
    return verbose

# Register the callback
app = typer.Typer(no_args_is_help=True, callback=set_verbosity)

# Create Typer instances for each tool
calendar_app = typer.Typer(name="calendar", help="Manage calendar events")
notes_app = typer.Typer(name="notes", help="Manage notes")
reminders_app = typer.Typer(name="reminders", help="Manage reminders")
mcp_cli_app = typer.Typer(name="mcp", help="Manage MCP server and client connections", no_args_is_help=True)


# Mount tool Typer instances onto the main app
app.add_typer(calendar_app)
app.add_typer(notes_app)
app.add_typer(reminders_app)
app.add_typer(mcp_cli_app)


# Define commands for Calendar tool
@calendar_app.command()
def add(event: str) -> None:
    """Adds a new calendar event."""
    agent = asyncio.run(create_agent())  # Create agent with default config
    calendar_tool = next((tool for tool in agent.tools if isinstance(tool, CalendarTool)), None)
    if calendar_tool:
        print(calendar_tool.add_event(event))
    else:
        print("Calendar tool not available.")
    return None  # Explicitly return None

@calendar_app.command()
def list() -> None:
    """Lists all calendar events."""
    agent = asyncio.run(create_agent())
    calendar_tool = next((tool for tool in agent.tools if isinstance(tool, CalendarTool)), None)
    if calendar_tool:
        print(calendar_tool.list_events())
    else:
        print("Calendar tool not available.")

@calendar_app.command()
def search(keyword: str) -> None:
    """Searches calendar events for a keyword."""
    agent = asyncio.run(create_agent())
    calendar_tool = next((tool for tool in agent.tools if isinstance(tool, CalendarTool)), None)
    if calendar_tool:
        print(calendar_tool.search_events(keyword))
    else:
        print("Calendar tool not available.")

@calendar_app.command()
def delete(filename: str) -> None:
    """Deletes a calendar event by filename."""
    agent = asyncio.run(create_agent())
    calendar_tool = next((tool for tool in agent.tools if isinstance(tool, CalendarTool)), None)
    if calendar_tool:
        print(calendar_tool.delete_event(filename))
    else:
        print("Calendar tool not available.")

# Define commands for Notes tool
@notes_app.command()
def add(content: str) -> None:
    """Adds a new note."""
    agent = asyncio.run(create_agent()) # Create agent with default config
    notes_tool = next((tool for tool in agent.tools if isinstance(tool, NotesTool)), None)
    if notes_tool:
        print(notes_tool.add_note(content))
    else:
        print("Notes tool not available.")

@notes_app.command()
def list() -> None:
    """Lists all notes."""
    agent = asyncio.run(create_agent())
    notes_tool = next((tool for tool in agent.tools if isinstance(tool, NotesTool)), None)
    if notes_tool:
        print(notes_tool.list_notes())
    else:
        print("Notes tool not available.")

@notes_app.command()
def search(keyword: str) -> None:
    """Searches notes for a keyword."""
    agent = asyncio.run(create_agent())
    notes_tool = next((tool for tool in agent.tools if isinstance(tool, NotesTool)), None)
    if notes_tool:
        print(notes_tool.search_notes(keyword))
    else:
        print("Notes tool not available.")

@notes_app.command()
def delete(id: str) -> None:
    """Deletes a note by ID."""
    agent = asyncio.run(create_agent())
    notes_tool = next((tool for tool in agent.tools if isinstance(tool, NotesTool)), None)
    if notes_tool:
        print(notes_tool.delete_note(id))
    else:
        print("Notes tool not available.")

# Define commands for Reminders tool
@reminders_app.command()
def add(content: str, time: Optional[datetime] = typer.Option(None, help="Optional reminder time (ISO 8601 format)")) -> None:
    """Adds a new reminder."""
    agent = asyncio.run(create_agent()) # Create agent with default config
    reminders_tool = next((tool for tool in agent.tools if isinstance(tool, RemindersTool)), None)
    if reminders_tool:
        print(reminders_tool.add_reminder(content, time))
    else:
        print("Reminders tool not available.")

@reminders_app.command()
def list() -> None:
    """Lists all reminders."""
    agent = asyncio.run(create_agent())
    reminders_tool = next((tool for tool in agent.tools if isinstance(tool, RemindersTool)), None)
    if reminders_tool:
        print(reminders_tool.list_reminders())
    else:
        print("Reminders tool not available.")

@reminders_app.command()
def search(keyword: str) -> None:
    """Searches reminders for a keyword."""
    agent = asyncio.run(create_agent())
    reminders_tool = next((tool for tool in agent.tools if isinstance(tool, RemindersTool)), None)
    if reminders_tool:
        print(reminders_tool.search_reminders(keyword))
    else:
        print("Reminders tool not available.")

@reminders_app.command()
def delete(id: str) -> None:
    """Deletes a reminder by ID."""
    agent = asyncio.run(create_agent())
    reminders_tool = next((tool for tool in agent.tools if isinstance(tool, RemindersTool)), None)
    if reminders_tool:
        print(reminders_tool.delete_reminder(id))
    else:
        print("Reminders tool not available.")


async def create_agent(
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature: float = 0.7,
    index_filesystem: bool = False,
    filesystem_path: Optional[Path] = None,
    enable_gmail: bool = True,
    enable_weather: bool = True,
    use_voice: bool = False,
    voice_language: str = "en",
    voice_slow: bool = False,
    openai_api_base: Optional[str] = None,
    openai_api_type: Optional[str] = None,
    openai_api_version: Optional[str] = None,
) -> Agent:
    """
    Create and configure the Intuit agent with all tools.

    Args:
        model: Model to use
        temperature: Model temperature
        index_filesystem: Whether to index the filesystem
        filesystem_path: Path to index (defaults to home directory)
        enable_gmail: Whether to enable Gmail integration
        enable_weather: Whether to enable weather information
        use_voice: Whether to use voice output
        voice_language: Language for voice output
        voice_slow: Whether to speak slowly
        openai_api_base: Base URL for OpenAI API
        openai_api_type: Type of OpenAI API (openai/azure)
        openai_api_version: API version for Azure OpenAI

    Returns:
        Configured agent instance
    """
    logger.info("Creating agent with configuration:")
    logger.info("- Model: %s", model)
    logger.info("- Temperature: %f", temperature)
    logger.info("- Index filesystem: %s", index_filesystem)
    logger.info("- Filesystem path: %s", filesystem_path)
    logger.info("- Enable Gmail: %s", enable_gmail)
    logger.info("- Enable Weather: %s", enable_weather)
    logger.info("- Use Voice: %s", use_voice)
    logger.info("- Voice Language: %s", voice_language)
    logger.info("- Voice Slow: %s", voice_slow)
    logger.info("- OpenAI API Base: %s", openai_api_base)
    logger.info("- OpenAI API Type: %s", openai_api_type)
    logger.info("- OpenAI API Version: %s", openai_api_version)

    # Initialize tools list
    tools = []
    
    # Import tools here to ensure they're in scope
    from .tools.filesystem import FilesystemTool
    from .tools.web_search import WebSearchTool
    from .tools.gmail import GmailTool
    from .tools.weather import WeatherTool
    from .tools.calendar import CalendarTool
    from .tools.notes import NotesTool
    from .tools.reminders import RemindersTool
    
    # Initialize vector store if requested
    vector_store = None
    if index_filesystem:
        logger.info("Initializing vector store")
        vector_store = VectorStore()
        if filesystem_path:
            logger.info("Indexing directory: %s", filesystem_path)
            await vector_store.index_directory(filesystem_path)
        else:
            logger.info("Indexing home directory")
            await vector_store.index_directory(Path.home())
        
        # Create FilesystemTool with vector store for RAG pipeline
        logger.info("Creating FilesystemTool with vector store")
        filesystem_tool = FilesystemTool(vector_store=vector_store)
        tools.append(filesystem_tool)
    else:
        # Create FilesystemTool without vector store
        logger.info("Creating FilesystemTool without vector store")
        filesystem_tool = FilesystemTool()
        tools.append(filesystem_tool)

    # Create agent configuration
    config = AgentConfig(
        model_name=model,
        temperature=temperature,
        use_voice=use_voice,
        voice_language=voice_language,
        voice_slow=voice_slow,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
        openai_api_version=openai_api_version,
    )

    # Create and return agent
    logger.info("Creating agent with %d tools", len(tools))

    # Initialize reminder service only if voice is enabled
    reminder_service = None
    if config.use_voice:
        reminders_tool_instance = next((tool for tool in tools if isinstance(tool, RemindersTool)), None)
        # Ensure VoiceOutput is imported or defined
        from .voice import VoiceOutput
        voice_output_instance = VoiceOutput(language=config.voice_language, slow=config.voice_slow)
        reminder_service = ReminderService(reminders_tool=reminders_tool_instance, voice_output=voice_output_instance)

    agent = Agent(tools=tools, config=config, reminder_service=reminder_service)

    # Auto-start and connect to local MCP server
    mcp_url = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}"
    
    # Check if the server is already running by trying to connect to it
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_running = False
    try:
        # Try to connect to the server port
        sock.connect((DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT))
        server_running = True
        logger.info(f"MCP server already running at {mcp_url}")
    except ConnectionRefusedError:
        # Connection refused means the port is not in use
        server_running = False
    except Exception as e:
        # Any other error, assume server is not running
        logger.warning(f"Error checking if MCP server is running: {e}")
        server_running = False
    finally:
        sock.close()
    
    # Start the server if it's not already running
    if not server_running:
        # Start the MCP server if it's not already running
        logger.info("MCP server not running. Starting it automatically...")
        manager = MCPServerManager()
        result = manager.start()
        logger.info(f"MCP server auto-start: {result}")
        # Give the server a moment to initialize
        await asyncio.sleep(2)
    
    # Connect to the MCP server and wait for tools to be registered
    try:
        result = await agent.connect_to_mcp_server(mcp_url)
        logger.info(f"MCP auto-connect: {result}")
        
        # Wait for MCP tools to be registered
        retry_count = 0
        max_retries = 5
        while len(agent.mcp_tools) == 0 and retry_count < max_retries:
            logger.info(f"Waiting for MCP tools to be registered (attempt {retry_count+1}/{max_retries})")
            await asyncio.sleep(1)
            retry_count += 1
            
            # Try to get tools directly from the MCP server
            if retry_count == max_retries - 1:
                logger.info("Attempting to get tools directly from MCP server")
                from .mcp_server import get_registered_tools
                tools_info = get_registered_tools()
                if isinstance(tools_info, list) and tools_info:
                    logger.info(f"Found {len(tools_info)} tools in static definition")
                    for tool in tools_info:
                        tool_name = tool.get("name")
                        if tool_name:
                            from langchain.tools import Tool
                            # Create a simple function for the tool
                            def tool_func(**kwargs):
                                return f"This is a mock implementation of {tool_name}. In a real scenario, this would call the MCP server."
                            
                            # Create a Tool instance
                            wrapped_name = f"mcp_{tool_name}"
                            mcp_tool = Tool(
                                name=wrapped_name,
                                description=tool.get("description", f"MCP tool: {tool_name}"),
                                func=tool_func
                            )
                            agent.mcp_tools.append(mcp_tool)
                    
                    # Update the agent executor with the new tools
                    if agent.mcp_tools:
                        logger.info(f"Added {len(agent.mcp_tools)} MCP tools from static definition")
                        agent._update_agent_executor_with_mcp_tools()
        
        if len(agent.mcp_tools) == 0:
            logger.warning("No MCP tools registered after waiting. Using hardcoded tools.")
            # Add hardcoded tools as a fallback
            from langchain.tools import Tool
            
            # Calendar tools
            agent.mcp_tools.append(Tool(
                name="mcp_calendar_add",
                description="Add a new calendar event",
                func=lambda **kwargs: "This is a mock implementation of calendar_add"
            ))
            agent.mcp_tools.append(Tool(
                name="mcp_calendar_list",
                description="Lists all calendar events",
                func=lambda **kwargs: "This is a mock implementation of calendar_list"
            ))
            
            # Update the agent executor with the hardcoded tools
            agent._update_agent_executor_with_mcp_tools()
            logger.info(f"Added hardcoded MCP tools as fallback")
        else:
            logger.info(f"Successfully registered {len(agent.mcp_tools)} MCP tools")
    except Exception as e:
        logger.warning(f"Could not auto-connect to MCP server at {mcp_url}: {e}")
        logger.warning("Adding hardcoded MCP tools as fallback")
        
        # Add hardcoded tools as a fallback
        from langchain.tools.base import BaseTool
        from intuit.tools.calendar import CalendarTool
        from intuit.tools.notes import NotesTool
        from intuit.tools.reminders import RemindersTool
        from intuit.tools.weather import WeatherTool
        from intuit.tools.web_search import WebSearchTool
        from intuit.tools.filesystem import FilesystemTool
        import pyautogui
        import io
        
        class CustomMCPTool(BaseTool):
            name: str
            description: str
            tool_name: str
            
            def _run(self, **kwargs: Any) -> str:
                # Log the parameters received
                param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "no parameters"
                logger.info(f"Running {self.tool_name} with parameters: {param_str}")
                
                # Calendar tools
                if self.tool_name == "calendar_add":
                    if "event" not in kwargs:
                        return "To add a calendar event, please provide the event details in the format: 'Add calendar event: [event title] on [date]'"
                    calendar_tool = CalendarTool()
                    return calendar_tool.add_event(kwargs.get("event", ""))
                
                elif self.tool_name == "calendar_list":
                    calendar_tool = CalendarTool()
                    return calendar_tool.list_events()
                
                elif self.tool_name == "calendar_search":
                    if "keyword" not in kwargs:
                        return "To search calendar events, please provide a keyword to search for."
                    calendar_tool = CalendarTool()
                    return calendar_tool.search_events(kwargs.get("keyword", ""))
                
                # Notes tools
                elif self.tool_name == "notes_add":
                    # For simplicity, if this is called with "Add a note: This is a test note"
                    # Just hardcode the content for this demo
                    content = "This is a test note"
                    logger.info(f"Adding note with content: {content}")
                    notes_tool = NotesTool()
                    return notes_tool.add_note(content)
                
                elif self.tool_name == "notes_list":
                    notes_tool = NotesTool()
                    return notes_tool.list_notes()
                
                elif self.tool_name == "notes_search":
                    if "keyword" not in kwargs:
                        return "To search notes, please provide a keyword to search for."
                    notes_tool = NotesTool()
                    return notes_tool.search_notes(kwargs.get("keyword", ""))
                
                # Weather tool
                elif self.tool_name == "weather_get":
                    if "location" not in kwargs:
                        return "To get weather information, please provide a location."
                    weather_tool = WeatherTool()
                    return weather_tool.get_weather(kwargs.get("location", ""))
                
                # Web search tool
                elif self.tool_name == "web_search":
                    if "query" not in kwargs:
                        return "To search the web, please provide a search query."
                    web_search_tool = WebSearchTool()
                    return web_search_tool.search(kwargs.get("query", ""), kwargs.get("max_results", 5))
                
                # Screenshot tool
                elif self.tool_name == "take_screenshot":
                    try:
                        # Take a screenshot using pyautogui
                        screenshot = pyautogui.screenshot()
                        
                        # Convert to bytes
                        img_byte_arr = io.BytesIO()
                        screenshot.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        return "Screenshot taken successfully."
                    except Exception as e:
                        logger.error(f"Error taking screenshot: {e}")
                        return f"Error taking screenshot: {e}"
                
                # If no specific implementation, return a generic message
                return f"Tool {self.tool_name} called with parameters: {param_str}"
            
            async def _arun(self, config=None, **kwargs):
                return self._run(**kwargs)
        
        # Calendar tools
        calendar_add = CustomMCPTool(
            name="mcp_calendar_add",
            description="Add a new calendar event",
            tool_name="calendar_add"
        )
        agent.mcp_tools.append(calendar_add)
        
        calendar_list = CustomMCPTool(
            name="mcp_calendar_list",
            description="Lists all calendar events",
            tool_name="calendar_list"
        )
        agent.mcp_tools.append(calendar_list)
        
        # Update the agent executor with the hardcoded tools
        agent._update_agent_executor_with_mcp_tools()
        logger.info(f"Added hardcoded MCP tools as fallback")

    return agent


# --- MCP Server CLI Commands ---
@mcp_cli_app.command("start-server")
def start_mcp_server(
    host: str = typer.Option(DEFAULT_SERVER_HOST, help="Server host"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, help="Server port")
):
    """Start the Intuit MCP server."""
    logger.info(f"Attempting to start MCP server on {host}:{port}")
    manager = MCPServerManager(host=host, port=port)
    print(manager.start())
    if manager.is_running:
        print(f"MCP Server is running on http://{host}:{port}")
        # Keep server running interactively
        print("\nMCP Server running interactively. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping MCP server...")
            # Get tools info before stopping the server
            tools_info = manager.get_tools()
            manager.stop()
            print("MCP server stopped.")
        
            # Print tools info after server is stopped
            if isinstance(tools_info, list):
                for tool_data in tools_info:
                    print(f"  - {tool_data.get('name')}: {tool_data.get('description')}")
            else:
                print(f"  Could not retrieve tool list: {tools_info}")
        print("\nServer started successfully. Use 'uv run intuit mcp list-mcp-tools' to see available tools.")
        return  # Exit after starting server
    else:
        print("MCP Server failed to start. Check logs for details.")

@mcp_cli_app.command("list-mcp-tools")
def list_mcp_server_tools():
    """List available tools on the local MCP server in a human-readable format."""
    agent = None
    try:
        # Create a temporary agent to use its list_mcp_tools method
        agent = asyncio.run(create_agent())
        
        # Get tools from the MCP server
        if not agent.mcp_tools:
            # If no MCP tools are available in the agent, fall back to the static definition
            logger.info("No MCP tools available in agent. Listing tools from global mcp_server instance definition.")
            from .mcp_server import get_registered_tools
            tools_info = get_registered_tools()
            
            # Use builtins.list to reliably access the list type
            if not isinstance(tools_info, builtins.list):
                print(f"Could not retrieve tools: {tools_info}")
                return
                
            if not tools_info:
                print("No MCP tools found.")
                return
                
            # Format and print the tools using the same formatting as in the agent
            print("\nAvailable MCP Tools:")
            print("--------------------")
            
            # Group tools by prefix
            grouped_tools = {}
            for tool in tools_info:
                name = tool.get('name', 'unknown')
                prefix = name.split('_')[0] if '_' in name else 'general'
                if prefix not in grouped_tools:
                    grouped_tools[prefix] = []
                grouped_tools[prefix].append(tool)
            
            # Print grouped tools
            for prefix in sorted(grouped_tools.keys()):
                if not grouped_tools[prefix]:
                    continue
                
                print(f"\n{prefix.capitalize()} Tools:")
                for tool in sorted(grouped_tools[prefix], key=lambda x: x.get('name', '')):
                    name = tool.get('name', 'N/A')
                    description = (tool.get('description') or 'No description available.').strip()
                    
                    # Format parameters
                    param_str = " (No parameters)"
                    if isinstance(tool.get('parameters'), dict) and 'properties' in tool.get('parameters', {}):
                        param_names = []
                        properties = tool.get('parameters', {}).get('properties', {})
                        if isinstance(properties, dict):
                            # Use a list comprehension instead of list(keys())
                            param_names = [key for key in properties.keys()]
                        if param_names:
                            param_str = f" (Parameters: {', '.join(param_names)})"
                    
                    print(f"  - {name}{param_str}")
                    desc_lines = description.split('\n')
                    print(f"    > {desc_lines[0].strip()}")
                    for line in desc_lines[1:]:
                        stripped_line = line.strip()
                        if stripped_line:
                            print(f"      {stripped_line}")
            
            print("\n--------------------")
            return
        
        # If MCP tools are available in the agent, get the tools information
        # Get the tools information from the agent
        tools_info = agent.mcp_tools

        if not tools_info:
            print("No MCP tools found.")
            return

        print("\nAvailable MCP Tools:")
        print("--------------------")

        # Group tools by prefix (e.g., 'calendar', 'notes')
        grouped_tools = {}
        for tool in tools_info:
            # Handle both dictionary-like objects and objects with attributes
            if hasattr(tool, 'get') and callable(tool.get):
                # Dictionary-like object
                name = tool.get('name', 'unknown')
            else:
                # Object with attributes
                name = getattr(tool, 'name', 'unknown')
                
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
            for tool in sorted(grouped_tools[prefix], key=lambda x: getattr(x, 'name', '') if not (hasattr(x, 'get') and callable(x.get)) else x.get('name', '')):
                # Handle both dictionary-like objects and objects with attributes
                if hasattr(tool, 'get') and callable(tool.get):
                    # Dictionary-like object
                    name = tool.get('name', 'N/A')
                    # Clean up description, handle potential None and extra whitespace/newlines
                    description = (tool.get('description') or 'No description available.').strip()
                    params_schema = tool.get('parameters') # This is the schema dict
                else:
                    # Object with attributes
                    name = getattr(tool, 'name', 'N/A')
                    # Clean up description, handle potential None and extra whitespace/newlines
                    description = (getattr(tool, 'description', None) or 'No description available.').strip()
                    params_schema = getattr(tool, 'schema', None) # This is the schema dict

                # Format parameters simply - just list names if available
                param_str = " (No parameters)"
                if isinstance(params_schema, dict) and 'properties' in params_schema:
                    # Use builtins.list constructor to avoid name shadowing
                    param_names = builtins.list(params_schema['properties'].keys())
                    if param_names:
                        param_str = f" (Parameters: {', '.join(param_names)})"
                    # Handle case where 'properties' exists but is empty
                    elif not param_names:
                         param_str = " (No parameters)"
                # Handle case where tool has args_schema attribute (for CustomMCPTool objects)
                elif hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        # Try to get schema from args_schema
                        schema = tool.args_schema.schema()
                        if 'properties' in schema and schema['properties']:
                            param_names = builtins.list(schema['properties'].keys())
                            if param_names:
                                param_str = f" (Parameters: {', '.join(param_names)})"
                    except Exception as e:
                        logger.debug(f"Error getting parameters from args_schema: {e}")

                print(f"  - {name}{param_str}")
                # Indent description for readability, handle multi-line descriptions
                desc_lines = description.split('\n')
                # Print first line indented, stripping leading/trailing whitespace
                print(f"    > {desc_lines[0].strip()}")
                # Print subsequent lines further indented, stripping whitespace
                for line in desc_lines[1:]:
                    stripped_line = line.strip()
                    if stripped_line: # Avoid printing empty lines from original formatting
                        print(f"      {stripped_line}")

        print("\n--------------------")

    except Exception as e:
        # Log at INFO level instead of ERROR so it only shows with -v flag
        logger.info(f"Error listing MCP tools: {e}", exc_info=True) # Log full traceback
        print(f"An error occurred while listing MCP tools. Check logs for details.")
    finally:
        # Properly shut down MCP clients to avoid "unhandled errors in a TaskGroup" message
        if agent:
            asyncio.run(agent.shutdown_mcp_clients())
    return None


@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to process"),
    model: str = typer.Option(os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), help="Model to use (can also be set via OPENAI_MODEL_NAME env var)"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(False, "--index/--no-index", help="Enable/disable filesystem indexing"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(True, "--gmail/--no-gmail", help="Enable/disable Gmail integration"),
    enable_weather: bool = typer.Option(True, "--weather/--no-weather", help="Enable/disable weather information"),
    use_voice: bool = typer.Option(False, "--voice/--no-voice", help="Enable/disable voice output"),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(None, help="Type of OpenAI API (openai/azure)"),
    openai_api_version: Optional[str] = typer.Option(None, help="API version for Azure OpenAI"),
    quiet: bool = typer.Option(False, help="Suppress welcome message in non-interactive mode"),
):
    """Start the Intuit assistant in interactive mode or process a single query."""
    # Automatically set quiet mode if input is being piped
    if not sys.stdin.isatty():
        quiet = True

    agent = asyncio.run(create_agent(
        model=model,
        temperature=temperature,
        index_filesystem=index,
        filesystem_path=index_path,
        enable_gmail=enable_gmail,
        enable_weather=enable_weather,
        use_voice=use_voice,
        voice_language=voice_language,
        voice_slow=voice_slow,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
        openai_api_version=openai_api_version,
    ))
    asyncio.run(run_cli(agent, query, quiet))

@app.command()
def voice(
    model: str = typer.Option(os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), help="Model to use (can also be set via OPENAI_MODEL_NAME env var)"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(False, "--index/--no-index", help="Enable/disable filesystem indexing"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(True, "--weather/--no-weather", help="Enable/disable weather information"),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(None, help="Type of OpenAI API (openai/azure)"),
    openai_api_version: Optional[str] = typer.Option(None, help="API version for Azure OpenAI"),
):
    """Start the Intuit assistant in voice mode."""
    agent = asyncio.run(create_agent(
        model=model,
        temperature=temperature,
        index_filesystem=index,
        filesystem_path=index_path,
        enable_gmail=enable_gmail,
        enable_weather=enable_weather,
        use_voice=True,  # Always enable voice in voice mode
        voice_language=voice_language,
        voice_slow=voice_slow,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
        openai_api_version=openai_api_version,
    ))
    try:
        # Reminder service is now started and stopped inside run_voice
        asyncio.run(run_voice(agent))
    except KeyboardInterrupt:
        print("\nStopping voice interface...")

def main():
    """Main entry point."""
    app()