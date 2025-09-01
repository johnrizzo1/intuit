"""
Agent factory for creating and configuring Intuit agents.
"""

import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from typing import Optional, Dict

from .agent import Agent, AgentConfig
from .tools.web_search import WebSearchTool
from .tools.gmail import GmailTool
from .tools.weather import WeatherTool
from .tools.hackernews import HackerNewsTool
from .tools.filesystem import FilesystemTool
from .tools.calendar import CalendarTool
from .tools.notes import NotesTool
from .tools.reminders import RemindersTool
from .vector_store.indexer import VectorStore
from .reminders_service import ReminderService
from .memory.chroma_store import ChromaMemoryStore
from .memory.manager import IntuitMemoryManager
from .mcp_server import MCPServerManager, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from .logging_config import configure_logging, aggressive_silence

logger = logging.getLogger(__name__)


async def create_agent(
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
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
    quiet: bool = True,
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
        quiet: Whether to silence loggers

    Returns:
        Configured agent instance
    """
    # Configure logging
    if quiet:
        aggressive_silence()
    else:
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

    # Initialize tools
    tools = await _create_tools(
        index_filesystem=index_filesystem,
        filesystem_path=filesystem_path,
        enable_gmail=enable_gmail,
        enable_weather=enable_weather,
    )

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

    # Initialize reminder service if voice is enabled
    reminder_service = None
    if config.use_voice:
        reminders_tool_instance = next(
            (tool for tool in tools if isinstance(tool, RemindersTool)), None
        )
        if reminders_tool_instance:
            from .voice import VoiceOutput

            voice_output_instance = VoiceOutput(
                language=config.voice_language, slow=config.voice_slow
            )
            reminder_service = ReminderService(
                reminders_tool=reminders_tool_instance,
                voice_output=voice_output_instance,
            )

    # Initialize memory store and manager
    memory_store, memory_manager = await _create_memory_components(model)

    # Create agent
    logger.info("Creating agent with %d tools", len(tools))
    agent = Agent(tools=tools, config=config, reminder_service=reminder_service)

    # Set up memory components
    if memory_store:
        agent.memory_store = memory_store
    if memory_manager:
        agent.memory_manager = memory_manager

    # Auto-connect to MCP server
    await _setup_mcp_connection(agent)

    return agent


async def _create_tools(
    index_filesystem: bool = False,
    filesystem_path: Optional[Path] = None,
    enable_gmail: bool = True,
    enable_weather: bool = True,
) -> list:
    """Create and configure all tools for the agent."""
    tools = []

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

    # Add web search tool
    logger.info("Adding WebSearchTool")
    web_search_tool = WebSearchTool()
    tools.append(web_search_tool)

    # Add weather tool if enabled
    if enable_weather:
        logger.info("Adding WeatherTool")
        weather_tool = WeatherTool()
        tools.append(weather_tool)

    # Add HackerNews tool
    logger.info("Adding HackerNewsTool")
    hackernews_tool = HackerNewsTool()
    tools.append(hackernews_tool)

    # Add Gmail tool if enabled
    if enable_gmail:
        logger.info("Adding GmailTool")
        gmail_tool = GmailTool()
        tools.append(gmail_tool)

    # Add calendar tool
    logger.info("Adding CalendarTool")
    calendar_tool = CalendarTool()
    tools.append(calendar_tool)

    # Add notes tool
    logger.info("Adding NotesTool")
    notes_tool = NotesTool()
    tools.append(notes_tool)

    # Add reminders tool
    logger.info("Adding RemindersTool")
    reminders_tool = RemindersTool()
    tools.append(reminders_tool)

    return tools


async def _create_memory_components(model: str) -> tuple:
    """Create memory store and manager components."""
    memory_store = None
    memory_manager = None

    try:
        logger.info("Initializing memory store")
        memory_store = ChromaMemoryStore(model=model)
        logger.info("Initializing memory manager")
        memory_manager = IntuitMemoryManager(store=memory_store, model=model)
        logger.info("Memory store and manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory store or manager: {e}")
        memory_store = None
        memory_manager = None

    return memory_store, memory_manager


async def _setup_mcp_connection(agent: Agent) -> None:
    """Set up MCP server connection and tools."""
    mcp_url = f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}"

    # Check if the server is already running
    server_running = _is_mcp_server_running()

    # Start the server if it's not already running
    if not server_running:
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
        await _wait_for_mcp_tools(agent)

        if len(agent.mcp_tools) == 0:
            logger.warning(
                "No MCP tools registered after waiting. Using hardcoded tools."
            )
            _add_fallback_mcp_tools(agent)
        else:
            logger.info(f"Successfully registered {len(agent.mcp_tools)} MCP tools")
    except Exception as e:
        logger.warning(f"Could not auto-connect to MCP server at {mcp_url}: {e}")
        logger.warning("Adding hardcoded MCP tools as fallback")
        _add_fallback_mcp_tools(agent)


def _is_mcp_server_running() -> bool:
    """Check if MCP server is already running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_running = False
    try:
        sock.connect((DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT))
        server_running = True
        logger.info(
            f"MCP server already running at http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT}"
        )
    except ConnectionRefusedError:
        server_running = False
    except Exception as e:
        logger.warning(f"Error checking if MCP server is running: {e}")
        server_running = False
    finally:
        sock.close()

    return server_running


async def _wait_for_mcp_tools(agent: Agent) -> None:
    """Wait for MCP tools to be registered with the agent."""
    retry_count = 0
    max_retries = 5
    while len(agent.mcp_tools) == 0 and retry_count < max_retries:
        logger.info(
            f"Waiting for MCP tools to be registered (attempt {retry_count+1}/{max_retries})"
        )
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
                        # Create a BaseTool instance using the project's BaseTool
                        from .tools.basetool import BaseTool

                        class SimpleMCPTool(BaseTool):
                            def __init__(self, name: str, description: str, **kwargs):
                                super().__init__(
                                    name=name, description=description, **kwargs
                                )

                            async def _arun(self, **kwargs):
                                return {
                                    "result": f"This is a mock implementation of {tool_name}. In a real scenario, this would call the MCP server."
                                }

                        wrapped_name = f"mcp_{tool_name}"
                        description = tool.get("description", f"MCP tool: {tool_name}")
                        if not isinstance(description, str):
                            description = str(description)
                        mcp_tool = SimpleMCPTool(
                            name=wrapped_name, description=description
                        )
                        agent.mcp_tools.append(mcp_tool)

                # Update the agent executor with the new tools
                if agent.mcp_tools:
                    logger.info(
                        f"Added {len(agent.mcp_tools)} MCP tools from static definition"
                    )
                    agent._update_agent_executor_with_mcp_tools()


def _add_fallback_mcp_tools(agent: Agent) -> None:
    """Add hardcoded MCP tools as fallback when server connection fails."""
    from .tools.basetool import BaseTool
    from typing import Any
    import pyautogui
    import io

    class CustomMCPTool(BaseTool):
        tool_name: str

        def __init__(self, name: str, description: str, tool_name: str, **kwargs):
            super().__init__(name=name, description=description, **kwargs)
            self.tool_name = tool_name

        async def _arun(self, **kwargs: Any) -> Dict[str, Any]:
            # Log the parameters received
            param_str = (
                ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                if kwargs
                else "no parameters"
            )
            logger.info(f"Running {self.tool_name} with parameters: {param_str}")

            # Calendar tools
            if self.tool_name == "calendar_add":
                if "event" not in kwargs:
                    return {
                        "result": "To add a calendar event, please provide the event details in the format: 'Add calendar event: [event title] on [date]'"
                    }
                from .tools.calendar import CalendarTool

                calendar_tool = CalendarTool()
                result = calendar_tool.add_event(kwargs.get("event", ""))
                return {"result": result}

            elif self.tool_name == "calendar_list":
                from .tools.calendar import CalendarTool

                calendar_tool = CalendarTool()
                result = calendar_tool.list_events()
                return {"result": result}

            elif self.tool_name == "calendar_search":
                if "keyword" not in kwargs:
                    return {
                        "result": "To search calendar events, please provide a keyword to search for."
                    }
                from .tools.calendar import CalendarTool

                calendar_tool = CalendarTool()
                result = calendar_tool.search_events(kwargs.get("keyword", ""))
                return {"result": result}

            # Notes tools
            elif self.tool_name == "notes_add":
                content = kwargs.get("content", "This is a test note")
                logger.info(f"Adding note with content: {content}")
                from .tools.notes import NotesTool

                notes_tool = NotesTool()
                result = notes_tool.add_note(content)
                return {"result": result}

            elif self.tool_name == "notes_list":
                from .tools.notes import NotesTool

                notes_tool = NotesTool()
                result = notes_tool.list_notes()
                return {"result": result}

            elif self.tool_name == "notes_search":
                if "keyword" not in kwargs:
                    return {
                        "result": "To search notes, please provide a keyword to search for."
                    }
                from .tools.notes import NotesTool

                notes_tool = NotesTool()
                result = notes_tool.search_notes(kwargs.get("keyword", ""))
                return {"result": result}

            # Weather tool
            elif self.tool_name == "weather_get":
                if "location" not in kwargs:
                    return {
                        "result": "To get weather information, please provide a location."
                    }
                from .tools.weather import WeatherTool

                weather_tool = WeatherTool()
                result = await weather_tool._arun(location=kwargs.get("location", ""))
                return result

            # Web search tool
            elif self.tool_name == "web_search":
                if "query" not in kwargs:
                    return {
                        "result": "To search the web, please provide a search query."
                    }
                from .tools.web_search import WebSearchTool

                web_search_tool = WebSearchTool()
                result = await web_search_tool._arun(
                    kwargs.get("query", ""), kwargs.get("max_results", 5)
                )
                return result

            # Screenshot tool
            elif self.tool_name == "take_screenshot":
                try:
                    # Take a screenshot using pyautogui
                    screenshot = pyautogui.screenshot()

                    # Convert to bytes
                    img_byte_arr = io.BytesIO()
                    screenshot.save(img_byte_arr, format="PNG")
                    img_bytes = img_byte_arr.getvalue()

                    return {"result": "Screenshot taken successfully."}
                except Exception as e:
                    logger.error(f"Error taking screenshot: {e}")
                    return {"result": f"Error taking screenshot: {e}"}

            # If no specific implementation, return a generic message
            return {
                "result": f"Tool {self.tool_name} called with parameters: {param_str}"
            }

    # Calendar tools
    calendar_add = CustomMCPTool(
        name="mcp_calendar_add",
        description="Add a new calendar event",
        tool_name="calendar_add",
    )
    agent.mcp_tools.append(calendar_add)

    calendar_list = CustomMCPTool(
        name="mcp_calendar_list",
        description="Lists all calendar events",
        tool_name="calendar_list",
    )
    agent.mcp_tools.append(calendar_list)

    # Update the agent executor with the hardcoded tools
    agent._update_agent_executor_with_mcp_tools()
    logger.info(f"Added hardcoded MCP tools as fallback")
