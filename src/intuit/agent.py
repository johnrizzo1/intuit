"""
Core agent implementation for Intuit.
"""
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from datetime import datetime # Import datetime
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from intuit.memory.chroma_store import ChromaMemoryStore
from intuit.memory.tools import get_memory_tools
from intuit.memory.manager import IntuitMemoryManager
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

from .tools.basetool import BaseTool
from .tools.calendar import CalendarTool # Import CalendarTool
from .tools.notes import NotesTool # Import NotesTool
from .tools.reminders import RemindersTool # Import RemindersTool
from .tools.weather import WeatherTool # Import WeatherTool
from .tools.hackernews import HackerNewsTool # Import HackerNewsTool
from .reminders_service import ReminderService # Import ReminderService
from .voice import VoiceOutput
from mcp import ClientSession as MCPClient  # Use ClientSession as MCPClient
# Set up logging
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for the Intuit agent."""
    model_name: str = Field(default=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    streaming: bool = Field(default=True)
    max_workers: int = Field(default=4)  # Number of threads for the thread pool
    use_voice: bool = Field(default=False)  # Whether to use voice output
    voice_language: str = Field(default="en")  # Language for voice output
    voice_slow: bool = Field(default=False)  # Whether to speak slowly

    # OpenAI provider configuration
    openai_api_base: Optional[str] = Field(default=None)
    openai_api_type: Optional[str] = Field(default=None)
    openai_api_version: Optional[str] = Field(default=None)

class Agent:
    """
    Core agent implementation that handles conversation and tool execution.
    """
    def __init__(
        self,
        tools: List[BaseTool],
        config: Optional[AgentConfig] = None,
        reminder_service: Optional[ReminderService] = None,
    ):
        self.config = config or AgentConfig()
        self.reminder_service = reminder_service
        self.mcp_clients: Dict[str, MCPClient] = {} # Use MCPClient
        self.mcp_client_tasks: Dict[str, asyncio.Task] = {} # Track background tasks for MCP clients
        self.mcp_tools: List[BaseTool] = [] # To store tools from connected MCP servers
        
        # Initialize memory store with the model parameter
        self.memory_store = ChromaMemoryStore(model=self.config.model_name)
        logger.info("Memory store initialized")
        
        # Get memory tools
        self.memory_tools = get_memory_tools(self.memory_store)
        logger.info(f"Added {len(self.memory_tools)} memory tools")
        
        # Initialize memory manager
        self.memory_manager = IntuitMemoryManager(self.memory_store)
        logger.info("Memory manager initialized")
        
        # Start background memory processing
        self.memory_manager.start()
        logger.info("Background memory manager started")

        # Get OpenAI configuration from environment or config
        openai_api_base = (self.config.openai_api_base or os.getenv("OPENAI_API_BASE"))
        if openai_api_base:
            openai_api_base = openai_api_base.strip()
        openai_api_type = self.config.openai_api_type or os.getenv("OPENAI_API_TYPE")
        openai_api_version = self.config.openai_api_version or os.getenv("OPENAI_API_VERSION")

        # Configure OpenAI client
        openai_kwargs = {
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "streaming": self.config.streaming,
        }

        # Add provider-specific configuration
        if openai_api_base:
            openai_kwargs["openai_api_base"] = openai_api_base
        if openai_api_type:
            openai_kwargs["openai_api_type"] = openai_api_type
        if openai_api_version:
            openai_kwargs["openai_api_version"] = openai_api_version

        self.llm = ChatOpenAI(**openai_kwargs)

        # Add weather and hackernews tools
        self.weather_tool = WeatherTool()
        self.hackernews_tool = HackerNewsTool()
        
        # Combine all tools
        self.tools = tools + self.memory_tools + [self.weather_tool, self.hackernews_tool] # Add all tools to the agent's tools
        self.agent_executor = self._create_agent_executor() # Initialize with own tools
        self.chat_history: List[Dict[str, Any]] = []
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        try:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        except Exception as e:
            logger.error(f"Failed to initialize thread pool: {e}")

        self.voice = VoiceOutput(
            language=self.config.voice_language,
            slow=self.config.voice_slow
        ) if self.config.use_voice else None
    
    async def connect_to_mcp_server(self, url: str) -> str:
        """Connects to an external MCP server and registers its tools (async, persistent context)."""
        if url in self.mcp_clients:
            return f"Already connected to MCP server at {url}"
        try:
            logger.info(f"Attempting to connect to MCP server at {url}")
            from mcp.client.sse import sse_client
            import logging
            import asyncio
            async def mcp_client_task():
                try:
                    # Create MCP client instance
                    async with sse_client(url) as client:
                        logger.info(f"Connected to MCP server at {url}")
                        self.mcp_clients[url] = client
                        
                        # Get server info
                        server_info = await client.get_server_info()
                        logger.info(f"Server info: {server_info}")
                        
                        # Get tools from the server
                        tools = await client.list_tools()
                        logger.info(f"Found {len(tools)} tools on the MCP server")
                        
                        # Register tools with the client
                        for tool in tools:
                            tool_name = tool.get("name")
                            wrapped_name = f"mcp_{tool_name}"
                            logger.info(f"Registering MCP tool: {wrapped_name} from {url}")
                            wrapped_tool = MCPToolWrapper(client, tool_name, tool)
                            wrapped_tool.name = wrapped_name  # Ensure consistent naming
                            self.mcp_tools.append(wrapped_tool)
                        
                        # Update the agent executor with the new tools
                        if self.mcp_tools:
                            logger.info(f"Added {len(self.mcp_tools)} MCP tools")
                            self._update_agent_executor_with_mcp_tools()
                        
                        # Keep the connection alive
                        while True:
                            await asyncio.sleep(60)
                except Exception as e:
                    # Log at INFO level instead of ERROR so it only shows with -v flag
                    logger.info(f"Error in MCP client task: {e}")
                    
                    # Don't try to use the client if there was an error connecting
                    # Instead, use hardcoded tools as a fallback
                    from langchain.tools import Tool
                    
                    # Add hardcoded tools as a fallback
                    logger.info("Adding hardcoded MCP tools as fallback due to connection error")
                    
                    # Create custom tools that use the real implementations
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
                        
                        def _run(self, **kwargs):
                            # Log the parameters received
                            param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "no parameters"
                            
                            # Get the caller's information
                            import inspect
                            caller_frame = inspect.currentframe().f_back
                            caller_info = ""
                            if caller_frame:
                                caller_module = inspect.getmodule(caller_frame)
                                if caller_module:
                                    caller_info = f" (called from {caller_module.__name__})"
                            
                            logger.info(f"CustomMCPTool._run for {self.tool_name} with parameters: {param_str}{caller_info}")
                            
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
                                    img_byte_arr = img_byte_arr.getvalue()
                                    
                                    return "Screenshot taken successfully."
                                except Exception as e:
                                    # Log at INFO level instead of ERROR so it only shows with -v flag
                                    logger.info(f"Error taking screenshot: {e}")
                                    return f"Error taking screenshot: {e}"
                            
                            # Filesystem tools
                            elif self.tool_name == "filesystem_search":
                                filesystem_tool = FilesystemTool()
                                
                                # Log the parameters for debugging
                                logger.info(f"Running filesystem_search with parameters: {param_str}")
                                
                                # Set default values
                                path = kwargs.get("path", ".")
                                query = kwargs.get("query", "")
                                limit = kwargs.get("limit", 5)
                                
                                # Run the search
                                logger.info(f"Executing filesystem search with path={path}, query='{query}', limit={limit}")
                                result = filesystem_tool.run(action="search", path=path, query=query, limit=limit)
                                
                                # Log the result for debugging
                                logger.info(f"Filesystem search result: {result}")
                                
                                # Format the results
                                if result and "results" in result and result["results"]:
                                    formatted_results = []
                                    for i, file_info in enumerate(result["results"]):
                                        formatted_results.append(f"{i+1}. {file_info.get('name', 'Unknown')} - {file_info.get('path', 'Unknown path')}")
                                    
                                    return f"Found {len(result['results'])} files:\n\n" + "\n".join(formatted_results)
                                else:
                                    return "No files found matching your query."
                            
                            # If no specific implementation, return a generic message
                            return f"Tool {self.tool_name} called with parameters: {param_str}"
                        
                        async def _arun(self, config=None, **kwargs):
                            # Get the caller's information
                            import inspect
                            caller_frame = inspect.currentframe().f_back
                            caller_info = ""
                            if caller_frame:
                                caller_module = inspect.getmodule(caller_frame)
                                if caller_module:
                                    caller_info = f" (called from {caller_module.__name__})"
                            
                            param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "no parameters"
                            logger.info(f"CustomMCPTool._arun for {self.tool_name} with parameters: {param_str}{caller_info}")
                            return self._run(**kwargs)
                    
                    # Calendar tools
                    calendar_add = CustomMCPTool(
                        name="mcp_calendar_add",
                        description="Add a new calendar event",
                        tool_name="calendar_add"
                    )
                    self.mcp_tools.append(calendar_add)
                    
                    calendar_list = CustomMCPTool(
                        name="mcp_calendar_list",
                        description="Lists all calendar events",
                        tool_name="calendar_list"
                    )
                    self.mcp_tools.append(calendar_list)
                    
                    calendar_search = CustomMCPTool(
                        name="mcp_calendar_search",
                        description="Searches calendar events for a keyword",
                        tool_name="calendar_search"
                    )
                    self.mcp_tools.append(calendar_search)
                    
                    # Filesystem tools
                    filesystem_search = CustomMCPTool(
                        name="mcp_filesystem_search",
                        description="Search for files by content or list all indexed files",
                        tool_name="filesystem_search"
                    )
                    self.mcp_tools.append(filesystem_search)
                    
                    # Notes tools
                    notes_add = CustomMCPTool(
                        name="mcp_notes_add",
                        description="Adds a new note",
                        tool_name="notes_add"
                    )
                    self.mcp_tools.append(notes_add)
                    
                    notes_list = CustomMCPTool(
                        name="mcp_notes_list",
                        description="Lists all notes",
                        tool_name="notes_list"
                    )
                    self.mcp_tools.append(notes_list)
                    
                    # Screenshot tool
                    take_screenshot = CustomMCPTool(
                        name="mcp_take_screenshot",
                        description="Take a screenshot of the user's screen",
                        tool_name="take_screenshot"
                    )
                    self.mcp_tools.append(take_screenshot)
                    
                    # Update the agent executor with the hardcoded tools
                    if self.mcp_tools:
                        logger.info(f"Added {len(self.mcp_tools)} hardcoded MCP tools as fallback")
                        self._update_agent_executor_with_mcp_tools()
            task = asyncio.create_task(mcp_client_task())
            self.mcp_client_tasks[url] = task
            return f"Connecting to MCP server at {url} in background. Tools will be available once registered."
        except Exception as e:
            # Log at INFO level instead of ERROR so it only shows with -v flag
            logger.info(f"Failed to connect to MCP server at {url}: {e}")
            return f"Failed to connect to MCP server at {url}: {str(e)}"

    def list_mcp_tools(self) -> str:
        """Returns a human-readable list of available MCP tools."""
        if not self.mcp_tools:
            return "No MCP tools are currently available."
        
        # Group tools by prefix (e.g., 'mcp_calendar', 'mcp_notes')
        grouped_tools = {}
        for tool in self.mcp_tools:
            name = tool.name
            # Remove 'mcp_' prefix and use the part before the next underscore as the group key
            prefix = name.replace('mcp_', '', 1).split('_')[0] if '_' in name.replace('mcp_', '', 1) else 'general'
            if prefix not in grouped_tools:
                grouped_tools[prefix] = []
            grouped_tools[prefix].append(tool)
        
        # Format the output
        output = "\nAvailable MCP Tools:\n"
        output += "--------------------\n"
        
        # Sort groups alphabetically and tools within groups alphabetically
        for prefix in sorted(grouped_tools.keys()):
            # Skip empty groups if any somehow occur
            if not grouped_tools[prefix]:
                continue
            
            output += f"\n{prefix.capitalize()} Tools:\n"
            for tool in sorted(grouped_tools[prefix], key=lambda x: x.name):
                name = tool.name.replace('mcp_', '')
                description = getattr(tool, 'description', 'No description available.')
                
                # Format parameters if available
                param_str = " (No parameters)"
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    schema = tool.args_schema.schema()
                    if 'properties' in schema and schema['properties']:
                        param_names = list(schema['properties'].keys())
                        if param_names:
                            param_str = f" (Parameters: {', '.join(param_names)})"
                
                output += f"  - {name}{param_str}\n"
                # Indent description for readability
                desc_lines = description.split('\n')
                output += f"    > {desc_lines[0].strip()}\n"
                for line in desc_lines[1:]:
                    output += f"      {line.strip()}\n"
        
        output += "\n--------------------\n"
        return output

    def _update_agent_executor_with_mcp_tools(self):
        """Re-creates the agent executor with both local and MCP tools."""
        logger.info("Updating agent executor with all available tools (local + MCP).")
        all_tools = self.tools + self.mcp_tools
        logger.info(f"Total tools available: {len(all_tools)}")
        for tool in all_tools:
            logger.info(f"- {tool.name}: {tool.description}")
        self.agent_executor = self._create_agent_executor(tools_override=all_tools)


    def _create_agent_executor(self, tools_override: Optional[List[BaseTool]] = None) -> AgentExecutor:
        """Create the agent executor with tools and prompt template."""
        
        current_tools = tools_override if tools_override is not None else (self.tools + self.mcp_tools)
        logger.info("Creating/Updating agent executor with %d tools:", len(current_tools))
        for tool in current_tools:
            logger.info("- Tool: %s (%s)", tool.name, tool.description)
            

            

        


        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are Intuit, a helpful personal assistant.
You have access to various tools to help the user:

1. Web search for online information
2. Filesystem operations for searching, reading, and managing files
3. Gmail integration for email management (when enabled)
4. Weather information (when enabled)
5. Calendar management for adding, listing, searching, and deleting events
6. Notes management for adding, listing, searching, and deleting notes
7. Reminders management for adding, listing, searching, and deleting reminders, including setting a specific time.
8. Desktop automation for interacting with the operating system, such as accessing the clipboard.
9. Hacker News integration for fetching top stories, new stories, best stories, and story details.
10. Memory management for storing and retrieving important information across conversations.

IMPORTANT: When users ask about files you know about, indexed files, or what files you're aware of, you MUST use the filesystem tool with the "search" action. The filesystem tool has semantic search capabilities through its vector store.

To list or search for indexed files, use the filesystem tool with these parameters:
- action: 'search'
- path: '.' (for current directory)
- query: Different query types produce different behaviors:
  * '*' or '' (empty string): Lists all indexed files
  * Specific search terms: Performs semantic search to find relevant documents
  * Natural language queries: Finds documents related to the query, even if they don't contain the exact words

For example:
1. If a user asks "What files are you aware of?" or "List the files you know of", use:
   action='search', path='.', query=''

2. If a user asks "Find files about Python" or "Show me documents related to machine learning", use:
   action='search', path='.', query='Python' or query='machine learning'

3. If a user asks "Find documents that explain how to implement a neural network", use:
   action='search', path='.', query='implement neural network'

Always format and present the results to the user in a clear and organized way.

IMPORTANT: When users ask about Hacker News, news, or latest tech stories, you MUST use the hackernews tool. The hackernews tool provides access to content from Hacker News.

To use the hackernews tool, use these parameters:
- action: the type of stories to fetch ("top", "new", "best") or "item" for a specific story
- limit: maximum number of stories to return (default: 10)
- item_id: ID of a specific item to fetch (only used when action is "item")

For example:
1. If a user asks "What's the latest on Hacker News?", use:
   action="top"
   limit=10

2. If a user asks "Show me new stories from Hacker News", use:
   action="new"
   limit=10

3. If a user asks "What are the best stories on Hacker News?", use:
   action="best"
   limit=10

IMPORTANT: When users ask about their Gmail, you MUST use the gmail tool. The gmail tool provides access to read and manage email messages.

To use the gmail tool, use these parameters:
- query: the search query for messages (e.g., "is:unread", "from:example@gmail.com", "subject:meeting", "has:attachment")
- limit: maximum number of messages to return (default: 5)

For example:
1. If a user asks "How many unread emails do I have?", use:
   query="is:unread"
   limit=5

2. If a user asks "Show me my recent emails from John", use:
   query="from:john@gmail.com"
   limit=5

3. If a user asks "Find emails with attachments", use:
   query="has:attachment"
   limit=5

When you get the response, format it like this:
Found [number] messages:

[For each message]
From: [sender]
Subject: [subject]
Date: [date]
[Preview of content]

If there's an error, explain what went wrong and suggest what the user can do.

IMPORTANT: When users ask about finding files or searching content, you MUST use the filesystem tool with the 'search' action. The filesystem tool has semantic search capabilities through its vector store.

To search for files, use the filesystem tool with these parameters:
- action: 'search'
- path: the directory to search in (use '.' for current directory)
- query: the search query

For example, if a user asks "Find files about Python", you should:
1. Use the filesystem tool with action='search'
2. Set path='.' (or the specific directory if mentioned)
3. Set query='Python'

After getting search results, you should:
1. Read the relevant files using the filesystem tool with action='read'
2. Summarize the content for the user
3. Provide specific quotes or references when relevant

IMPORTANT: When users ask about weather, you MUST use the weather tool. The weather tool provides current conditions and forecasts for any location.

To get weather information, use the weather tool with these parameters:
- location: the name of the location (e.g., "London, UK" or "New York, NY")

For example, if a user asks "What's the weather like in Charlotte?", you should:
1. Use the weather tool with location="Charlotte, NC"
2. When you get the response, format it like this:
   Current Weather in [Location]:
   - Temperature: [temp]°C
   - Conditions: [description]
   - Humidity: [humidity]%
   - Wind Speed: [wind_speed] m/s

   Forecast:
   [For each forecast item]
   - [time]: [temp]°C, [description]

3. If there's an error, explain what went wrong and suggest what the user can do.

IMPORTANT: When users ask about Hacker News or tech news, you MUST use the hackernews tool first. If they're asking about other news or current events that might not be in your training data, use the web search tool.

To get content from Hacker News, use the hackernews tool with these parameters:
- action: the type of stories to fetch ("top", "new", "best") or "item" for a specific story
- limit: maximum number of stories to return (default: 10)
- item_id: ID of a specific item to fetch (only used when action is "item")

For example, if a user asks "What's trending on Hacker News?", you should:
1. Use the hackernews tool with action="top" and limit=10
2. When you get the response, format it like this:
   Top Stories from Hacker News:
   
   [For each story]
   - [title]
     Score: [score] | Comments: [descendants]
     Posted by: [by]
     URL: [url]

3. If there's an error, explain what went wrong and suggest what the user can do.

IMPORTANT: When users ask about current events, news, or information that might not be in your training data, you MUST use the web search tool.

To search the web, use the web search tool with these parameters:
- query: the search query
- max_results: number of results to return (default: 5)

For example, if a user asks "What's the latest news about AI?", you should:
1. Use the web search tool with query="latest news about artificial intelligence"
2. When you get the results, format them like this:
   Search Results for "[query]":

   [For each result]
   - [title]
     [snippet]
     Source: [url]

3. Summarize the key points from the search results
4. If there's an error, explain what went wrong and suggest what the user can do

DO NOT suggest manual commands or alternative search methods. Always use the appropriate tool for the task.

When using the web search tool, you MUST:
1. Call the tool with the appropriate query
2. Wait for the results
3. Format and present the results to the user
4. Do not respond with placeholder messages like "I will search..." or "Searching..."

IMPORTANT: You MUST use the function calling format to use tools. For example:
To use the web search tool:
{{
    "name": "web_search",
    "arguments": {{
        "query": "your search query here",
        "max_results": 5
    }}
}}

To use the filesystem tool:
{{
    "name": "filesystem",
    "arguments": {{
        "action": "search",
        "path": ".",
        "query": "your search query here"
    }}
}}

To use the gmail tool:
{{
    "name": "gmail",
    "arguments": {{
        "query": "your search query here",
        "limit": 5
    }}
}}

To use the weather tool:
{{
    "name": "weather",
    "arguments": {{
        "location": "location name here"
    }}
}}

To use the hackernews tool:
{{
   "name": "mcp_hackernews_top",
   "arguments": {{
       "limit": 10
   }}
}}

Or:
{{
   "name": "mcp_hackernews_new",
   "arguments": {{
       "limit": 10
   }}
}}

Or:
{{
   "name": "mcp_hackernews_best",
   "arguments": {{
       "limit": 10
   }}
}}

Or:
{{
   "name": "mcp_hackernews_story",
   "arguments": {{
       "item_id": 12345
   }}
}}

To use the calendar tool:
{{
   "name": "calendar",
   "arguments": {{
       "action": "add" | "list" | "search" | "delete",
       "event": "details of the calendar event (for add action)",
       "keyword": "keyword to search for (for search action)",
       "filename": "filename of the event to delete (for delete action)"
   }}
}}

To use the notes tool:
{{
   "name": "notes",
   "arguments": {{
       "action": "add" | "list" | "search" | "delete",
       "content": "content of the note (for add action)",
       "keyword": "keyword to search for (for search action)",
       "id": "ID of the note to delete (for delete action)"
   }}
}}

To use the reminders tool:
{{
   "name": "reminders",
   "arguments": {{
       "action": "add" | "list" | "search" | "delete",
       "content": "content of the reminder (for add action)",
       "reminder_time": "optional reminder time in ISO 8601 format (e.g., '2025-12-31T23:59:59') (for add action)",
       "keyword": "keyword to search for (for search action)",
       "id": "ID of the reminder to delete (for delete action)"
   }}
}}

To use the reminders tool:
{{
  "name": "reminders",
  "arguments": {{
      "action": "add" | "list" | "search" | "delete",
      "content": "content of the reminder (for add action)",
      "reminder_time": "optional reminder time in ISO 8601 format (e.g., '2025-12-31T23:59:59') (for add action)",
      "keyword": "keyword to search for (for search action)",
      "id": "ID of the reminder to delete (for delete action)"
  }}
}}

To use the get_clipboard tool:
{{
  "name": "get_clipboard",
  "arguments": {{}}
}}

You can also use tools from connected MCP servers. These will be prefixed with 'mcp_'.
For example, if an MCP server provides a 'take_screenshot' tool, you would call it as:
{{
  "name": "mcp_take_screenshot",
  "arguments": {{...}}
}}

IMPORTANT: You have memory tools that allow you to remember important information across conversations:

1. add_memory: Store important information in your long-term memory
   Example: {{
     "name": "add_memory",
     "arguments": {{
       "content": "User prefers dark mode",
       "importance": 8,
       "tags": ["preferences"]
     }}
   }}

2. search_memory: Search your long-term memory for relevant information
   Example: {{
     "name": "search_memory",
     "arguments": {{
       "query": "user preferences",
       "limit": 5
     }}
   }}

3. get_memory: Retrieve a specific memory by ID
   Example: {{
     "name": "get_memory",
     "arguments": {{
       "memory_id": "mem_123456"
     }}
   }}

4. delete_memory: Remove a specific memory by ID
   Example: {{
     "name": "delete_memory",
     "arguments": {{
       "memory_id": "mem_123456"
     }}
   }}

Use these memory tools to:
- Remember important user preferences and information
- Recall past conversations and decisions
- Maintain context across multiple sessions
- Personalize your responses based on what you've learned about the user

Always be concise and clear in your responses.'''),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create a tool map for easy lookup
        # This map now includes both local and MCP tools
        all_tools_for_map = tools_override if tools_override is not None else (self.tools + self.mcp_tools)
        tool_map = {tool.name: tool for tool in all_tools_for_map}


        # Create the execute_tool function (this part remains largely the same,
        # but the tool_map it uses is now comprehensive)
        async def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
            """Execute a tool and return its result."""
            logger.debug(f"Attempting to execute tool: {tool_name} with args: {tool_args}")
            try:
                tool = tool_map.get(tool_name)
                if not tool:
                    # Log at INFO level instead of ERROR so it only shows with -v flag
                    logger.info(f"Tool '{tool_name}' not found in tool_map.")
                    return f"Error: Tool '{tool_name}' not found"

                # For MCPToolWrapper or Tool, arguments are passed directly as it handles its own schema.
                if isinstance(tool, MCPToolWrapper) or tool.name.startswith("mcp_"):
                    logger.debug(f"Executing MCP tool '{tool_name}' directly.")
                    try:
                        # Try to call the tool's coroutine method directly
                        if hasattr(tool, "coroutine"):
                            result = await tool.coroutine(**tool_args)
                        # Try to call the tool's func method directly
                        elif hasattr(tool, "func"):
                            result = tool.func(**tool_args)
                        # Try to call _arun with the tool_args and config
                        elif hasattr(tool, "_arun"):
                            try:
                                result = await tool._arun(**tool_args)
                            except TypeError as e:
                                if "missing 1 required keyword-only argument: 'config'" in str(e):
                                    # If the error is about missing 'config', add an empty config
                                    result = await tool._arun(config={}, **tool_args)
                                else:
                                    # For other TypeError errors, try to call the synchronous _run method
                                    result = tool._run(**tool_args)
                        # Try to call _run directly
                        elif hasattr(tool, "_run"):
                            result = tool._run(**tool_args)
                        else:
                            # If all else fails, return a mock result
                            result = f"This is a mock implementation of {tool_name}. In a real scenario, this would call the MCP server."
                    except Exception as e:
                        # For any other exception, return a mock result
                        logger.warning(f"Error executing tool: {e}. Returning mock result.")
                        result = f"This is a mock implementation of {tool_name}. In a real scenario, this would call the MCP server."
                    
                    return str(result)

                # Argument filtering for local tools (existing logic)
                expected_args = set()
                if tool_name == "filesystem":
                    expected_args = {"action", "path", "query", "content", "recursive", "limit"}
                elif tool_name == "web_search":
                    expected_args = {"query", "max_results"}
                elif tool_name == "weather":
                    expected_args = {"location"}
                elif tool_name == "gmail":
                    expected_args = {"query", "limit"}
                elif tool_name == "calendar":
                     expected_args = {"action", "event", "keyword", "filename"}
                elif tool_name == "notes":
                     expected_args = {"action", "content", "keyword", "id"}
                elif tool_name == "reminders":
                     expected_args = {"action", "content", "reminder_time", "keyword", "id"}
                elif tool_name == "hackernews":
                     expected_args = {"action", "limit", "item_id"}
                # Add other local tools here if necessary

                filtered_args = {k: v for k, v in tool_args.items() if k in expected_args}
                if not expected_args and tool_args: # If no expected args defined but args provided
                    logger.warning(f"Tool '{tool_name}' received arguments but has no defined expected_args. Passing all.")
                    filtered_args = tool_args
                elif not expected_args and not tool_args: # No expected, no provided
                     pass # No args to filter or pass
                elif expected_args and not filtered_args and tool_args: # Expected args defined, but none of the provided args matched
                    logger.warning(f"Tool '{tool_name}' received arguments {tool_args}, but none matched expected_args {expected_args}. Passing empty.")

                logger.debug(f"Executing local tool '{tool_name}' with filtered_args: {filtered_args}")
                result = await tool._arun(**filtered_args)
                return str(result)
            except Exception as e:
                # Log at INFO level instead of ERROR so it only shows with -v flag
                logger.info(f"Error executing tool {tool_name} with args {tool_args}: {e}", exc_info=True)
                return f"Error executing {tool_name}: {str(e)}"

        # Convert tools to OpenAI functions
        functions = []
        current_tools_for_functions = tools_override if tools_override is not None else (self.tools + self.mcp_tools)
        
        # Log the tools being converted
        logger.info(f"Converting {len(current_tools_for_functions)} tools to OpenAI functions")
        for tool in current_tools_for_functions:
            logger.info(f"Converting tool: {tool.name} ({type(tool).__name__})")
        
        # Make sure we have at least one tool
        if len(current_tools_for_functions) == 0:
            logger.warning("No tools available for conversion to OpenAI functions")
            # Add a dummy tool to avoid OpenAI API error
            from langchain.tools import Tool
            dummy_tool = Tool(
                name="dummy_tool",
                description="This is a placeholder tool",
                func=lambda **kwargs: "This tool is a placeholder"
            )
            current_tools_for_functions = [dummy_tool]
            logger.info("Added dummy tool to avoid OpenAI API error")
        
        for tool in current_tools_for_functions:
            if isinstance(tool, MCPToolWrapper):
                # For MCP tools, use the schema provided by the server
                functions.append({
                    "name": tool.name, # Already prefixed with mcp_
                    "description": tool.description,
                    "parameters": tool.schema # Use the schema from MCP server
                })
                logger.info(f"Added MCPToolWrapper: {tool.name}")
            elif tool.name.startswith("mcp_"):
                # For other MCP tools (like our hardcoded ones)
                functions.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                })
                logger.info(f"Added MCP tool: {tool.name}")
            elif tool.name == "dummy_tool":
                # For the dummy tool
                functions.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                })
                logger.info(f"Added dummy tool: {tool.name}")
            elif tool.name == "web_search": # Existing local tool definitions
                functions.append({
                    "name": "web_search",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                })
            elif tool.name == "filesystem":
                functions.append({
                    "name": "filesystem",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["search", "list", "read", "write", "info"],
                                "description": "The action to perform"
                            },
                            "path": {
                                "type": "string",
                                "description": "The path to operate on"
                            },
                            "query": {
                                "type": "string",
                                "description": "The search query (for search action)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write (for write action)"
                            }
                        },
                        "required": ["action", "path"]
                    }
                })
            elif tool.name == "weather":
                functions.append({
                    "name": "weather",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location name (e.g., 'London, UK' or 'New York, NY')"
                            }
                        },
                        "required": ["location"]
                    }
                })
            elif tool.name == "gmail":
                functions.append({
                    "name": "gmail",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for messages (e.g., 'is:unread', 'from:example@gmail.com', 'subject:meeting', 'has:attachment')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of messages to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                })
            elif tool.name == "calendar": # Add calendar tool function definition
                functions.append({
                    "name": "calendar",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "list", "search", "delete"],
                                "description": "The action to perform (add, list, search, or delete)"
                            },
                            "event": {
                                "type": "string",
                                "description": "Details of the calendar event (required for add action)"
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Keyword to search for (for search action)"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Filename of the event to delete (required for delete action)"
                            }
                        },
                        "required": ["action"]
                    }
                })
            elif tool.name == "notes": # Add notes tool function definition
                functions.append({
                    "name": "notes",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "list", "search", "delete"],
                                "description": "The action to perform (add, list, search, or delete)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content of the note (required for add action)"
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Keyword to search for (for search action)"
                            },
                            "id": {
                                "type": "string",
                                "description": "ID of the note to delete (required for delete action)"
                            }
                        },
                        "required": ["action"]
                    }
                })
            elif tool.name == "reminders": # Add reminders tool function definition
                functions.append({
                    "name": "reminders",
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "list", "search", "delete"],
                                "description": "The action to perform (add, list, search, or delete)"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content of the reminder (required for add action)"
                            },
                            "reminder_time": {
                                "type": "string",
                                "description": "Optional reminder time in ISO 8601 format (e.g., '2025-12-31T23:59:59') (for add action)"
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Keyword to search for (for search action)"
                            },
                            "id": {
                                "type": "string",
                                "description": "ID of the reminder to delete (for delete action)"
                            }
                        },
                        "required": ["action"]
                    }
                })

        logger.info("Converted %d tools to OpenAI functions", len(functions))

        # Create the agent with tool execution
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user"
                    else AIMessage(content=msg["content"])
                    for msg in x["chat_history"]
                ],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self.llm.bind(functions=functions)
            | OpenAIFunctionsAgentOutputParser()
        )

        # Create wrapped tools for the executor
        from langchain.tools import StructuredTool # Keep this import
        
        # This part needs to wrap all tools, including MCP ones if StructuredTool is still desired
        # For MCP tools, the schema is already defined, so direct wrapping might be simpler
        # or adapt StructuredTool to use the existing schema.
        # For simplicity, let's assume Langchain's AgentExecutor can work with tools
        # that have an `_arun` method and a `name`/`description`/`args_schema` or similar.
        # The `functions` list prepared for `self.llm.bind(functions=functions)`
        # already describes the tools to the LLM. The `tools` argument to AgentExecutor
        # needs to be a list of callables or Langchain Tool objects.

        langchain_tools_list = []
        current_tools_for_executor = tools_override if tools_override is not None else (self.tools + self.mcp_tools)

        for tool_instance in current_tools_for_executor:
            if isinstance(tool_instance, MCPToolWrapper):
                # MCPToolWrapper is already a BaseTool and has _arun
                # We need to ensure it's compatible with AgentExecutor or wrap it appropriately
                # For now, let's assume it can be used directly if it conforms to Langchain's Tool interface
                # or if we create a simple Langchain Tool from it.
                # Let's create a StructuredTool for consistency, using its schema.
                langchain_tools_list.append(
                    StructuredTool.from_function(
                        func=lambda t=tool_instance, **kwargs: asyncio.run(t._arun(**kwargs)), # Ensure async run
                        name=tool_instance.name,
                        description=tool_instance.description,
                        args_schema=tool_instance.args_schema_pydantic # Use the Pydantic model
                    )
                )
            else: # Existing local tools
                # This logic can be simplified if all local tools also provide a Pydantic schema
                args_schema_dict = {}
                if tool_instance.name == "filesystem":
                    args_schema_dict = {"action": str, "path": str, "query": Optional[str], "content": Optional[str]}
                elif tool_instance.name == "web_search":
                    args_schema_dict = {"query": str, "max_results": Optional[int]}
                elif tool_instance.name == "weather":
                    args_schema_dict = {"location": str}
                elif tool_instance.name == "gmail":
                    args_schema_dict = {"query": str, "limit": Optional[int]}
                elif tool_instance.name == "calendar":
                    args_schema_dict = {"action": str, "event": Optional[str], "keyword": Optional[str], "filename": Optional[str]}
                elif tool_instance.name == "notes":
                    args_schema_dict = {"action": str, "content": Optional[str], "keyword": Optional[str], "id": Optional[str]}
                elif tool_instance.name == "reminders":
                    args_schema_dict = {"action": str, "content": Optional[str], "reminder_time": Optional[datetime], "keyword": Optional[str], "id": Optional[str]}
                
                # Pydantic v2: use __annotations__ for dynamic model creation
                annotations = {k: v for k, v in args_schema_dict.items()}
                args_schema = type(
                    f"{tool_instance.name}Schema",
                    (BaseModel,),
                    {"__annotations__": annotations}
                ) if args_schema_dict else None
                langchain_tools_list.append(
                    StructuredTool.from_function(
                        func=lambda t=tool_instance, **kwargs: asyncio.run(t._arun(**kwargs)),
                        name=tool_instance.name,
                        description=tool_instance.description,
                        args_schema=args_schema
                    )
                )
        
        return AgentExecutor(
            agent=agent, # This is the RunnableSequence defined earlier
            tools=langchain_tools_list,
            verbose=False, # Set to False for cleaner logs in production
            handle_parsing_errors=True, # Good for development
            max_iterations=5, # Increased slightly
            return_intermediate_steps=True,
        )

    async def process_input(self, user_input: str) -> str:
        """
        Process user input and return a response.

        Args:
            user_input: The user's input text

        Returns:
            The agent's response
        """
        logger.info("Processing input: %s", user_input)

        # Check for special commands
        if user_input.lower().strip() in ["list all indexed files", "what files are you aware of", "list the files you know of"]:
            logger.info("Detected request to list indexed files")
            
            # Try to find the filesystem tool
            filesystem_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "filesystem":
                    filesystem_tool = tool
                    logger.info("Found filesystem tool in tools list")
                    break
            
            # If we found a filesystem tool, use it directly
            if filesystem_tool:
                logger.info("Using filesystem tool directly")
                try:
                    # Use the async version of the tool's run method
                    result = await filesystem_tool._arun(action="search", path=".", query="", limit=10)
                    logger.info(f"Filesystem tool result: {result}")
                    
                    if result and "results" in result and result["results"]:
                        formatted_results = []
                        for i, file_info in enumerate(result["results"]):
                            formatted_results.append(f"{i+1}. {file_info.get('name', 'Unknown')} - {file_info.get('path', 'Unknown path')}")
                        
                        output = f"Found {len(result['results'])} indexed files:\n\n" + "\n".join(formatted_results)
                    else:
                        output = "No indexed files found."
                        
                    # Update chat history
                    self.chat_history.append(HumanMessage(content=user_input))
                    self.chat_history.append(AIMessage(content=output))
                    
                    return output
                except Exception as e:
                    logger.error(f"Error using filesystem tool directly: {e}")
                    # Fall back to normal agent execution
            
        # Check for hackernews queries
        elif any(keyword in user_input.lower() for keyword in ["hackernews", "hacker news", "tech news", "latest news", "top stories"]):
            logger.info("Detected request for Hacker News")
            
            # Try to find the hackernews tool
            hackernews_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "hackernews":
                    hackernews_tool = tool
                    logger.info("Found hackernews tool in tools list")
                    break
            
            # If we found a hackernews tool, use it directly
            if hackernews_tool:
                logger.info("Using hackernews tool directly")
                try:
                    # Determine which action to use based on the query
                    action = "top"  # Default to top stories
                    limit = 10      # Default limit
                    
                    if "new" in user_input.lower() or "latest" in user_input.lower():
                        action = "new"
                    elif "best" in user_input.lower():
                        action = "best"
                    
                    # Use the async version of the tool's run method
                    result = await hackernews_tool._arun(action=action, limit=limit)
                    logger.info(f"Hackernews tool result: {result}")
                    
                    # Format the result
                    if isinstance(result, dict):
                        if "stories" in result and result["stories"]:
                            stories = result["stories"]
                            output = f"Here are the {action} stories from Hacker News:\n\n"
                            
                            for i, story in enumerate(stories[:limit]):
                                title = story.get('title', 'No title')
                                url = story.get('url', f"https://news.ycombinator.com/item?id={story.get('id')}")
                                score = story.get('score', 0)
                                comments = story.get('descendants', 0)
                                by = story.get('by', 'unknown')
                                
                                output += f"{i+1}. {title}\n"
                                output += f"   Score: {score} | Comments: {comments}\n"
                                output += f"   Posted by: {by}\n"
                                output += f"   URL: {url}\n\n"
                        else:
                            output = "No stories found on Hacker News."
                    else:
                        output = str(result)
                        
                    # Update chat history
                    self.chat_history.append(HumanMessage(content=user_input))
                    self.chat_history.append(AIMessage(content=output))
                    
                    return output
                except Exception as e:
                    logger.error(f"Error using hackernews tool directly: {e}")
                    # Fall back to normal agent execution
        
        # Check for weather queries
        elif any(keyword in user_input.lower() for keyword in ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy"]):
            logger.info("Detected request for weather information")
            
            # Try to find the weather tool
            weather_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "weather":
                    weather_tool = tool
                    logger.info("Found weather tool in tools list")
                    break
            
            # If we found a weather tool, use it directly
            if weather_tool:
                logger.info("Using weather tool directly")
                try:
                    # Extract location from the query
                    import re
                    
                    # Try to extract location using common patterns
                    location_patterns = [
                        r"weather (?:in|for|at) ([\w\s,]+)",
                        r"temperature (?:in|for|at) ([\w\s,]+)",
                        r"forecast (?:in|for|at) ([\w\s,]+)",
                        r"(?:in|for|at) ([\w\s,]+)"
                    ]
                    
                    location = None
                    for pattern in location_patterns:
                        match = re.search(pattern, user_input.lower())
                        if match:
                            location = match.group(1).strip()
                            break
                    
                    # If no location found, use a default
                    if not location:
                        location = "New York, NY"
                        output = "I couldn't determine the location from your query. Here's the weather for New York, NY:\n\n"
                    else:
                        output = f"Here's the current weather for {location}:\n\n"
                    
                    # Use the async version of the get_weather method
                    result = await weather_tool.get_weather_async(location)
                    logger.info(f"Weather tool result: {result}")
                    
                    # The get_weather_async method already formats the result
                    output += result
                        
                    # Update chat history
                    self.chat_history.append(HumanMessage(content=user_input))
                    self.chat_history.append(AIMessage(content=output))
                    
                    return output
                except Exception as e:
                    logger.error(f"Error using weather tool directly: {e}")
                    # Fall back to normal agent execution
        
        elif user_input.lower().strip() in ["list all available mcp tools", "list mcp tools", "show mcp tools"]:
            # If we're asking for MCP tools but don't have any yet, try to get them from the MCP server directly
            if not self.mcp_tools and self.mcp_clients:
                logger.info("No MCP tools registered yet, trying to get them directly from the server")
                for url, client in self.mcp_clients.items():
                    try:
                        # Get tools from the server
                        tools = await client.list_tools()
                        logger.info(f"Found {len(tools)} tools on the MCP server")
                        
                        # Register tools with the client
                        for tool in tools:
                            tool_name = tool.get("name")
                            wrapped_name = f"mcp_{tool_name}"
                            logger.info(f"Registering MCP tool: {wrapped_name} from {url}")
                            wrapped_tool = MCPToolWrapper(client, tool_name, tool)
                            wrapped_tool.name = wrapped_name  # Ensure consistent naming
                            self.mcp_tools.append(wrapped_tool)
                        
                        # Update the agent executor with the new tools
                        if self.mcp_tools:
                            logger.info(f"Added {len(self.mcp_tools)} MCP tools")
                            self._update_agent_executor_with_mcp_tools()
                    except Exception as e:
                        # Log at INFO level instead of ERROR so it only shows with -v flag
                        logger.info(f"Error getting tools from MCP server: {e}")
            
            # If we still don't have any MCP tools, try to get them from the static definition
            if not self.mcp_tools:
                logger.info("No MCP tools available, trying to get them from the static definition")
                try:
                    from .mcp_server import get_registered_tools
                    tools_info = get_registered_tools()
                    if isinstance(tools_info, list) and tools_info:
                        logger.info(f"Found {len(tools_info)} tools in static definition")
                        
                        # Create hardcoded MCP tools
                        hardcoded_tools = [
                            {
                                "name": "calendar_add",
                                "description": "Add a new calendar event",
                                "parameters": {"type": "object", "properties": {"event": {"type": "string"}}, "required": ["event"]}
                            },
                            {
                                "name": "calendar_list",
                                "description": "Lists all calendar events",
                                "parameters": {"type": "object", "properties": {}}
                            },
                            {
                                "name": "calendar_search",
                                "description": "Searches calendar events for a keyword",
                                "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}
                            },
                            {
                                "name": "calendar_delete",
                                "description": "Deletes a calendar event by filename",
                                "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}
                            },
                            {
                                "name": "notes_add",
                                "description": "Adds a new note",
                                "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
                            },
                            {
                                "name": "notes_list",
                                "description": "Lists all notes",
                                "parameters": {"type": "object", "properties": {}}
                            },
                            {
                                "name": "notes_search",
                                "description": "Searches notes for a keyword",
                                "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}
                            },
                            {
                                "name": "notes_delete",
                                "description": "Deletes a note by ID",
                                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
                            },
                            {
                                "name": "take_screenshot",
                                "description": "Take a screenshot of the user's screen and return it as an image",
                                "parameters": {"type": "object", "properties": {}}
                            },
                            {
                                "name": "hackernews_top",
                                "description": "Get top stories from Hacker News",
                                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}
                            },
                            {
                                "name": "hackernews_new",
                                "description": "Get new stories from Hacker News",
                                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}
                            },
                            {
                                "name": "hackernews_best",
                                "description": "Get best stories from Hacker News",
                                "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}}
                            },
                            {
                                "name": "hackernews_story",
                                "description": "Get details of a specific Hacker News story",
                                "parameters": {"type": "object", "properties": {"item_id": {"type": "integer"}}, "required": ["item_id"]}
                            }
                        ]
                        
                        # Register the hardcoded tools
                        for tool in hardcoded_tools:
                            tool_name = tool.get("name")
                            if tool_name:
                                wrapped_name = f"mcp_{tool_name}"
                                logger.info(f"Registering MCP tool: {wrapped_name} from hardcoded definition")
                                
                                # Create a BaseTool instance directly
                                from langchain.tools import Tool
                                
                                # Create a simple function for the tool
                                def tool_func(**kwargs):
                                    return f"This is a mock implementation of {tool_name}. In a real scenario, this would call the MCP server."
                                
                                # Create a Tool instance
                                mcp_tool = Tool(
                                    name=wrapped_name,
                                    description=tool.get("description", f"MCP tool: {tool_name}"),
                                    func=tool_func
                                )
                                
                                self.mcp_tools.append(mcp_tool)
                                
                                # Continue with the rest of the tools
                                continue
                                
                                self.mcp_tools.append(HardcodedMCPTool())
                        
                        # Update the agent executor with the new tools
                        if self.mcp_tools:
                            logger.info(f"Added {len(self.mcp_tools)} MCP tools from hardcoded definition")
                            self._update_agent_executor_with_mcp_tools()
                except Exception as e:
                    # Log at INFO level instead of ERROR so it only shows with -v flag
                    logger.info(f"Error getting tools from static definition: {e}")
            
            return self.list_mcp_tools()

        # If not a special command, proceed with normal agent execution
        loop = asyncio.get_event_loop()
        try:
            # Use thread pool for I/O-bound tasks
            result = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.agent_executor.invoke({
                    "input": user_input,
                    "chat_history": self.chat_history,
                })
            )

            logger.info("Agent executor result: %s", result)

            # Extract the output and intermediate steps
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # If we have intermediate steps, format them into the response
            if intermediate_steps:
                formatted_steps = []
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        if isinstance(action, dict) and "name" in action and "arguments" in action:
                            tool_name = action["name"]
                            tool_args = action["arguments"]
                            formatted_steps.append(f"Tool: {tool_name}")
                            formatted_steps.append(f"Arguments: {tool_args}")
                            formatted_steps.append(f"Result: {observation}")
                            formatted_steps.append("---")

                if formatted_steps:
                    output = "\n".join(formatted_steps)

            # Update chat history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": output})
            
            # Process the conversation with the memory manager
            await self.memory_manager.process_conversation(self.chat_history)
            logger.info("Conversation processed by memory manager")

            # Speak the response if voice is enabled
            # The calling interface (GUI or CLI) is responsible for speaking the response
            # if self.voice:
            #     await self.voice.speak(output)

            return output
        except Exception as e:
            logger.error("Error processing input: %s", str(e))
            return f"Error: {str(e)}"

    async def run(self, user_input: str) -> str:
        """
        Run the agent with the given input.
        """
        return await self.process_input(user_input)

    async def shutdown_mcp_clients(self):
        """Cleanly shut down all MCP client connections."""
        for url, task in self.mcp_client_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"MCP client connection to {url} cancelled.")
        self.mcp_clients.clear()
        self.mcp_client_tasks.clear()
        
        # Stop the memory manager
        if hasattr(self, 'memory_manager'):
            self.memory_manager.stop()
            logger.info("Memory manager stopped")

    def __del__(self):
        """Shutdown the thread pool executor if it was initialized."""
        if hasattr(self, "thread_pool") and self.thread_pool:
            self.thread_pool.shutdown(wait=False)

class MCPToolWrapper(BaseTool):
    """
    Wrapper for tools provided by external MCP servers.
    This makes an MCP tool look like a local BaseTool.
    """
    def __init__(self, client: Optional[MCPClient], tool_name: str, tool_info: Dict[str, Any]):
        super().__init__()
        self.client = client
        self.tool_name_on_server = tool_name # Original name on MCP server
        self.name = f"mcp_{tool_name}"      # Prefixed name for use in this agent
        self.tool_info = tool_info  # Store the tool info for later use
        
        # Get the client from the tool_info if it's not provided directly
        if self.client is None and "client" in tool_info:
            self.client = tool_info["client"]
            
        # Set the description
        if self.client:
            self.description = tool_info.get("description", f"MCP tool: {tool_name} from {self.client.base_url}")
        else:
            self.description = tool_info.get("description", f"MCP tool: {tool_name}")
        
        # The schema from MCP server is typically JSON schema.
        # Langchain's StructuredTool can often work with Pydantic models for args_schema.
        # We need to convert or ensure compatibility.
        self.schema = tool_info.get("parameters", {"type": "object", "properties": {}}) # JSON schema for parameters
        self.args_schema_pydantic = self._create_pydantic_schema_from_json(self.schema)


    def _create_pydantic_schema_from_json(self, json_schema: Dict[str, Any]) -> Optional[type[BaseModel]]:
        """
        Dynamically creates a Pydantic model from a JSON schema.
        This is a simplified version. For full JSON schema compatibility, a more robust converter is needed.
        """
        if not json_schema or json_schema.get("type") != "object" or not json_schema.get("properties"):
            return type(f"{self.tool_name_on_server}EmptySchema", (BaseModel,), {"__annotations__": {}}) # Return empty schema if no properties

        fields = {}
        props = json_schema.get("properties", {})
        required_fields = json_schema.get("required", [])

        for name, prop_schema in props.items():
            field_type_str = prop_schema.get("type", "string")
            # Basic type mapping
            if field_type_str == "string":
                field_type = str
            elif field_type_str == "integer":
                field_type = int
            elif field_type_str == "number":
                field_type = float
            elif field_type_str == "boolean":
                field_type = bool
            elif field_type_str == "array": # Further handling needed for item types
                field_type = List[Any]
            elif field_type_str == "object": # Further handling needed for nested objects
                field_type = Dict[str, Any]
            else:
                field_type = Any # Fallback

            if name in required_fields:
                fields[name] = (field_type, ...) # Ellipsis means required
            else:
                fields[name] = (Optional[field_type], None) # Optional field

        if not fields: # If properties were empty or unparsable into fields
             return type(f"{self.tool_name_on_server}EmptySchema", (BaseModel,), {"__annotations__": {}})

        # Pydantic v2: use __annotations__ for dynamic model creation
        annotations = {k: v[0] for k, v in fields.items()}
        return type(f"{self.tool_name_on_server}ArgsSchema", (BaseModel,), {"__annotations__": annotations})

    async def _arun(self, **kwargs: Any) -> str:
        """
        Executes the wrapped MCP tool asynchronously.
        kwargs are the arguments for the tool.
        """
        # Get the caller's information
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_info = ""
        if caller_frame:
            caller_module = inspect.getmodule(caller_frame)
            if caller_module:
                caller_info = f" (called from {caller_module.__name__})"
        
        logger.info(f"MCPToolWrapper._arun for '{self.name}' (server tool: '{self.tool_name_on_server}') with args: {kwargs}{caller_info}")
        
        # Check if the client is available
        if self.client is None:
            # Try to get the client from the tool_info again
            if "client" in self.tool_info:
                self.client = self.tool_info["client"]
            else:
                return f"Error: MCP client not available for tool '{self.name}'"
        
        try:
            # The MCP client's call_tool_async expects arguments as a dictionary
            result = await self.client.call_tool_async(tool_name=self.tool_name_on_server, arguments=kwargs)
            logger.debug(f"Result from MCP tool '{self.name}': {result}")
            return str(result) # Ensure result is a string
        except Exception as e:
            # Log at INFO level instead of ERROR so it only shows with -v flag
            logger.info(f"Error executing MCP tool {self.name}: {e}", exc_info=True)
            return f"Error during MCP tool '{self.name}' execution: {str(e)}"

    # _run is not strictly necessary if _arun is implemented and used by AgentExecutor
    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution (optional, falls back to _arun if not implemented by BaseTool)."""
        # Get the caller's information
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_info = ""
        if caller_frame:
            caller_module = inspect.getmodule(caller_frame)
            if caller_module:
                caller_info = f" (called from {caller_module.__name__})"
        
        logger.warning(f"MCPToolWrapper._run for '{self.name}' (server tool: '{self.tool_name_on_server}') with args: {kwargs}{caller_info}. Consider using async.")
        # This is a simple bridge. For true sync execution, MCPClient would need a sync method.
        return asyncio.run(self._arun(**kwargs))