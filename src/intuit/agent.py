"""
Core agent implementation for Intuit.
"""
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

from .tools.base import BaseTool
from .voice import VoiceOutput

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
    ):
        self.config = config or AgentConfig()
        
        # Get OpenAI configuration from environment or config
        openai_api_base = self.config.openai_api_base or os.getenv("OPENAI_API_BASE")
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
        
        self.tools = tools
        self.agent_executor = self._create_agent_executor()
        self.chat_history: List[Dict[str, Any]] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Initialize voice output if enabled
        self.voice = VoiceOutput(
            language=self.config.voice_language,
            slow=self.config.voice_slow
        ) if self.config.use_voice else None

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt template."""
        logger.info("Creating agent executor with %d tools:", len(self.tools))
        for tool in self.tools:
            logger.info("- Tool: %s (%s)", tool.name, tool.description)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Intuit, a helpful personal assistant. 
You have access to various tools to help the user:

1. Web search for online information
2. Filesystem operations for searching, reading, and managing files
3. Gmail integration for email management (when enabled)
4. Weather information (when enabled)

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

Always be concise and clear in your responses."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create a tool map for easy lookup
        tool_map = {tool.name: tool for tool in self.tools}
        
        # Create the execute_tool function
        async def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
            """Execute a tool and return its result."""
            try:
                tool = tool_map.get(tool_name)
                if not tool:
                    return f"Error: Tool '{tool_name}' not found"
                
                # Filter arguments based on the tool
                if tool_name == "filesystem":
                    expected_args = {"action", "path", "query", "content", "recursive", "limit"}
                elif tool_name == "web_search":
                    expected_args = {"query", "max_results"}
                elif tool_name == "weather":
                    expected_args = {"location"}
                elif tool_name == "gmail":
                    expected_args = {"query", "limit"}
                else:
                    expected_args = set()
                
                # Filter out unexpected arguments
                filtered_args = {k: v for k, v in tool_args.items() if k in expected_args}
                
                # Execute the tool
                result = await tool._arun(**filtered_args)
                return str(result)
            except Exception as e:
                logger.error("Error executing tool %s: %s", tool_name, str(e))
                return f"Error executing {tool_name}: {str(e)}"

        # Convert tools to OpenAI functions
        functions = []
        for tool in self.tools:
            if tool.name == "web_search":
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
        from langchain.tools import StructuredTool
        wrapped_tools = []
        for tool in self.tools:
            if tool.name == "filesystem":
                wrapped_tools.append(
                    StructuredTool.from_function(
                        func=lambda t=tool, **kwargs: asyncio.run(t._arun(**kwargs)),
                        name=tool.name,
                        description=tool.description,
                        args_schema={
                            "action": str,
                            "path": str,
                            "query": str,
                            "content": str
                        }
                    )
                )
            elif tool.name == "web_search":
                wrapped_tools.append(
                    StructuredTool.from_function(
                        func=lambda t=tool, **kwargs: asyncio.run(t._arun(**kwargs)),
                        name=tool.name,
                        description=tool.description,
                        args_schema={
                            "query": str,
                            "max_results": int
                        }
                    )
                )
            elif tool.name == "weather":
                wrapped_tools.append(
                    StructuredTool.from_function(
                        func=lambda t=tool, **kwargs: asyncio.run(t._arun(**kwargs)),
                        name=tool.name,
                        description=tool.description,
                        args_schema={
                            "location": str
                        }
                    )
                )
            elif tool.name == "gmail":
                wrapped_tools.append(
                    StructuredTool.from_function(
                        func=lambda t=tool, **kwargs: asyncio.run(t._arun(**kwargs)),
                        name=tool.name,
                        description=tool.description,
                        args_schema={
                            "query": str,
                            "limit": int
                        }
                    )
                )

        # Create the agent executor with the wrapped tools
        return AgentExecutor(
            agent=agent,
            tools=wrapped_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
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
        
        # Use thread pool for I/O-bound tasks
        loop = asyncio.get_event_loop()
        try:
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
            
            # Speak the response if voice is enabled
            if self.voice:
                await self.voice.speak(output)
            
            return output
        except Exception as e:
            logger.error("Error processing input: %s", str(e))
            return f"Error: {str(e)}"

    async def run(self, user_input: str) -> str:
        """
        Run the agent on a single input.
        This is a wrapper around process_input for compatibility with the CLI interface.
        
        Args:
            user_input: The user's input text
            
        Returns:
            The agent's response
        """
        return await self.process_input(user_input)

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []

    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False) 