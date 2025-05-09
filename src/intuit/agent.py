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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Intuit, a helpful personal assistant. 
You have access to various tools to help the user:

1. Web search for online information
2. Filesystem operations for searching, reading, and managing files
3. Gmail integration for email management (when enabled)
4. Weather information (when enabled)

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

Always be concise and clear in your responses."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        functions = [convert_to_openai_function(t) for t in self.tools]
        logger.info("Converted %d tools to OpenAI functions", len(functions))
        
        agent = RunnableParallel({
            "input": RunnablePassthrough(),
            "chat_history": lambda x: [
                HumanMessage(content=msg["content"]) if msg["role"] == "user"
                else AIMessage(content=msg["content"])
                for msg in x["chat_history"]
            ],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }) | prompt | self.llm | OpenAIFunctionsAgentOutputParser()

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
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
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": result["output"]})
            
            # Speak the response if voice is enabled
            if self.voice:
                await self.voice.speak(result["output"])
            
            return result["output"]
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