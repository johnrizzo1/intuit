"""
Core agent implementation for Intuit.
"""
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

# Set up logging
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for the Intuit agent."""
    model_name: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    streaming: bool = Field(default=True)
    max_workers: int = Field(default=4)  # Number of threads for the thread pool

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
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=self.config.streaming,
        )
        self.tools = tools
        self.agent_executor = self._create_agent_executor()
        self.chat_history: List[Dict[str, Any]] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt template."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Intuit, a helpful personal assistant. "
                      "You have access to various tools to help the user:\n"
                      "1. Web search for online information\n"
                      "2. Filesystem operations for searching, reading, and managing files\n"
                      "3. Gmail integration for email management (when enabled)\n"
                      "4. Weather information (when enabled)\n\n"
                      "When users ask about finding files or searching content, "
                      "you MUST use the filesystem tool with the 'search' action. "
                      "The filesystem tool has semantic search capabilities through its vector store. "
                      "To search for files, use the filesystem tool with these parameters:\n"
                      "- action: 'search'\n"
                      "- path: the directory to search in\n"
                      "- query: the search query\n\n"
                      "DO NOT suggest manual commands or alternative search methods. "
                      "Always use the filesystem tool for file searches.\n\n"
                      "Always be concise and clear in your responses."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        functions = [convert_to_openai_function(t) for t in self.tools]
        
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
        )

    async def process_input(self, user_input: str) -> str:
        """
        Process user input and return a response.
        
        Args:
            user_input: The user's input text
            
        Returns:
            The agent's response
        """
        # Use thread pool for I/O-bound tasks
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            lambda: self.agent_executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history,
            })
        )
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": result["output"]})
        
        return result["output"]

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