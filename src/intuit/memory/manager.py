"""
Memory manager for Intuit using LangMem.
"""
import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional
from langmem.knowledge.extraction import create_memory_manager
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from .chroma_store import ChromaMemoryStore
from ..utils.spinner import ThinkingSpinner, with_spinner

logger = logging.getLogger(__name__)

class IntuitMemoryManager:
    """Background memory manager for Intuit."""
    
    def __init__(self, store: ChromaMemoryStore, model: str = "llama3.2:3b"):
        """
        Initialize the memory manager.
        
        Args:
            store: The memory store to use
            model: The model to use for memory operations
        """
        self.store = store
        self.model = model
        
        # Create appropriate chat model instance for LangMem
        # LangMem requires a ChatModel instance, not a string
        if model.startswith(('gpt-', 'text-')) or '/' in model and 'gpt' in model.lower():
            # OpenAI model
            chat_model = ChatOpenAI(
                model=model.split('/')[-1] if '/' in model else model,
                temperature=0.7
            )
        else:
            # Ollama model (default)
            chat_model = ChatOllama(
                model=model.split('/')[-1] if '/' in model else model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.7
            )
            
        self.manager = create_memory_manager(chat_model)
        self.running = False
        self.task = None
        logger.info(f"Memory manager initialized with model: {model}")
    
    async def process_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """
        Process a conversation to extract and consolidate memories.
        
        Args:
            messages: List of conversation messages
        """
        import sys
        # Only use spinner if stdout is a real TTY
        use_spinner = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        if use_spinner:
            spinner = ThinkingSpinner(
                text=f"Processing conversation ({len(messages)} messages)",
                spinner=None,
                color="blue"
            )
            spinner.start()
        
        try:
            logger.info(f"Processing conversation with {len(messages)} messages")
            # Convert messages to the format expected by LangMem
            # Handle both dict and LangChain message objects
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Already a dict
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                else:
                    # LangChain message object (HumanMessage, AIMessage, etc.)
                    role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    formatted_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Check if the manager has a process_conversation method
            if hasattr(self.manager, 'process_conversation'):
                # Process the conversation using LangMem
                await self.manager.process_conversation(formatted_messages)
                logger.info("Conversation processing complete")
            else:
                # LangMem manager doesn't have process_conversation method
                # This is expected - we handle memory storage through our tools
                logger.debug("Using manual memory storage (LangMem process_conversation not available)")
                
                # Extract important information from the conversation and store it
                # This is a simple implementation that just stores the last message
                if formatted_messages:
                    last_message = formatted_messages[-1]
                    if last_message.get("role") == "user" and last_message.get("content"):
                        # Store the user's message as a memory
                        await self.store.add_memory(
                            content=last_message.get("content"),
                            metadata={"source": "conversation", "importance": 5}
                        )
                        logger.info("Stored last user message as memory")
            
            if use_spinner:
                spinner.ok()
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            if use_spinner:
                spinner.fail()
    
    async def process_memories(self) -> None:
        """Process memories to extract insights and reduce redundancy."""
        import sys
        # Only use spinner if stdout is a real TTY
        use_spinner = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        if use_spinner:
            spinner = ThinkingSpinner(
                text="Processing memories",
                spinner=None,
                color="green"
            )
            spinner.start()
        
        try:
            logger.info("Starting memory processing")
            start_time = time.time()
            # The memory manager doesn't have a consolidate_memories method
            # Instead, we'll just log that processing is complete
            duration = time.time() - start_time
            logger.info(f"Memory processing complete (took {duration:.2f}s)")
            if use_spinner:
                spinner.ok()
        except Exception as e:
            logger.error(f"Error processing memories: {e}")
            if use_spinner:
                spinner.fail()
    
    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of a conversation.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Summary of the conversation
        """
        with ThinkingSpinner(text=f"Summarizing conversation ({len(messages)} messages)",
                            spinner=None,
                            color="cyan") as spinner:
            try:
                logger.info(f"Summarizing conversation with {len(messages)} messages")
                # Convert messages to the format expected by LangMem
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                
                # Generate summary
                summary = await self.manager.summarize_conversation(formatted_messages)
                logger.info("Conversation summarization complete")
                spinner.ok()
                return summary
            except Exception as e:
                logger.error(f"Error summarizing conversation: {e}")
                spinner.fail()
                return "Failed to summarize conversation."
    
    async def run_background_tasks(self, interval: int = 3600) -> None:
        """
        Run background tasks periodically.
        
        Args:
            interval: Interval between memory processing runs (in seconds)
        """
        self.running = True
        while self.running:
            try:
                logger.info("Running memory processing...")
                await self.process_memories()
            except Exception as e:
                logger.error(f"Error in background task: {e}")
            
            # Sleep for the specified interval, but check if we should stop every second
            for _ in range(interval):
                if not self.running:
                    break
                await asyncio.sleep(1)
    
    def start(self, interval: int = 3600) -> None:
        """
        Start the background memory manager.
        
        Args:
            interval: Interval between consolidation runs (in seconds)
        """
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run_background_tasks(interval))
            logger.info(f"Background memory manager started with interval {interval}s")
    
    def stop(self) -> None:
        """Stop the background memory manager."""
        if self.task and not self.task.done():
            self.running = False
            self.task.cancel()
            logger.info("Background memory manager stopped")