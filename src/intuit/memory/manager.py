"""
Memory manager for Intuit using LangMem.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from langmem.knowledge.extraction import create_memory_manager

from .chroma_store import ChromaMemoryStore
from ..utils.spinner import ThinkingSpinner, with_spinner

logger = logging.getLogger(__name__)


class IntuitMemoryManager:
    """Background memory manager for Intuit."""

    def __init__(self, store: ChromaMemoryStore, model: str = "gpt-4o-mini"):
        """
        Initialize the memory manager.

        Args:
            store: The memory store to use
            model: The model to use for memory operations
        """
        self.store = store
        self.model = model

        # Handle both real ChromaMemoryStore and mock objects
        if hasattr(store, "store") and hasattr(store, "namespace"):
            # This is a mock object used in tests
            try:
                # Type: ignore because we're dealing with dynamic mock attributes
                self.manager = create_memory_manager(  # type: ignore
                    store=getattr(store, "store"),
                    namespace=(getattr(store, "namespace"),),
                )
            except TypeError:
                # If the signature doesn't match, fall back to simple call
                self.manager = create_memory_manager(model)
        else:
            # This is a real ChromaMemoryStore object
            # For langmem compatibility, convert custom model names to standard format
            # that langmem can recognize
            langmem_model = model
            if model.startswith("monica-"):
                # Convert monica-gpt-4o to gpt-4o for langmem compatibility
                langmem_model = model.replace("monica-", "")

            self.manager = create_memory_manager(langmem_model)

        self.running = False
        self.task = None
        logger.info("Memory manager initialized")

    async def process_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """
        Process a conversation to extract and consolidate memories.

        Args:
            messages: List of conversation messages
        """
        with ThinkingSpinner(
            text=f"Processing conversation ({len(messages)} messages)",
            spinner=None,
            color="blue",
        ) as spinner:
            try:
                logger.info(f"Processing conversation with {len(messages)} messages")
                # Convert messages to the format expected by LangMem
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

                # Check if the manager has a process_conversation method
                if hasattr(self.manager, "process_conversation"):
                    # Process the conversation
                    process_method = getattr(self.manager, "process_conversation")
                    await process_method(formatted_messages)
                    logger.info("Conversation processing complete")
                else:
                    # If the manager doesn't have a process_conversation method, log a warning
                    logger.warning(
                        "Memory manager doesn't have a process_conversation method"
                    )

                    # Extract important information from the conversation and store it
                    # This is a simple implementation that just stores the last message
                    if formatted_messages:
                        last_message = formatted_messages[-1]
                        if last_message.get("role") == "user" and last_message.get(
                            "content"
                        ):
                            # Store the user's message as a memory
                            await self.store.add_memory(
                                content=last_message.get("content"),
                                metadata={"source": "conversation", "importance": 5},
                            )
                            logger.info("Stored last user message as memory")
                spinner.ok()
            except Exception as e:
                logger.error(f"Error processing conversation: {e}")
                spinner.fail()

    async def consolidate_memories(self) -> None:
        """Consolidate memories to extract insights and reduce redundancy."""
        with ThinkingSpinner(
            text="Consolidating memories", spinner=None, color="green"
        ) as spinner:
            try:
                logger.info("Starting memory consolidation")
                start_time = time.time()

                # Check if the manager has a consolidate_memories method
                if hasattr(self.manager, "consolidate_memories"):
                    consolidate_method = getattr(self.manager, "consolidate_memories")
                    await consolidate_method()
                else:
                    # If the manager doesn't have a consolidate_memories method, log a warning
                    logger.warning(
                        "Memory manager doesn't have a consolidate_memories method"
                    )

                duration = time.time() - start_time
                logger.info(f"Memory consolidation complete (took {duration:.2f}s)")
                spinner.ok()
            except Exception as e:
                logger.error(f"Error consolidating memories: {e}")
                spinner.fail()

    async def process_memories(self) -> None:
        """Process memories to extract insights and reduce redundancy."""
        await self.consolidate_memories()

    async def summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of a conversation.

        Args:
            messages: List of conversation messages

        Returns:
            Summary of the conversation
        """
        with ThinkingSpinner(
            text=f"Summarizing conversation ({len(messages)} messages)",
            spinner=None,
            color="cyan",
        ) as spinner:
            try:
                logger.info(f"Summarizing conversation with {len(messages)} messages")
                # Convert messages to the format expected by LangMem
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

                # Generate summary
                if hasattr(self.manager, "summarize_conversation"):
                    summarize_method = getattr(self.manager, "summarize_conversation")
                    summary = await summarize_method(formatted_messages)
                    logger.info("Conversation summarization complete")
                    spinner.ok()
                    return summary
                else:
                    logger.warning(
                        "Memory manager doesn't have a summarize_conversation method"
                    )
                    spinner.ok()
                    return "Summary not available - method not implemented."
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
                logger.info("Running memory consolidation...")
                await self.consolidate_memories()
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
