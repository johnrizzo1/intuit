"""
Memory CLI commands.
"""

import asyncio
import typer
from typing import Optional, List
from .shared import create_agent_sync
from ..logging_config import configure_logging

# Create the memory app
memory_app = typer.Typer(name="memory", help="Manage memory")


def _create_agent_with_logging_control(verbose: int = 0):
    """Helper function to create an agent with proper logging control."""
    configure_logging(verbose=verbose, quiet=(verbose == 0))
    return create_agent_sync()


@memory_app.command()
def add(
    content: str,
    importance: int = typer.Option(5, help="Importance level (1-10)"),
    tags: Optional[List[str]] = typer.Option(
        None, help="Tags for categorizing the memory"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
) -> None:
    """Adds a new memory."""
    agent = _create_agent_with_logging_control(verbose)
    try:
        if hasattr(agent, "memory_store"):
            memory_store = agent.memory_store
            memory_id = asyncio.run(
                memory_store.add_memory(
                    content=content,
                    metadata={"importance": importance, "tags": tags or []},
                )
            )
            print(f"Memory added with ID: {memory_id}")
        else:
            print("Memory store not available.")
    except Exception as e:
        print(f"Error adding memory: {str(e)}")


@memory_app.command()
def search(
    query: str,
    limit: int = typer.Option(5, help="Maximum number of results to return"),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
) -> None:
    """Searches memories by semantic similarity."""
    agent = _create_agent_with_logging_control(verbose)
    try:
        if hasattr(agent, "memory_store"):
            memory_store = agent.memory_store
            memories = asyncio.run(memory_store.search_memories(query, limit))
            if not memories:
                print("No memories found matching your query.")
            else:
                print(f"Found {len(memories)} memories:")
                for i, memory in enumerate(memories):
                    print(f"{i+1}. {memory['content']}")
        else:
            print("Memory store not available.")
    except Exception as e:
        print(f"Error searching memories: {str(e)}")


@memory_app.command()
def get(
    memory_id: str,
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
) -> None:
    """Gets a specific memory by ID."""
    agent = _create_agent_with_logging_control(verbose)
    try:
        if hasattr(agent, "memory_store"):
            memory_store = agent.memory_store
            memory = asyncio.run(memory_store.get_memory(memory_id))
            if memory:
                print(f"Memory {memory_id}: {memory['content']}")
            else:
                print(f"Memory with ID {memory_id} not found.")
        else:
            print("Memory store not available.")
    except Exception as e:
        print(f"Error getting memory: {str(e)}")


@memory_app.command()
def delete(
    memory_id: str,
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    ),
) -> None:
    """Deletes a memory by ID."""
    agent = _create_agent_with_logging_control(verbose)
    try:
        if hasattr(agent, "memory_store"):
            memory_store = agent.memory_store
            success = asyncio.run(memory_store.delete_memory(memory_id))
            if success:
                print(f"Memory with ID {memory_id} deleted.")
            else:
                print(f"Failed to delete memory with ID {memory_id}.")
        else:
            print("Memory store not available.")
    except Exception as e:
        print(f"Error deleting memory: {str(e)}")


@memory_app.command()
def clear(
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity"
    )
) -> None:
    """Clears all memories."""
    agent = _create_agent_with_logging_control(verbose)
    try:
        if hasattr(agent, "memory_store"):
            memory_store = agent.memory_store
            success = asyncio.run(memory_store.clear_memories())
            if success:
                print("All memories cleared.")
            else:
                print("Failed to clear memories.")
        else:
            print("Memory store not available.")
    except Exception as e:
        print(f"Error clearing memories: {str(e)}")
