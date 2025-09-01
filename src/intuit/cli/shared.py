"""
Shared utilities for CLI commands.
"""

import asyncio
import functools
from typing import Type, Any, Callable
from ..agent_factory import create_agent
from ..tools.basetool import BaseTool


def tool_command(tool_cls: Type, method_name: str):
    """
    Decorator factory for creating CLI commands that use tools.

    Args:
        tool_cls: The tool class to find in the agent
        method_name: The method name to call on the tool

    Returns:
        Decorator function
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> None:
            agent = asyncio.run(create_agent())
            tool = next((t for t in agent.tools if isinstance(t, tool_cls)), None)
            if not tool:
                print(f"{tool_cls.__name__} not available.")
                return

            # Call the method on the tool
            result = getattr(tool, method_name)(*args, **kwargs)
            print(result)

        return wrapper

    return decorator


def get_tool_from_agent(agent, tool_cls: Type) -> Any:
    """
    Get a tool instance from the agent.

    Args:
        agent: The agent instance
        tool_cls: The tool class to find

    Returns:
        Tool instance or None if not found
    """
    return next((tool for tool in agent.tools if isinstance(tool, tool_cls)), None)


def create_agent_sync() -> Any:
    """
    Create an agent synchronously.

    Returns:
        Agent instance
    """
    return asyncio.run(create_agent())
