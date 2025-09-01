"""
Reminders CLI commands.
"""

import typer
from typing import Optional
from datetime import datetime
from .shared import create_agent_sync, get_tool_from_agent
from ..tools.reminders import RemindersTool

# Create the reminders app
reminders_app = typer.Typer(name="reminders", help="Manage reminders")


@reminders_app.command()
def add(
    content: str,
    time: Optional[datetime] = typer.Option(
        None, help="Optional reminder time (ISO 8601 format)"
    ),
) -> None:
    """Adds a new reminder."""
    agent = create_agent_sync()
    reminders_tool = get_tool_from_agent(agent, RemindersTool)
    if reminders_tool:
        print(reminders_tool.add_reminder(content, time))
    else:
        print("Reminders tool not available.")


@reminders_app.command()
def list() -> None:
    """Lists all reminders."""
    agent = create_agent_sync()
    reminders_tool = get_tool_from_agent(agent, RemindersTool)
    if reminders_tool:
        print(reminders_tool.list_reminders())
    else:
        print("Reminders tool not available.")


@reminders_app.command()
def search(keyword: str) -> None:
    """Searches reminders for a keyword."""
    agent = create_agent_sync()
    reminders_tool = get_tool_from_agent(agent, RemindersTool)
    if reminders_tool:
        print(reminders_tool.search_reminders(keyword))
    else:
        print("Reminders tool not available.")


@reminders_app.command()
def delete(id: str) -> None:
    """Deletes a reminder by ID."""
    agent = create_agent_sync()
    reminders_tool = get_tool_from_agent(agent, RemindersTool)
    if reminders_tool:
        print(reminders_tool.delete_reminder(id))
    else:
        print("Reminders tool not available.")
