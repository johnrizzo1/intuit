"""
Calendar CLI commands.
"""

import typer
from .shared import create_agent_sync, get_tool_from_agent
from ..tools.calendar import CalendarTool

# Create the calendar app
calendar_app = typer.Typer(name="calendar", help="Manage calendar events")


@calendar_app.command()
def add(event: str) -> None:
    """Adds a new calendar event."""
    agent = create_agent_sync()
    calendar_tool = get_tool_from_agent(agent, CalendarTool)
    if calendar_tool:
        print(calendar_tool.add_event(event))
    else:
        print("Calendar tool not available.")


@calendar_app.command()
def list() -> None:
    """Lists all calendar events."""
    agent = create_agent_sync()
    calendar_tool = get_tool_from_agent(agent, CalendarTool)
    if calendar_tool:
        print(calendar_tool.list_events())
    else:
        print("Calendar tool not available.")


@calendar_app.command()
def search(keyword: str) -> None:
    """Searches calendar events for a keyword."""
    agent = create_agent_sync()
    calendar_tool = get_tool_from_agent(agent, CalendarTool)
    if calendar_tool:
        print(calendar_tool.search_events(keyword))
    else:
        print("Calendar tool not available.")


@calendar_app.command()
def delete(filename: str) -> None:
    """Deletes a calendar event by filename."""
    agent = create_agent_sync()
    calendar_tool = get_tool_from_agent(agent, CalendarTool)
    if calendar_tool:
        print(calendar_tool.delete_event(filename))
    else:
        print("Calendar tool not available.")
