"""
Notes CLI commands.
"""

import typer
from .shared import create_agent_sync, get_tool_from_agent
from ..tools.notes import NotesTool

# Create the notes app
notes_app = typer.Typer(name="notes", help="Manage notes")


@notes_app.command()
def add(content: str) -> None:
    """Adds a new note."""
    agent = create_agent_sync()
    notes_tool = get_tool_from_agent(agent, NotesTool)
    if notes_tool:
        print(notes_tool.add_note(content))
    else:
        print("Notes tool not available.")


@notes_app.command()
def list() -> None:
    """Lists all notes."""
    agent = create_agent_sync()
    notes_tool = get_tool_from_agent(agent, NotesTool)
    if notes_tool:
        print(notes_tool.list_notes())
    else:
        print("Notes tool not available.")


@notes_app.command()
def search(keyword: str) -> None:
    """Searches notes for a keyword."""
    agent = create_agent_sync()
    notes_tool = get_tool_from_agent(agent, NotesTool)
    if notes_tool:
        print(notes_tool.search_notes(keyword))
    else:
        print("Notes tool not available.")


@notes_app.command()
def delete(id: str) -> None:
    """Deletes a note by ID."""
    agent = create_agent_sync()
    notes_tool = get_tool_from_agent(agent, NotesTool)
    if notes_tool:
        print(notes_tool.delete_note(id))
    else:
        print("Notes tool not available.")
