import os
import json
import sys
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class Note(BaseModel):
    """Pydantic model for a note."""
    id: str = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)

class NotesTool:
    name = "notes"
    description = "A tool for managing notes (add, list, search, delete)."

    def __init__(self, data_dir="data/notes"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_filepath(self, note_id: str) -> str:
        """Generates the file path for a note."""
        return os.path.join(self.data_dir, f"{note_id}.json")

    def add_note(self, note_content: str) -> str:
        """Adds a new note."""
        note_id = datetime.now().strftime("%Y%m%d%H%M%S%f") # Use microsecond for uniqueness
        note = Note(id=note_id, content=note_content)
        filepath = self._get_filepath(note_id)
        with open(filepath, "w") as f:
            f.write(note.model_dump_json(indent=4)) # Use model_dump_json for correct serialization and indentation
        return f"Note added with ID: {note_id}"

    def list_notes(self) -> str:
        """Lists all notes."""
        notes = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        note = Note.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        notes.append(f"ID: {note.id}, Content: {note.content}, Timestamp: {note.timestamp}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading note file {filename}: {e}", file=sys.stderr) # Log error instead of failing
                    continue
        return "\n".join(notes) if notes else "No notes found."

    def search_notes(self, keyword: str) -> str:
        """Searches notes for a keyword."""
        matching_notes = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        note = Note.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        if keyword.lower() in note.content.lower():
                            matching_notes.append(f"ID: {note.id}, Content: {note.content}, Timestamp: {note.timestamp}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading note file {filename}: {e}", file=sys.stderr)
                    continue
        return "\n".join(matching_notes) if matching_notes else f"No notes found matching '{keyword}'."

    def delete_note(self, note_id: str) -> str:
        """Deletes a note by ID."""
        filepath = self._get_filepath(note_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"Note deleted with ID: {note_id}"
        return f"Error: Note with ID '{note_id}' not found at {filepath}"