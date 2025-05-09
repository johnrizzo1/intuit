import os
import json
import sys
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class CalendarEvent(BaseModel):
    """Pydantic model for a calendar event."""
    id: str = Field(...)
    details: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)

class CalendarTool:
    name = "calendar"
    description = "A tool for managing calendar events (add, list, search, delete)."

    def __init__(self, data_dir="data/calendar"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_filepath(self, event_id: str) -> str:
        """Generates the file path for an event."""
        return os.path.join(self.data_dir, f"{event_id}.json")

    def add_event(self, event_details: str) -> str:
        """Adds a new calendar event."""
        event_id = datetime.now().strftime("%Y%m%d%H%M%S%f") # Use microsecond for uniqueness
        event = CalendarEvent(id=event_id, details=event_details)
        filepath = self._get_filepath(event_id)
        with open(filepath, "w") as f:
            f.write(event.model_dump_json(indent=4)) # Use model_dump_json for correct serialization and indentation
        return f"Event added with ID: {event_id}"

    def list_events(self) -> str:
        """Lists all calendar events."""
        events = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        event = CalendarEvent.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        events.append(f"ID: {event.id}, Details: {event.details}, Timestamp: {event.timestamp}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading event file {filename}: {e}", file=sys.stderr) # Log error instead of failing
                    continue
        return "\n".join(events) if events else "No events found."

    def search_events(self, keyword: str) -> str:
        """Searches calendar events for a keyword."""
        matching_events = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        event = CalendarEvent.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        if keyword.lower() in event.details.lower():
                            matching_events.append(f"ID: {event.id}, Details: {event.details}, Timestamp: {event.timestamp}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading event file {filename}: {e}", file=sys.stderr)
                    continue
        return "\n".join(matching_events) if matching_events else f"No events found matching '{keyword}'."

    def delete_event(self, event_id: str) -> str:
        """Deletes a calendar event by ID."""
        filepath = self._get_filepath(event_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"Event deleted with ID: {event_id}"
        return f"Error: Event with ID '{event_id}' not found at {filepath}"