import os
import json
import sys
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class Reminder(BaseModel):
    """Pydantic model for a reminder."""
    id: str = Field(...)
    content: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    reminder_time: Optional[datetime] = Field(default=None) # Add reminder_time field

class RemindersTool:
    name = "reminders"
    description = "A tool for managing reminders (add, list, search, delete)."

    def __init__(self, data_dir="data/reminders"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_filepath(self, reminder_id: str) -> str:
        """Generates the file path for a reminder."""
        return os.path.join(self.data_dir, f"{reminder_id}.json")

    def add_reminder(self, reminder_details: str, reminder_time: Optional[datetime] = None) -> str:
        """Adds a new reminder."""
        reminder_id = datetime.now().strftime("%Y%m%d%H%M%S%f") # Use microsecond for uniqueness
        reminder = Reminder(id=reminder_id, content=reminder_details, reminder_time=reminder_time)
        filepath = self._get_filepath(reminder_id)
        with open(filepath, "w") as f:
            f.write(reminder.model_dump_json(indent=4)) # Use model_dump_json for correct serialization and indentation
        return f"Reminder added with ID: {reminder_id}"

    def list_reminders(self) -> str:
        """Lists all reminders."""
        reminders = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        reminder = Reminder.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        reminders.append(f"ID: {reminder.id}, Content: {reminder.content}, Timestamp: {reminder.timestamp}, Reminder Time: {reminder.reminder_time}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading reminder file {filename}: {e}", file=sys.stderr) # Log error instead of failing
                    continue
        return "\n".join(reminders) if reminders else "No reminders found."

    def search_reminders(self, keyword: str) -> str:
        """Searches reminders for a keyword."""
        matching_reminders = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        reminder = Reminder.model_validate_json(f.read()) # Use model_validate_json to deserialize
                        if keyword.lower() in reminder.content.lower():
                            matching_reminders.append(f"ID: {reminder.id}, Content: {reminder.content}, Timestamp: {reminder.timestamp}, Reminder Time: {reminder.reminder_time}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading reminder file {filename}: {e}", file=sys.stderr)
                    continue
        return "\n".join(matching_reminders) if matching_reminders else f"No reminders found matching '{keyword}'."

    def delete_reminder(self, reminder_id: str) -> str:
        """Deletes a reminder by ID."""
        filepath = self._get_filepath(reminder_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"Reminder deleted with ID: {reminder_id}"
        return f"Error: Reminder with ID '{reminder_id}' not found at {filepath}"
    
    async def _arun(self, action: str, content: Optional[str] = None,
                    reminder_time: Optional[str] = None, keyword: Optional[str] = None,
                    id: Optional[str] = None) -> str:
        """
        Async run method for LangChain compatibility.
        Routes the action to the appropriate method.
        """
        if action == "add":
            if not content:
                return "Error: 'content' is required for add action"
            # Parse reminder_time if provided
            parsed_time = None
            if reminder_time:
                try:
                    parsed_time = datetime.fromisoformat(reminder_time)
                except ValueError:
                    return f"Error: Invalid reminder_time format: {reminder_time}"
            return self.add_reminder(content, parsed_time)
        elif action == "list":
            return self.list_reminders()
        elif action == "search":
            if not keyword:
                return "Error: 'keyword' is required for search action"
            return self.search_reminders(keyword)
        elif action == "delete":
            if not id:
                return "Error: 'id' is required for delete action"
            return self.delete_reminder(id)
        else:
            return f"Error: Unknown action '{action}'"