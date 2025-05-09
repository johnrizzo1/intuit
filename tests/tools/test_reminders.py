import os
import pytest
import json
from datetime import datetime
from intuit.tools.reminders import RemindersTool, Reminder

@pytest.fixture
def reminders_tool(tmp_path):
    """Fixture to create a RemindersTool instance with a temporary data directory."""
    data_dir = tmp_path / "reminders_data"
    return RemindersTool(data_dir=str(data_dir))

def test_add_reminder(reminders_tool):
    """Test adding a reminder."""
    reminder_content = "Buy milk."
    reminder_time = datetime(2025, 12, 31, 10, 0, 0)
    result = reminders_tool.add_reminder(reminder_content, reminder_time)
    assert result.startswith("Reminder added with ID:")
    reminder_id = result.split(": ")[1]
    filepath = os.path.join(reminders_tool.data_dir, f"{reminder_id}.json")
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        reminder_data = json.load(f)
        reminder = Reminder(**reminder_data)
        assert reminder.id == reminder_id
        assert reminder.content == reminder_content
        assert isinstance(reminder.timestamp, datetime)
        assert reminder.reminder_time == reminder_time

def test_add_reminder_without_time(reminders_tool):
    """Test adding a reminder without a specific time."""
    reminder_content = "Walk the dog."
    result = reminders_tool.add_reminder(reminder_content)
    assert result.startswith("Reminder added with ID:")
    reminder_id = result.split(": ")[1]
    filepath = os.path.join(reminders_tool.data_dir, f"{reminder_id}.json")
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        reminder_data = json.load(f)
        reminder = Reminder(**reminder_data)
        assert reminder.id == reminder_id
        assert reminder.content == reminder_content
        assert isinstance(reminder.timestamp, datetime)
        assert reminder.reminder_time is None

def test_list_reminders(reminders_tool):
    """Test listing reminders."""
    reminder1_content = "Reminder 1 content."
    reminder2_content = "Reminder 2 content."
    reminders_tool.add_reminder(reminder1_content)
    reminders_tool.add_reminder(reminder2_content)
    reminders_output = reminders_tool.list_reminders()
    assert reminder1_content in reminders_output
    assert reminder2_content in reminders_output
    assert "ID:" in reminders_output
    assert "Timestamp:" in reminders_output
    assert "Reminder Time:" in reminders_output

def test_search_reminders(reminders_tool):
    """Test searching reminders."""
    reminder1_content = "Meeting reminder."
    reminder2_content = "Call John reminder."
    reminder3_content = "Team meeting reminder."
    reminders_tool.add_reminder(reminder1_content)
    reminders_tool.add_reminder(reminder2_content)
    reminders_tool.add_reminder(reminder3_content)
    
    search_results = reminders_tool.search_reminders("Meeting")
    assert reminder1_content in search_results
    assert reminder2_content not in search_results
    assert reminder3_content in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results
    assert "Reminder Time:" in search_results

    search_results = reminders_tool.search_reminders("John")
    assert reminder1_content not in search_results
    assert reminder2_content in search_results
    assert reminder3_content not in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results
    assert "Reminder Time:" in search_results

def test_delete_reminder(reminders_tool):
    """Test deleting a reminder by ID."""
    reminder_content = "Reminder to delete."
    result = reminders_tool.add_reminder(reminder_content)
    reminder_id = result.split(": ")[1]
    filepath = os.path.join(reminders_tool.data_dir, f"{reminder_id}.json")
    assert os.path.exists(filepath)
    
    delete_result = reminders_tool.delete_reminder(reminder_id)
    assert delete_result.startswith("Reminder deleted with ID:")
    assert not os.path.exists(filepath)

def test_delete_nonexistent_reminder(reminders_tool):
    """Test deleting a non-existent reminder."""
    reminder_id = "nonexistent_id"
    filepath = os.path.join(reminders_tool.data_dir, f"{reminder_id}.json")
    delete_result = reminders_tool.delete_reminder(reminder_id)
    assert delete_result == f"Error: Reminder with ID '{reminder_id}' not found at {filepath}"