import os
import pytest
import json
from datetime import datetime
from intuit.tools.calendar import CalendarTool, CalendarEvent

@pytest.fixture
def calendar_tool(tmp_path):
    """Fixture to create a CalendarTool instance with a temporary data directory."""
    data_dir = tmp_path / "calendar_data"
    return CalendarTool(data_dir=str(data_dir))

def test_add_event(calendar_tool):
    """Test adding a calendar event."""
    event_details = "Meeting with John tomorrow at 10 AM"
    result = calendar_tool.add_event(event_details)
    assert result.startswith("Event added with ID:")
    event_id = result.split(": ")[1]
    filepath = os.path.join(calendar_tool.data_dir, f"{event_id}.json")
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        event_data = json.load(f)
        event = CalendarEvent(**event_data)
        assert event.id == event_id
        assert event.details == event_details
        assert isinstance(event.timestamp, datetime)

def test_list_events(calendar_tool):
    """Test listing calendar events."""
    event1_details = "Event 1"
    event2_details = "Event 2"
    calendar_tool.add_event(event1_details)
    calendar_tool.add_event(event2_details)
    events_output = calendar_tool.list_events()
    assert event1_details in events_output
    assert event2_details in events_output
    assert "ID:" in events_output
    assert "Timestamp:" in events_output

def test_search_events(calendar_tool):
    """Test searching calendar events."""
    event1_details = "Meeting with Jane"
    event2_details = "Call with John"
    event3_details = "Team meeting"
    calendar_tool.add_event(event1_details)
    calendar_tool.add_event(event2_details)
    calendar_tool.add_event(event3_details)
    
    search_results = calendar_tool.search_events("Jane")
    assert event1_details in search_results
    assert event2_details not in search_results
    assert event3_details not in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results

    search_results = calendar_tool.search_events("meeting")
    assert event1_details in search_results
    assert event2_details not in search_results
    assert event3_details in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results

def test_delete_event(calendar_tool):
    """Test deleting a calendar event by ID."""
    event_details = "Event to delete"
    result = calendar_tool.add_event(event_details)
    event_id = result.split(": ")[1]
    filepath = os.path.join(calendar_tool.data_dir, f"{event_id}.json")
    assert os.path.exists(filepath)
    
    delete_result = calendar_tool.delete_event(event_id)
    assert delete_result.startswith("Event deleted with ID:")
    assert not os.path.exists(filepath)

def test_delete_nonexistent_event(calendar_tool):
    """Test deleting a non-existent event."""
    event_id = "nonexistent_id"
    filepath = os.path.join(calendar_tool.data_dir, f"{event_id}.json")
    delete_result = calendar_tool.delete_event(event_id)
    assert delete_result == f"Error: Event with ID '{event_id}' not found at {filepath}"