import os
import pytest
import json
from datetime import datetime
from intuit.tools.notes import NotesTool, Note

@pytest.fixture
def notes_tool(tmp_path):
    """Fixture to create a NotesTool instance with a temporary data directory."""
    data_dir = tmp_path / "notes_data"
    return NotesTool(data_dir=str(data_dir))

def test_add_note(notes_tool):
    """Test adding a note."""
    note_content = "This is a test note."
    result = notes_tool.add_note(note_content)
    assert result.startswith("Note added with ID:")
    note_id = result.split(": ")[1]
    filepath = os.path.join(notes_tool.data_dir, f"{note_id}.json")
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        note_data = json.load(f)
        note = Note(**note_data)
        assert note.id == note_id
        assert note.content == note_content
        assert isinstance(note.timestamp, datetime)

def test_list_notes(notes_tool):
    """Test listing notes."""
    note1_content = "Note 1 content."
    note2_content = "Note 2 content."
    notes_tool.add_note(note1_content)
    notes_tool.add_note(note2_content)
    notes_output = notes_tool.list_notes()
    assert note1_content in notes_output
    assert note2_content in notes_output
    assert "ID:" in notes_output
    assert "Timestamp:" in notes_output

def test_search_notes(notes_tool):
    """Test searching notes."""
    note1_content = "This note is about cats."
    note2_content = "This note is about dogs."
    note3_content = "Another note about cats and dogs."
    notes_tool.add_note(note1_content)
    notes_tool.add_note(note2_content)
    notes_tool.add_note(note3_content)
    
    search_results = notes_tool.search_notes("cats")
    assert note1_content in search_results
    assert note2_content not in search_results
    assert note3_content in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results

    search_results = notes_tool.search_notes("dogs")
    assert note1_content not in search_results
    assert note2_content in search_results
    assert note3_content in search_results
    assert "ID:" in search_results
    assert "Timestamp:" in search_results

def test_delete_note(notes_tool):
    """Test deleting a note by ID."""
    note_content = "Note to delete."
    result = notes_tool.add_note(note_content)
    note_id = result.split(": ")[1]
    filepath = os.path.join(notes_tool.data_dir, f"{note_id}.json")
    assert os.path.exists(filepath)
    
    delete_result = notes_tool.delete_note(note_id)
    assert delete_result.startswith("Note deleted with ID:")
    assert not os.path.exists(filepath)

def test_delete_nonexistent_note(notes_tool):
    """Test deleting a non-existent note."""
    note_id = "nonexistent_id"
    filepath = os.path.join(notes_tool.data_dir, f"{note_id}.json")
    delete_result = notes_tool.delete_note(note_id)
    assert delete_result == f"Error: Note with ID '{note_id}' not found at {filepath}"