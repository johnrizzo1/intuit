"""
Tests for dictation mode functionality.
"""
import pytest
from unittest.mock import Mock
import speech_recognition as sr

from src.intuit.dictation import DictationMode, DictationState


class TestDictationMode:
    """Test suite for DictationMode class."""
    
    @pytest.fixture
    def recognizer(self):
        """Create a mock recognizer."""
        return Mock(spec=sr.Recognizer)
    
    @pytest.fixture
    def dictation_mode(self, recognizer):
        """Create a DictationMode instance."""
        return DictationMode(
            recognizer=recognizer,
            silence_threshold=30.0,
            sample_rate=16000
        )
    
    def test_initialization(self, dictation_mode):
        """Test DictationMode initialization."""
        assert dictation_mode.state == DictationState.IDLE
        assert dictation_mode.silence_threshold == 30.0
        assert dictation_mode.sample_rate == 16000
        assert len(dictation_mode.transcription_buffer) == 0
    
    def test_wake_word_detection(self, dictation_mode):
        """Test wake word detection."""
        assert dictation_mode._is_wake_word("hey intuit")
        assert dictation_mode._is_wake_word("Hey Intuit")
        assert dictation_mode._is_wake_word("HEY INTUIT")
        assert not dictation_mode._is_wake_word("hello there")
    
    def test_pause_command_detection(self, dictation_mode):
        """Test pause command detection."""
        assert dictation_mode._is_pause_command("pause dictation")
        assert dictation_mode._is_pause_command("Pause Dictation")
        assert dictation_mode._is_pause_command("PAUSE DICTATION")
        assert not dictation_mode._is_pause_command("pause")
    
    def test_resume_command_detection(self, dictation_mode):
        """Test resume command detection."""
        assert dictation_mode._is_resume_command("resume dictation")
        assert dictation_mode._is_resume_command("Resume Dictation")
        assert not dictation_mode._is_resume_command("resume")
    
    def test_end_command_detection(self, dictation_mode):
        """Test end command detection."""
        assert dictation_mode._is_end_command("end dictation")
        assert dictation_mode._is_end_command("End Dictation")
        assert not dictation_mode._is_end_command("end")
    
    def test_start_command_detection(self, dictation_mode):
        """Test start command detection."""
        assert dictation_mode._is_start_command("start dictation")
        assert dictation_mode._is_start_command("Start Dictation")
        assert not dictation_mode._is_start_command("start")
    
    def test_remove_commands_from_text(self, dictation_mode):
        """Test command removal from text."""
        text = "hey intuit this is a test"
        clean = dictation_mode._remove_commands_from_text(text)
        assert "hey intuit" not in clean.lower()
        assert "this is a test" in clean
        
        text = "pause dictation now"
        clean = dictation_mode._remove_commands_from_text(text)
        assert "pause dictation" not in clean.lower()
        assert "now" in clean
    
    def test_add_transcription(self, dictation_mode):
        """Test adding transcription text."""
        callback_called = False
        received_text = None
        
        def callback(text):
            nonlocal callback_called, received_text
            callback_called = True
            received_text = text
        
        dictation_mode.on_transcription = callback
        dictation_mode._add_transcription("Hello world")
        
        assert callback_called
        assert received_text == "Hello world"
        assert "Hello world" in dictation_mode.transcription_buffer
    
    def test_get_full_transcription(self, dictation_mode):
        """Test getting full transcription."""
        dictation_mode._add_transcription("First sentence.")
        dictation_mode._add_transcription("Second sentence.")
        
        full = dictation_mode.get_full_transcription()
        assert full == "First sentence. Second sentence."
    
    def test_clear_transcription(self, dictation_mode):
        """Test clearing transcription."""
        dictation_mode._add_transcription("Some text")
        assert len(dictation_mode.transcription_buffer) > 0
        
        dictation_mode.clear_transcription()
        assert len(dictation_mode.transcription_buffer) == 0
    
    def test_state_changes(self, dictation_mode):
        """Test state change notifications."""
        state_changes = []
        
        def callback(state):
            state_changes.append(state)
        
        dictation_mode.on_state_change = callback
        
        dictation_mode._set_state(DictationState.LISTENING)
        assert dictation_mode.state == DictationState.LISTENING
        assert DictationState.LISTENING in state_changes
        
        dictation_mode._set_state(DictationState.PAUSED)
        assert dictation_mode.state == DictationState.PAUSED
        assert DictationState.PAUSED in state_changes
    
    @pytest.mark.asyncio
    async def test_save_transcription(self, dictation_mode, tmp_path):
        """Test saving transcription to file."""
        dictation_mode._add_transcription("Test transcription content")
        
        filepath = await dictation_mode.save_transcription(tmp_path)
        
        assert filepath.exists()
        assert filepath.parent == tmp_path
        assert filepath.name.startswith("dictation_")
        assert filepath.suffix == ".txt"
        
        content = filepath.read_text()
        assert "Test transcription content" in content
    
    def test_stop(self, dictation_mode):
        """Test stopping dictation."""
        dictation_mode._set_state(DictationState.LISTENING)
        dictation_mode.running = True
        
        dictation_mode.stop()
        
        assert not dictation_mode.running
        assert dictation_mode.state == DictationState.ENDED


class TestDictationIntegration:
    """Integration tests for dictation mode."""
    
    @pytest.mark.asyncio
    async def test_dictation_workflow(self):
        """Test a complete dictation workflow."""
        recognizer = Mock(spec=sr.Recognizer)
        transcriptions = []
        states = []
        
        def on_transcription(text):
            transcriptions.append(text)
        
        def on_state_change(state):
            states.append(state)
        
        dictation = DictationMode(
            recognizer=recognizer,
            silence_threshold=30.0,
            on_transcription=on_transcription,
            on_state_change=on_state_change
        )
        
        # Simulate state changes
        dictation._set_state(DictationState.LISTENING)
        assert DictationState.LISTENING in states
        
        # Simulate transcription
        dictation._add_transcription("This is a test")
        assert "This is a test" in transcriptions
        
        # Simulate pause
        dictation._set_state(DictationState.PAUSED)
        assert DictationState.PAUSED in states
        
        # Simulate resume
        dictation._set_state(DictationState.LISTENING)
        
        # Add more transcription
        dictation._add_transcription("More content")
        assert "More content" in transcriptions
        
        # End dictation
        dictation._set_state(DictationState.ENDED)
        assert DictationState.ENDED in states
        
        # Verify full transcription
        full = dictation.get_full_transcription()
        assert "This is a test" in full
        assert "More content" in full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])