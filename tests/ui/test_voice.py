"""
Tests for the voice interface.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import speech_recognition as sr
import numpy as np
import sounddevice as sd

from intuit.ui.voice import VoiceInterface

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = AsyncMock()
    agent.run.return_value = "Test response"
    return agent

@pytest.fixture
def voice_interface(mock_agent):
    """Create a voice interface with a mock agent."""
    return VoiceInterface(mock_agent)

@pytest.mark.asyncio
async def test_voice_interface_initialization(voice_interface, mock_agent):
    """Test that the voice interface initializes correctly."""
    assert voice_interface.agent == mock_agent
    assert isinstance(voice_interface.recognizer, sr.Recognizer)
    assert voice_interface.sample_rate == 16000
    assert voice_interface.channels == 1
    assert voice_interface.dtype == np.float32

@pytest.mark.asyncio
async def test_audio_callback(voice_interface):
    """Test the audio callback function."""
    # Create mock audio data
    mock_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    # Call the callback
    voice_interface._audio_callback(mock_data, 2, None, None)
    
    # Check that data was added to queue
    assert not voice_interface.audio_queue.empty()
    queued_data = voice_interface.audio_queue.get()
    np.testing.assert_array_equal(queued_data, mock_data)

@pytest.mark.asyncio
async def test_voice_interface_listen(voice_interface):
    """Test the listen method."""
    # Mock audio data
    mock_audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    with patch('sounddevice.InputStream') as mock_stream, \
         patch.object(voice_interface.recognizer, 'recognize_google') as mock_recognize:
        # Set up the mock stream
        mock_stream.return_value.__enter__.return_value = None
        
        # Set up the mock recognition
        mock_recognize.return_value = "test query"
        
        # Add mock audio data to queue
        voice_interface.audio_queue.put(mock_audio)
        
        # Test listening
        result = await voice_interface._listen()
        assert result == "test query"

@pytest.mark.asyncio
async def test_voice_interface_speak(voice_interface):
    """Test the speak method."""
    with patch('gtts.gTTS.save') as mock_save, \
         patch('os.system') as mock_system, \
         patch('os.unlink') as mock_unlink:
        await voice_interface._speak("test message")
        mock_save.assert_called_once()
        mock_system.assert_called_once()
        mock_unlink.assert_called_once()

@pytest.mark.asyncio
async def test_voice_interface_run(voice_interface, mock_agent):
    """Test the run method."""
    with patch.object(voice_interface, '_listen') as mock_listen, \
         patch.object(voice_interface, '_speak') as mock_speak:
        # Test normal operation
        mock_listen.side_effect = ["test query", "exit"]
        await voice_interface.run()
        mock_agent.run.assert_called_once_with("test query")
        mock_speak.assert_called()

@pytest.mark.asyncio
async def test_voice_interface_error_handling(voice_interface, mock_agent):
    """Test error handling in the voice interface."""
    with patch.object(voice_interface, '_listen') as mock_listen, \
         patch.object(voice_interface, '_speak') as mock_speak:
        # Mock an error
        mock_agent.run.side_effect = Exception("Test error")
        mock_listen.side_effect = ["test query", "exit"]
        
        await voice_interface.run()
        
        # Verify both error and goodbye messages were spoken
        assert mock_speak.call_count == 2
        assert mock_speak.call_args_list[0] == call("Error: Test error")
        assert mock_speak.call_args_list[1] == call("Goodbye!") 