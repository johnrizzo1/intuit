"""
Tests for the voice interface.
"""
import pytest
from unittest.mock import AsyncMock, patch, call
import numpy as np

from intuit.ui.voice import VoiceInterface
from intuit.audio.stt_base import STTProvider
from intuit.config.audio_config import AudioConfig, STTConfig


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing."""
    
    def __init__(self):
        self.transcribe_called = False
        self.last_audio = None
    
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Mock transcribe method."""
        self.transcribe_called = True
        self.last_audio = audio_data
        return "test query"
    
    async def cleanup(self):
        """Mock cleanup method."""
        pass


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = AsyncMock()
    agent.run.return_value = "Test response"
    agent.voice = False
    return agent


@pytest.fixture
def mock_stt_provider():
    """Create a mock STT provider."""
    return MockSTTProvider()


@pytest.fixture
def mock_audio_config():
    """Create a mock audio configuration."""
    return AudioConfig(
        stt=STTConfig(
            provider="whisper",
            model_size="base",
            device="cpu"
        )
    )


@pytest.fixture
def voice_interface(mock_agent, mock_stt_provider, mock_audio_config):
    """Create a voice interface with mocks."""
    with patch('intuit.ui.voice.STTFactory.create',
               return_value=mock_stt_provider):
        interface = VoiceInterface(
            mock_agent,
            audio_config=mock_audio_config
        )
        return interface


@pytest.mark.asyncio
async def test_voice_interface_initialization(
    voice_interface, mock_agent, mock_stt_provider
):
    """Test that the voice interface initializes correctly."""
    assert voice_interface.agent == mock_agent
    assert voice_interface.stt_provider == mock_stt_provider
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
async def test_voice_interface_listen(voice_interface, mock_stt_provider):
    """Test the listen method with new STT provider."""
    # Mock audio data
    mock_audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    with patch('sounddevice.InputStream') as mock_stream:
        # Set up the mock stream
        mock_stream.return_value.__enter__.return_value = None
        
        # Add mock audio data to queue
        voice_interface.audio_queue.put(mock_audio)
        
        # Test listening
        result = await voice_interface._listen()
        assert result == "test query"
        assert mock_stt_provider.transcribe_called
        assert mock_stt_provider.last_audio is not None


@pytest.mark.asyncio
async def test_voice_interface_speak(voice_interface):
    """Test the speak method."""
    with patch('gtts.gTTS.save') as mock_save, \
         patch('asyncio.create_subprocess_exec') as mock_exec, \
         patch('os.unlink') as mock_unlink:
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_exec.return_value = mock_process
        
        await voice_interface._speak("test message")
        mock_save.assert_called_once()
        mock_exec.assert_called_once()
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
        
        # Verify error and goodbye messages were spoken
        assert mock_speak.call_count == 2
        # First call is error message with session_id
        assert "Error: Test error" in str(mock_speak.call_args_list[0])
        # Second call is goodbye
        assert mock_speak.call_args_list[1] == call("Goodbye!")


@pytest.mark.asyncio
async def test_voice_interface_with_whisper_provider():
    """Test voice interface with Whisper STT provider."""
    mock_agent = AsyncMock()
    mock_agent.run.return_value = "Response"
    
    # Create config for Whisper
    config = AudioConfig(
        stt=STTConfig(
            provider="whisper",
            model_size="tiny",
            device="cpu"
        )
    )
    
    with patch('intuit.ui.voice.STTFactory.create') as mock_create:
        mock_provider = MockSTTProvider()
        mock_create.return_value = mock_provider
        
        interface = VoiceInterface(mock_agent, audio_config=config)
        
        # Verify Whisper provider was created
        mock_create.assert_called_once()
        assert interface.stt_provider == mock_provider


@pytest.mark.asyncio
async def test_voice_interface_with_google_provider():
    """Test voice interface with Google STT provider."""
    mock_agent = AsyncMock()
    mock_agent.run.return_value = "Response"
    
    # Create config for Google
    config = AudioConfig(
        stt=STTConfig(
            provider="google",
            language="en-US"
        )
    )
    
    with patch('intuit.ui.voice.STTFactory.create') as mock_create:
        mock_provider = MockSTTProvider()
        mock_create.return_value = mock_provider
        
        interface = VoiceInterface(mock_agent, audio_config=config)
        
        # Verify Google provider was created
        mock_create.assert_called_once()
        assert interface.stt_provider == mock_provider


@pytest.mark.asyncio
async def test_voice_interface_stt_error_handling(
    voice_interface, mock_stt_provider
):
    """Test STT error handling."""
    # Make STT provider raise an error
    async def error_transcribe(audio_data):
        raise Exception("STT error")
    
    mock_stt_provider.transcribe = error_transcribe
    
    with patch('sounddevice.InputStream') as mock_stream:
        mock_stream.return_value.__enter__.return_value = None
        
        # Add mock audio data
        mock_audio = np.array([[0.1, 0.2]], dtype=np.float32)
        voice_interface.audio_queue.put(mock_audio)
        
        # Should return None on error
        result = await voice_interface._listen()
        assert result is None