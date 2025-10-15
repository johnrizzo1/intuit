"""
Tests for the voice TUI interface.
"""
import pytest
from unittest.mock import Mock, AsyncMock
from intuit.ui.voice_tui import (
    AudioMeter,
    ConversationMessage,
    StatusBar,
    VoiceMetrics,
    VoiceTUI,
)
from intuit.agent import Agent


class TestAudioMeter:
    """Tests for AudioMeter widget."""
    
    def test_initialization(self):
        """Test AudioMeter initialization."""
        meter = AudioMeter()
        assert meter.level == 0.0
        assert meter.max_level == 0.0
    
    def test_update_level(self):
        """Test updating audio level."""
        meter = AudioMeter()
        meter.update_level(0.5)
        assert meter.level == 0.5
        
        # Test clamping to 1.0
        meter.update_level(1.5)
        assert meter.level == 1.0


class TestConversationMessage:
    """Tests for ConversationMessage widget."""
    
    def test_initialization(self):
        """Test ConversationMessage initialization."""
        msg = ConversationMessage("User", "Hello", "12:00:00")
        assert msg.speaker == "User"
        assert msg.message == "Hello"
        assert msg.timestamp == "12:00:00"


class TestStatusBar:
    """Tests for StatusBar widget."""
    
    def test_initialization(self):
        """Test StatusBar initialization."""
        status = StatusBar()
        assert status.status == "Initializing..."
        assert status.listening is False
        assert status.processing is False
        assert status.speaking is False
    
    def test_status_updates(self):
        """Test status updates."""
        status = StatusBar()
        status.listening = True
        assert status.listening is True
        
        status.processing = True
        assert status.processing is True
        
        status.speaking = True
        assert status.speaking is True


class TestVoiceMetrics:
    """Tests for VoiceMetrics widget."""
    
    def test_initialization(self):
        """Test VoiceMetrics initialization."""
        metrics = VoiceMetrics()
        assert metrics.latency == 0.0
        assert metrics.confidence == 0.0
        assert metrics.duration == 0.0
    
    def test_metric_updates(self):
        """Test metric updates."""
        metrics = VoiceMetrics()
        metrics.latency = 1.5
        metrics.confidence = 0.85
        metrics.duration = 3.2
        
        assert metrics.latency == 1.5
        assert metrics.confidence == 0.85
        assert metrics.duration == 3.2


class TestVoiceTUI:
    """Tests for VoiceTUI application."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock(spec=Agent)
        agent.run = AsyncMock(return_value="Test response")
        agent.reminder_service = None
        agent.shutdown_mcp_clients = AsyncMock()
        return agent
    
    def test_initialization(self, mock_agent):
        """Test VoiceTUI initialization."""
        app = VoiceTUI(mock_agent)
        assert app.agent == mock_agent
        assert app.running is True
        assert app.sample_rate == 16000
    
    def test_bindings(self, mock_agent):
        """Test that key bindings are defined."""
        app = VoiceTUI(mock_agent)
        binding_keys = [b[0] for b in app.BINDINGS]
        assert "q" in binding_keys
        assert "c" in binding_keys