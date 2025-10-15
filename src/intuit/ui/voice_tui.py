"""
Voice TUI interface for Intuit using Textual.

This module provides a rich terminal user interface for voice interactions,
displaying conversation history, audio metrics, and real-time status.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import sounddevice as sd
from textual.app import App, ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    ScrollableContainer,
)
from textual.widgets import Header, Footer, Static, TextArea, Button
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

from ..agent import Agent


class AudioMeter(Static):
    """Widget to display audio input level as a meter."""
    
    level = reactive(0.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_level = 0.0
        self.decay_rate = 0.95
    
    def render(self) -> Panel:
        """Render the audio meter."""
        # Create a visual meter
        bar_width = 40
        filled = int(self.level * bar_width)
        empty = bar_width - filled
        
        # Color based on level
        if self.level > 0.8:
            color = "red"
        elif self.level > 0.5:
            color = "yellow"
        else:
            color = "green"
        
        meter = f"[{color}]{'█' * filled}[/]{' ' * empty}"
        level_text = f"{int(self.level * 100):3d}%"
        
        content = f"🎤 Input Level: {meter} {level_text}"
        return Panel(content, title="Microphone", border_style="blue")
    
    def update_level(self, level: float) -> None:
        """Update the audio level."""
        self.level = min(1.0, level)
        self.max_level = max(self.max_level * self.decay_rate, self.level)


class ConversationMessage(Static):
    """Widget to display a single conversation message."""
    
    def __init__(self, speaker: str, message: str, timestamp: str, **kwargs):
        super().__init__(**kwargs)
        self.speaker = speaker
        self.message = message
        self.timestamp = timestamp
    
    def render(self) -> Panel:
        """Render the message."""
        if self.speaker == "You":
            style = "cyan"
            align = "right"
        else:
            style = "green"
            align = "left"
        
        content = Text()
        content.append(f"{self.timestamp}\n", style="dim")
        content.append(self.message)
        
        if align == "right":
            content = Align.right(content)
        else:
            content = Align.left(content)
        
        return Panel(
            content,
            title=f"[bold {style}]{self.speaker}[/]",
            border_style=style,
            padding=(0, 1)
        )


class StatusBar(Static):
    """Widget to display current status and metrics."""
    
    status = reactive("Initializing...")
    listening = reactive(False)
    processing = reactive(False)
    speaking = reactive(False)
    
    def render(self) -> Panel:
        """Render the status bar."""
        status_icons = []
        
        if self.listening:
            status_icons.append("[cyan]🎤 Listening[/]")
        if self.processing:
            status_icons.append("[yellow]⚙️  Processing[/]")
        if self.speaking:
            status_icons.append("[green]🔊 Speaking[/]")
        
        if not status_icons:
            status_icons.append("[dim]⏸️  Idle[/]")
        
        content = " | ".join(status_icons) + f"\n{self.status}"
        return Panel(content, title="Status", border_style="magenta")


class ConversationHistory(ScrollableContainer):
    """Widget to display conversation history with auto-scroll."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[ConversationMessage] = []
    
    def add_message(self, speaker: str, message: str) -> None:
        """Add a message to the conversation history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg_widget = ConversationMessage(speaker, message, timestamp)
        self.mount(msg_widget)
        self.messages.append(msg_widget)
        
        # Auto-scroll to bottom
        self.scroll_end(animate=False)


class VoiceMetrics(Static):
    """Widget to display voice quality metrics."""
    
    latency = reactive(0.0)
    confidence = reactive(0.0)
    duration = reactive(0.0)
    
    def render(self) -> Panel:
        """Render the metrics."""
        content = f"""
[bold]Voice Quality Metrics[/]

Latency:    {self.latency:.2f}s
Confidence: {int(self.confidence * 100)}%
Duration:   {self.duration:.1f}s
        """.strip()
        
        return Panel(content, title="Metrics", border_style="yellow")


class InputArea(Container):
    """Widget for text input with send button."""
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Horizontal(id="input-container"):
            yield TextArea(id="text-input")
            yield Button("Send", id="send-button", variant="primary")


class VoiceTUI(App):
    """Textual TUI application for voice interface."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    Header {
        dock: top;
        height: 3;
    }
    
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    #chat-panel {
        width: 2fr;
        height: 100%;
        layout: vertical;
    }
    
    #conversation {
        height: 1fr;
        border: solid cyan;
        padding: 1;
    }
    
    #input-area {
        height: auto;
        border: solid green;
        padding: 1;
    }
    
    #input-container {
        height: auto;
        align: center middle;
    }
    
    #text-input {
        width: 1fr;
        height: 5;
        border: solid blue;
    }
    
    #send-button {
        width: auto;
        height: 5;
        margin-left: 1;
    }
    
    #metrics-panel {
        width: 1fr;
        height: 100%;
        layout: vertical;
        border: solid magenta;
        padding: 1;
    }
    
    AudioMeter, VoiceMetrics, StatusBar {
        height: auto;
        margin: 1 0;
    }
    
    ConversationMessage {
        margin: 1 0;
    }
    
    Footer {
        dock: bottom;
        height: 1;
    }
    """
    
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "clear", "Clear"),
        ("ctrl+l", "toggle_listen", "Listen"),
    ]
    
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.audio_meter: Optional[AudioMeter] = None
        self.status_bar: Optional[StatusBar] = None
        self.conversation: Optional[ConversationHistory] = None
        self.metrics: Optional[VoiceMetrics] = None
        self.text_input: Optional[TextArea] = None
        self.running = True
        self.listening_mode = True  # Toggle between voice and text input
        
        # Audio monitoring
        self.audio_queue = asyncio.Queue()
        self.sample_rate = 16000
        self.monitoring_task: Optional[asyncio.Task] = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        
        # Main container with horizontal layout
        with Container(id="main-container"):
            # Chat panel (2/3 width)
            with Container(id="chat-panel"):
                self.conversation = ConversationHistory(id="conversation")
                yield self.conversation
                
                # Input area
                with Container(id="input-area"):
                    with Horizontal(id="input-container"):
                        self.text_input = TextArea(id="text-input")
                        yield self.text_input
                        yield Button(
                            "Send", id="send-button", variant="primary"
                        )
            
            # Metrics panel (1/3 width)
            with Vertical(id="metrics-panel"):
                self.audio_meter = AudioMeter()
                yield self.audio_meter
                
                self.metrics = VoiceMetrics()
                yield self.metrics
                
                self.status_bar = StatusBar()
                yield self.status_bar
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "Intuit Voice Assistant"
        self.sub_title = "Ctrl+Q: Quit | Ctrl+C: Clear | Ctrl+L: Toggle Listen"
        
        if self.status_bar:
            self.status_bar.status = "Ready - Voice mode active"
        
        # Focus on text input
        if self.text_input:
            self.text_input.focus()
        
        # Start audio monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_audio())
        
        # Start voice interaction loop
        asyncio.create_task(self._voice_loop())
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-button":
            self._send_text_message()
    
    def _send_text_message(self) -> None:
        """Send text message from input area."""
        if not self.text_input:
            return
        
        text = self.text_input.text.strip()
        if not text:
            return
        
        # Clear input
        self.text_input.clear()
        
        # Process the message
        asyncio.create_task(self._process_text_input(text))
    
    async def _process_text_input(self, text: str) -> None:
        """Process text input from the user."""
        try:
            # Add user message
            if self.conversation:
                self.conversation.add_message("You", text)
            
            # Handle exit command
            if text.lower() in ["exit", "quit", "stop"]:
                if self.conversation:
                    self.conversation.add_message("System", "Goodbye!")
                await asyncio.sleep(1)
                self.exit()
                return
            
            # Update status
            if self.status_bar:
                self.status_bar.processing = True
                self.status_bar.status = "Processing your request..."
            
            # Process with agent
            start_time = time.time()
            response = await self.agent.run(text)
            process_duration = time.time() - start_time
            
            # Update metrics
            if self.metrics:
                self.metrics.latency = process_duration
            
            if self.status_bar:
                self.status_bar.processing = False
            
            # Add response
            if self.conversation:
                self.conversation.add_message("Assistant", response)
            
            # Speak response if in listening mode
            if self.listening_mode:
                from ..ui.voice import VoiceInterface
                voice_interface = VoiceInterface(self.agent)
                
                if self.status_bar:
                    self.status_bar.speaking = True
                    self.status_bar.status = "Speaking response..."
                
                await voice_interface._speak(response)
                
                if self.status_bar:
                    self.status_bar.speaking = False
                    self.status_bar.status = "Ready - Voice mode active"
            else:
                if self.status_bar:
                    self.status_bar.status = "Ready - Text mode active"
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.conversation:
                self.conversation.add_message("System", error_msg)
            if self.status_bar:
                self.status_bar.status = error_msg
                self.status_bar.processing = False
    
    async def _monitor_audio(self) -> None:
        """Monitor audio input levels."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                self.log(f"Audio status: {status}")
            # Calculate RMS level
            level = np.sqrt(np.mean(indata**2))
            # Use call_soon_threadsafe for thread-safe queue operations
            try:
                self.audio_queue.put_nowait(level)
            except asyncio.QueueFull:
                pass  # Skip if queue is full
        
        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024
            ):
                while self.running:
                    try:
                        level = await asyncio.wait_for(
                            self.audio_queue.get(),
                            timeout=0.1
                        )
                        if self.audio_meter:
                            # Scale level (RMS is typically 0-1)
                            scaled_level = float(level) * 20
                            self.audio_meter.update_level(scaled_level)
                    except asyncio.TimeoutError:
                        # Decay the meter
                        if self.audio_meter:
                            current = self.audio_meter.level
                            self.audio_meter.update_level(current * 0.85)
                    await asyncio.sleep(0.03)  # ~30 FPS update rate
        except Exception as e:
            self.log(f"Audio monitoring error: {e}")
    
    async def _voice_loop(self) -> None:
        """Main voice interaction loop."""
        from ..ui.voice import VoiceInterface
        
        try:
            voice_interface = VoiceInterface(self.agent)
            
            if self.conversation:
                self.conversation.add_message(
                    "System",
                    "Voice assistant started. Speak to interact!"
                )
            
            while self.running:
                try:
                    # Only listen if not currently speaking
                    if self.status_bar and self.status_bar.speaking:
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Update status
                    if self.status_bar:
                        self.status_bar.listening = True
                        self.status_bar.status = "Listening for your voice..."
                    
                    # Listen for input
                    start_time = time.time()
                    query = await voice_interface._listen()
                    listen_duration = time.time() - start_time
                    
                    if self.status_bar:
                        self.status_bar.listening = False
                    
                    if not query:
                        continue
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.duration = listen_duration
                        self.metrics.confidence = 0.85  # Placeholder
                    
                    # Add user message to conversation
                    if self.conversation:
                        self.conversation.add_message("You", query)
                    
                    # Handle exit command
                    if query.strip().lower() in ["exit", "quit", "stop"]:
                        if self.conversation:
                            self.conversation.add_message(
                                "System",
                                "Goodbye!"
                            )
                        await asyncio.sleep(1)
                        self.exit()
                        break
                    
                    # Process query
                    if self.status_bar:
                        self.status_bar.processing = True
                        self.status_bar.status = "Processing your request..."
                    
                    start_time = time.time()
                    response = await self.agent.run(query)
                    process_duration = time.time() - start_time
                    
                    if self.status_bar:
                        self.status_bar.processing = False
                    
                    # Update latency metric
                    if self.metrics:
                        self.metrics.latency = process_duration
                    
                    # Add assistant response to conversation
                    if self.conversation:
                        self.conversation.add_message("Assistant", response)
                    
                    # Speak response (mark as speaking to prevent listening)
                    if self.status_bar:
                        self.status_bar.speaking = True
                        self.status_bar.status = "Speaking response..."
                    
                    await voice_interface._speak(response)
                    
                    # Add a small delay after speaking to avoid picking up echo
                    await asyncio.sleep(0.5)
                    
                    if self.status_bar:
                        self.status_bar.speaking = False
                        self.status_bar.status = "Ready to listen..."
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    if self.conversation:
                        self.conversation.add_message("System", error_msg)
                    if self.status_bar:
                        self.status_bar.status = error_msg
                        self.status_bar.listening = False
                        self.status_bar.processing = False
                        self.status_bar.speaking = False
                    await asyncio.sleep(2)
        
        except Exception as e:
            self.log(f"Voice loop error: {e}")
            if self.conversation:
                self.conversation.add_message(
                    "System",
                    f"Fatal error: {str(e)}"
                )
    
    def action_clear(self) -> None:
        """Clear conversation history."""
        if self.conversation:
            for msg in self.conversation.messages:
                msg.remove()
            self.conversation.messages.clear()
            self.conversation.add_message(
                "System",
                "Conversation history cleared."
            )
    
    def action_toggle_listen(self) -> None:
        """Toggle between voice and text-only mode."""
        self.listening_mode = not self.listening_mode
        mode = "Voice" if self.listening_mode else "Text"
        if self.status_bar:
            self.status_bar.status = f"Switched to {mode} mode"
        if self.conversation:
            self.conversation.add_message(
                "System",
                f"Switched to {mode} mode"
            )
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.running = False
        self.exit()


async def run_voice_tui(agent: Agent) -> None:
    """Run the voice TUI interface."""
    try:
        # Start reminder service if initialized
        if agent.reminder_service:
            agent.reminder_service.start()
        
        app = VoiceTUI(agent)
        await app.run_async()
    
    finally:
        # Stop reminder service if it was started
        if agent.reminder_service:
            agent.reminder_service.stop()
        
        # Properly shut down MCP clients
        await agent.shutdown_mcp_clients()