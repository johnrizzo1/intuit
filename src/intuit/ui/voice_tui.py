"""
Voice TUI interface for Intuit using Textual.

This module provides a rich terminal user interface for voice interactions,
displaying conversation history, audio metrics, and real-time status.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import sounddevice as sd
from textual.app import App, ComposeResult
from textual.binding import Binding
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
from ..dictation import DictationMode, DictationState
from ..logging_config import PipelineLogger, get_pipeline_logger

logger = logging.getLogger(__name__)


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
    dictation_mode = reactive(False)
    dictation_state = reactive("idle")
    privacy_mode = reactive(False)
    
    def render(self) -> Panel:
        """Render the status bar."""
        status_icons = []
        
        # Privacy mode takes precedence
        if self.privacy_mode:
            status_icons.append("[red]🔒 Privacy Mode[/]")
        elif self.dictation_mode:
            # Dictation mode indicators
            if self.dictation_state == "listening":
                status_icons.append("[green]📝 Dictating[/]")
            elif self.dictation_state == "paused":
                status_icons.append("[yellow]⏸️  Dictation Paused[/]")
            elif self.dictation_state == "ended":
                status_icons.append("[red]🛑 Dictation Ended[/]")
        else:
            # Normal mode indicators
            if self.listening:
                status_icons.append("[cyan]🎤 Listening[/]")
            if self.processing:
                status_icons.append("[yellow]⚙️  Processing[/]")
            if self.speaking:
                status_icons.append("[green]🔊 Speaking[/]")
        
        if not status_icons:
            status_icons.append("[dim]⏸️  Idle[/]")
        
        content = " | ".join(status_icons) + f"\n{self.status}"
        border_color = "red" if self.privacy_mode else "magenta"
        return Panel(content, title="Status", border_style=border_color)


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


class DictationDisplay(ScrollableContainer):
    """Widget to display live dictation transcription."""
    
    transcription = reactive("")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_widget: Optional[Static] = None
    
    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.text_widget = Static("", id="dictation-text")
        self.mount(self.text_widget)
    
    def update_transcription(self, text: str) -> None:
        """Update the displayed transcription."""
        self.transcription = text
        if self.text_widget:
            self.text_widget.update(text)
        # Auto-scroll to bottom
        self.scroll_end(animate=False)
    
    def clear_transcription(self) -> None:
        """Clear the transcription display."""
        self.update_transcription("")


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
    
    #dictation-display {
        height: 1fr;
        border: solid green;
        padding: 1;
        background: $surface;
    }
    
    #dictation-text {
        color: $text;
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
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+c", "clear_history", "Clear History", show=True),
        Binding("ctrl+l", "toggle_listen", "Privacy Mode", show=True),
        Binding("ctrl+d", "toggle_dictation", "Dictation", show=True),
    ]
    
    def __init__(
        self,
        agent: Agent,
        enable_dictation: bool = False,
        pipeline_logger: Optional[PipelineLogger] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.agent = agent
        self.audio_meter: Optional[AudioMeter] = None
        self.status_bar: Optional[StatusBar] = None
        self.conversation: Optional[ConversationHistory] = None
        self.metrics: Optional[VoiceMetrics] = None
        self.text_input: Optional[TextArea] = None
        self.dictation_display: Optional[DictationDisplay] = None
        self.running = True
        self.listening_mode = True  # Toggle between voice and text input
        self.voice_loop_active = True  # Control voice loop
        self.pipeline_logger = pipeline_logger or get_pipeline_logger()
        
        # Dictation mode
        self.dictation_enabled = enable_dictation
        self.dictation_mode: Optional[DictationMode] = None
        self.dictation_task: Optional[asyncio.Task] = None
        self.in_dictation = False
        
        # Privacy mode
        self.privacy_mode = False
        
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
                # Always create both displays, show/hide based on mode
                self.conversation = ConversationHistory(
                    id="conversation"
                )
                self.conversation.display = not self.dictation_enabled
                yield self.conversation
                
                self.dictation_display = DictationDisplay(
                    id="dictation-display"
                )
                self.dictation_display.display = self.dictation_enabled
                yield self.dictation_display
                
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
        bindings = (
            "Ctrl+Q: Quit | Ctrl+C: Clear History | "
            "Ctrl+L: Privacy Mode | Ctrl+D: Dictation"
        )
        self.sub_title = bindings
        
        if self.status_bar:
            mode = "Dictation" if self.dictation_enabled else "Voice"
            self.status_bar.status = f"Ready - {mode} mode available"
        
        # Focus on text input (if not in dictation mode)
        if self.text_input:
            self.text_input.focus()
        
        # Start audio monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_audio())
        
        # Start voice interaction loop (unless starting in dictation mode)
        if not self.dictation_enabled:
            asyncio.create_task(self._voice_loop())
        else:
            # Start dictation mode automatically if enabled
            asyncio.create_task(self._start_dictation())
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-button":
            self._send_text_message()
    
    async def on_key(self, event) -> None:
        """Handle key press events."""
        # Check if we're in the text input and Enter is pressed
        if (self.text_input and
                self.text_input.has_focus and
                event.key == "enter" and
                not event.shift):
            # Prevent default behavior
            event.prevent_default()
            # Send the message
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
            # Temporarily pause voice loop to prevent double processing
            self.voice_loop_active = False
            
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
                
                # Add delay after speaking
                await asyncio.sleep(0.5)
                
                if self.status_bar:
                    self.status_bar.speaking = False
                    self.status_bar.status = "Ready - Voice mode active"
            else:
                if self.status_bar:
                    self.status_bar.status = "Ready - Text mode active"
            
            # Resume voice loop
            self.voice_loop_active = True
        
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
            voice_interface = VoiceInterface(
                self.agent,
                pipeline_logger=self.pipeline_logger
            )
            
            if self.conversation:
                self.conversation.add_message(
                    "System",
                    "Voice assistant started. Speak to interact!"
                )
            
            while self.running:
                try:
                    # Skip if voice loop is paused (text input active)
                    if not self.voice_loop_active:
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Skip if in text-only mode (not listening for voice)
                    if not self.listening_mode:
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Only listen if not currently speaking
                    if self.status_bar and self.status_bar.speaking:
                        await asyncio.sleep(0.5)
                        continue
                    
                    # Clear any residual audio from the queue before listening
                    # This prevents the assistant's speech from being picked up
                    while not voice_interface.audio_queue.empty():
                        try:
                            voice_interface.audio_queue.get_nowait()
                        except Exception:
                            break
                    
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
                    
                    session_id = (
                        voice_interface.pipeline_logger.current_session_id
                    )
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.duration = listen_duration
                        self.metrics.confidence = 0.85  # Placeholder
                    
                    # Add user message to conversation
                    if self.conversation:
                        self.conversation.add_message("You", query)
                    
                    # Handle exit command
                    if query.strip().lower() in [
                        "exit", "quit", "stop", "goodbye", "bye"
                    ]:
                        goodbye_msg = "Goodbye! Have a great day!"
                        
                        # Add to conversation
                        if self.conversation:
                            self.conversation.add_message(
                                "Assistant",
                                goodbye_msg
                            )
                        
                        # Speak goodbye
                        if self.status_bar:
                            self.status_bar.speaking = True
                            self.status_bar.status = "Speaking..."
                        
                        await voice_interface._speak(goodbye_msg, session_id)
                        
                        if self.status_bar:
                            self.status_bar.speaking = False
                        
                        # Exit after speaking
                        await asyncio.sleep(0.5)
                        self.exit()
                        break
                    
                    # Check for privacy mode commands
                    import re
                    if re.search(r"(enable|start|activate)\s+privacy\s+mode", query.lower()):
                        self.privacy_mode = True
                        if self.status_bar:
                            self.status_bar.privacy_mode = True
                            self.status_bar.status = "Privacy mode enabled - listening only"
                        if self.conversation:
                            self.conversation.add_message(
                                "System",
                                "🔒 Privacy mode enabled - I'm listening but won't respond"
                            )
                        continue
                    
                    if re.search(r"(disable|stop|deactivate)\s+privacy\s+mode", query.lower()):
                        self.privacy_mode = False
                        if self.status_bar:
                            self.status_bar.privacy_mode = False
                            self.status_bar.status = "Privacy mode disabled"
                        if self.conversation:
                            self.conversation.add_message(
                                "System",
                                "🔓 Privacy mode disabled - I'm back to normal"
                            )
                        continue
                    
                    # If in privacy mode, just acknowledge but don't process
                    if self.privacy_mode:
                        if self.conversation:
                            self.conversation.add_message(
                                "System",
                                "🔒 [Privacy mode - not responding]"
                            )
                        continue
                    
                    # Check for dictation start command
                    dictation_pattern = (
                        r"(start|take|begin)\s+(a\s+)?dictation"
                    )
                    if re.search(dictation_pattern, query.lower()):
                        # Start dictation mode instead of sending to agent
                        if self.conversation:
                            self.conversation.add_message(
                                "Assistant",
                                "Starting dictation mode..."
                            )
                        await self._start_dictation()
                        continue
                    
                    # Process query
                    if self.status_bar:
                        self.status_bar.processing = True
                        self.status_bar.status = "Processing your request..."
                    
                    self.pipeline_logger.log_agent_start(query, session_id)
                    start_time = time.time()
                    response = await self.agent.run(query)
                    process_duration = time.time() - start_time
                    self.pipeline_logger.log_agent_complete(
                        response, session_id
                    )
                    
                    if self.status_bar:
                        self.status_bar.processing = False
                    
                    # Update latency metric
                    if self.metrics:
                        self.metrics.latency = process_duration
                    
                    # Add assistant response to conversation
                    if self.conversation:
                        self.conversation.add_message("Assistant", response)
                    
                    # Check if response is long and needs summarization
                    response_to_speak = response
                    full_response = None
                    
                    # Count sentences (rough estimate)
                    sentence_count = (
                        response.count('.') +
                        response.count('!') +
                        response.count('?')
                    )
                    word_count = len(response.split())
                    
                    # If response is longer than 30 words AND has
                    # multiple sentences (truly long)
                    # This prevents short, single-sentence responses
                    # from being summarized
                    if word_count > 30 and sentence_count > 1:
                        full_response = response
                        
                        # Ask agent to create a one-sentence summary
                        if self.status_bar:
                            self.status_bar.status = "Creating summary..."
                        
                        summary_prompt = (
                            f"Summarize the following response in "
                            f"exactly one short sentence "
                            f"(10 words or less):\n\n{response}"
                        )
                        summary = await self.agent.run(summary_prompt)
                        response_to_speak = summary.strip()
                        
                        logger.info(
                            f"Long response detected. "
                            f"Summary: {response_to_speak}"
                        )
                    
                    # Speak response (mark as speaking to prevent listening)
                    if self.status_bar:
                        self.status_bar.speaking = True
                        self.status_bar.status = "Speaking response..."
                    
                    await voice_interface._speak(response_to_speak, session_id)
                    
                    # If we summarized, ask if they want the full response
                    if full_response:
                        await asyncio.sleep(0.5)
                        
                        prompt_text = (
                            "Would you like me to read the full response?"
                        )
                        await voice_interface._speak(prompt_text, session_id)
                        
                        # Listen for yes/no response
                        if self.status_bar:
                            self.status_bar.speaking = False
                            self.status_bar.listening = True
                            self.status_bar.status = "Listening for yes/no..."
                        
                        user_response_session = (
                            voice_interface.pipeline_logger.start_session()
                        )
                        user_response = await voice_interface._listen()
                        
                        if user_response:
                            user_response_lower = user_response.lower().strip()
                            yes_words = [
                                'yes', 'yeah', 'yep',
                                'sure', 'please', 'read'
                            ]
                            if any(word in user_response_lower
                                   for word in yes_words):
                                if self.status_bar:
                                    self.status_bar.speaking = True
                                    self.status_bar.status = (
                                        "Reading full response..."
                                    )
                                
                                await voice_interface._speak(
                                    full_response,
                                    user_response_session
                                )
                    
                    # End session
                    self.pipeline_logger.end_session(session_id)
                    
                    # Add a small delay after speaking to avoid echo
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
    
    def action_clear_history(self) -> None:
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
    
    def action_toggle_dictation(self) -> None:
        """Toggle dictation mode on/off."""
        if not self.in_dictation:
            # Start dictation - use run_worker for async tasks in Textual
            self.run_worker(self._start_dictation(), exclusive=True)
        else:
            # End dictation
            if self.dictation_mode:
                self.dictation_mode.stop()
    
    async def _start_dictation(self) -> None:
        """Start dictation mode."""
        try:
            self.in_dictation = True
            self.voice_loop_active = False  # Pause normal voice loop
            
            # Show dictation display, hide conversation
            if self.conversation:
                self.conversation.display = False
            if self.dictation_display:
                self.dictation_display.display = True
            if self.text_input:
                self.text_input.display = False
            
            # Initialize dictation mode
            self.dictation_mode = DictationMode(
                stt_provider=None,  # Will use default from config
                silence_threshold=30.0,
                sample_rate=self.sample_rate,
                on_transcription=self._on_dictation_text,
                on_state_change=self._on_dictation_state_change,
            )
            
            # Update UI
            if self.status_bar:
                self.status_bar.dictation_mode = True
                self.status_bar.status = "Dictation mode started"
            
            # Start dictation
            self.dictation_mode._set_state(DictationState.LISTENING)
            self.dictation_task = asyncio.create_task(
                self.dictation_mode.run_dictation_loop()
            )
            
            # Wait for dictation to end
            await self.dictation_task
            
            # Save transcription
            filepath = await self.dictation_mode.save_transcription()
            
            # Update UI - switch back to conversation
            if self.dictation_display:
                self.dictation_display.display = False
            if self.conversation:
                self.conversation.display = True
            if self.text_input:
                self.text_input.display = True
            
            if self.status_bar:
                self.status_bar.dictation_mode = False
                self.status_bar.status = f"Dictation saved to: {filepath}"
            
            if self.conversation:
                self.conversation.add_message(
                    "System",
                    f"Dictation saved to: {filepath}"
                )
            
            # Clean up
            self.in_dictation = False
            self.voice_loop_active = True
            
        except Exception as e:
            self.log(f"Error in dictation mode: {e}")
            
            # Restore UI on error
            if self.dictation_display:
                self.dictation_display.display = False
            if self.conversation:
                self.conversation.display = True
            if self.text_input:
                self.text_input.display = True
            
            if self.status_bar:
                self.status_bar.status = f"Dictation error: {str(e)}"
            self.in_dictation = False
            self.voice_loop_active = True
    
    def _on_dictation_text(self, text: str) -> None:
        """Callback for new dictation text."""
        logger.info(f"Dictation callback received: {text}")
        if self.dictation_display and self.dictation_mode:
            # Append new text to display
            current = self.dictation_mode.get_full_transcription()
            logger.info(f"Updating display with: {current}")
            self.dictation_display.update_transcription(current)
        else:
            logger.warning(
                f"Display not ready: display={self.dictation_display}, "
                f"mode={self.dictation_mode}"
            )
    
    def _on_dictation_state_change(self, state: DictationState) -> None:
        """Callback for dictation state changes."""
        if self.status_bar:
            self.status_bar.dictation_state = state.value
            
            if state == DictationState.LISTENING:
                self.status_bar.status = "Listening for dictation..."
            elif state == DictationState.PAUSED:
                self.status_bar.status = "Dictation paused"
            elif state == DictationState.ENDED:
                self.status_bar.status = "Dictation ended"
    
    def action_quit(self) -> None:
        """Quit the application."""
        # Stop dictation if active
        if self.in_dictation and self.dictation_mode:
            self.dictation_mode.stop()
        
        self.running = False
        self.exit()


async def run_voice_tui(
    agent: Agent,
    enable_dictation: bool = False,
    pipeline_logger: Optional[PipelineLogger] = None
) -> None:
    """Run the voice TUI interface."""
    try:
        # Start reminder service if initialized
        if agent.reminder_service:
            agent.reminder_service.start()
        
        app = VoiceTUI(
            agent,
            enable_dictation=enable_dictation,
            pipeline_logger=pipeline_logger
        )
        await app.run_async()
    
    finally:
        # Stop reminder service if it was started
        if agent.reminder_service:
            agent.reminder_service.stop()
        
        # Properly shut down MCP clients
        await agent.shutdown_mcp_clients()