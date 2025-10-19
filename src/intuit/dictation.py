"""
Dictation mode for Intuit voice assistant.

This module provides continuous transcription with command detection,
allowing users to dictate text with sporadic periods of silence.
"""
import asyncio
import logging
import queue
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List
import speech_recognition as sr
import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)


class DictationState(Enum):
    """States for dictation mode."""
    IDLE = "idle"
    LISTENING = "listening"
    PAUSED = "paused"
    ENDED = "ended"


class DictationMode:
    """
    Manages dictation mode with continuous transcription and command detection.
    
    Features:
    - Continuous listening with silence detection
    - Wake word detection ("Hey Intuit")
    - Dictation commands (Pause/Resume/End)
    - Real-time transcription display
    - Automatic file saving with timestamps
    """
    
    # Command patterns
    WAKE_WORD = r"hey\s+intuit"
    PAUSE_COMMAND = r"pause\s+dictation"
    RESUME_COMMAND = r"resume\s+dictation"
    END_COMMAND = r"end\s+dictation"
    START_COMMAND = r"(start|take|begin)\s+(a\s+)?dictation"
    
    def __init__(
        self,
        recognizer: sr.Recognizer,
        silence_threshold: float = 30.0,
        sample_rate: int = 16000,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[DictationState], None]] = None,
    ):
        """
        Initialize dictation mode.
        
        Args:
            recognizer: Speech recognition instance
            silence_threshold: Seconds of silence before pause (default: 30)
            sample_rate: Audio sample rate
            on_transcription: Callback for new transcription text
            on_state_change: Callback for state changes
        """
        self.recognizer = recognizer
        # Configure recognizer for dictation (longer pauses allowed)
        self.recognizer.pause_threshold = 2.0  # 2 seconds of silence
        self.recognizer.energy_threshold = 300  # Lower for quieter speech
        self.recognizer.dynamic_energy_threshold = True
        
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.on_transcription = on_transcription
        self.on_state_change = on_state_change
        
        self.state = DictationState.IDLE
        self.transcription_buffer: List[str] = []
        self.last_audio_time = 0.0
        self.running = False
        
        # Track silence for newline insertion
        self.silence_newline_added = False
        
        # Audio monitoring - use thread-safe queue for callback
        self.audio_queue = queue.Queue()
        self.channels = 1
        self.dtype = np.float32
    
    def _set_state(self, new_state: DictationState) -> None:
        """Update state and notify callback."""
        if self.state != new_state:
            self.state = new_state
            logger.info(f"Dictation state changed to: {new_state.value}")
            if self.on_state_change:
                self.on_state_change(new_state)
    
    def _is_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        return bool(re.search(self.WAKE_WORD, text.lower()))
    
    def _is_pause_command(self, text: str) -> bool:
        """Check if text contains pause command."""
        return bool(re.search(self.PAUSE_COMMAND, text.lower()))
    
    def _is_resume_command(self, text: str) -> bool:
        """Check if text contains resume command."""
        return bool(re.search(self.RESUME_COMMAND, text.lower()))
    
    def _is_end_command(self, text: str) -> bool:
        """Check if text contains end command."""
        return bool(re.search(self.END_COMMAND, text.lower()))
    
    def _is_start_command(self, text: str) -> bool:
        """Check if text contains start dictation command."""
        return bool(re.search(self.START_COMMAND, text.lower()))
    
    def _remove_commands_from_text(self, text: str) -> str:
        """Remove command phrases from transcription text."""
        # Remove wake word
        text = re.sub(self.WAKE_WORD, "", text, flags=re.IGNORECASE)
        # Remove dictation commands
        text = re.sub(self.PAUSE_COMMAND, "", text, flags=re.IGNORECASE)
        text = re.sub(self.RESUME_COMMAND, "", text, flags=re.IGNORECASE)
        text = re.sub(self.END_COMMAND, "", text, flags=re.IGNORECASE)
        text = re.sub(self.START_COMMAND, "", text, flags=re.IGNORECASE)
        # Clean up extra whitespace
        text = " ".join(text.split())
        return text.strip()
    
    def _add_transcription(self, text: str) -> None:
        """Add text to transcription buffer and notify callback."""
        if text:
            self.transcription_buffer.append(text)
            # Reset silence newline flag when new speech is added
            self.silence_newline_added = False
            if self.on_transcription:
                self.on_transcription(text)
    
    def _add_newline(self) -> None:
        """Add a newline to transcription buffer."""
        self.transcription_buffer.append("\n")
        if self.on_transcription:
            self.on_transcription("\n")
    
    def get_full_transcription(self) -> str:
        """Get the complete transcription."""
        return " ".join(self.transcription_buffer)
    
    def clear_transcription(self) -> None:
        """Clear the transcription buffer."""
        self.transcription_buffer.clear()
    
    async def save_transcription(
        self, directory: Optional[Path] = None
    ) -> Path:
        """
        Save transcription to a file with timestamp.
        
        Args:
            directory: Directory to save file (default: current directory)
            
        Returns:
            Path to saved file
        """
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dictation_{timestamp}.txt"
        filepath = directory / filename
        
        # Save transcription
        transcription = self.get_full_transcription()
        filepath.write_text(transcription, encoding="utf-8")
        
        logger.info(f"Saved dictation to: {filepath}")
        return filepath
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream (runs in audio thread)."""
        if status:
            logger.debug(f"Audio status: {status}")
        # Put audio data in thread-safe queue
        try:
            self.audio_queue.put_nowait(indata.copy())
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
    
    async def _listen_once(self) -> Optional[str]:
        """
        Listen for a single utterance and transcribe it.
        
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Record audio for a period
            audio_data = []
            recording_duration = 3.0  # Record for 3 seconds at a time
            
            logger.debug(f"Starting audio recording for {recording_duration}s")
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            ):
                start_time = asyncio.get_event_loop().time()
                while (asyncio.get_event_loop().time() - start_time <
                       recording_duration):
                    try:
                        # Get from thread-safe queue with timeout
                        data = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.audio_queue.get(timeout=0.1)
                        )
                        audio_data.append(data)
                    except queue.Empty:
                        continue
            
            logger.debug(f"Recorded {len(audio_data)} audio chunks")
            
            if not audio_data:
                return None
            
            # Convert to format expected by speech_recognition
            audio_data = np.concatenate(audio_data)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create AudioData object
            audio = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=self.sample_rate,
                sample_width=2
            )
            
            # Transcribe (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                self.recognizer.recognize_google,
                audio
            )
            self.last_audio_time = asyncio.get_event_loop().time()
            return text
            
        except sr.UnknownValueError:
            # No speech detected
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in listen_once: {e}")
            return None
    
    async def run_dictation_loop(self) -> None:
        """
        Main dictation loop that continuously listens and transcribes.
        
        This loop handles:
        - Continuous listening in LISTENING state
        - Command detection (pause, resume, end)
        - Wake word detection when paused
        - Silence detection
        """
        self.running = True
        self.last_audio_time = asyncio.get_event_loop().time()
        
        logger.info("Starting dictation loop")
        
        while self.running and self.state != DictationState.ENDED:
            try:
                current_time = asyncio.get_event_loop().time()
                
                # Check for silence timeout
                silence_duration = current_time - self.last_audio_time
                if (self.state == DictationState.LISTENING and
                        silence_duration > self.silence_threshold):
                    logger.info("Silence threshold reached")
                    # Don't auto-pause, just note it
                    # User must explicitly pause or end
                
                # Listen for speech
                text = await self._listen_once()
                
                if not text:
                    # Silence detected - add newline if we haven't already
                    if (self.state == DictationState.LISTENING and
                            not self.silence_newline_added and
                            len(self.transcription_buffer) > 0):
                        logger.debug("Adding newline after silence")
                        self._add_newline()
                        self.silence_newline_added = True
                    await asyncio.sleep(0.1)
                    continue
                
                logger.info(f"Dictation heard: {text}")
                
                # Check for commands based on current state
                if self.state == DictationState.LISTENING:
                    # In listening mode, check for pause or end commands
                    if self._is_pause_command(text):
                        logger.info("Pause command detected")
                        self._set_state(DictationState.PAUSED)
                        continue
                    elif self._is_end_command(text):
                        logger.info("End command detected")
                        self._set_state(DictationState.ENDED)
                        break
                    elif self._is_wake_word(text):
                        logger.info("Wake word detected")
                        # Wake word triggers command mode - don't transcribe
                        # Text after wake word is processed as command
                        clean_text = self._remove_commands_from_text(text)
                        if clean_text:
                            # If there's text after wake word, it's a command
                            # Don't add to transcription
                            pass
                        continue
                    else:
                        # Normal dictation - add to transcription
                        clean_text = self._remove_commands_from_text(text)
                        if clean_text:
                            logger.info(
                                f"Adding to transcription: {clean_text}"
                            )
                            self._add_transcription(clean_text)
                        else:
                            logger.info(
                                "No clean text after removing commands"
                            )
                
                elif self.state == DictationState.PAUSED:
                    # In paused mode, only listen for resume, end, or wake word
                    logger.info(f"In PAUSED state, heard: {text}")
                    if self._is_resume_command(text):
                        logger.info("Resume command detected")
                        self._set_state(DictationState.LISTENING)
                        continue
                    elif self._is_end_command(text):
                        logger.info("End command detected (from paused)")
                        self._set_state(DictationState.ENDED)
                        break
                    elif self._is_wake_word(text):
                        logger.info("Wake word detected (in paused mode)")
                        # Wake word in paused mode - process as command
                        continue
                    # Ignore other speech while paused
                    logger.info("Ignoring speech while paused")
                
            except Exception as e:
                logger.error(f"Error in dictation loop: {e}")
                await asyncio.sleep(1)
        
        logger.info("Dictation loop ended")
        self.running = False
    
    def stop(self) -> None:
        """Stop the dictation loop."""
        self.running = False
        self._set_state(DictationState.ENDED)