"""
Voice interface for Intuit.
"""
import asyncio
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from typing import Optional
from pathlib import Path
import sounddevice as sd
import numpy as np
import queue
import threading

from ..agent import Agent

class VoiceInterface:
    """Voice interface for Intuit."""
    
    def __init__(self, agent: Agent, stop_event=None):
        """
        Initialize the voice interface.
        
        Args:
            agent: The agent to use for processing queries
            stop_event: Optional event to signal when to stop the interface
        """
        self.agent = agent
        self.recognizer = sr.Recognizer()
        self.running = True
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.stop_event = stop_event
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def _listen(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for user input.
        
        Args:
            timeout: Time to listen for in seconds
            
        Returns:
            The recognized text or None if no speech was detected
        """
        print("Listening...")
        
        # Check if we should stop
        if self.stop_event and self.stop_event.is_set():
            self.running = False
            return None
        
        # Start recording
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        ):
            # Record for the specified timeout
            if timeout > 0:
                # Break the sleep into smaller chunks to check stop_event
                chunks = int(timeout / 0.1)
                for _ in range(chunks):
                    await asyncio.sleep(0.1)
                    if self.stop_event and self.stop_event.is_set():
                        self.running = False
                        return None
            else:
                await asyncio.sleep(5)  # Default to 5 seconds
        
        # Process recorded audio
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if not audio_data:
            print("No audio recorded")
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
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None
    
    async def _speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            os.system(f"afplay {fp.name}")  # macOS specific
            os.unlink(fp.name)
    
    async def run(self, timeout: float = 5.0) -> None:
        """
        Run the voice interface.
        
        Args:
            timeout: Time to listen for in seconds for each iteration
        """
        try:
            while self.running:
                # Check if we should stop
                if self.stop_event and self.stop_event.is_set():
                    break
                
                # Listen for input
                query = await self._listen(timeout)
                if not query:
                    # Check again if we should stop
                    if self.stop_event and self.stop_event.is_set():
                        break
                    continue
                
                # Handle exit command
                if query.strip().lower() in ["exit", "quit", "stop"]:
                    break
                
                # Process query
                try:
                    print("Processing...")
                    # Check if we should stop
                    if self.stop_event and self.stop_event.is_set():
                        break
                        
                    response = await self.agent.run(query)
                    
                    # Check if we should stop
                    if self.stop_event and self.stop_event.is_set():
                        break
                        
                    # Don't call self._speak if the agent already has voice output
                    # This prevents the response from being spoken twice
                    if not self.agent.voice:
                        await self._speak(response)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    
                    # Check if we should stop
                    if not self.stop_event or not self.stop_event.is_set():
                        await self._speak(error_msg)
                    break  # Exit after error
        except Exception as e:
            print(f"Fatal error: {str(e)}")
        finally:
            self.running = False
            await self._speak("Goodbye!")

async def run_voice(agent: Agent, timeout: float = 5.0, stop_event=None) -> None:
    """
    Run the voice interface.
    
    Args:
        agent: The agent to use for processing queries
        timeout: Time to listen for in seconds for each iteration
        stop_event: Optional event to signal when to stop the interface
    """
    try:
        # Start reminder service if initialized (inside the event loop)
        if agent.reminder_service:
            agent.reminder_service.start()
            
        interface = VoiceInterface(agent, stop_event)
        await interface.run(timeout)
    except asyncio.CancelledError:
        print("Voice interface cancelled")
    except KeyboardInterrupt:
        print("Voice interface interrupted")
    except Exception as e:
        print(f"Voice interface error: {e}")
    finally:
        # Stop reminder service if it was started
        if agent.reminder_service:
            agent.reminder_service.stop()
            
        # Properly shut down MCP clients to avoid "unhandled errors in a TaskGroup" message
        await agent.shutdown_mcp_clients()