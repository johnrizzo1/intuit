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
    
    def __init__(self, agent: Agent):
        """Initialize the voice interface."""
        self.agent = agent
        self.recognizer = sr.Recognizer()
        self.running = True
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def _listen(self) -> Optional[str]:
        """Listen for user input."""
        print("Listening...")
        
        # Start recording
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        ):
            # Record for 5 seconds
            await asyncio.sleep(5)
        
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
    
    async def run(self) -> None:
        """Run the voice interface."""
        try:
            while self.running:
                # Listen for input
                query = await self._listen()
                if not query:
                    continue
                
                # Handle exit command
                if query.strip().lower() in ["exit", "quit", "stop"]:
                    break
                
                # Process query
                try:
                    print("Processing...")
                    response = await self.agent.run(query)
                    # Don't call self._speak if the agent already has voice output
                    # This prevents the response from being spoken twice
                    if not self.agent.voice:
                        await self._speak(response)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    await self._speak(error_msg)
                    break  # Exit after error
        except Exception as e:
            print(f"Fatal error: {str(e)}")
        finally:
            self.running = False
            await self._speak("Goodbye!")

async def run_voice(agent: Agent) -> None:
    """Run the voice interface."""
    try:
        # Start reminder service if initialized (inside the event loop)
        if agent.reminder_service:
            agent.reminder_service.start()
            
        interface = VoiceInterface(agent)
        await interface.run()
    finally:
        # Stop reminder service if it was started
        if agent.reminder_service:
            agent.reminder_service.stop()
            
        # Properly shut down MCP clients to avoid "unhandled errors in a TaskGroup" message
        await agent.shutdown_mcp_clients()