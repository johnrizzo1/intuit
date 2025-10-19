"""
Voice interface for Intuit.
"""
import asyncio
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import logging
from typing import Optional
import sounddevice as sd
import numpy as np
import queue

from ..agent import Agent
from ..logging_config import PipelineLogger, get_pipeline_logger

logger = logging.getLogger(__name__)


class VoiceInterface:
    """Voice interface for Intuit."""
    
    def __init__(
        self,
        agent: Agent,
        pipeline_logger: Optional[PipelineLogger] = None
    ):
        """Initialize the voice interface."""
        self.agent = agent
        self.recognizer = sr.Recognizer()
        self.running = True
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.pipeline_logger = pipeline_logger or get_pipeline_logger()
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def _listen(self) -> Optional[str]:
        """Listen for user input."""
        session_id = self.pipeline_logger.start_session()
        self.pipeline_logger.log_stt_start(session_id)
        logger.info("Listening for voice input...")
        
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
            logger.warning("No audio recorded")
            self.pipeline_logger.log_stt_error("No audio data", session_id)
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
            logger.info(f"Transcribed: {text}")
            self.pipeline_logger.log_stt_complete(text, session_id)
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            self.pipeline_logger.log_stt_error(
                "Speech not recognized", session_id
            )
            return None
        except sr.RequestError as e:
            error_msg = f"Speech recognition service error: {e}"
            logger.error(error_msg)
            self.pipeline_logger.log_stt_error(error_msg, session_id)
            return None
    
    async def _speak(
        self, text: str, session_id: Optional[str] = None
    ) -> None:
        """Convert text to speech and play it."""
        if session_id:
            self.pipeline_logger.log_tts_start(text, session_id)
        logger.info(f"Speaking: {text[:50]}...")
        
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp3'
            ) as fp:
                temp_file = fp.name
            
            # Generate TTS audio file
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)
            
            # Play audio asynchronously (non-blocking)
            process = await asyncio.create_subprocess_exec(
                'afplay', temp_file,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if session_id:
                self.pipeline_logger.log_tts_complete(session_id)
            logger.info("Speech playback complete")
        except Exception as e:
            error_msg = f"TTS error: {e}"
            logger.error(error_msg)
            if session_id:
                self.pipeline_logger.log_tts_error(error_msg, session_id)
    
    async def run(self) -> None:
        """Run the voice interface."""
        logger.info("Starting voice interface")
        try:
            while self.running:
                # Listen for input
                query = await self._listen()
                if not query:
                    continue
                
                session_id = self.pipeline_logger.current_session_id
                
                # Handle exit command
                if query.strip().lower() in ["exit", "quit", "stop"]:
                    logger.info("Exit command received")
                    break
                
                # Process query
                try:
                    self.pipeline_logger.log_agent_start(query, session_id)
                    logger.info(f"Processing query: {query}")
                    response = await self.agent.run(query)
                    self.pipeline_logger.log_agent_complete(
                        response, session_id
                    )
                    
                    # Don't call self._speak if agent has voice output
                    # This prevents the response from being spoken twice
                    if not self.agent.voice:
                        await self._speak(response, session_id)
                    
                    self.pipeline_logger.end_session(session_id)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(error_msg)
                    self.pipeline_logger.log_agent_error(error_msg, session_id)
                    await self._speak(error_msg, session_id)
                    self.pipeline_logger.end_session(session_id)
                    break  # Exit after error
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
        finally:
            self.running = False
            logger.info("Voice interface shutting down")
            await self._speak("Goodbye!")


async def run_voice(
    agent: Agent,
    pipeline_logger: Optional[PipelineLogger] = None
) -> None:
    """Run the voice interface."""
    try:
        # Start reminder service if initialized (inside the event loop)
        if agent.reminder_service:
            agent.reminder_service.start()
            
        interface = VoiceInterface(agent, pipeline_logger=pipeline_logger)
        await interface.run()
    finally:
        # Stop reminder service if it was started
        if agent.reminder_service:
            agent.reminder_service.stop()
            
        # Properly shut down MCP clients
        await agent.shutdown_mcp_clients()