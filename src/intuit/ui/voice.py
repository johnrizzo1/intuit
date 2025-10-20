"""
Voice interface for Intuit.
"""
import asyncio
import tempfile
import os
import logging
import wave
from typing import Optional
import sounddevice as sd
import numpy as np
import queue

from ..agent import Agent
from ..logging_config import PipelineLogger, get_pipeline_logger
from ..audio.stt_factory import STTFactory
from ..audio.tts_factory import TTSFactory
from ..config.audio_config import AudioConfig

logger = logging.getLogger(__name__)


def get_audio_duration(audio_file: str) -> Optional[float]:
    """Get the duration of an audio file in seconds."""
    try:
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return None


class VoiceInterface:
    """Voice interface for Intuit."""
    
    def __init__(
        self,
        agent: Agent,
        pipeline_logger: Optional[PipelineLogger] = None,
        audio_config: Optional[AudioConfig] = None
    ):
        """Initialize the voice interface.
        
        Args:
            agent: The agent to process queries
            pipeline_logger: Optional pipeline logger for tracking
            audio_config: Optional audio configuration
                (loads from env if not provided)
        """
        self.agent = agent
        self.running = True
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.pipeline_logger = pipeline_logger or get_pipeline_logger()
        
        # Initialize audio configuration and providers
        self.audio_config = audio_config or AudioConfig.from_env()
        self.stt_provider = STTFactory.create(self.audio_config.stt)
        self.tts_provider = TTSFactory.create(self.audio_config.tts)
        
        stt_name = self.audio_config.stt.provider
        tts_name = self.audio_config.tts.provider
        logger.info(f"Initialized STT provider: {stt_name}")
        logger.info(f"Initialized TTS provider: {tts_name}")
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def _listen(self) -> Optional[str]:
        """Listen for user input using configured STT provider."""
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
        
        # Convert to numpy array and normalize to int16
        audio_array = np.concatenate(audio_data)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        try:
            # Use STT provider to transcribe
            text = await self.stt_provider.transcribe(
                audio_array, self.sample_rate
            )
            
            if text:
                logger.info(f"Transcribed: {text}")
                self.pipeline_logger.log_stt_complete(text, session_id)
                return text
            else:
                logger.warning("No speech detected in audio")
                self.pipeline_logger.log_stt_error(
                    "No speech detected", session_id
                )
                return None
        except Exception as e:
            error_msg = f"Speech recognition error: {e}"
            logger.error(error_msg)
            self.pipeline_logger.log_stt_error(error_msg, session_id)
            return None
    
    async def _speak(
        self, text: str, session_id: Optional[str] = None
    ) -> None:
        """Convert text to speech and play it using TTS provider."""
        if session_id:
            self.pipeline_logger.log_tts_start(text, session_id)
        logger.info(f"Speaking: {text[:50]}...")
        
        try:
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='.wav'
            ) as fp:
                temp_file = fp.name
            
            # Synthesize speech using TTS provider
            await self.tts_provider.synthesize(
                text,
                output_path=temp_file
            )
            
            # Get audio duration before playing
            audio_duration = get_audio_duration(temp_file)
            
            # Play audio asynchronously (non-blocking)
            process = await asyncio.create_subprocess_exec(
                'afplay', temp_file,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
            
            if session_id:
                self.pipeline_logger.log_tts_complete(
                    session_id, audio_duration=audio_duration
                )
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