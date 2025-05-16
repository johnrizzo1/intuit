"""
Voice processing module for the GUI interface.

This module runs in a separate process and handles voice input/output,
communicating with the main GUI process via multiprocessing queues.
"""
import asyncio
import multiprocessing
import queue
import time
import logging
import tempfile
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import optional dependencies
try:
    import speech_recognition as sr
    import numpy as np
    import sounddevice as sd
    from gtts import gTTS
    VOICE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    VOICE_DEPENDENCIES_AVAILABLE = False
    print("Warning: Some voice dependencies are not installed. Voice functionality will be limited.")
    print("To install required packages, run: pip install SpeechRecognition gTTS sounddevice numpy soundfile")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Handles voice input/output in a separate process.
    
    This class manages speech recognition, text-to-speech conversion,
    and communication with the GUI process.
    """
    
    # Check if dependencies are available
    if not VOICE_DEPENDENCIES_AVAILABLE:
        raise ImportError(
            "Required voice dependencies are not installed. "
            "Please install them using: pip install SpeechRecognition gTTS sounddevice numpy soundfile"
        )
    
    def __init__(self, 
                 input_queue: multiprocessing.Queue, 
                 output_queue: multiprocessing.Queue,
                 language: str = 'en',
                 slow: bool = False):
        """
        Initialize the voice processor.
        
        Args:
            input_queue: Queue for receiving commands from the GUI process
            output_queue: Queue for sending data to the GUI process
            language: Language code for speech synthesis
            slow: Whether to speak slowly
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.language = language
        self.slow = slow
        self.running = False
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.temp_dir = Path(tempfile.gettempdir()) / "intuit_voice"
        self.temp_dir.mkdir(exist_ok=True)
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.info(f"Status: {status}")
        
        # Calculate audio metrics
        if indata.size > 0:
            # Calculate volume (RMS amplitude)
            volume = float(np.sqrt(np.mean(indata**2)))
            
            # Calculate approximate pitch using zero-crossing rate
            # This is a simple approximation, not accurate for actual pitch detection
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(indata).astype(int))))
            pitch = min(1.0, zero_crossings / (frames * 0.1))
            
            # Send metrics to GUI
            self.output_queue.put({
                'type': 'metrics',
                'volume': min(1.0, volume * 5),  # Scale up for better visualization
                'pitch': pitch
            })
        
        # Store audio data for recognition
        self.audio_queue.put(indata.copy())
    
    async def _listen(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for user input.
        
        Args:
            timeout: Time to listen for in seconds
            
        Returns:
            The recognized text or None if no speech was detected
        """
        logger.info("Listening...")
        
        # Check if we should stop
        if not self.running:
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
                # Break the sleep into smaller chunks to check running state
                chunks = int(timeout / 0.1)
                for _ in range(chunks):
                    await asyncio.sleep(0.1)
                    if not self.running:
                        return None
            else:
                await asyncio.sleep(5)  # Default to 5 seconds
        
        # Process recorded audio
        audio_data = []
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
        
        if not audio_data:
            logger.info("No audio recorded")
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
            logger.info(f"Recognized: {text}")
            
            # Send recognized text to GUI
            self.output_queue.put({
                'type': 'text',
                'content': text
            })
            
            return text
        except sr.UnknownValueError:
            logger.info("Could not understand audio")
            
            # Send a message to the GUI that no speech was recognized
            self.output_queue.put({
                'type': 'text',
                'content': ''
            })
            
            # Also send a notification that we couldn't understand
            self.output_queue.put({
                'type': 'process_text',
                'content': 'I could not understand what you said. Could you please try again?'
            })
            
            return None
        except sr.RequestError as e:
            logger.error(f"Error with speech recognition service: {e}")
            
            # Send error to GUI
            self.output_queue.put({
                'type': 'error',
                'message': f"Speech recognition service error: {e}"
            })
            
            return None
    
    async def _speak(self, text: str) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
        """
        try:
            # Create temporary file for audio
            temp_file = self.temp_dir / "speech.mp3"
            
            # Convert text to speech
            logger.info("Converting text to speech")
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(str(temp_file))
            
            # Notify GUI that speech is starting
            self.output_queue.put({
                'type': 'speaking',
                'state': 'start'
            })
            
            # Play the audio using a separate thread to avoid blocking
            logger.info("Playing audio")
            
            # Define a function to play audio in a thread
            def play_audio_thread():
                try:
                    if sys.platform == 'darwin':  # macOS
                        import subprocess
                        subprocess.run(['afplay', str(temp_file)], check=True)
                    else:  # Use sounddevice for other platforms
                        import soundfile as sf
                        data, samplerate = sf.read(str(temp_file))
                        sd.play(data, samplerate)
                        sd.wait()  # Wait until audio is finished playing
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
                finally:
                    # Notify GUI that speech is finished
                    self.output_queue.put({
                        'type': 'speaking',
                        'state': 'stop'
                    })
                    
                    # Clean up
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.error(f"Error cleaning up temp file: {e}")
            
            # Start audio playback in a separate thread
            audio_thread = threading.Thread(target=play_audio_thread)
            audio_thread.daemon = True  # Allow the thread to be terminated when the process exits
            audio_thread.start()
            
            # Don't wait for the thread to complete - continue processing commands
            # The thread will send the 'stop' message when done
            return
            
            # Note: Notification and cleanup are handled in the audio thread
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            
            # Notify GUI that speech has stopped due to error
            self.output_queue.put({
                'type': 'speaking',
                'state': 'stop',
                'error': str(e)
            })
    
    async def process_command(self, command: Dict[str, Any]) -> None:
        """
        Process a command from the GUI process.
        
        Args:
            command: Command dictionary
        """
        cmd_type = command.get('type')
        
        if cmd_type == 'speak':
            # Speak the provided text
            text = command.get('text', '')
            if text:
                await self._speak(text)
        elif cmd_type == 'listen':
            # Listen for user input
            timeout = command.get('timeout', 5.0)
            text = await self._listen(timeout)
            
            # The _listen method now handles sending the text to the GUI
            # We don't need to do anything else here
            
        elif cmd_type == 'stop':
            # Stop all processing
            self.running = False
    
    async def run(self) -> None:
        """Run the voice processor main loop."""
        self.running = True
        
        try:
            while self.running:
                # Check for commands from the GUI process
                try:
                    if not self.input_queue.empty():
                        command = self.input_queue.get_nowait()
                        await self.process_command(command)
                except queue.Empty:
                    pass
                
                # Short sleep to prevent CPU hogging
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in voice processor: {e}")
            self.output_queue.put({
                'type': 'error',
                'message': str(e)
            })
        finally:
            self.running = False
            logger.info("Voice processor stopped")
    
    def __del__(self):
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.glob("*"):
                file.unlink()
            self.temp_dir.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up voice files: {e}")


def voice_process_main(input_queue: multiprocessing.Queue, 
                       output_queue: multiprocessing.Queue,
                       language: str = 'en',
                       slow: bool = False) -> None:
    """
    Main function for the voice process.
    
    Args:
        input_queue: Queue for receiving commands from the GUI process
        output_queue: Queue for sending data to the GUI process
        language: Language code for speech synthesis
        slow: Whether to speak slowly
    """
    try:
        # Check if dependencies are available
        if not VOICE_DEPENDENCIES_AVAILABLE:
            output_queue.put({
                'type': 'error',
                'message': "Required voice dependencies are not installed. "
                          "Please install them using: pip install SpeechRecognition gTTS sounddevice numpy soundfile"
            })
            return
            
        # Initialize the voice processor
        processor = VoiceProcessor(input_queue, output_queue, language, slow)
        
        # Run the voice processor
        asyncio.run(processor.run())
    except Exception as e:
        logger.error(f"Fatal error in voice process: {e}")
        output_queue.put({
            'type': 'error',
            'message': str(e)
        })


class VoiceProcessManager:
    """
    Manages the voice process from the GUI side.
    
    This class handles starting and stopping the voice process,
    and communicating with it via queues.
    """
    
    def __init__(self, language: str = 'en', slow: bool = False):
        """
        Initialize the voice process manager.
        
        Args:
            language: Language code for speech synthesis
            slow: Whether to speak slowly
        """
        self.language = language
        self.slow = slow
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.running = False
    
    def start(self) -> bool:
        """
        Start the voice process.
        
        Returns:
            True if the process was started successfully, False otherwise
        """
        if self.process is not None and self.process.is_alive():
            logger.warning("Voice process is already running")
            return False
        
        try:
            # Check if dependencies are available
            if not VOICE_DEPENDENCIES_AVAILABLE:
                print("Warning: Voice dependencies are not installed. Voice functionality will be limited.")
                print("To install required packages, run: pip install SpeechRecognition gTTS sounddevice numpy soundfile")
            
            # Create and start the process
            self.process = multiprocessing.Process(
                target=voice_process_main,
                args=(self.input_queue, self.output_queue, self.language, self.slow),
                daemon=True
            )
            self.process.start()
            self.running = True
            logger.info("Voice process started")
            return True
        except Exception as e:
            logger.error(f"Error starting voice process: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the voice process.
        
        Returns:
            True if the process was stopped successfully, False otherwise
        """
        if self.process is None or not self.process.is_alive():
            logger.warning("Voice process is not running")
            return False
        
        try:
            # Send stop command to the process
            self.input_queue.put({'type': 'stop'})
            
            # Wait for the process to terminate (with timeout)
            self.process.join(timeout=3.0)
            
            # If the process is still alive, terminate it forcefully
            if self.process.is_alive():
                logger.warning("Voice process did not terminate gracefully, forcing termination")
                self.process.terminate()
                self.process.join(timeout=1.0)
                
                # If it's still alive, kill it
                if self.process.is_alive():
                    logger.warning("Voice process did not terminate, killing it")
                    self.process.kill()
                    self.process.join(timeout=1.0)
            
            self.running = False
            logger.info("Voice process stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping voice process: {e}")
            return False
    
    def speak(self, text: str) -> None:
        """
        Send a speak command to the voice process.
        
        Args:
            text: Text to speak
        """
        if not self.running:
            logger.warning("Voice process is not running")
            return
        
        self.input_queue.put({
            'type': 'speak',
            'text': text
        })
    
    def listen(self, timeout: float = 5.0, process: bool = False) -> None:
        """
        Send a listen command to the voice process.
        
        Args:
            timeout: Time to listen for in seconds
            process: Whether to process the recognized text
        """
        if not self.running:
            logger.warning("Voice process is not running")
            return
        
        self.input_queue.put({
            'type': 'listen',
            'timeout': timeout,
            'process': process
        })
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """
        Get data from the voice process.
        
        Returns:
            Data from the voice process, or None if no data is available
        """
        try:
            if not self.output_queue.empty():
                return self.output_queue.get_nowait()
        except queue.Empty:
            pass
        
        return None
    
    def is_running(self) -> bool:
        """
        Check if the voice process is running.
        
        Returns:
            True if the voice process is running, False otherwise
        """
        return self.running and self.process is not None and self.process.is_alive()
    
    def __del__(self):
        """Clean up resources."""
        self.stop()