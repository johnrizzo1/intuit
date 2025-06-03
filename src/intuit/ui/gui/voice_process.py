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
import collections
from pathlib import Path
from typing import Optional, Dict, Any, List, Deque

# Try to import optional dependencies
try:
    import speech_recognition as sr
    import numpy as np
    import sounddevice as sd
    from gtts import gTTS
    import scipy.signal
    VOICE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    VOICE_DEPENDENCIES_AVAILABLE = False
    print("Warning: Some voice dependencies are not installed. Voice functionality will be limited.")
    print("To install required packages, run: pip install SpeechRecognition gTTS sounddevice numpy soundfile scipy")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechDetector:
    """
    Advanced speech detection with noise filtering capabilities.
    
    This class implements various techniques to differentiate between
    actual speech and background noise.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 volume_threshold: float = 0.02,
                 speech_window: float = 0.3,
                 frequency_range: tuple = (300, 3000)):
        """
        Initialize the speech detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            volume_threshold: RMS volume threshold for speech detection
            speech_window: Minimum duration (in seconds) for a sound to be considered speech
            frequency_range: Frequency range (in Hz) for bandpass filtering
        """
        self.sample_rate = sample_rate
        self.volume_threshold = volume_threshold
        self.speech_window_samples = int(speech_window * sample_rate)
        self.frequency_range = frequency_range
        
        # Buffer for temporal analysis
        self.audio_buffer: Deque[np.ndarray] = collections.deque(maxlen=50)  # ~0.5s at 16kHz with 160 samples per frame
        self.volume_history: Deque[float] = collections.deque(maxlen=20)  # For volume averaging
        
        # Speech detection state
        self.is_speech_detected = False
        self.speech_start_time = 0
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        
        # Design bandpass filter
        nyquist = sample_rate / 2
        low = frequency_range[0] / nyquist
        high = frequency_range[1] / nyquist
        self.b, self.a = scipy.signal.butter(4, [low, high], btype='band')
    
    def reset(self):
        """Reset the detector state."""
        self.audio_buffer.clear()
        self.volume_history.clear()
        self.is_speech_detected = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
    
    def process_audio(self, audio_data: np.ndarray) -> tuple:
        """
        Process audio data to detect speech.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (is_speech, volume, pitch, filtered_audio)
        """
        # Apply bandpass filter to focus on speech frequencies
        try:
            filtered_audio = scipy.signal.lfilter(self.b, self.a, audio_data.flatten())
            filtered_audio = filtered_audio.reshape(audio_data.shape)
        except Exception as e:
            logger.error(f"Error applying bandpass filter: {e}")
            filtered_audio = audio_data
        
        # Calculate RMS volume
        rms_volume = float(np.sqrt(np.mean(filtered_audio**2)))
        
        # Add to volume history for averaging
        self.volume_history.append(rms_volume)
        
        # Calculate averaged volume
        if len(self.volume_history) > 0:
            avg_volume = sum(self.volume_history) / len(self.volume_history)
        else:
            avg_volume = rms_volume
        
        # Calculate approximate pitch using zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(filtered_audio).astype(int))))
        frames = filtered_audio.shape[0]
        pitch = min(1.0, zero_crossings / (frames * 0.1))
        
        # Add to audio buffer for temporal analysis
        self.audio_buffer.append(filtered_audio)
        
        # Determine if this is speech based on volume threshold and temporal analysis
        is_frame_speech = avg_volume > self.volume_threshold
        
        # Update consecutive frame counters
        if is_frame_speech:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
        else:
            self.consecutive_silence_frames += 1
            # Only reset speech frames if we've had enough silence
            if self.consecutive_silence_frames > 3:  # ~300ms of silence
                self.consecutive_speech_frames = 0
        
        # Determine overall speech state with hysteresis
        # Need more consecutive frames to start speech than to end it
        if not self.is_speech_detected and self.consecutive_speech_frames > 5:  # ~500ms of speech
            self.is_speech_detected = True
            self.speech_start_time = time.time()
        elif self.is_speech_detected and self.consecutive_silence_frames > 10:  # ~1s of silence
            self.is_speech_detected = False
        
        # Scale volume for visualization
        display_volume = min(1.0, avg_volume * 5)
        
        return (self.is_speech_detected, display_volume, pitch, filtered_audio)

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
        
        # Speech detection
        self.speech_detector = SpeechDetector(
            sample_rate=self.sample_rate,
            volume_threshold=0.02,  # Adjust based on testing
            speech_window=0.3,
            frequency_range=(300, 3000)
        )
        
        # Speaking state
        self.is_speaking = False
        self.current_audio_thread = None
        self.should_interrupt = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio stream status issue: {status}")
        
        # Process audio with advanced speech detection
        if indata.size > 0:
            is_speech, volume, pitch, filtered_audio = self.speech_detector.process_audio(indata)
            
            # Send metrics to GUI
            self.output_queue.put({
                'type': 'metrics',
                'volume': volume,
                'pitch': pitch,
                'is_speech': is_speech
            })
            
            # Check if we should interrupt current speech
            # Note: Main interruption logic moved to _continuous_listen_thread for more reliable detection
            if is_speech and self.is_speaking and volume > 0.4:  # Higher threshold for interruption
                logger.info(f"User speech detected while speaking - volume: {volume:.2f}")
                # Don't set should_interrupt here - let the continuous thread handle it
                # This avoids race conditions in the callback
        
        # Store filtered audio data for recognition
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
                self.is_speaking = True
                self.should_interrupt = False
                
                try:
                    if sys.platform == 'darwin':  # macOS
                        import subprocess
                        # Use popen instead of run to be able to terminate the process
                        process = subprocess.Popen(['afplay', str(temp_file)])
                        
                        # Check for interruption while playing
                        while process.poll() is None:
                            if self.should_interrupt:
                                logger.warning("INTERRUPTING SPEECH PLAYBACK - User is speaking")
                                process.terminate()
                                process.wait()
                                break
                            time.sleep(0.05)  # Check more frequently for interruption
                    else:  # Use sounddevice for other platforms
                        import soundfile as sf
                        data, samplerate = sf.read(str(temp_file))
                        
                        # Play with callback to check for interruption
                        def callback(outdata, frames, time, status):
                            if self.should_interrupt:
                                raise sd.CallbackStop
                            
                            if len(data) > 0:
                                outdata[:] = data[:frames]
                                data = data[frames:]
                            else:
                                raise sd.CallbackStop
                        
                        with sd.OutputStream(
                            samplerate=samplerate,
                            channels=data.shape[1] if len(data.shape) > 1 else 1,
                            callback=callback
                        ):
                            while not self.should_interrupt:
                                time.sleep(0.1)
                                
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
                finally:
                    # Notify GUI that speech is finished
                    self.output_queue.put({
                        'type': 'speaking',
                        'state': 'stop',
                        'interrupted': self.should_interrupt
                    })
                    
                    # Clean up
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.error(f"Error cleaning up temp file: {e}")
                    
                    self.is_speaking = False
                    self.should_interrupt = False
            
            # Start audio playback in a separate thread
            audio_thread = threading.Thread(target=play_audio_thread)
            audio_thread.daemon = True  # Allow the thread to be terminated when the process exits
            audio_thread.start()
            self.current_audio_thread = audio_thread
            
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
            
            # If text was recognized and process flag is set, send it for processing
            if text and command.get('process', False):
                # Send a message to the GUI that we're processing the text
                self.output_queue.put({
                    'type': 'processing',
                    'content': text
                })
                
                # The GUI will handle sending the text to the agent and returning the response
                # We don't need to do anything else here
            
        elif cmd_type == 'process_response':
            # Process a response from the agent
            response = command.get('response', '')
            if response:
                # Speak the response
                await self._speak(response)
                
                # After speaking, automatically start listening again for a continuous conversation
                # Send a message to the GUI to indicate we're ready to listen again
                self.output_queue.put({
                    'type': 'ready_to_listen'
                })
                
        elif cmd_type == 'stop':
            # Stop all processing
            self.running = False
    
    def stop_speaking(self):
        """Stop the current speech playback."""
        if self.is_speaking:
            logger.warning("STOPPING CURRENT SPEECH - Interruption requested")
            if self.current_audio_thread is not None and self.current_audio_thread.is_alive():
                logger.info("Audio playback thread detected - sending termination signal")
            else:
                logger.warning("No active audio playback thread found at interruption attempt")
            
            # Force interruption
            self.should_interrupt = True
            
            # Wait briefly to ensure the interruption takes effect
            time.sleep(0.1)
            
            # Double-check that interruption was successful
            if self.is_speaking:
                logger.warning("Speech interruption may not have succeeded - forcing state reset")
                self.is_speaking = False
            
            return True
        return False
    
    async def run(self) -> None:
        """Run the voice processor main loop."""
        self.running = True
        
        # Start a continuous listening thread
        continuous_listen_thread = threading.Thread(target=self._continuous_listen_thread)
        continuous_listen_thread.daemon = True
        continuous_listen_thread.start()
        
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
    
    def _continuous_listen_thread(self):
        """
        Continuously monitor audio input in a separate thread.
        This allows for speech detection even while the assistant is speaking.
        """
        logger.info("Starting continuous listening thread")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            ):
                while self.running:
                    logger.debug("Listening... waiting for input")
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in continuous listening thread: {e}")
        finally:
            logger.info("Stopping continuous listening thread. Cleanup completed.")
    
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
                       slow: bool = False,
                       volume_threshold: float = 0.02,
                       speech_window: float = 0.3) -> None:
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
            
        # Initialize the voice processor with sensitivity settings
        processor = VoiceProcessor(input_queue, output_queue, language, slow)
        processor.speech_detector.volume_threshold = volume_threshold
        processor.speech_detector.speech_window = speech_window * processor.sample_rate
        
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
    
    def __init__(self, language: str = 'en', slow: bool = False,
                 volume_threshold: float = 0.02, speech_window: float = 0.3):
        """
        Initialize the voice process manager.
        
        Args:
            language: Language code for speech synthesis
            slow: Whether to speak slowly
            volume_threshold: RMS volume threshold for speech detection
            speech_window: Minimum duration (in seconds) for a sound to be considered speech
        """
        self.language = language
        self.slow = slow
        self.volume_threshold = volume_threshold
        self.speech_window = speech_window
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.running = False
        self.is_speaking = False
    
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
                kwargs={
                    'volume_threshold': self.volume_threshold,
                    'speech_window': self.speech_window
                },
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
        
        self.is_speaking = True
        self.input_queue.put({
            'type': 'speak',
            'text': text
        })
    
    def stop_speaking(self) -> bool:
        """
        Stop the current speech playback.
        
        Returns:
            True if speech was stopped, False otherwise
        """
        if not self.running or not self.is_speaking:
            return False
        
        self.input_queue.put({
            'type': 'stop_speaking'
        })
        self.is_speaking = False
        return True
    
    def listen(self, timeout: float = 5.0, process: bool = True) -> None:
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
        
    def process_response(self, response: str) -> None:
        """
        Send an agent response to be spoken by the voice process.
        
        Args:
            response: The agent's response text
        """
        if not self.running:
            logger.warning("Voice process is not running")
            return
        
        self.is_speaking = True
        self.input_queue.put({
            'type': 'process_response',
            'response': response
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