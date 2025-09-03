import time
import queue
import logging
import multiprocessing
import psutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _audio_callback(self, indata, frames, time, status):
    if status:
        logger.warning(f"Audio input status: {status}")

    # Process audio with advanced speech detection
    if indata.size > 0:
        is_speech, volume, pitch, filtered_audio = self.speech_detector.process_audio(
            indata
        )

        # Log speech detection details
        logger.debug(
            f"Speech detection: is_speech={is_speech}, volume={volume}, pitch={pitch}"
        )

        # Send metrics to GUI
        self.output_queue.put(
            {
                "type": "metrics",
                "volume": volume,
                "pitch": pitch,
                "is_speech": is_speech,
            }
        )

        # Interrupt speech if user speech is detected
        if (
            is_speech and self.is_speaking and volume > 0.4
        ):  # Higher threshold for interruption
            logger.info("User speech detected while speaking - triggering interruption")
            self.should_interrupt = True


def get_data(self) -> Optional[Dict[str, Any]]:
    try:
        if not self.output_queue.empty():
            data = self.output_queue.get_nowait()
            logger.debug(f"Data received from voice process: {data} at {time.time()}")
            return data
    except queue.Empty:
        pass
    return None


def start(self) -> bool:
    logger.debug("Attempting to start the voice process")
    if self.process is not None and self.process.is_alive():
        logger.warning("Voice process is already running")
        return False

    try:
        # Log configuration details
        logger.debug(
            f"Voice process configuration: language={self.language}, slow={self.slow}, volume_threshold={self.volume_threshold}, speech_window={self.speech_window}"
        )

        # Create and start the process
        self.process = multiprocessing.Process(
            target=voice_process_main,
            args=(self.input_queue, self.output_queue, self.language, self.slow),
            kwargs={
                "volume_threshold": self.volume_threshold,
                "speech_window": self.speech_window,
            },
            daemon=True,
        )
        self.process.start()
        self.running = True
        logger.info("Voice process started successfully")

        # Log resource usage
        self.log_resource_usage()

        return True
    except Exception as e:
        logger.error(f"Error starting voice process: {e}")
        return False


def process_voice_data(self):
    """Process data from the voice process."""
    if self.voice_manager.is_running():
        data = self.voice_manager.get_data()
        if data:
            logger.debug(f"Processing voice data: {data}")
            if data["type"] == "metrics":
                # Update visualization with speech metrics
                self.reactToSpeech(data["volume"], data["pitch"])
            elif data["type"] == "text":
                # Display recognized text
                print(f"Recognized text: {data['content']}")
                self.voice_manager.speak(f"You said: {data['content']}")
            elif data["type"] == "error":
                # Display error messages
                print(f"Voice process error: {data['message']}")


def log_resource_usage(self):
    if self.process is not None and self.process.is_alive():
        process = psutil.Process(self.process.pid)
        cpu_usage = process.cpu_percent(interval=0.1)
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        logger.info(
            f"Voice process resource usage: CPU={cpu_usage}%, Memory={memory_usage}MB"
        )
