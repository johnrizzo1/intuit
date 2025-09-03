"""
Voice output functionality for Intuit.
"""

import os
import tempfile
import logging
from pathlib import Path

# Try to import optional voice dependencies
try:
    from gtts import gTTS
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    VOICE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    VOICE_DEPENDENCIES_AVAILABLE = False
    print(
        "Warning: Voice dependencies are not installed. Voice functionality will be disabled."
    )
    print(
        "To install required packages, run: pip install gtts sounddevice soundfile numpy"
    )

# Set up logging
logger = logging.getLogger(__name__)


class VoiceOutput:
    """Handles text-to-speech conversion and audio playback."""

    def __init__(self, language: str = "en", slow: bool = False):
        """
        Initialize voice output.

        Args:
            language: Language code for speech synthesis
            slow: Whether to speak slowly
        """
        if not VOICE_DEPENDENCIES_AVAILABLE:
            logger.warning(
                "Voice dependencies not available. Voice output will be disabled."
            )

        self.language = language
        self.slow = slow
        self.temp_dir = Path(tempfile.gettempdir()) / "intuit_voice"
        self.temp_dir.mkdir(exist_ok=True)

    async def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.

        Args:
            text: Text to convert to speech
        """
        if not VOICE_DEPENDENCIES_AVAILABLE:
            logger.warning(
                f"Voice output disabled (dependencies not available): {text}"
            )
            return

        try:
            # Create temporary file for audio
            temp_file = self.temp_dir / "speech.mp3"

            # Convert text to speech
            logger.info("Converting text to speech")
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(str(temp_file))

            # Play the audio
            logger.info("Playing audio")
            data, samplerate = sf.read(str(temp_file))
            sd.play(data, samplerate)
            sd.wait()  # Wait until audio is finished playing

            # Clean up
            temp_file.unlink()

        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            raise

    def __del__(self):
        """Clean up temporary files."""
        try:
            for file in self.temp_dir.glob("*"):
                file.unlink()
            self.temp_dir.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up voice files: {e}")
