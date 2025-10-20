"""Audio processing modules for Intuit."""

from .stt_base import STTProvider
from .stt_google import GoogleSTT
from .stt_factory import STTFactory
from .tts_base import TTSProvider
from .tts_gtts import GTTSProvider
from .tts_factory import TTSFactory

# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "WhisperSTT":
        from .stt_whisper import WhisperSTT
        return WhisperSTT
    elif name == "CoquiTTSProvider":
        from .tts_coqui import CoquiTTSProvider
        return CoquiTTSProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "STTProvider",
    "WhisperSTT",
    "GoogleSTT",
    "STTFactory",
    "TTSProvider",
    "CoquiTTSProvider",
    "GTTSProvider",
    "TTSFactory",
]