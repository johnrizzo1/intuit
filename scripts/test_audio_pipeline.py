#!/usr/bin/env python3
"""
Test script to diagnose audio pipeline issues.
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n=== Testing Imports ===")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"  MPS available: {torch.backends.mps.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import whisper
        print(f"✓ Whisper imported successfully")
    except ImportError as e:
        print(f"✗ Whisper import failed: {e}")
        return False
    
    try:
        import sounddevice as sd
        print(f"✓ sounddevice imported successfully")
        print(f"  Available devices: {len(sd.query_devices())}")
    except ImportError as e:
        print(f"✗ sounddevice import failed: {e}")
        return False
    
    return True


def test_hardware_detection():
    """Test hardware detection."""
    print("\n=== Testing Hardware Detection ===")
    
    try:
        from intuit.hardware.detector import HardwareDetector
        
        device = HardwareDetector.get_device_string()
        print(f"✓ Detected device: {device}")
        
        info = HardwareDetector.get_device_info()
        print(f"  Device info: {info}")
        
        return True
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_config():
    """Test audio configuration."""
    print("\n=== Testing Audio Configuration ===")
    
    try:
        from intuit.config.audio_config import AudioConfig
        
        config = AudioConfig.from_env()
        print(f"✓ Audio config loaded")
        print(f"  STT Provider: {config.stt.provider}")
        print(f"  STT Device: {config.stt.device}")
        print(f"  TTS Provider: {config.tts.provider}")
        print(f"  TTS Device: {config.tts.device}")
        print(f"  LLM Provider: {config.llm.provider}")
        
        return True
    except Exception as e:
        print(f"✗ Audio config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stt_provider():
    """Test STT provider initialization."""
    print("\n=== Testing STT Provider ===")
    
    try:
        from intuit.audio.stt_factory import STTFactory
        from intuit.config.audio_config import AudioConfig
        
        config = AudioConfig.from_env()
        stt = STTFactory.create(config.stt)
        print(f"✓ STT provider created: {type(stt).__name__}")
        
        if hasattr(stt, 'device'):
            print(f"  Device: {stt.device}")
        if hasattr(stt, 'model_size'):
            print(f"  Model size: {stt.model_size}")
        
        return True
    except Exception as e:
        print(f"✗ STT provider failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_provider():
    """Test TTS provider initialization."""
    print("\n=== Testing TTS Provider ===")
    
    try:
        from intuit.audio.tts_factory import TTSFactory
        from intuit.config.audio_config import AudioConfig
        
        config = AudioConfig.from_env()
        tts = TTSFactory.create(config.tts)
        print(f"✓ TTS provider created: {type(tts).__name__}")
        
        if hasattr(tts, 'device'):
            print(f"  Device: {tts.device}")
        
        return True
    except Exception as e:
        print(f"✗ TTS provider failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_devices():
    """Test audio device availability."""
    print("\n=== Testing Audio Devices ===")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        print(f"✓ Found {len(devices)} audio devices")
        
        # Find default input device
        default_input = sd.query_devices(kind='input')
        print(f"  Default input: {default_input['name']}")
        print(f"  Channels: {default_input['max_input_channels']}")
        print(f"  Sample rate: {default_input['default_samplerate']}")
        
        return True
    except Exception as e:
        print(f"✗ Audio device query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("=" * 60)
    print("Audio Pipeline Diagnostic Tool")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Hardware Detection": test_hardware_detection(),
        "Audio Config": test_audio_config(),
        "STT Provider": test_stt_provider(),
        "TTS Provider": test_tts_provider(),
        "Audio Devices": test_audio_devices(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Audio pipeline should work.")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())