"""Hardware capability detection for acceleration."""
import torch
import platform
from enum import Enum
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AcceleratorType(Enum):
    """Available hardware accelerators."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal
    CPU = "cpu"


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities."""
    accelerator: AcceleratorType
    device_name: str
    compute_capability: Optional[str]
    memory_gb: float
    supports_fp16: bool
    supports_int8: bool


class HardwareDetector:
    """Detect available hardware acceleration."""
    
    @staticmethod
    def detect() -> HardwareCapabilities:
        """Detect best available hardware accelerator."""
        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            logger.info(f"CUDA device detected: {device.name}")
            return HardwareCapabilities(
                accelerator=AcceleratorType.CUDA,
                device_name=device.name,
                compute_capability=f"{device.major}.{device.minor}",
                memory_gb=device.total_memory / (1024**3),
                supports_fp16=device.major >= 7,
                supports_int8=device.major >= 7,
            )
        
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Metal) device detected")
            return HardwareCapabilities(
                accelerator=AcceleratorType.MPS,
                device_name=platform.processor(),
                compute_capability=None,
                memory_gb=0.0,  # MPS shares system memory
                supports_fp16=True,
                supports_int8=False,
            )
        
        # Fallback to CPU
        logger.warning("No GPU detected, falling back to CPU")
        return HardwareCapabilities(
            accelerator=AcceleratorType.CPU,
            device_name=platform.processor(),
            compute_capability=None,
            memory_gb=0.0,
            supports_fp16=False,
            supports_int8=False,
        )
    
    @staticmethod
    def get_device_string() -> str:
        """Get PyTorch device string."""
        caps = HardwareDetector.detect()
        return caps.accelerator.value
    
    @staticmethod
    def print_capabilities() -> None:
        """Print hardware capabilities to console."""
        caps = HardwareDetector.detect()
        print("\n🔧 Hardware Capabilities:")
        print(f"  Accelerator: {caps.accelerator.value.upper()}")
        print(f"  Device: {caps.device_name}")
        if caps.compute_capability:
            print(f"  Compute Capability: {caps.compute_capability}")
        if caps.memory_gb > 0:
            print(f"  Memory: {caps.memory_gb:.1f} GB")
        print(f"  FP16 Support: {'✅' if caps.supports_fp16 else '❌'}")
        print(f"  INT8 Support: {'✅' if caps.supports_int8 else '❌'}\n")
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if any GPU is available."""
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    
    @staticmethod
    def get_recommended_compute_type() -> str:
        """Get recommended compute type based on hardware."""
        caps = HardwareDetector.detect()
        if caps.supports_fp16:
            return "float16"
        return "float32"