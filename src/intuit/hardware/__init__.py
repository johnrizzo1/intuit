"""Hardware detection and configuration for Intuit."""

from .detector import (
    HardwareDetector,
    HardwareCapabilities,
    AcceleratorType,
)

__all__ = [
    "HardwareDetector",
    "HardwareCapabilities",
    "AcceleratorType",
]