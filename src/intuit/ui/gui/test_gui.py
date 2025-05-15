#!/usr/bin/env python3
"""
Test script for the Intuit AI GUI.

This script demonstrates the isometric hockey puck visualization
with reactive lighting effects.
"""

import sys
import time
import random
import sys
import os

from PySide6.QtWidgets import QApplication

from .main_gui import IntuitGUI


def simulate_speech(gui):
    """
    Simulate speech by randomly varying volume and pitch.
    
    Args:
        gui: IntuitGUI instance
    """
    # Generate random volume (0.3 to 1.0)
    volume = random.uniform(0.3, 1.0)
    
    # Generate random pitch (0.0 to 1.0)
    pitch = random.uniform(0.0, 1.0)
    
    # Update the GUI
    gui.reactToSpeech(volume, pitch)
    
    # Print the values
    print(f"Volume: {volume:.2f}, Pitch: {pitch:.2f}")


def main():
    """Run the test GUI."""
    # Create the application
    app = QApplication(sys.argv)
    
    # Create the main window
    window = IntuitGUI()
    window.show()
    
    # Start the test animation
    window.testAnimation()
    
    # Print instructions
    print("Hockey Puck GUI Controls:")
    print("- Drag with mouse to move the window")
    print("- Press Space to toggle between pulse and ripple effects")
    print("- Press T to start/stop test animation")
    print("- Press Escape to close")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()