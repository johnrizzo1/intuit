#!/usr/bin/env python3
"""
Demo script for the Intuit AI Hockey Puck GUI.

This script demonstrates how to integrate the hockey puck GUI with the AI system.
It simulates speech data and shows how the GUI reacts to it.
"""

import sys
import time
import random
import threading
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from .main_gui import IntuitGUI


class SpeechSimulator:
    """
    Simulates speech data for demonstration purposes.
    
    This class generates random speech data (volume and pitch) to demonstrate
    how the GUI reacts to speech.
    """
    
    def __init__(self, gui):
        """Initialize the speech simulator."""
        self.gui = gui
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the speech simulation."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the speech simulation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
    def _run_simulation(self):
        """Run the speech simulation in a separate thread."""
        print("Speech simulation started")
        
        # Simulate different speech patterns
        patterns = [
            self._simulate_conversation,
            self._simulate_excited_speech,
            self._simulate_calm_speech,
            self._simulate_silence_then_speech
        ]
        
        while self.running:
            # Choose a random pattern
            pattern = random.choice(patterns)
            pattern()
            
            # Pause between patterns
            time.sleep(random.uniform(1.0, 3.0))
            
        print("Speech simulation stopped")
        
    def _simulate_conversation(self):
        """Simulate a conversation pattern."""
        print("Simulating conversation pattern")
        
        # Simulate 5-10 speech segments
        segments = random.randint(5, 10)
        
        for _ in range(segments):
            if not self.running:
                break
                
            # Random volume (medium to high) and pitch
            volume = random.uniform(0.4, 0.8)
            pitch = random.uniform(0.3, 0.7)
            
            # Send to GUI
            self.gui.reactToSpeech(volume, pitch)
            
            # Short pause between segments
            time.sleep(random.uniform(0.3, 0.8))
            
    def _simulate_excited_speech(self):
        """Simulate excited speech pattern."""
        print("Simulating excited speech pattern")
        
        # Simulate 3-6 speech segments
        segments = random.randint(3, 6)
        
        for _ in range(segments):
            if not self.running:
                break
                
            # High volume and pitch
            volume = random.uniform(0.7, 1.0)
            pitch = random.uniform(0.7, 1.0)
            
            # Send to GUI
            self.gui.reactToSpeech(volume, pitch)
            
            # Very short pause between segments
            time.sleep(random.uniform(0.1, 0.4))
            
    def _simulate_calm_speech(self):
        """Simulate calm speech pattern."""
        print("Simulating calm speech pattern")
        
        # Simulate 3-5 speech segments
        segments = random.randint(3, 5)
        
        for _ in range(segments):
            if not self.running:
                break
                
            # Low to medium volume and pitch
            volume = random.uniform(0.2, 0.5)
            pitch = random.uniform(0.1, 0.4)
            
            # Send to GUI
            self.gui.reactToSpeech(volume, pitch)
            
            # Longer pause between segments
            time.sleep(random.uniform(0.8, 1.5))
            
    def _simulate_silence_then_speech(self):
        """Simulate silence followed by speech."""
        print("Simulating silence then speech pattern")
        
        # Silence
        self.gui.reactToSpeech(0.0, 0.0)
        time.sleep(random.uniform(1.0, 2.0))
        
        if not self.running:
            return
            
        # Sudden speech
        volume = random.uniform(0.6, 0.9)
        pitch = random.uniform(0.4, 0.8)
        self.gui.reactToSpeech(volume, pitch)
        
        # Hold for a moment
        time.sleep(random.uniform(0.5, 1.0))


def main():
    """Run the demo."""
    # Create the application
    app = QApplication(sys.argv)
    
    # Create the main window
    window = IntuitGUI()
    window.show()
    
    # Create and start the speech simulator
    simulator = SpeechSimulator(window)
    
    # Use a timer to start the simulator after the GUI is shown
    QTimer.singleShot(500, simulator.start)
    
    # Print instructions
    print("Hockey Puck GUI Demo")
    print("-------------------")
    print("This demo simulates AI speech and shows how the GUI reacts to it.")
    print("Controls:")
    print("- Drag with mouse to move the window")
    print("- Press Space to toggle between pulse and ripple effects")
    print("- Press T to start/stop test animation")
    print("- Press Escape to close")
    
    # Run the application
    try:
        sys.exit(app.exec())
    finally:
        # Stop the simulator when the application exits
        simulator.stop()


if __name__ == "__main__":
    main()