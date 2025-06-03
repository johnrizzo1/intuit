#!/usr/bin/env python3
"""
Agent-integrated GUI for the Intuit AI Hockey Puck.

This script integrates the Intuit agent with the hockey puck GUI.
"""

import sys
import os
import json
import asyncio
import threading
from typing import Optional, Dict, Any

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Import the GUI components
try:
    from .standalone_gui import IntuitGUI, HockeyPuckView, HockeyPuckItem
    from .voice_process import VoiceProcessManager
except ImportError:
    # When run as a standalone script
    import os
    import sys
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from standalone_gui import IntuitGUI, HockeyPuckView, HockeyPuckItem
    from intuit.utils.voice_process import VoiceProcessManager


class AgentGUI(IntuitGUI):
    """
    Extended GUI class that integrates with the Intuit agent.
    
    This class overrides the voice processing methods to use the actual agent
    instead of simulated responses.
    """
    
    def __init__(self, agent=None, config=None):
        """
        Initialize the agent-integrated GUI.
        
        Args:
            agent: The Intuit agent to use for processing queries
            config: Optional configuration dictionary
        """
        # Store the agent reference
        self.agent = agent
        
        # Initialize the base GUI
        super().__init__(config)
        
        # Print status
        if self.agent:
            print("Agent integration enabled")
        else:
            print("Warning: No agent provided, using simulated responses")
    
    def process_voice_data(self):
        """Process data from the voice process using the agent."""
        if not self.voice_manager.is_running():
            return
            
        # Get data from the voice process
        data = self.voice_manager.get_data()
        if data:
            data_type = data.get('type')
            
            if data_type == 'metrics':
                # Update the puck visualization with speech metrics
                volume = data.get('volume', 0.0)
                pitch = data.get('pitch', 0.5)
                self.reactToSpeech(volume, pitch)
            
            elif data_type == 'text':
                # Handle recognized text
                content = data.get('content', '')
                # Just log the recognized text - let the GUIManager handle processing with the agent
                print(f"Recognized: {content}")
                # The GUIManager's _monitor_voice_process will pick this up and send to the agent
                
            elif data_type == 'speaking':
                # Handle speaking state changes
                state = data.get('state')
                if state == 'start':
                    print("AI is speaking...")
                elif state == 'stop':
                    print("AI finished speaking")
                    
                    # If there was an error, log it
                    if 'error' in data:
                        print(f"Speech error: {data['error']}")
                    
                    # Always start listening again after speaking stops
                    # This creates a continuous conversation loop
                    if self.voice_active:
                        print("Listening for next command...")
                        QTimer.singleShot(1000, lambda: self.voice_manager.listen(timeout=5.0, process=True))
            
            elif data_type == 'ready_to_listen':
                # Voice process is ready to listen again
                if self.voice_active:
                    print("Listening for next command...")
                    self.voice_manager.listen(timeout=5.0, process=True)
            
            elif data_type == 'error':
                # Handle errors
                error_msg = data.get('message', 'Unknown error')
                print(f"Voice process error: {error_msg}")
                
                # If the error is about missing dependencies, show a more helpful message
                if "dependencies are not installed" in error_msg:
                    print("To install required packages, run:")
                    print("pip install SpeechRecognition gTTS sounddevice numpy soundfile")
                    
                    # Disable voice to prevent further errors
                    self.voice_active = False
    
    # This method is now handled by GUIManager in integration.py
    # Keeping this as a stub for backward compatibility
    def _process_with_agent(self, text):
        """
        This method is deprecated. Agent processing is now handled by GUIManager.
        This stub exists only for backward compatibility.
        """
        print("Warning: AgentGUI._process_with_agent is deprecated. Use GUIManager._process_with_agent instead.")
        pass


def run_agent_gui(agent=None, config=None, block=True):
    """
    Run the agent-integrated GUI.
    
    Args:
        agent: The Intuit agent to use for processing queries
        config: Optional configuration dictionary
        block: Whether to block until the GUI is closed
        
    Returns:
        True if the GUI was started successfully, False otherwise
    """
    try:
        # Create the application
        app = QApplication.instance() or QApplication([])
        
        # Create the main window
        window = AgentGUI(agent, config)
        window.show()
        
        # Print instructions
        print("Intuit Agent GUI")
        print("---------------")
        print("Controls:")
        print("- Drag with mouse to move the window")
        print("- Press Space to toggle between smoke, pulse and ripple effects")
        print("- Press T to stop/restart animation (enabled by default)")
        print("- Press V to toggle voice interface")
        print("- Right-click for menu")
        print("- Press Escape to close")
        
        if block:
            # Run the application and block until it's closed
            return app.exec() == 0
        else:
            # Start the event loop but don't block
            return True
            
    except Exception as e:
        print(f"Error starting agent GUI: {e}")
        return False


if __name__ == "__main__":
    # This is just for testing without an agent
    run_agent_gui()