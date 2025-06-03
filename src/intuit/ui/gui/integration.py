#!/usr/bin/env python3
"""
Integration module for the Intuit AI Hockey Puck GUI.

This module provides classes and functions to integrate the hockey puck GUI
with the existing AI system.
"""

import threading
import queue
import time
from typing import Dict, Any, Optional, Callable

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QObject, Signal, Slot, QThread

from .main_gui import IntuitGUI
from .light_effects import PulseEffect, RippleEffect
from .standalone_gui import HockeyPuckItem  # Import from standalone_gui instead of puck_widget

class SpeechProcessor(QObject):
    """
    Process speech data from the AI system and update the GUI.
    
    This class runs in a separate thread and processes speech data from the AI
    system, then updates the GUI accordingly.
    """
    
    # Signal to update the GUI from the worker thread
    speech_signal = Signal(float, float)
    
    def __init__(self, parent=None):
        """Initialize the speech processor."""
        super().__init__(parent)
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the speech processor thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_speech)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the speech processor thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
    def process_speech_data(self, data: Dict[str, Any]):
        """
        Process speech data from the AI system.
        
        Args:
            data: Speech data dictionary with at least 'volume' and 'pitch' keys.
        """
        self.queue.put(data)
        
    def _process_speech(self):
        """Process speech data in a separate thread."""
        while self.running:
            try:
                # Get speech data from the queue with a timeout
                data = self.queue.get(timeout=0.1)
                
                # Extract volume and pitch
                volume = data.get('volume', 0.0)
                pitch = data.get('pitch', 0.5)
                
                # Emit signal to update the GUI
                self.speech_signal.emit(volume, pitch)
                
                # Mark the task as done
                self.queue.task_done()
                
            except queue.Empty:
                # No data in the queue, just continue
                pass
            except Exception as e:
                print(f"Error processing speech data: {e}")

class GUIManager:
    """
    Manage the lifecycle of the GUI.
    
    This class provides a high-level interface for integrating the GUI with
    the AI system.
    """
    
    def __init__(self):
        """Initialize the GUI manager."""
        self.app = None
        self.window = None
        self.speech_processor = None
        self.app_thread = None
        self.running = False
        self.agent = None
        self.voice_manager = None
        
    def start(self, block: bool = False, agent=None, enable_voice: bool = True) -> bool:
        """
        Start the GUI.
        
        Args:
            block: If True, block until the GUI is closed. If False, run the GUI
                  in a separate thread and return immediately.
            agent: The agent to use for processing queries
            enable_voice: Whether to enable voice functionality
            
        Returns:
            True if the GUI was started successfully, False otherwise.
        """
        if self.running:
            return True
            
        try:
            # Store the agent reference
            self.agent = agent
            
            # Create the application
            self.app = QApplication([])
            
            # Create the main window
            self.window = IntuitGUI()
            
            # Create and start the speech processor
            self.speech_processor = SpeechProcessor()
            self.speech_processor.speech_signal.connect(self.window.reactToSpeech)
            self.speech_processor.start()
            
            # Initialize voice processing if enabled
            if enable_voice and agent:
                # Import here to avoid circular imports
                from .voice_process import VoiceProcessManager
                
                # Create and start the voice process manager
                # Create voice manager with adjusted sensitivity settings
                self.voice_manager = VoiceProcessManager(
                    volume_threshold=0.03,  # Slightly higher threshold for better noise rejection
                    speech_window=0.25      # Shorter window for faster response
                )
                if self.voice_manager.start():
                    print("Voice functionality enabled")
                    
                    # Start a thread to monitor the voice process output queue
                    self.voice_thread = threading.Thread(target=self._monitor_voice_process)
                    self.voice_thread.daemon = True
                    self.voice_thread.start()
                    
                    # Start with a greeting
                    # Use process_response instead of speak to ensure proper state handling
                    self.voice_manager.process_response("Hello! I'm Intuit. How can I help you today?")
                    
                    # Start listening after the greeting
                    QThread.msleep(2000)  # Wait for greeting to finish
                    self.voice_manager.listen()
                else:
                    print("Failed to start voice process")
            
            # Show the window
            self.window.show()
            
            self.running = True
            
            if block:
                # Run the application and block until it's closed
                return self.app.exec() == 0
            else:
                # Run the application in a separate thread
                self.app_thread = threading.Thread(target=self._run_app)
                self.app_thread.daemon = True
                self.app_thread.start()
                return True
                
        except Exception as e:
            print(f"Error starting GUI: {e}")
            self.cleanup()
            return False
            
    def stop(self):
        """Stop the GUI."""
        if not self.running:
            return
            
        self.running = False
        
        # Stop the voice manager
        if self.voice_manager:
            self.voice_manager.stop()
            self.voice_manager = None
        
        # Stop the speech processor
        if self.speech_processor:
            self.speech_processor.stop()
            self.speech_processor = None
            
        # Close the window
        if self.window:
            self.window.close()
            self.window = None
            
        # Quit the application
        if self.app:
            self.app.quit()
            self.app = None
            
        # Wait for the thread to finish
        if self.app_thread:
            self.app_thread.join(timeout=1.0)
            self.app_thread = None
            
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        
    def _run_app(self):
        """Run the application in a separate thread."""
        try:
            self.app.exec()
        except Exception as e:
            print(f"Error running GUI: {e}")
        finally:
            self.running = False
            
    def process_voice_data(self):
        """
        Process data from the voice process.
        
        Note: This method is called by the main event loop and should not block.
        The actual processing of voice data is done in the _monitor_voice_process thread.
        """
        # This method is intentionally left empty as voice processing is now
        # handled by the _monitor_voice_process thread.
        pass
    
    def _monitor_voice_process(self):
        """Monitor the voice process output queue and process messages."""
        print("Starting voice process monitor thread")
        while self.running:
            if not self.voice_manager or not self.voice_manager.is_running():
                time.sleep(0.1)
                continue
                
            # Get data from the voice process
            data = self.voice_manager.get_data()
            if data:
                data_type = data.get('type')
                print(f"Received voice process message: {data_type}")
                
                if data_type == 'metrics':
                    # Update the puck visualization with speech metrics
                    volume = data.get('volume', 0.0)
                    pitch = data.get('pitch', 0.5)
                    if self.speech_processor:
                        self.speech_processor.process_speech_data({
                            'volume': volume,
                            'pitch': pitch
                        })
                
                elif data_type == 'text':
                    # Handle recognized text
                    content = data.get('content')
                    if content:
                        print(f"Recognized: {content}")
                        
                        # Process the text with the agent
                        if self.agent:
                            print("Processing with Intuit agent...")
                            threading.Thread(
                                target=self._process_with_agent,
                                args=(content,)
                            ).start()
                
                elif data_type == 'speaking':
                    # Handle speaking state changes
                    state = data.get('state')
                    if state == 'start':
                        print("AI is speaking...")
                    elif state == 'stop':
                        interrupted = data.get('interrupted', False)
                        if interrupted:
                            print("AI speech was interrupted by user")
                        else:
                            print("AI finished speaking")
                        
                        # Start listening again after speaking stops
                        # This creates a continuous conversation loop
                        print("Starting to listen again after speaking...")
                        time.sleep(1)  # Short delay before listening again
                        self.voice_manager.listen()
                        print("Listen command sent")
                    
                elif data_type == 'ready_to_listen':
                    # Voice process is ready to listen again
                    print("Ready to listen...")
                    self.voice_manager.listen()
                
                elif data_type == 'metrics':
                    # Update the puck visualization with speech metrics
                    volume = data.get('volume', 0.0)
                    pitch = data.get('pitch', 0.5)
                    is_speech = data.get('is_speech', False)
                    
                    # If user is speaking while AI is speaking, interrupt the AI
                    # Use a higher threshold for interruption to avoid false positives
                    if is_speech and self.voice_manager.is_speaking and volume > 0.4:
                        print(f"User is speaking (volume: {volume:.2f}) - interrupting AI speech")
                        # Stop the current speech
                        if self.voice_manager.stop_speaking():
                            print("Successfully sent interruption signal to voice process")
                    
                    # Process speech metrics for visualization
                    if self.speech_processor:
                        self.speech_processor.process_speech_data({
                            'volume': volume,
                            'pitch': pitch
                        })
                
                elif data_type == 'error':
                    # Handle errors
                    message = data.get('message', 'Unknown error')
                    print(f"Voice process error: {message}")
            
            # Short sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def _process_with_agent(self, text):
        """Process text with the agent and send the response to the voice process."""
        try:
            # Import asyncio here to avoid circular imports
            import asyncio
            
            # Process the text with the agent
            response = asyncio.run(self.agent.run(text))
            
            # Send the response to the voice process
            if response and self.voice_manager:
                print(f"Agent response: {response}")
                # Use process_response instead of speak to trigger the ready_to_listen message
                self.voice_manager.process_response(response)
                
        except Exception as e:
            print(f"Error in agent processing: {e}")
            if self.voice_manager:
                self.voice_manager.process_response(f"Sorry, I encountered an error: {str(e)}")


# Singleton instance for easy access
_manager = None


def get_gui_manager() -> GUIManager:
    """
    Get the singleton GUI manager instance.
    
    Returns:
        The GUI manager instance.
    """
    global _manager
    if _manager is None:
        _manager = GUIManager()
    return _manager


def start_gui(block: bool = False, agent=None, enable_voice: bool = True) -> bool:
    """
    Start the GUI.
    
    Args:
        block: If True, block until the GUI is closed. If False, run the GUI
              in a separate thread and return immediately.
        agent: The agent to use for processing queries
        enable_voice: Whether to enable voice functionality
              
    Returns:
        True if the GUI was started successfully, False otherwise.
    """
    return get_gui_manager().start(block, agent, enable_voice)


def stop_gui():
    """Stop the GUI."""
    get_gui_manager().stop()


def process_speech(data: Dict[str, Any]):
    """
    Process speech data from the AI system.
    
    Args:
        data: Speech data dictionary with at least 'volume' and 'pitch' keys.
    """
    get_gui_manager().process_speech(data)


def set_speech_callback(callback: Optional[Callable[[Dict[str, Any]], None]]):
    """
    Set a callback function to be called when speech is detected.
    
    Args:
        callback: Function to call with speech data, or None to remove the callback.
    """
    get_gui_manager().set_speech_callback(callback)


def is_gui_running() -> bool:
    """
    Check if the GUI is running.
    
    Returns:
        True if the GUI is running, False otherwise.
    """
    return get_gui_manager().is_running()


# Example usage
if __name__ == "__main__":
    # Start the GUI
    start_gui(block=True)