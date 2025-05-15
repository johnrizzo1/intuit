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
        
    def start(self, block: bool = False) -> bool:
        """
        Start the GUI.
        
        Args:
            block: If True, block until the GUI is closed. If False, run the GUI
                  in a separate thread and return immediately.
                  
        Returns:
            True if the GUI was started successfully, False otherwise.
        """
        if self.running:
            return True
            
        try:
            # Create the application
            self.app = QApplication([])
            
            # Create the main window
            self.window = IntuitGUI()
            
            # Create and start the speech processor
            self.speech_processor = SpeechProcessor()
            self.speech_processor.speech_signal.connect(self.window.reactToSpeech)
            self.speech_processor.start()
            
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
            
    def process_speech(self, data: Dict[str, Any]):
        """
        Process speech data from the AI system.
        
        Args:
            data: Speech data dictionary with at least 'volume' and 'pitch' keys.
        """
        if self.running and self.speech_processor:
            self.speech_processor.process_speech_data(data)
            
    def set_speech_callback(self, callback: Optional[Callable[[Dict[str, Any]], None]]):
        """
        Set a callback function to be called when speech is detected.
        
        Args:
            callback: Function to call with speech data, or None to remove the callback.
        """
        if self.running and self.window:
            self.window.setSpeechCallback(callback)
            
    def is_running(self) -> bool:
        """
        Check if the GUI is running.
        
        Returns:
            True if the GUI is running, False otherwise.
        """
        return self.running


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


def start_gui(block: bool = False) -> bool:
    """
    Start the GUI.
    
    Args:
        block: If True, block until the GUI is closed. If False, run the GUI
              in a separate thread and return immediately.
              
    Returns:
        True if the GUI was started successfully, False otherwise.
    """
    return get_gui_manager().start(block)


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