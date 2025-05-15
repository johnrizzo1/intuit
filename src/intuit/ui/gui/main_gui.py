"""
Main GUI module for the Intuit AI.

This module provides the main GUI window for the Intuit AI,
featuring an isometric hockey puck visualization with reactive lighting effects.
"""

import sys
import asyncio
import threading
from typing import Optional, Callable, Dict, Any

from PySide6.QtCore import Qt, QSize, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QIcon, QAction, QColor, QPainter, QPainterPath, QRegion
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsDropShadowEffect
)

from . import puck_widget
from puck_widget import HockeyPuckView


class IntuitGUI(QWidget):
    """
    Main GUI window for the Intuit AI.
    
    This class provides a frameless, circular window shaped like a hockey puck
    with dynamic lighting effects that react to AI speech.
    """
    
    # Signal to update the GUI from a non-GUI thread
    speech_signal = Signal(float, float)
    
    def __init__(self, parent=None):
        """Initialize the main GUI window."""
        super().__init__(parent)
        
        # Window setup - frameless and always on top
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set size
        self.puck_size = 300
        self.setFixedSize(self.puck_size, self.puck_size)
        
        # Create the main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the hockey puck view
        self.puck_view = HockeyPuckView()
        self.main_layout.addWidget(self.puck_view)
        
        # Add drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        
        # Connect signals
        self.speech_signal.connect(self.onSpeechData)
        
        # Test animation timer
        self.test_timer = None
        self.test_animation_active = False
        
        # Speech callback
        self.speech_callback = None
        
        # Variables for dragging the window
        self.dragging = False
        self.drag_position = QPoint()
        
        # Create a circular mask for the window
        self.createCircularMask()
        
    def createCircularMask(self):
        """Set a circular mask for the window."""
        path = QPainterPath()
        path.addEllipse(0, 0, self.puck_size, self.puck_size)
        region = QRegion(path.toFillPolygon().toPolygon())
        super().setMask(region)
        
    def paintEvent(self, event):
        """Paint the window background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the circular background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(40, 44, 52))  # Dark base color
        painter.drawEllipse(0, 0, self.puck_size, self.puck_size)
        
    def mousePressEvent(self, event):
        """Handle mouse press events for dragging the window."""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging the window."""
        if event.buttons() & Qt.LeftButton and self.dragging:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for dragging the window."""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()
            
    def toggleEffect(self):
        """Toggle between pulse and ripple effects."""
        self.puck_view.toggleEffectType()
        
    def testAnimation(self):
        """Start or stop a test animation."""
        if self.test_animation_active:
            # Stop the test animation
            if self.test_timer:
                self.test_timer.stop()
                self.test_timer = None
            self.test_animation_active = False
        else:
            # Start the test animation
            self.test_animation_active = True
            self.test_timer = QTimer(self)
            self.test_timer.timeout.connect(self.updateTestAnimation)
            self.test_timer.start(100)  # Update every 100ms
            
            # Initialize test animation state
            self.test_volume = 0.0
            self.test_volume_direction = 0.1
            self.test_pitch = 0.0
            self.test_pitch_direction = 0.05
            
    def updateTestAnimation(self):
        """Update the test animation state."""
        # Update test volume (oscillate between 0.0 and 1.0)
        self.test_volume += self.test_volume_direction
        if self.test_volume >= 1.0 or self.test_volume <= 0.0:
            self.test_volume_direction *= -1
            self.test_volume = max(0.0, min(1.0, self.test_volume))
            
        # Update test pitch (oscillate between 0.0 and 1.0)
        self.test_pitch += self.test_pitch_direction
        if self.test_pitch >= 1.0 or self.test_pitch <= 0.0:
            self.test_pitch_direction *= -1
            self.test_pitch = max(0.0, min(1.0, self.test_pitch))
            
        # Update the puck view
        self.puck_view.reactToSpeech(self.test_volume, self.test_pitch)
        
    @Slot(float, float)
    def onSpeechData(self, volume: float, pitch: float):
        """
        Handle speech data updates.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch level (0.0 to 1.0)
        """
        self.puck_view.reactToSpeech(volume, pitch)
        
    def reactToSpeech(self, volume: float, pitch: float = 0.5):
        """
        React to speech by updating the visualization.
        
        This method can be called from any thread.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch level (0.0 to 1.0)
        """
        # Emit signal to update the GUI from the main thread
        self.speech_signal.emit(volume, pitch)
        
    def setSpeechCallback(self, callback):
        """
        Set a callback function to be called when speech is detected.
        
        Args:
            callback: Function to call with speech data
        """
        self.speech_callback = callback
        
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any active animations
        if self.test_timer:
            self.test_timer.stop()
            self.test_timer = None
            
        # Accept the close event
        event.accept()
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Close the window when Escape is pressed
        if event.key() == Qt.Key_Escape:
            self.close()
        # Toggle the effect when Space is pressed
        elif event.key() == Qt.Key_Space:
            self.toggleEffect()
        # Start/stop test animation when T is pressed
        elif event.key() == Qt.Key_T:
            self.testAnimation()
        else:
            super().keyPressEvent(event)


def run_gui():
    """Run the GUI as a standalone application."""
    app = QApplication(sys.argv)
    window = IntuitGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()