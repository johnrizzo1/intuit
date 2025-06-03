#!/usr/bin/env python3
"""
Standalone GUI script for the Intuit AI Hockey Puck.

This script contains all the necessary code to run the hockey puck GUI
without relying on imports from the package.
"""

import sys
import time
import random
import threading
import math  # Added for math functions used in SmokeEffect
import multiprocessing
import json
import os
from PySide6.QtCore import Qt, QSize, Signal, Slot, QTimer, QPoint, QPointF, QRectF
from PySide6.QtGui import (
    QIcon, QAction, QColor, QPainter, QPainterPath, QRegion,
    QBrush, QPen, QLinearGradient, QRadialGradient
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsDropShadowEffect,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsItem, QStyleOptionGraphicsItem, QMenu
)
# Import the voice process manager
try:
    # When imported as a module
    from .voice_process import VoiceProcessManager
except ImportError:
    # When run as a standalone script
    import os
    import sys
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from intuit.utils.voice_process import VoiceProcessManager



# Light Effects
class PulseEffect:
    """
    Creates a pulsing light effect.
    
    This effect causes the light to pulse in intensity based on the volume
    and pitch of speech.
    """
    
    def __init__(self):
        """Initialize the pulse effect."""
        self.intensity = 0.0
        self.target_intensity = 0.0
        self.color = QColor(0, 120, 255)  # Default blue color
        self.transition_speed = 0.05  # Slower transition speed (was 0.1)
        
    def update(self, volume: float, pitch: float, dt: float) -> None:
        """
        Update the pulse effect based on speech volume and pitch.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
            dt: Time delta since last update
        """
        # Set target intensity based on volume
        self.target_intensity = volume
        
        # Adjust color based on pitch
        if pitch > 0.7:
            # High pitch: more blue
            self.color = QColor(0, 100, 255)
        elif pitch > 0.4:
            # Medium pitch: purple
            self.color = QColor(100, 0, 255)
        else:
            # Low pitch: red
            self.color = QColor(255, 0, 100)
            
        # Smoothly transition to target intensity
        diff = self.target_intensity - self.intensity
        self.intensity += diff * self.transition_speed
        
    def apply(self, painter: QPainter, rect: QRectF) -> None:
        """
        Apply the pulse effect to the painter.
        
        Args:
            painter: QPainter to apply the effect to
            rect: Rectangle to apply the effect within
        """
        if self.intensity <= 0.01:
            return
            
        # Create a radial gradient for the glow
        center = rect.center()
        radius = rect.width() / 2
        gradient = QRadialGradient(center, radius * 1.2)  # Reduced radius multiplier (was 1.5)
        
        # Set up the gradient colors
        color = QColor(self.color)
        color.setAlphaF(self.intensity * 0.6)  # Reduced alpha (was 0.7)
        
        gradient.setColorAt(0.0, color)
        color.setAlphaF(0.0)
        gradient.setColorAt(1.0, color)
        
        # Draw the glow
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(rect.adjusted(-radius/3, -radius/3, radius/3, radius/3))  # Reduced adjustment (was radius/2)
        painter.restore()


class RippleEffect:
    """
    Creates a rippling light effect.
    
    This effect causes light to ripple outward from the center based on the
    volume and pitch of speech.
    """
    
    def __init__(self):
        """Initialize the ripple effect."""
        self.ripples = []
        self.color = QColor(0, 200, 255)  # Default cyan color
        self.spawn_timer = 0.0
        self.spawn_interval = 0.5  # Time between ripples
        self.min_spawn_interval = 0.1
        self.max_spawn_interval = 1.0
        
    def update(self, volume: float, pitch: float, dt: float) -> None:
        """
        Update the ripple effect based on speech volume and pitch.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
            dt: Time delta since last update
        """
        # Adjust spawn interval based on volume
        if volume > 0.1:
            self.spawn_interval = self.max_spawn_interval - volume * (self.max_spawn_interval - self.min_spawn_interval)
            self.spawn_timer += dt
            
            # Spawn new ripples
            if self.spawn_timer >= self.spawn_interval:
                self.spawn_timer = 0.0
                
                # Adjust color based on pitch
                if pitch > 0.7:
                    # High pitch: cyan
                    self.color = QColor(0, 200, 255)
                elif pitch > 0.4:
                    # Medium pitch: green
                    self.color = QColor(0, 255, 100)
                else:
                    # Low pitch: yellow
                    self.color = QColor(255, 200, 0)
                    
                # Add a new ripple
                self.ripples.append({
                    'radius': 0.0,
                    'max_radius': 1.0 + volume * 0.5,  # Larger ripples for louder sounds
                    'speed': 0.5 + volume * 1.0,  # Faster ripples for louder sounds
                    'alpha': 0.7,
                    'color': QColor(self.color)
                })
        
        # Update existing ripples
        i = 0
        while i < len(self.ripples):
            ripple = self.ripples[i]
            ripple['radius'] += ripple['speed'] * dt
            
            # Fade out as the ripple expands
            progress = ripple['radius'] / ripple['max_radius']
            ripple['alpha'] = 0.7 * (1.0 - progress)
            
            # Remove ripples that have expanded beyond their max radius
            if ripple['radius'] >= ripple['max_radius']:
                self.ripples.pop(i)
            else:
                i += 1
                
    def apply(self, painter: QPainter, rect: QRectF) -> None:
        """
        Apply the ripple effect to the painter.
        
        Args:
            painter: QPainter to apply the effect to
            rect: Rectangle to apply the effect within
        """
        if not self.ripples:
            return
            
        center = rect.center()
        base_radius = rect.width() / 2
        
        painter.save()
        painter.setPen(Qt.NoPen)
        
        for ripple in self.ripples:
            color = QColor(ripple['color'])
            color.setAlphaF(ripple['alpha'])
            
            painter.setBrush(Qt.NoBrush)
            pen = QPen(color, 2)
            painter.setPen(pen)
            
            radius = base_radius * (1.0 + ripple['radius'])
            painter.drawEllipse(center, radius, radius)
            
        painter.restore()


class SmokeEffect:
    """
    Creates a smoke/energy swirling effect.
    
    This effect simulates smoke or energy swirling around inside a transparent globe,
    creating a dynamic, organic visual representation of AI activity.
    """
    
    def __init__(self):
        """Initialize the smoke effect."""
        self.particles = []
        self.max_particles = 30
        self.color = QColor(0, 150, 255)  # Default blue color
        self.spawn_timer = 0.0
        self.spawn_interval = 0.1
        self.base_intensity = 0.0
        self.target_intensity = 0.0
        self.transition_speed = 0.03  # Very slow transition
        
    def update(self, volume: float, pitch: float, dt: float) -> None:
        """
        Update the smoke effect based on speech volume and pitch.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
            dt: Time delta since last update
        """
        # Set target intensity based on volume
        self.target_intensity = volume
        
        # Smoothly transition base intensity
        diff = self.target_intensity - self.base_intensity
        self.base_intensity += diff * self.transition_speed
        
        # Adjust color based on pitch
        if pitch > 0.7:
            # High pitch: blue/cyan
            self.color = QColor(0, 150, 255)
        elif pitch > 0.4:
            # Medium pitch: purple
            self.color = QColor(130, 0, 255)
        else:
            # Low pitch: red/orange
            self.color = QColor(255, 60, 0)
            
        # Spawn new particles based on volume
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.particles) < self.max_particles:
            self.spawn_timer = 0.0
            
            # Only spawn particles if there's some volume
            if self.base_intensity > 0.1:
                # Calculate random position within the globe
                angle = random.uniform(0, 2 * 3.14159)
                distance = random.uniform(0, 0.8)  # Keep within 80% of radius
                
                # Add some randomness to the color
                color_var = 30
                r = max(0, min(255, self.color.red() + random.randint(-color_var, color_var)))
                g = max(0, min(255, self.color.green() + random.randint(-color_var, color_var)))
                b = max(0, min(255, self.color.blue() + random.randint(-color_var, color_var)))
                
                # Create a new particle
                self.particles.append({
                    'x': math.cos(angle) * distance,
                    'y': math.sin(angle) * distance,
                    'size': random.uniform(0.05, 0.15),
                    'life': 1.0,
                    'decay': random.uniform(0.2, 0.4),
                    'vx': random.uniform(-0.2, 0.2),
                    'vy': random.uniform(-0.2, 0.2),
                    'color': QColor(r, g, b),
                    'alpha': random.uniform(0.5, 0.8) * self.base_intensity
                })
        
        # Update existing particles
        i = 0
        while i < len(self.particles):
            particle = self.particles[i]
            
            # Update position with swirling motion
            cx = particle['x']  # Current x position relative to center
            cy = particle['y']  # Current y position relative to center
            dist = math.sqrt(cx*cx + cy*cy)
            
            # Add swirling motion (stronger near center, weaker near edges)
            swirl_factor = max(0, 0.8 - dist) * 2.0
            particle['vx'] += -cy * swirl_factor * dt
            particle['vy'] += cx * swirl_factor * dt
            
            # Apply velocity
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            
            # Keep particles inside the globe
            dist = math.sqrt(particle['x']*particle['x'] + particle['y']*particle['y'])
            if dist > 0.9:  # If near the edge
                # Normalize and scale back
                particle['x'] = (particle['x'] / dist) * 0.9
                particle['y'] = (particle['y'] / dist) * 0.9
                
                # Bounce off the edge (reduce velocity and reverse direction)
                particle['vx'] *= -0.5
                particle['vy'] *= -0.5
            
            # Reduce life
            particle['life'] -= particle['decay'] * dt
            
            # Remove dead particles
            if particle['life'] <= 0:
                self.particles.pop(i)
            else:
                i += 1
                
    def apply(self, painter: QPainter, rect: QRectF) -> None:
        """
        Apply the smoke effect to the painter.
        
        Args:
            painter: QPainter to apply the effect to
            rect: Rectangle to apply the effect within
        """
        if not self.particles:
            return
            
        center = rect.center()
        radius = rect.width() / 2
        
        painter.save()
        
        for particle in self.particles:
            # Calculate position in screen coordinates
            x = center.x() + particle['x'] * radius
            y = center.y() + particle['y'] * radius
            size = particle['size'] * radius
            
            # Set color with alpha based on life
            color = QColor(particle['color'])
            color.setAlphaF(particle['alpha'] * particle['life'])
            
            # Draw the particle as a soft gradient circle
            gradient = QRadialGradient(x, y, size)
            gradient.setColorAt(0, color)
            color.setAlphaF(0)
            gradient.setColorAt(1, color)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(x, y), size, size)
        
        painter.restore()


# Hockey Puck Widget
class HockeyPuckItem(QGraphicsEllipseItem):
    """
    A QGraphicsEllipseItem that represents a hockey puck.
    
    This item draws an isometric hockey puck with dynamic lighting effects.
    """
    
    def __init__(self, x: float, y: float, width: float, height: float):
        """
        Initialize the hockey puck item.
        
        Args:
            x: X coordinate
            y: Y coordinate
            width: Width of the puck
            height: Height of the puck
        """
        super().__init__(x, y, width, height)
        
        # Set up puck properties
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        
        # Set up colors
        self.base_color = QColor(30, 30, 30)  # Dark gray
        self.highlight_color = QColor(60, 60, 60)  # Light gray
        self.shadow_color = QColor(10, 10, 10)  # Almost black
        
        # Set up lighting effects
        self.pulse_effect = PulseEffect()
        self.ripple_effect = RippleEffect()
        self.smoke_effect = SmokeEffect()  # New smoke effect
        self.current_effect = self.smoke_effect  # Set smoke effect as default
        
        # Set up animation properties
        self.last_update_time = time.time()
        self.volume = 0.0
        self.pitch = 0.0
        self.test_animation = True  # Enable test animation by default
        self.test_animation_time = 0.0
        
    def toggle_effect(self) -> None:
        """Toggle between smoke, pulse and ripple effects."""
        if self.current_effect is self.smoke_effect:
            self.current_effect = self.pulse_effect
        elif self.current_effect is self.pulse_effect:
            self.current_effect = self.ripple_effect
        else:
            self.current_effect = self.smoke_effect
            
    def toggle_test_animation(self) -> None:
        """Toggle the test animation on/off."""
        self.test_animation = not self.test_animation
        self.test_animation_time = 0.0
        
    def update_animation(self) -> None:
        """Update the animation state."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # If test animation is enabled, generate synthetic volume and pitch
        if self.test_animation:
            self.test_animation_time += dt
            
            # Create a pulsing pattern
            self.volume = (1 + math.sin(self.test_animation_time * 2)) / 2
            self.pitch = (1 + math.sin(self.test_animation_time * 1.5)) / 2
            
        # Update the current effect
        self.current_effect.update(self.volume, self.pitch, dt)
        
        # Request a redraw
        self.update()
        
    def set_speech_data(self, volume: float, pitch: float) -> None:
        """
        Set the speech data for the lighting effects.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        self.pitch = max(0.0, min(1.0, pitch))
        
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None) -> None:
        """
        Paint the hockey puck.
        
        Args:
            painter: QPainter to use for drawing
            option: Style options
            widget: Widget being painted on
        """
        # Get the rect
        rect = self.rect()
        
        # Calculate dimensions
        width = rect.width()
        height = rect.height()
        center_x = rect.center().x()
        center_y = rect.center().y()
        
        # Set up the painter
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        # Draw the base (bottom of the puck)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.base_color))
        painter.drawEllipse(rect)
        
        # Draw the top of the puck (slightly smaller)
        top_rect = QRectF(
            rect.x() + width * 0.05,
            rect.y() + height * 0.05,
            width * 0.9,
            height * 0.9
        )
        
        # Create a gradient for the top
        gradient = QLinearGradient(
            top_rect.topLeft(),
            top_rect.bottomRight()
        )
        gradient.setColorAt(0, self.highlight_color)
        gradient.setColorAt(1, self.shadow_color)
        
        painter.setBrush(gradient)
        painter.drawEllipse(top_rect)
        
        # Apply the current lighting effect
        self.current_effect.apply(painter, rect)


class HockeyPuckView(QGraphicsView):
    """
    A QGraphicsView that displays the hockey puck.
    
    This view provides a container for the hockey puck item and handles
    user interaction.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the hockey puck view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set up the scene with transparent background
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent background
        self.setScene(self.scene)
        
        # Set up the view
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent background
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        
        # Make sure there's no border or frame
        self.setStyleSheet("background: transparent; border: none;")
        self.viewport().setStyleSheet("background: transparent; border: none;")
        
        # Create the hockey puck item
        self.puck_size = 200
        self.puck_item = HockeyPuckItem(0, 0, self.puck_size, self.puck_size)
        self.scene.addItem(self.puck_item)
        
        # Set up the animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        # Set up mouse tracking
        self.setMouseTracking(True)
        self.drag_position = None
        
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        
        # Update the scene rect to match the view size
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        
        # Center the puck in the view
        self.puck_item.setPos(
            (self.width() - self.puck_size) / 2,
            (self.height() - self.puck_size) / 2
        )
        
    def update_animation(self):
        """Update the animation."""
        self.puck_item.update_animation()
        
    def set_speech_data(self, volume: float, pitch: float):
        """
        Set the speech data for the lighting effects.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
        """
        self.puck_item.set_speech_data(volume, pitch)
        
    def toggle_effect(self):
        """Toggle between smoke, pulse and ripple effects."""
        self.puck_item.toggle_effect()
        
    def toggle_test_animation(self):
        """Toggle the test animation on/off."""
        self.puck_item.toggle_test_animation()
        
    def mousePressEvent(self, event):
        """Handle mouse press events for dragging."""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging."""
        if event.buttons() & Qt.LeftButton and self.drag_position is not None:
            # Calculate the movement delta
            delta = event.position() - self.drag_position
            self.drag_position = event.position()
            
            # Move the window
            self.window().move(self.window().pos() + QPoint(int(delta.x()), int(delta.y())))
            event.accept()
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for dragging."""
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# Main GUI Window
class IntuitGUI(QWidget):
    """
    Main GUI window for the Intuit AI.
    
    This window displays the hockey puck visualization and handles user input.
    """
    
    def __init__(self, config=None):
        """
        Initialize the main GUI window.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Set up window properties
        self.setWindowTitle("Intuit AI")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # Remove spacing between widgets
        self.setLayout(layout)
        
        # Create the hockey puck view
        self.puck_view = HockeyPuckView(self)
        layout.addWidget(self.puck_view)
        
        # Set up the window size
        self.resize(250, 250)
        
        # No drop shadow effect to avoid visible box
        
        # Set up the speech callback
        self.speech_callback = None
        
        # Process configuration
        voice_enabled = False
        voice_language = "en"
        voice_slow = False
        
        if config:
            voice_enabled = config.get("voice_enabled", False)
            voice_language = config.get("voice_language", "en")
            voice_slow = config.get("voice_slow", False)
        
        # Set up voice process manager
        self.voice_manager = VoiceProcessManager(language=voice_language, slow=voice_slow)
        self.voice_active = False
        self.voice_timer = QTimer(self)
        self.voice_timer.timeout.connect(self.process_voice_data)
        self.voice_timer.start(50)  # Check for voice data every 50ms
        
        # Auto-start voice if enabled in config
        if voice_enabled:
            # Use a timer to start voice after the GUI is fully initialized
            QTimer.singleShot(500, self.toggle_voice)
        
    def reactToSpeech(self, volume: float, pitch: float):
        """
        React to speech by updating the lighting effects.
        
        Args:
            volume: Speech volume (0.0 to 1.0)
            pitch: Speech pitch (0.0 to 1.0)
        """
        self.puck_view.set_speech_data(volume, pitch)
        
        # Call the speech callback if set
        if self.speech_callback:
            self.speech_callback({
                'volume': volume,
                'pitch': pitch
            })
            
    def setSpeechCallback(self, callback):
        """
        Set a callback function to be called when speech is detected.
        
        Args:
            callback: Function to call with speech data, or None to remove the callback
        """
        self.speech_callback = callback
        
    def process_voice_data(self):
        """Process data from the voice process."""
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
                if content:
                    print(f"Recognized: {content}")
                    
                    # Simulate querying the agent/model
                    print("Querying AI model with recognized text...")
                    
                    # In a real implementation, this would call the Intuit AI agent
                    # For demonstration, we'll simulate different responses based on the input
                    if "weather" in content.lower():
                        response = "The weather today is sunny with a high of 75 degrees."
                    elif "time" in content.lower():
                        response = "The current time is " + time.strftime("%I:%M %p") + "."
                    elif "hello" in content.lower() or "hi" in content.lower():
                        response = "Hello there! How can I assist you today?"
                    elif "help" in content.lower():
                        response = "I can help you with various tasks. Try asking about the weather, time, or say hello!"
                    else:
                        response = f"I received your request about '{content}'. In a real implementation, this would be processed by the Intuit AI agent."
                    
                    print(f"AI response: {response}")
                    
                    # Convert the text response to voice
                    if self.voice_active:
                        self.voice_manager.speak(response)
                else:
                    # Empty content means speech wasn't recognized
                    # We'll continue listening in the 'speaking' handler after the error message
                    pass
                
            elif data_type == 'process_text':
                # Handle text that needs to be processed by the AI model
                content = data.get('content', '')
                print(f"Processing: {content}")
                
                # In a real implementation, this would call the Intuit AI agent
                # For now, we'll use the provided content
                if content.startswith("I could not understand"):
                    # This is a notification from the voice process
                    if self.voice_active:
                        self.voice_manager.speak(content)
            
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
    
    def toggle_voice(self):
        """Toggle voice functionality on/off."""
        if self.voice_active:
            # Stop the voice process
            self.voice_manager.stop()
            self.voice_active = False
            print("Voice interface deactivated")
        else:
            # Start the voice process
            if self.voice_manager.start():
                self.voice_active = True
                print("Voice interface activated")
                
                # Welcome message
                self.voice_manager.speak("Voice interface activated. How can I help you?")
                
                # Start a timer to begin listening after the welcome message
                QTimer.singleShot(2000, lambda: self.voice_manager.listen(timeout=5.0, process=True))
    
    def contextMenuEvent(self, event):
        """Show context menu on right-click."""
        menu = QMenu(self)
        
        # Add menu items
        toggle_effect_action = menu.addAction("Toggle Effect")
        toggle_animation_action = menu.addAction("Toggle Animation")
        toggle_voice_action = menu.addAction("Toggle Voice")
        menu.addSeparator()
        exit_action = menu.addAction("Exit")
        
        # Update voice action text based on current state
        if self.voice_active:
            toggle_voice_action.setText("Disable Voice")
        else:
            toggle_voice_action.setText("Enable Voice")
        
        # Show the menu and get the selected action
        action = menu.exec(self.mapToGlobal(event.pos()))
        
        # Handle the selected action
        if action == toggle_effect_action:
            self.puck_view.toggle_effect()
        elif action == toggle_animation_action:
            self.puck_view.toggle_test_animation()
        elif action == toggle_voice_action:
            self.toggle_voice()
        elif action == exit_action:
            self.close()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            # Close the window
            self.close()
        elif event.key() == Qt.Key_Space:
            # Toggle between pulse and ripple effects
            self.puck_view.toggle_effect()
        elif event.key() == Qt.Key_T:
            # Toggle test animation
            self.puck_view.toggle_test_animation()
        elif event.key() == Qt.Key_V:
            # Toggle voice functionality
            self.toggle_voice()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop the voice process if it's running
        if self.voice_active:
            self.voice_manager.stop()
            self.voice_active = False
        
        # Accept the close event
        event.accept()


# Main function
def main():
    """Run the standalone GUI."""
    # Create the application
    app = QApplication(sys.argv)
    
    # Load configuration if provided
    config = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        try:
            with open(sys.argv[1], 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Create the main window
    window = IntuitGUI(config)
    window.show()
    
    # Print instructions
    print("Hockey Puck GUI")
    print("---------------")
    print("Controls:")
    print("- Drag with mouse to move the window")
    print("- Press Space to toggle between smoke, pulse and ripple effects")
    print("- Press T to stop/restart animation (enabled by default)")
    print("- Press V to toggle voice interface")
    print("- Right-click for menu")
    print("- Press Escape to close")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    import math  # Import math here for the test animation
    main()