"""
Light effects module for the isometric hockey puck visualization.

This module provides classes and functions to create dynamic lighting effects
that react to AI speech patterns.
"""

from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property, QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QGraphicsEffect


class PulseEffect(QGraphicsEffect):
    """
    A pulsing light effect that can be applied to QGraphicsItems.
    
    This effect creates a pulsing light that emanates from the center of the item,
    with intensity and color that can be dynamically adjusted based on speech patterns.
    """
    
    def __init__(self, parent=None):
        """Initialize the pulse effect with default values."""
        super().__init__(parent)
        self._intensity = 0.5  # Range from 0.0 to 1.0
        self._color = QColor(64, 224, 208)  # Default: Turquoise
        self._animation = None
        self._center = QPoint(0, 0)
        
    def intensity(self):
        """Get the current intensity of the pulse effect."""
        return self._intensity
        
    def setIntensity(self, intensity):
        """Set the intensity of the pulse effect."""
        self._intensity = max(0.0, min(1.0, intensity))  # Clamp between 0 and 1
        self.update()
        
    def color(self):
        """Get the current color of the pulse effect."""
        return self._color
        
    def setColor(self, color):
        """Set the color of the pulse effect."""
        self._color = color
        self.update()
        
    def setCenter(self, center):
        """Set the center point of the pulse effect."""
        self._center = center
        self.update()
        
    # Define properties for animation
    intensity_prop = Property(float, intensity, setIntensity)
    
    def draw(self, painter):
        """Draw the pulse effect."""
        if self.sourceIsPixmap():
            pixmap = self.sourcePixmap(Qt.LogicalCoordinates)
            painter.drawPixmap(0, 0, pixmap)
        else:
            self.drawSource(painter)
            
        # Draw the pulse effect
        source_rect = self.boundingRectFor(self.sourceBoundingRect())
        center_x = source_rect.width() / 2
        center_y = source_rect.height() / 2
        
        # Create a radial gradient for the pulse
        gradient = QRadialGradient(center_x, center_y, source_rect.width() / 2)
        
        # Adjust color based on intensity
        color = QColor(self._color)
        color.setAlphaF(self._intensity * 0.7)  # Adjust alpha for visibility
        
        gradient.setColorAt(0, color)
        color.setAlphaF(0)
        gradient.setColorAt(1, color)
        
        painter.setOpacity(self._intensity)
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(source_rect)
        
    def startPulse(self, duration=1000, min_intensity=0.2, max_intensity=0.8):
        """
        Start a pulsing animation.
        
        Args:
            duration: Duration of one pulse cycle in milliseconds
            min_intensity: Minimum intensity value (0.0 to 1.0)
            max_intensity: Maximum intensity value (0.0 to 1.0)
        """
        if self._animation:
            self._animation.stop()
            
        self._animation = QPropertyAnimation(self, b"intensity_prop")
        self._animation.setDuration(duration)
        self._animation.setStartValue(min_intensity)
        self._animation.setEndValue(max_intensity)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._animation.setLoopCount(-1)  # Infinite loop
        self._animation.setDirection(QPropertyAnimation.Direction.Forward)
        self._animation.start()
        
    def stopPulse(self):
        """Stop the pulsing animation."""
        if self._animation:
            self._animation.stop()
            self._animation = None


class RippleEffect(QGraphicsEffect):
    """
    A rippling light effect that can be applied to QGraphicsItems.
    
    This effect creates concentric rings of light that emanate from the center of the item,
    with speed and intensity that can be dynamically adjusted based on speech patterns.
    """
    
    def __init__(self, parent=None):
        """Initialize the ripple effect with default values."""
        super().__init__(parent)
        self._intensity = 0.5  # Range from 0.0 to 1.0
        self._color = QColor(64, 224, 208)  # Default: Turquoise
        self._ripple_count = 3
        self._ripple_width = 20
        self._ripple_speed = 1.0
        self._animations = []
        self._ripple_positions = [0.0] * self._ripple_count  # Position of each ripple (0.0 to 1.0)
        
    def intensity(self):
        """Get the current intensity of the ripple effect."""
        return self._intensity
        
    def setIntensity(self, intensity):
        """Set the intensity of the ripple effect."""
        self._intensity = max(0.0, min(1.0, intensity))  # Clamp between 0 and 1
        self.update()
        
    def color(self):
        """Get the current color of the ripple effect."""
        return self._color
        
    def setColor(self, color):
        """Set the color of the ripple effect."""
        self._color = color
        self.update()
        
    def setRippleSpeed(self, speed):
        """Set the speed of the ripple animation."""
        self._ripple_speed = max(0.1, min(3.0, speed))  # Clamp between 0.1 and 3.0
        
    def draw(self, painter):
        """Draw the ripple effect."""
        if self.sourceIsPixmap():
            pixmap = self.sourcePixmap(Qt.LogicalCoordinates)
            painter.drawPixmap(0, 0, pixmap)
        else:
            self.drawSource(painter)
            
        # Draw the ripple effect
        source_rect = self.boundingRectFor(self.sourceBoundingRect())
        center_x = source_rect.width() / 2
        center_y = source_rect.height() / 2
        max_radius = source_rect.width() / 2
        
        painter.setOpacity(self._intensity)
        painter.setPen(Qt.NoPen)
        
        # Draw each ripple
        for pos in self._ripple_positions:
            if pos > 0:
                radius = pos * max_radius
                color = QColor(self._color)
                # Fade out as the ripple expands
                alpha = (1.0 - pos) * self._intensity * 255
                color.setAlpha(int(alpha))
                painter.setBrush(color)
                painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
                
    def updateRipples(self, delta_time):
        """
        Update the ripple positions based on elapsed time.
        
        Args:
            delta_time: Time elapsed since last update in seconds
        """
        # Update each ripple position
        for i in range(self._ripple_count):
            self._ripple_positions[i] += delta_time * self._ripple_speed * 0.5
            
            # Reset ripple when it reaches the edge
            if self._ripple_positions[i] >= 1.0:
                # Stagger the ripples
                self._ripple_positions[i] = i * (1.0 / self._ripple_count)
                
        self.update()
        
    def startRipple(self):
        """Start the ripple animation."""
        # Initialize ripple positions to be staggered
        for i in range(self._ripple_count):
            self._ripple_positions[i] = i * (1.0 / self._ripple_count)
            
    def reactToSpeech(self, volume, pitch=0.5):
        """
        React to speech by adjusting the ripple effect.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch level (0.0 to 1.0)
        """
        # Adjust intensity based on volume
        self.setIntensity(volume)
        
        # Adjust speed based on pitch
        self.setRippleSpeed(0.5 + pitch * 2.5)
        
        # Adjust color based on pitch (blue for low, green for mid, red for high)
        if pitch < 0.33:
            self.setColor(QColor(100, 100, 255))  # Blue-ish
        elif pitch < 0.66:
            self.setColor(QColor(100, 255, 100))  # Green-ish
        else:
            self.setColor(QColor(255, 100, 100))  # Red-ish