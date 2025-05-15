"""
Isometric hockey puck widget for the Intuit AI visualization.

This module provides a QGraphicsView-based widget that renders an isometric hockey puck
with dynamic lighting effects that react to AI speech.
"""

import time
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRectF
from PySide6.QtGui import (
    QPainter, QBrush, QPen, QColor, QLinearGradient, 
    QRadialGradient, QPainterPath
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, 
    QGraphicsItem, QStyleOptionGraphicsItem, QWidget
)

from . import light_effects
from .light_effects import PulseEffect, RippleEffect


class HockeyPuckItem(QGraphicsEllipseItem):
    """
    A QGraphicsEllipseItem that represents an isometric hockey puck.
    
    This class renders a circular puck with 3D isometric effects and
    can be animated with dynamic lighting.
    """
    
    def __init__(self, x, y, width, height, parent=None):
        """
        Initialize the hockey puck item.
        
        Args:
            x: X-coordinate of the top-left corner
            y: Y-coordinate of the top-left corner
            width: Width of the puck
            height: Height of the puck
            parent: Parent item
        """
        super().__init__(x, y, width, height, parent)
        
        # Set up appearance
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        
        # Colors
        self.base_color = QColor(40, 44, 52)  # Dark base color
        self.highlight_color = QColor(60, 64, 72)  # Slightly lighter for highlights
        self.shadow_color = QColor(20, 22, 26)  # Darker for shadows
        self.edge_color = QColor(70, 74, 82)  # Light color for edges
        
        # Effects
        self.pulse_effect = PulseEffect()
        self.ripple_effect = RippleEffect()
        self.current_effect = None  # Will be set when an effect is activated
        
        # Animation state
        self.last_update_time = time.time()
        
    def paint(self, painter, option, widget):
        """
        Paint the hockey puck with isometric effects.
        
        Args:
            painter: QPainter to use for drawing
            option: Style options
            widget: Widget being painted on
        """
        # Disable default QGraphicsEllipseItem drawing
        option = QStyleOptionGraphicsItem(option)
        # In PySide6, we need to use the Qt.ItemFlag enum instead of State_Selected
        # Just skip this line as we're overriding the drawing anyway
        
        # Get the rect in local coordinates
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Enable antialiasing for smoother edges
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Draw the puck body (main circle)
        # Create a gradient for 3D effect (top to bottom)
        gradient = QLinearGradient(0, rect.top(), 0, rect.bottom())
        gradient.setColorAt(0, self.highlight_color)
        gradient.setColorAt(0.5, self.base_color)
        gradient.setColorAt(1, self.shadow_color)
        
        painter.setBrush(gradient)
        painter.setPen(QPen(self.edge_color, 1))
        painter.drawEllipse(rect)
        
        # Draw the top edge (ellipse with smaller height)
        top_height = height * 0.1
        top_rect = QRectF(rect.x(), rect.y(), width, top_height * 2)
        
        # Create a gradient for the top (lighter in center)
        top_gradient = QRadialGradient(rect.center().x(), rect.y() + top_height, width / 2)
        top_gradient.setColorAt(0, self.highlight_color)
        top_gradient.setColorAt(1, self.base_color)
        
        painter.setBrush(top_gradient)
        painter.setPen(QPen(self.edge_color, 1))
        painter.drawEllipse(top_rect)
        
        # Draw a subtle highlight on the top surface
        highlight_path = QPainterPath()
        highlight_path.addEllipse(rect.x() + width * 0.2, rect.y() + height * 0.2, 
                                 width * 0.6, height * 0.2)
        
        highlight_gradient = QLinearGradient(0, rect.top() + height * 0.2, 
                                           0, rect.top() + height * 0.4)
        highlight_color = QColor(self.highlight_color)
        highlight_color.setAlpha(40)  # Subtle highlight
        highlight_gradient.setColorAt(0, highlight_color)
        highlight_color.setAlpha(10)
        highlight_gradient.setColorAt(1, highlight_color)
        
        painter.setBrush(highlight_gradient)
        painter.setPen(Qt.NoPen)
        painter.drawPath(highlight_path)
        
    def boundingRect(self):
        """Return the bounding rectangle of the puck, including any effects."""
        base_rect = super().boundingRect()
        # Add some margin for effects
        margin = max(base_rect.width(), base_rect.height()) * 0.2
        return base_rect.adjusted(-margin, -margin, margin, margin)
        
    def activatePulseEffect(self, color=None):
        """
        Activate the pulse effect on the puck.
        
        Args:
            color: Optional QColor for the pulse effect
        """
        if color:
            self.pulse_effect.setColor(color)
        
        self.setGraphicsEffect(self.pulse_effect)
        self.current_effect = self.pulse_effect
        self.pulse_effect.startPulse()
        
    def activateRippleEffect(self, color=None):
        """
        Activate the ripple effect on the puck.
        
        Args:
            color: Optional QColor for the ripple effect
        """
        if color:
            self.ripple_effect.setColor(color)
            
        self.setGraphicsEffect(self.ripple_effect)
        self.current_effect = self.ripple_effect
        self.ripple_effect.startRipple()
        
    def deactivateEffects(self):
        """Deactivate all effects on the puck."""
        self.setGraphicsEffect(None)
        self.current_effect = None
        
    def update(self):
        """Update the puck and its effects."""
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update ripple effect if active
        if self.current_effect == self.ripple_effect:
            self.ripple_effect.updateRipples(delta_time)
            
        super().update()
        
    def reactToSpeech(self, volume, pitch=0.5, use_ripple=True):
        """
        React to speech by adjusting the active effect.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch level (0.0 to 1.0)
            use_ripple: Whether to use ripple effect (True) or pulse effect (False)
        """
        # Determine color based on pitch
        if pitch < 0.33:
            color = QColor(100, 100, 255)  # Blue-ish for low pitch
        elif pitch < 0.66:
            color = QColor(100, 255, 100)  # Green-ish for medium pitch
        else:
            color = QColor(255, 100, 100)  # Red-ish for high pitch
            
        # Activate the appropriate effect
        if use_ripple:
            if self.current_effect != self.ripple_effect:
                self.activateRippleEffect(color)
            self.ripple_effect.reactToSpeech(volume, pitch)
        else:
            if self.current_effect != self.pulse_effect:
                self.activatePulseEffect(color)
            self.pulse_effect.setIntensity(volume)
            self.pulse_effect.setColor(color)


class HockeyPuckView(QGraphicsView):
    """
    A QGraphicsView that displays the isometric hockey puck.
    
    This class provides a view for the hockey puck and handles updates
    and animations.
    """
    
    def __init__(self, parent=None):
        """Initialize the hockey puck view."""
        super().__init__(parent)
        
        # Set up the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Configure view
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(Qt.NoBrush)  # Transparent background
        self.setFrameShape(QGraphicsView.NoFrame)  # No frame
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Make the view transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Speech reaction settings
        self.use_ripple_effect = True
        self.current_volume = 0.0
        self.current_pitch = 0.5
        self.target_volume = 0.0
        self.target_pitch = 0.5
        self.volume_smoothing = 0.1  # Lower values = smoother transitions
        self.pitch_smoothing = 0.1
        
        # Create the puck
        self.puck = None
        self.createPuck()
        
        # Set up animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateAnimation)
        self.timer.start(16)  # ~60 FPS
        
    def createPuck(self):
        """Create the hockey puck item in the scene."""
        # Remove existing puck if any
        if self.puck:
            self.scene.removeItem(self.puck)
            
        # Create a new puck
        size = min(self.width(), self.height()) * 0.95  # Fill most of the view
        x = (self.width() - size) / 2
        y = (self.height() - size) / 2
        
        self.puck = HockeyPuckItem(x, y, size, size)
        self.scene.addItem(self.puck)
        
        # Activate the default effect
        if self.use_ripple_effect:
            self.puck.activateRippleEffect()
        else:
            self.puck.activatePulseEffect()
        
    def resizeEvent(self, event):
        """Handle resize events to adjust the puck size."""
        super().resizeEvent(event)
        
        # Resize the scene to match the view
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        
        # Recreate the puck with the new size
        self.createPuck()
        
    def updateAnimation(self):
        """Update the animation state."""
        if self.puck:
            # Smoothly interpolate volume and pitch
            self.current_volume += (self.target_volume - self.current_volume) * self.volume_smoothing
            self.current_pitch += (self.target_pitch - self.current_pitch) * self.pitch_smoothing
            
            # Update the puck's reaction to speech
            self.puck.reactToSpeech(
                self.current_volume,
                self.current_pitch,
                self.use_ripple_effect
            )
            
            # Update the puck
            self.puck.update()
            
    def reactToSpeech(self, volume, pitch=0.5):
        """
        React to speech by setting target animation parameters.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            pitch: Pitch level (0.0 to 1.0)
        """
        self.target_volume = max(0.0, min(1.0, volume))
        self.target_pitch = max(0.0, min(1.0, pitch))
        
    def toggleEffectType(self):
        """Toggle between pulse and ripple effects."""
        self.use_ripple_effect = not self.use_ripple_effect
        
        # Update the effect
        if self.puck:
            if self.use_ripple_effect:
                self.puck.activateRippleEffect()
            else:
                self.puck.activatePulseEffect()