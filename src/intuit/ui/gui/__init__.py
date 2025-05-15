"""
Intuit AI Hockey Puck GUI Package.

This package provides a frameless circular window shaped like a hockey puck
with dynamic lighting effects that react to AI speech.
"""

from .main_gui import IntuitGUI
from .puck_widget import HockeyPuckItem, HockeyPuckView
from .light_effects import PulseEffect, RippleEffect
from .integration import (
    GUIManager, get_gui_manager, start_gui, stop_gui,
    process_speech, set_speech_callback, is_gui_running
)

__all__ = [
    'IntuitGUI',
    'HockeyPuckItem',
    'HockeyPuckView',
    'PulseEffect',
    'RippleEffect',
    'GUIManager',
    'get_gui_manager',
    'start_gui',
    'stop_gui',
    'process_speech',
    'set_speech_callback',
    'is_gui_running',
]