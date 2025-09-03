#!/usr/bin/env python3
"""
Run script for the Intuit AI Hockey Puck GUI.

This script directly runs the GUI without any imports.
"""

import sys
import os
import time
import random
import threading
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the Python path
sys.path.insert(0, script_dir)

# Import the GUI modules directly with absolute paths
import intuit.ui.gui.main_gui as main_gui
import intuit.ui.gui.integration as integration

IntuitGUI = main_gui.IntuitGUI
start_gui = integration.start_gui
stop_gui = integration.stop_gui


def main():
    """Run the GUI."""
    print("Starting Intuit in GUI mode...")
    try:
        # Start the GUI in blocking mode
        app = QApplication(sys.argv)
        window = IntuitGUI()
        window.show()

        # Print instructions
        print("Hockey Puck GUI")
        print("---------------")
        print("Controls:")
        print("- Drag with mouse to move the window")
        print("- Press Space to toggle between pulse and ripple effects")
        print("- Press T to start/stop test animation")
        print("- Press Escape to close")

        # Run the application
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nExiting GUI mode...")
    except Exception as e:
        print(f"Error starting GUI: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
