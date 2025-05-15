# Implementation Plan: Isometric Hockey Puck AI Interface (GUI)

## Summary
This plan outlines the steps to implement an expressive isometric hockey puck interface for the AI using **PySide6**. The GUI will feature a frameless, circular window shaped like a hockey puck with dynamic lighting effects that respond to the AI's speech.

## Directory Structure
The GUI implementation follows this directory structure:

```
src/
└── intuit/
    └── ui/
        └── gui/
            ├── __init__.py       # Package initialization
            ├── main_gui.py       # Main GUI window implementation
            ├── puck_widget.py    # Hockey puck widget implementation
            ├── light_effects.py  # Dynamic lighting effects
            ├── integration.py    # Integration with AI system
            ├── test_gui.py       # Test script
            └── demo.py           # Demo script
```

## Design Features
1. **Frameless Circular Window:**
   - A completely circular window with no frame or title bar.
   - Draggable with mouse for repositioning.
   - Isometric 3D effect with shadowing for depth.

2. **Dynamic Lighting Effects:**
   - Two effect types: Pulse and Ripple.
   - Pulse effect: Smooth pulsing light from the center.
   - Ripple effect: Concentric rings emanating from the center.
   - Effects change color based on speech pitch (blue for low, green for mid, red for high).
   - Effect intensity varies with speech volume.

3. **Integration with AI Speech:**
   - Capture speech events from the existing AI system.
   - Map speech data (volume, pitch) to visual lighting animations.
   - Smooth transitions between animation states.

---

## Implementation Steps
### Step 1: Setup Project Directory
- **Status:** ✅ Completed
- **Actions:**
  - Created the `src/intuit/ui/gui` directory.
  - Added all necessary files for the implementation.

### Step 2: Create Frameless Circular Window
- **Status:** ✅ Completed
- **Actions:**
  - Implemented `IntuitGUI` class using `QWidget` with frameless window flags.
  - Added circular mask using `QPainterPath` and `QRegion`.
  - Implemented window dragging functionality.
  - Added keyboard shortcuts for controlling the interface.

### Step 3: Develop Puck Widget
- **Status:** ✅ Completed
- **Actions:**
  - Implemented `HockeyPuckItem` class using `QGraphicsEllipseItem`.
  - Added shadow and gradient effects for the isometric appearance.
  - Created `HockeyPuckView` class with transparent background.
  - Integrated the puck widget with the main window.

### Step 4: Implement Light Effects
- **Status:** ✅ Completed
- **Actions:**
  - Implemented `PulseEffect` class for pulsing light animations.
  - Implemented `RippleEffect` class for rippling light animations.
  - Added color and intensity controls for both effects.
  - Created smooth transitions between animation states.

### Step 5: Integrate with AI Speech
- **Status:** ✅ Completed
- **Actions:**
  - Created `integration.py` module to connect GUI with AI system.
  - Implemented `SpeechProcessor` class to process AI speech data.
  - Added `GUIManager` class to manage the GUI lifecycle.
  - Created test scripts with simulated speech data.

---

## Tools and Libraries
- **PySide6** for GUI and graphics.
- **Asyncio** for integrating real-time speech events.

---

## Progress Tracking
| Step                          | Status       | Notes                                       |
|-------------------------------|--------------|---------------------------------------------|
| Setup Project Directory       | ✅ Completed | Created directory structure and files       |
| Create Frameless Circular Window | ✅ Completed | Implemented frameless window with circular mask |
| Develop Puck Widget           | ✅ Completed | Implemented HockeyPuckItem and HockeyPuckView |
| Implement Light Effects       | ✅ Completed | Created PulseEffect and RippleEffect classes |
| Integrate with AI Speech      | ✅ Completed | Created integration module with SpeechProcessor |

---

## Deliverables
- **Frameless Circular Window:** ✅ Implemented a completely circular window with no frame or title bar, shaped like a hockey puck.
- **Isometric 3D Effect:** ✅ Created a 3D isometric appearance with shadowing and gradient effects.
- **Dynamic Lighting Effects:** ✅ Implemented both pulse and ripple effects that react to speech volume and pitch.
- **Interactive Controls:** ✅ Added keyboard shortcuts and mouse dragging functionality.
- **Integration Module:** ✅ Created an integration module to connect the GUI with the existing AI system.
- **Testing Tools:** ✅ Provided test_gui.py and demo.py scripts to demonstrate and test the functionality.
- **CLI Integration:** ✅ Added a `gui` command to the CLI to start the application in GUI mode.
- **Standalone Script:** ✅ Created a standalone script that can be run independently without package dependencies.

---

This document will be updated as tasks are completed.