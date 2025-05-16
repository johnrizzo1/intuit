# Implementation Plan: Voice Integration with GUI

## Summary
This document outlines the implementation of voice interface capabilities for the Intuit AI Hockey Puck GUI. The integration uses a multi-process architecture to ensure stability and responsiveness, with the voice processing running in a separate process to avoid blocking the GUI.

## Architecture

### Multi-Process Design
The voice integration follows a multi-process architecture:

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│   GUI Process   │◄────┤  Voice Process  │
│                 │     │                 │
└─────────────────┘     └─────────────────┘
        ▲                       ▲
        │                       │
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Input Queue    │     │  Output Queue   │
└─────────────────┘     └─────────────────┘
```

1. **GUI Process**: Handles the visual interface and user interactions
2. **Voice Process**: Handles speech recognition and synthesis
3. **Inter-Process Communication**: Uses multiprocessing.Queue for bidirectional communication

## Components

### 1. Voice Process Manager (`VoiceProcessManager`)
- Manages the lifecycle of the voice process
- Provides methods to start/stop the voice process
- Handles communication with the voice process via queues

### 2. Voice Processor (`VoiceProcessor`)
- Runs in a separate process
- Handles speech recognition and synthesis
- Extracts speech metrics (volume, pitch) for visualization
- Communicates with the GUI process via queues

### 3. GUI Integration
- Updates to `IntuitGUI` class to manage the voice process
- Timer-based polling for voice process data
- Visual feedback for speech via the hockey puck visualization

## Communication Protocol

### GUI to Voice Process
Commands sent from GUI to voice process:

1. **Speak**: Request to speak text
   ```json
   {
     "type": "speak",
     "text": "Text to speak"
   }
   ```

2. **Listen**: Request to listen for user input
   ```json
   {
     "type": "listen",
     "timeout": 5.0,
     "process": true
   }
   ```

3. **Stop**: Request to stop all processing
   ```json
   {
     "type": "stop"
   }
   ```

### Voice Process to GUI
Data sent from voice process to GUI:

1. **Metrics**: Speech metrics for visualization
   ```json
   {
     "type": "metrics",
     "volume": 0.5,
     "pitch": 0.7
   }
   ```

2. **Text**: Recognized speech text
   ```json
   {
     "type": "text",
     "content": "Recognized text"
   }
   ```

3. **Speaking**: Speaking state changes
   ```json
   {
     "type": "speaking",
     "state": "start|stop",
     "error": "Error message (optional)"
   }
   ```

4. **Error**: Error messages
   ```json
   {
     "type": "error",
     "message": "Error message"
   }
   ```

## User Interface

### Controls
- **V key**: Toggle voice interface on/off
- **Right-click menu**: Toggle voice interface on/off
- **Command-line option**: Start GUI with voice enabled (`--voice`)

### Visual Feedback
- Hockey puck visualization reacts to speech with dynamic lighting effects
- Different colors and intensities based on speech volume and pitch

## Configuration
The voice interface can be configured via:

1. **Command-line options**:
   - `--voice/--no-voice`: Enable/disable voice interface (default: enabled)
   - `--voice-language`: Language for voice output (default: "en")
   - `--slow/--no-slow`: Speak slowly (default: false)

2. **Configuration file**:
   - JSON format passed to the GUI process
   - Contains voice settings (enabled, language, slow)

## Implementation Details

### Voice Process
- Uses `speech_recognition` for speech recognition
- Uses `gTTS` for text-to-speech synthesis
- Implements audio playback in a separate thread to prevent hanging
- Runs in a separate process to avoid blocking the GUI
- Communicates with the GUI via multiprocessing queues

### GUI Integration
- Polls for voice process data every 50ms
- Updates the hockey puck visualization based on speech metrics
- Provides controls to toggle voice functionality
- Handles graceful shutdown of the voice process

## Error Handling
- Graceful termination of voice process on GUI exit
- Timeout-based forced termination if graceful shutdown fails
- Error reporting from voice process to GUI

## Dependencies

The voice integration requires the following Python packages:

```
speech_recognition  # For speech recognition
gtts                # For text-to-speech synthesis
sounddevice         # For audio input/output
numpy               # For audio processing
soundfile           # For audio file handling
```

These can be installed using pip:

```bash
pip install SpeechRecognition gTTS sounddevice numpy soundfile
```

## Conversation Flow

The voice interface implements a natural conversation flow:

1. **Activation**: When voice is activated, a welcome message is played
2. **Listening**: After the welcome message, the system begins listening for user input
3. **Recognition**: Speech is converted to text using speech recognition
4. **AI Processing**: Recognized text is sent to the AI model for processing
5. **Response Generation**: The AI model generates a contextual response
6. **Text-to-Speech**: The text response is converted to speech
7. **Continuation**: After speaking, the system automatically begins listening again
8. **Error Handling**: If speech isn't recognized, helpful feedback is provided

This creates a continuous conversation loop that feels natural to the user.

### AI Integration

The voice interface integrates with the AI model as follows:

1. **Speech Recognition**: User speech is converted to text
2. **Query Processing**: The text is sent to the AI model for processing
3. **Context-Aware Responses**: The AI model generates responses based on the query and conversation context
4. **Domain-Specific Handling**: Different types of queries (weather, time, greetings) receive appropriate responses
5. **Response Synthesis**: The AI response is converted back to speech

In the current implementation, the AI model is simulated with predefined responses for demonstration purposes. In a production environment, this would connect to the actual Intuit AI agent.

## Error Handling

The voice interface includes robust error handling:

1. **Unrecognized Speech**: When speech can't be recognized, the system provides feedback and prompts the user to try again
2. **Audio Playback Issues**: Audio playback is handled in a separate thread to prevent hanging
3. **Service Errors**: Speech recognition service errors are properly reported to the user
4. **Graceful Degradation**: If dependencies are missing, the system provides clear instructions for installation

## Future Enhancements
1. **Agent Integration**: Connect voice interface to the Intuit agent for full conversational capabilities
2. **Wake Word Detection**: Add wake word detection to activate listening
3. **Voice Activity Detection**: Improve speech detection accuracy
4. **Multiple Voice Options**: Support different voices and languages
5. **Voice Commands**: Add specific voice commands for controlling the GUI

## Conclusion
This implementation provides a stable and responsive voice interface for the Intuit AI Hockey Puck GUI. By running the voice processing in a separate process, we avoid blocking the GUI and ensure a smooth user experience. The integration is designed to be modular and extensible, allowing for future enhancements and improvements.