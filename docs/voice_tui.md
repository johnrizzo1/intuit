# Voice TUI Interface

The Voice TUI (Terminal User Interface) provides a rich, interactive terminal interface for voice conversations with Intuit.

## Overview

The Voice TUI transforms the voice interaction experience by providing:
- Real-time conversation display
- Audio level monitoring
- Status indicators
- Voice quality metrics
- Scrollable chat history

## Features

### 1. Chat Interface
- **Two-column layout**: Your messages on the right (cyan), assistant's on the left (green)
- **Timestamps**: Each message shows when it was sent
- **Auto-scroll**: Automatically scrolls to show the latest messages
- **Message history**: Full conversation history maintained during session

### 2. Audio Meter
- **Real-time visualization**: Shows microphone input level as a bar graph
- **Color-coded levels**:
  - Green: Normal (0-50%)
  - Yellow: Moderate (50-80%)
  - Red: High (80-100%)
- **Percentage display**: Shows exact input level

### 3. Status Bar
- **Activity indicators**:
  - 🎤 Listening: When capturing your voice
  - ⚙️ Processing: When the AI is thinking
  - 🔊 Speaking: When playing the response
  - ⏸️ Idle: When waiting for input
- **Status messages**: Detailed information about current activity

### 4. Voice Metrics
- **Latency**: Time taken to process your request
- **Confidence**: Speech recognition confidence level
- **Duration**: Length of your voice input

## Usage

### Starting the TUI

```bash
# Start with TUI (default)
uv run intuit voice --tui

# Start without TUI (basic voice mode)
uv run intuit voice --no-tui
```

### With Additional Options

```bash
# Use a specific model
uv run intuit voice --tui --model gpt-4o

# Enable filesystem indexing
uv run intuit voice --tui --index --index-path ~/Documents

# Set voice language
uv run intuit voice --tui --voice-language es

# Enable slow speech
uv run intuit voice --tui --slow
```

## Keyboard Shortcuts

- **q**: Quit the application
- **c**: Clear conversation history

## Voice Commands

- Say **"exit"**, **"quit"**, or **"stop"** to end the session
- Speak naturally for all other interactions

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Intuit Voice Assistant                          🕐 12:34:56 │
├──────────────────┬──────────────────────────────────────────┤
│ 🎤 Input Level:  │ ┌────────────────────────────────────┐   │
│ ████████░░░░░░░  │ │ You: What's the weather today?     │   │
│ 60%              │ │ 12:34:50                           │   │
│                  │ └────────────────────────────────────┘   │
│ Voice Metrics:   │ ┌────────────────────────────────────┐   │
│ Latency: 1.2s    │ │ Assistant: The weather today is... │   │
│ Confidence: 85%  │ │ 12:34:52                           │   │
│ Duration: 2.5s   │ └────────────────────────────────────┘   │
│                  │                                          │
│ Status:          │ (More messages...)                       │
│ 🎤 Listening     │                                          │
│ Ready to listen  │                                          │
├──────────────────┴──────────────────────────────────────────┤
│ q: Quit  c: Clear                                           │
└─────────────────────────────────────────────────────────────┘
```

## Technical Details

### Architecture

The Voice TUI is built using:
- **Textual**: Modern TUI framework for Python
- **Rich**: Terminal formatting and styling
- **sounddevice**: Real-time audio monitoring
- **asyncio**: Asynchronous voice processing

### Components

1. **VoiceTUI**: Main application class
2. **AudioMeter**: Real-time audio level display
3. **ConversationMessage**: Individual message widget
4. **ConversationHistory**: Scrollable message container
5. **StatusBar**: Activity and status display
6. **VoiceMetrics**: Performance metrics display

### Audio Monitoring

The TUI continuously monitors audio input levels:
- Samples at 16kHz
- Calculates RMS (Root Mean Square) level
- Updates display at ~20Hz
- Automatic decay for smooth visualization

### Message Flow

1. User speaks → Audio captured
2. Speech-to-text conversion
3. Message added to conversation (cyan, right-aligned)
4. AI processes request
5. Response generated
6. Message added to conversation (green, left-aligned)
7. Text-to-speech playback

## Troubleshooting

### No Audio Meter Movement

If the audio meter doesn't respond:
1. Check microphone permissions
2. Verify microphone is selected as default input
3. Test microphone in system settings
4. Try adjusting microphone volume

### TUI Not Displaying Correctly

If the TUI looks broken:
1. Ensure terminal supports 256 colors
2. Resize terminal window (minimum 80x24)
3. Update terminal emulator
4. Try a different terminal (iTerm2, Windows Terminal, etc.)

### Voice Recognition Issues

If speech isn't recognized:
1. Speak clearly and at normal pace
2. Reduce background noise
3. Check internet connection (uses Google Speech API)
4. Verify microphone quality

## Examples

### Basic Conversation

```bash
$ uv run intuit voice --tui

# TUI starts, shows "Ready to listen..."
# Speak: "What's the weather in San Francisco?"
# Watch audio meter respond to your voice
# See your message appear in conversation
# Wait for assistant's response
# Response is displayed and spoken
```

### With Filesystem Search

```bash
$ uv run intuit voice --tui --index --index-path ~/Projects

# Speak: "Find files about Python testing"
# Assistant searches indexed files
# Results displayed in conversation
```

### Multi-turn Conversation

```bash
$ uv run intuit voice --tui

# Turn 1
You: "What's the weather?"
Assistant: "The weather in your location is..."

# Turn 2
You: "What about tomorrow?"
Assistant: "Tomorrow's forecast shows..."

# Turn 3
You: "Thanks, goodbye"
Assistant: "You're welcome! Have a great day!"
```

## Best Practices

1. **Speak clearly**: Enunciate words for better recognition
2. **Wait for status**: Let each phase complete before speaking
3. **Monitor audio meter**: Ensure it responds to your voice
4. **Use natural language**: Speak as you would to a person
5. **Check metrics**: Monitor latency and confidence for quality

## Integration with Other Features

The Voice TUI works seamlessly with:
- **Calendar**: "Add a meeting tomorrow at 2pm"
- **Notes**: "Create a note about project ideas"
- **Reminders**: "Remind me to call John at 5pm"
- **Weather**: "What's the weather forecast?"
- **Web Search**: "Search for Python tutorials"
- **Gmail**: "Check my recent emails"

## Future Enhancements

Planned improvements:
- Voice activity detection (automatic start/stop)
- Multiple language support in UI
- Customizable themes
- Export conversation history
- Voice command shortcuts
- Noise cancellation indicators
- Speaker identification for multi-user

## See Also

- [Voice Interface Documentation](voice_interface.md)
- [CLI Documentation](cli.md)
- [Agent Configuration](agent_config.md)