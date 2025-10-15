# Voice TUI Layout Design

## Overview

The Voice TUI has been redesigned with an improved layout that provides better space utilization and user experience.

## Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Intuit Voice Assistant                    Ctrl+Q: Quit | Ctrl+C: Clear | .. │
├─────────────────────────────────────────────────────┬───────────────────────┤
│                                                     │                       │
│  CHAT PANEL (2/3 width)                            │  METRICS (1/3 width)  │
│  ┌───────────────────────────────────────────────┐ │  ┌─────────────────┐  │
│  │ Conversation History (scrollable)             │ │  │ 🎤 Microphone   │  │
│  │                                                │ │  │ ████████░░░░░░  │  │
│  │ ┌──────────────────────────────────────────┐  │ │  │ 60%             │  │
│  │ │ You: What's the weather?                 │  │ │  └─────────────────┘  │
│  │ │ 12:34:50                                 │  │ │                       │
│  │ └──────────────────────────────────────────┘  │ │  ┌─────────────────┐  │
│  │                                                │ │  │ Voice Metrics   │  │
│  │ ┌──────────────────────────────────────────┐  │ │  │ Latency: 1.2s   │  │
│  │ │ Assistant: The weather today is sunny... │  │ │  │ Confidence: 85% │  │
│  │ │ 12:34:52                                 │  │ │  │ Duration: 2.5s  │  │
│  │ └──────────────────────────────────────────┘  │ │  └─────────────────┘  │
│  │                                                │ │                       │
│  │ (more messages...)                             │ │  ┌─────────────────┐  │
│  │                                                │ │  │ Status          │  │
│  │                                                │ │  │ 🎤 Listening    │  │
│  │                                                │ │  │ Ready to listen │  │
│  └───────────────────────────────────────────────┘ │  └─────────────────┘  │
│                                                     │                       │
│  ┌───────────────────────────────────────────────┐ │                       │
│  │ Text Input Area (multi-line)                  │ │                       │
│  │ Type your message here...                     │ │                       │
│  │                                                │ │                       │
│  │                                    [Send]      │ │                       │
│  └───────────────────────────────────────────────┘ │                       │
└─────────────────────────────────────────────────────┴───────────────────────┘
```

## Key Features

### 1. Responsive Layout
- **Horizontal expansion**: Layout fills the entire terminal width
- **Vertical expansion**: Layout fills the entire terminal height
- **Proportional sizing**: Chat panel (2/3) and metrics panel (1/3)

### 2. Chat Panel (Left, 2/3 width)
- **Conversation History**: Scrollable area showing all messages
  - User messages: Cyan, right-aligned
  - Assistant messages: Green, left-aligned
  - System messages: Yellow, centered
  - Timestamps on each message
- **Text Input Area**: Multi-line text input at the bottom
  - Supports single or multi-line input
  - Send button for submitting messages
  - Enter key to send (Shift+Enter for new line)

### 3. Metrics Panel (Right, 1/3 width)
- **Audio Meter**: Real-time microphone input visualization
  - Color-coded: Green (normal), Yellow (moderate), Red (high)
  - Percentage display
- **Voice Metrics**: Performance indicators
  - Latency: Processing time
  - Confidence: Speech recognition confidence
  - Duration: Voice input duration
- **Status Bar**: Current activity indicators
  - 🎤 Listening
  - ⚙️ Processing
  - 🔊 Speaking
  - ⏸️ Idle

## Interaction Modes

### Voice Mode (Default)
- Continuously listens for voice input
- Displays audio meter activity
- Speaks responses aloud
- Can also type messages in input area

### Text Mode (Ctrl+L to toggle)
- Disables voice listening
- Only accepts text input
- Does not speak responses
- Useful for quiet environments

## Keyboard Shortcuts

- **Ctrl+Q**: Quit the application
- **Ctrl+C**: Clear conversation history
- **Ctrl+L**: Toggle between voice and text mode
- **Enter**: Send message (in text input)
- **Shift+Enter**: New line (in text input)

## CSS Styling

The layout uses Textual's CSS-like styling system:

```css
#main-container {
    layout: horizontal;  /* Side-by-side panels */
    height: 1fr;        /* Fill available height */
}

#chat-panel {
    width: 2fr;         /* 2/3 of width */
    layout: vertical;   /* Stack conversation and input */
}

#metrics-panel {
    width: 1fr;         /* 1/3 of width */
    layout: vertical;   /* Stack metrics vertically */
}

#conversation {
    height: 1fr;        /* Fill available space */
}

#input-area {
    height: auto;       /* Size to content */
}
```

## Benefits

1. **Better Space Utilization**: Chat area gets more space (2/3 vs 1/2)
2. **Improved Readability**: Larger conversation area
3. **Flexible Input**: Multi-line text input for complex queries
4. **Dual Mode**: Both voice and text input available
5. **Responsive**: Adapts to terminal size
6. **Professional**: Clean, organized layout

## Usage Examples

### Voice Interaction
1. Speak your query
2. Watch audio meter respond
3. See your message appear in chat
4. Wait for assistant's response
5. Response is displayed and spoken

### Text Interaction
1. Type in the input area
2. Press Enter or click Send
3. See your message in chat
4. Wait for assistant's response
5. Response is displayed (and spoken if in voice mode)

### Mixed Mode
1. Ask a question by voice
2. Follow up with typed clarification
3. Continue conversation naturally
4. Toggle modes as needed

## Technical Implementation

- **Textual Framework**: Modern TUI framework
- **Reactive Components**: Auto-updating widgets
- **Async Processing**: Non-blocking voice and text handling
- **Event-Driven**: Responds to user actions immediately
- **Flexible Layout**: CSS-like styling system