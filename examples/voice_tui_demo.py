"""
Demo script for the Voice TUI interface.

This script demonstrates how to use the voice TUI interface with Intuit.
Run with: uv run intuit voice --tui
"""

# Example usage documentation
USAGE = """
Voice TUI Demo
==============

The Voice TUI provides a rich terminal interface for voice interactions with Intuit.

Features:
---------
1. **Chat Interface**: Shows conversation history with messages from you and the assistant
2. **Audio Meter**: Real-time visualization of microphone input levels
3. **Status Bar**: Shows current activity (Listening, Processing, Speaking)
4. **Voice Metrics**: Displays latency, confidence, and duration metrics
5. **Auto-scroll**: Conversation automatically scrolls as new messages arrive

Usage:
------
To start the voice TUI:
    uv run intuit voice --tui

To start without TUI (basic voice mode):
    uv run intuit voice --no-tui

Keyboard Shortcuts:
-------------------
- q: Quit the application
- c: Clear conversation history

Voice Commands:
---------------
- Say "exit", "quit", or "stop" to end the session
- Speak naturally to interact with the assistant

Layout:
-------
┌─────────────────────────────────────────────────────────────┐
│ Header (with clock)                                         │
├──────────────────┬──────────────────────────────────────────┤
│ Left Panel:      │ Conversation Area:                       │
│ - Audio Meter    │ ┌────────────────────────────────────┐   │
│ - Voice Metrics  │ │ You: Hello                         │   │
│ - Status Bar     │ │ 12:34:56                           │   │
│                  │ └────────────────────────────────────┘   │
│                  │ ┌────────────────────────────────────┐   │
│                  │ │ Assistant: Hi! How can I help?     │   │
│                  │ │ 12:34:58                           │   │
│                  │ └────────────────────────────────────┘   │
│                  │ (scrollable)                             │
├──────────────────┴──────────────────────────────────────────┤
│ Footer (keyboard shortcuts)                                 │
└─────────────────────────────────────────────────────────────┘

Technical Details:
------------------
- Built with Textual for rich terminal UI
- Real-time audio level monitoring
- Async voice processing
- Automatic conversation history management
- Cross-platform support (Linux, macOS, Windows)

Example Session:
----------------
1. Start the TUI: uv run intuit voice --tui
2. Wait for "Ready to listen..." status
3. Speak your query (e.g., "What's the weather?")
4. Watch the audio meter respond to your voice
5. See your message appear in the conversation
6. Wait for the assistant's response
7. Continue the conversation or say "exit" to quit

Configuration:
--------------
You can customize the voice interface with these options:
    --model: Choose the AI model (default: gpt-4o-mini)
    --voice-language: Set voice language (default: en)
    --slow/--no-slow: Enable slow speech
    --index/--no-index: Enable filesystem indexing
    --weather/--no-weather: Enable weather information

Full example:
    uv run intuit voice --tui --model gpt-4o --voice-language en --index

For more information, see the README.md file.
"""

if __name__ == "__main__":
    print(USAGE)