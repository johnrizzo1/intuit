# Dictation Mode

Intuit's dictation mode provides continuous voice transcription with intelligent command detection, allowing you to dictate text with sporadic periods of silence.

## Overview

Dictation mode is designed for scenarios where you need to continuously speak and have your words transcribed in real-time. Unlike the standard voice mode where the assistant responds to each query, dictation mode focuses on capturing your speech as text.

## Features

- **Continuous Transcription**: Speaks continuously with automatic transcription
- **Real-time Display**: See your words appear as you speak
- **Smart Command Detection**: Voice commands work seamlessly during dictation
- **Wake Word Support**: Use "Hey Intuit" to enter command mode
- **Pause/Resume**: Control dictation flow with voice commands
- **Auto-save**: Transcriptions are automatically saved with timestamps
- **Silence Detection**: 30-second silence threshold for natural pauses

## Starting Dictation Mode

### From Command Line

Start Intuit in dictation mode:

```bash
devenv shell -- uv run intuit voice --dictation
```

This will:
1. Launch the TUI interface
2. Automatically enter dictation mode
3. Start listening for your speech

### From Voice Mode

If you're already in voice mode, you can enter dictation mode by saying:

```
"Start Dictation"
```

Or press `Ctrl+D` to toggle dictation mode.

## Using Dictation Mode

### Basic Dictation

Once in dictation mode, simply start speaking. Your words will appear in real-time on the screen:

```
You: "This is a test of the dictation feature. It works really well for 
capturing long-form content without interruption."
```

### Voice Commands

While dictating, you can use these commands:

#### Pause Dictation
```
"Pause Dictation"
```
Temporarily stops transcription. The system continues listening for resume/end commands.

#### Resume Dictation
```
"Resume Dictation"
```
Resumes transcription after a pause.

#### End Dictation
```
"End Dictation"
```
Ends the dictation session and saves the transcription to a file.

#### Command Mode (Hey Intuit)
```
"Hey Intuit [command]"
```
Enters command mode without adding to transcription. Use this to:
- Ask questions
- Execute commands
- Control the assistant

Example:
```
"Hey Intuit, what time is it?"
```
The assistant will respond, but "Hey Intuit, what time is it?" won't be added to your transcription.

## Dictation States

Dictation mode has three states, shown in the status bar:

1. **📝 Dictating** (Green): Actively listening and transcribing
2. **⏸️ Dictation Paused** (Yellow): Paused, listening for resume/end commands
3. **🛑 Dictation Ended** (Red): Session ended, file saved

## File Output

When you end a dictation session, the transcription is automatically saved to:

```
./dictation_YYYYMMDD_HHMMSS.txt
```

Example:
```
./dictation_20250118_143022.txt
```

The file contains the complete transcription without any command text.

## UI Elements

### Dictation Display
The main panel shows your transcription in real-time with auto-scrolling.

### Status Bar
Shows current dictation state and helpful status messages.

### Audio Meter
Visual feedback of microphone input levels.

### Metrics Panel
Displays voice quality metrics (latency, confidence, duration).

## Keyboard Shortcuts

- `Ctrl+D`: Toggle dictation mode on/off
- `Ctrl+Q`: Quit (saves dictation if active)
- `Ctrl+C`: Clear display
- `Ctrl+L`: Toggle listen mode

## Best Practices

### For Best Results

1. **Speak Naturally**: Don't pause too long between sentences
2. **Clear Audio**: Use a good microphone in a quiet environment
3. **Punctuation**: Say punctuation marks ("period", "comma", "question mark")
4. **Commands**: Use clear, distinct voice for commands
5. **Pauses**: The 30-second silence threshold allows for natural thinking pauses

### Example Workflow

```
1. Start dictation: "Start Dictation"
2. Dictate content: "Dear team, I wanted to share some thoughts..."
3. Pause if needed: "Pause Dictation"
4. Ask a question: "Hey Intuit, what's the weather?"
5. Resume: "Resume Dictation"
6. Continue: "As I was saying..."
7. End session: "End Dictation"
```

## Technical Details

### Silence Detection
- **Threshold**: 30 seconds of silence
- **Behavior**: Noted but doesn't auto-pause
- **Purpose**: Allows for natural thinking pauses

### Command Detection
Commands are detected using regex patterns and removed from transcription:
- Wake word: `hey intuit`
- Pause: `pause dictation`
- Resume: `resume dictation`
- End: `end dictation`
- Start: `start dictation`

### Audio Processing
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Format**: 16-bit PCM
- **Recognition**: Google Speech Recognition API

## Troubleshooting

### Dictation Not Starting
- Ensure TUI is enabled (`--tui` flag)
- Check microphone permissions
- Verify internet connection (for speech recognition)

### Commands Not Recognized
- Speak clearly and distinctly
- Ensure proper pronunciation of command phrases
- Check audio input levels in the meter

### Transcription Accuracy Issues
- Reduce background noise
- Use a better microphone
- Speak at a moderate pace
- Ensure good internet connection

### File Not Saved
- Check write permissions in current directory
- Ensure dictation was properly ended
- Look for error messages in status bar

## Examples

### Meeting Notes
```bash
devenv shell -- uv run intuit voice --dictation
```
Then dictate:
```
"Meeting notes for January 18th. Attendees: John, Sarah, Mike. 
Agenda item one: Project timeline. We discussed the Q1 deliverables..."
```

### Article Writing
```
"Start Dictation"
"The future of artificial intelligence. Paragraph. 
As we look ahead to the next decade, AI will transform..."
"Pause Dictation"
"Hey Intuit, search for AI statistics"
"Resume Dictation"
"According to recent research, AI adoption has increased..."
"End Dictation"
```

### Quick Notes
```
"Start Dictation"
"Remember to buy milk, eggs, and bread. 
Call dentist about appointment. 
Review project proposal by Friday."
"End Dictation"
```

## Integration with Other Features

Dictation mode works alongside other Intuit features:

- **Memory**: Dictations can be added to memory for later recall
- **Notes**: Save dictations as notes using the notes tool
- **Calendar**: Mention events that can be added to calendar
- **Reminders**: Create reminders from dictation content

## Future Enhancements

Planned improvements for dictation mode:

- Custom wake words
- Punctuation auto-detection
- Multiple language support
- Cloud storage integration
- Dictation templates
- Voice-activated formatting
- Real-time editing commands

## See Also

- [Voice TUI Documentation](voice_tui.md)
- [Voice Interface Documentation](voice_tui_layout.md)
- [Tools Documentation](tools_testing.md)