# Logging and Debug Mode

This document describes the logging configuration and debug mode features in Intuit.

## Overview

Intuit includes a comprehensive logging system that redirects all output to a log file instead of stdout. This keeps the terminal clean while providing detailed logs for debugging and monitoring.

## Features

- **File-based logging**: All logs are written to a file (default: `output.log`)
- **Debug mode**: Captures detailed pipeline stages (STT → Agent → TTS)
- **Console output option**: Optionally mirror logs to console
- **Pipeline tracking**: Tracks each voice interaction session through the complete pipeline
- **Configurable log levels**: Control verbosity with standard logging levels

## Usage

### Basic Voice Mode with Logging

```bash
# All logs go to output.log by default
intuit voice

# Specify a custom log file
intuit voice --log-file my-logs.log
```

### Debug Mode

Debug mode enables detailed pipeline logging that tracks each stage of voice interactions:

```bash
# Enable debug mode
intuit voice --debug

# Debug mode with custom log file
intuit voice --debug --log-file debug.log
```

### Console Output

By default, logs only go to the file. To also see them in the console:

```bash
# Mirror logs to both file and console
intuit voice --console

# Debug mode with console output
intuit voice --debug --console
```

### Combined Options

```bash
# Full debugging with console output and custom log file
intuit voice --debug --console --log-file detailed-debug.log

# Dictation mode with debug logging
intuit voice --dictation --debug --log-file dictation-debug.log
```

## Pipeline Logging

When debug mode is enabled, the system tracks each voice interaction through these stages:

### 1. STT (Speech-to-Text)
- **Start**: When audio recording begins
- **Complete**: When speech is successfully transcribed
- **Error**: If speech recognition fails

### 2. Agent Processing
- **Start**: When the agent begins processing the query
- **Complete**: When the agent generates a response
- **Error**: If agent processing fails

### 3. TTS (Text-to-Speech)
- **Start**: When text-to-speech conversion begins
- **Complete**: When speech playback finishes
- **Error**: If TTS fails

### Session Tracking

Each voice interaction is assigned a unique session ID that tracks it through all pipeline stages. This makes it easy to correlate logs for a single interaction.

## Log Format

### Standard Log Entry
```
2025-01-18 16:30:45,123 - intuit.ui.voice - INFO - Listening for voice input...
```

### Debug Pipeline Log Entry
```
2025-01-18 16:30:45,123 - intuit.logging_config - DEBUG - [SESSION abc123] STT_START
2025-01-18 16:30:47,456 - intuit.logging_config - DEBUG - [SESSION abc123] STT_COMPLETE: "What's the weather?"
2025-01-18 16:30:47,457 - intuit.logging_config - DEBUG - [SESSION abc123] AGENT_START: "What's the weather?"
2025-01-18 16:30:49,789 - intuit.logging_config - DEBUG - [SESSION abc123] AGENT_COMPLETE: "The current weather is..."
2025-01-18 16:30:49,790 - intuit.logging_config - DEBUG - [SESSION abc123] TTS_START: "The current weather is..."
2025-01-18 16:30:52,123 - intuit.logging_config - DEBUG - [SESSION abc123] TTS_COMPLETE
2025-01-18 16:30:52,124 - intuit.logging_config - DEBUG - [SESSION abc123] SESSION_END - Duration: 7.00s
```

## Log Levels

The logging system uses standard Python logging levels:

- **DEBUG**: Detailed pipeline information (enabled with `--debug`)
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## Configuration

### Environment Variables

You can configure logging behavior through environment variables:

```bash
# Set default log file
export INTUIT_LOG_FILE="intuit.log"

# Enable debug mode by default
export INTUIT_DEBUG_MODE="true"

# Enable console output by default
export INTUIT_CONSOLE_OUTPUT="true"
```

### Programmatic Configuration

```python
from intuit.logging_config import setup_logging

# Setup logging with custom configuration
pipeline_logger = setup_logging(
    log_file="custom.log",
    debug_mode=True,
    console_output=False
)
```

## Log File Management

### Log Rotation

For long-running sessions, consider implementing log rotation:

```bash
# Using logrotate (Linux/macOS)
# Add to /etc/logrotate.d/intuit
/path/to/output.log {
    daily
    rotate 7
    compress
    missingok
    notifcreate
}
```

### Viewing Logs

```bash
# View logs in real-time
tail -f output.log

# View with syntax highlighting
tail -f output.log | grep --color=auto "ERROR\|WARNING\|INFO"

# Search for specific sessions
grep "SESSION abc123" output.log

# View only pipeline stages
grep "STT_\|AGENT_\|TTS_" output.log
```

## Troubleshooting

### No Logs Generated

If logs aren't being generated:

1. Check file permissions in the current directory
2. Verify the log file path is writable
3. Ensure logging is not disabled by environment variables

### Logs Too Verbose

If logs are too detailed:

1. Disable debug mode (remove `--debug` flag)
2. Use a custom log level filter
3. Redirect specific loggers to `/dev/null`

### Missing Pipeline Logs

If pipeline logs aren't appearing:

1. Ensure `--debug` flag is enabled
2. Check that the pipeline logger is being passed to voice interfaces
3. Verify the session ID is being tracked correctly

## Best Practices

1. **Use debug mode during development**: Enable `--debug` to see detailed pipeline information
2. **Disable console output in production**: Use file-only logging for cleaner output
3. **Rotate logs regularly**: Prevent log files from growing too large
4. **Use unique log files for different modes**: Separate logs for dictation, voice, and chat modes
5. **Monitor log file size**: Set up alerts for unusually large log files
6. **Archive old logs**: Keep historical logs for troubleshooting

## Examples

### Development Workflow

```bash
# Start with full debugging
intuit voice --debug --console

# Review logs
tail -f output.log | grep "ERROR\|WARNING"
```

### Production Deployment

```bash
# Minimal logging to file only
intuit voice --log-file /var/log/intuit/voice.log

# With log rotation configured
intuit voice --log-file /var/log/intuit/voice.log --no-console
```

### Debugging Issues

```bash
# Enable full debugging with console output
intuit voice --debug --console --log-file debug-$(date +%Y%m%d-%H%M%S).log

# Review specific session
grep "SESSION abc123" debug-*.log
```

## Related Documentation

- [Voice TUI Documentation](voice_tui.md)
- [Dictation Mode Documentation](dictation_mode.md)
- [Architecture Overview](../README.md)