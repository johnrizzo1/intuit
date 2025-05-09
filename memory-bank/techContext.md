# Tech Context

This document details the technologies used, development setup, technical constraints, dependencies, and tool usage patterns.

## Technologies Used

- Python
- devenv for dependency management
- MCP protocol
- Vector database (for file content)
- Curses (CLI UI)
- Voice libraries (for real-time interaction)

## Development Setup

- Cross-platform: Linux, MacOS, Windows
- devenv manages all dependencies
- Test-driven development workflow

## Technical Constraints

- Must support extensibility for new tools
- Low latency for real-time interaction
- Robust voice handling

## Dependencies

- Python packages (see requirements.txt)
- devenv
- CLI/voice libraries

## Tool Usage Patterns

- CLI for command-based queries
- Curses for interactive UI
- Voice for real-time, full-duplex conversation