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
- devenv/Nix manages all dependencies and environment
- Test-driven development workflow
- **IMPORTANT**: All commands must be run within the devenv shell:
  - Use `devenv shell -- <command>` to run commands in the environment
  - Example: `devenv shell -- pytest tests/`
  - Example: `devenv shell -- python -m intuit.main`
  - This ensures all dependencies and environment variables are properly loaded

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
- Textual TUI for rich voice interface
- Voice for real-time, full-duplex conversation
- All commands run via: `devenv shell -- <command>`

## Command Execution

**Critical**: Since we use devenv/Nix for environment management, all commands MUST be executed within the devenv shell:

```bash
# Correct way to run commands
devenv shell -- uv run intuit voice
devenv shell -- pytest tests/
devenv shell -- python -m intuit.main

# Incorrect (will fail with missing dependencies)
uv run intuit voice
pytest tests/
```

The devenv shell ensures:
- All Python dependencies are available
- Environment variables are properly set
- System dependencies are accessible
- Consistent environment across all developers