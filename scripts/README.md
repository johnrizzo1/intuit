# Intuit Scripts

This directory contains utility scripts for the Intuit project.

## Memory Command Wrappers

**IMPORTANT**: The direct `intuit memory` commands currently show verbose INFO logs that can't be suppressed through normal means. Use these wrapper scripts instead for a cleaner output.

### `quiet_memory.py`

A wrapper script for memory commands that suppresses log output. This is useful when you want to use the memory commands without seeing all the INFO logs.

Usage:

```bash
python scripts/quiet_memory.py <command> [args...]
```

Commands:

- `add`: Add a new memory
- `search`: Search memories by semantic similarity
- `get`: Get a specific memory by ID
- `delete`: Delete a memory by ID
- `clear`: Clear all memories

Examples:

```bash
# Search memories without log output
python scripts/quiet_memory.py search "process_conversation"

# Add a memory without log output
python scripts/quiet_memory.py add "This is a new memory" --importance 8

# Get a memory by ID without log output
python scripts/quiet_memory.py get "memory-id-here"

# Delete a memory without log output
python scripts/quiet_memory.py delete "memory-id-here"

# Clear all memories without log output
python scripts/quiet_memory.py clear
```

If you want to see the log output, you can use the `-v` or `--verbose` flag:

```bash
python scripts/quiet_memory.py search "process_conversation" -v
```

### `quiet_memory_search.py`

A specialized wrapper for the memory search command. This script is included for backward compatibility, but `quiet_memory.py` is recommended for all memory commands.

Usage:

```bash
python scripts/quiet_memory_search.py "query" [options]
```

Example:

```bash
python scripts/quiet_memory_search.py "process_conversation"
```

## Why Use These Scripts?

The direct `intuit memory` commands show verbose INFO logs due to how the underlying libraries configure their loggers. Despite attempts to silence these logs in the main code, some libraries reset the logging configuration during initialization.

These wrapper scripts solve the problem by redirecting stderr (where the logs are written) to /dev/null, while preserving the actual command output to stdout. This gives you a clean, log-free experience when working with the memory system.
