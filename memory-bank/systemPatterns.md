# System Patterns

This document outlines the system architecture, key technical decisions, design patterns, component relationships, and critical implementation paths.

## System Architecture

- Modular agent core with pluggable tool architecture.
- Vector database for file content enrichment.
- Dual user interfaces: CLI (curses-like) and real-time voice.

## Key Technical Decisions

- Python as the implementation language.
- devenv for dependency management.
- MCP protocol for tool integration.
- Multi-threading and multi-processing for latency reduction.

## Design Patterns

- Plugin architecture for tools.
- Event-driven interaction model.
- Test-driven development.

## Component Relationships

- Agent Core ↔ Tool Plugins (web search, Gmail, weather, etc.)
- Agent Core ↔ Vector Database
- Agent Core ↔ CLI/Voice UI

## Critical Implementation Paths

- Tool registration and discovery.
- Query routing and enrichment.
- Real-time voice handling and full-duplex communication.

[2025-01-03 12:58:00] - Adoption of src/ layout pattern
The project now follows the src/ layout pattern, which is a Python packaging best practice. This pattern involves:

- Placing the main package under `src/package_name/` instead of directly in the project root
- Configuring setuptools to find packages in the `src/` directory using `[tool.setuptools.packages.find]`
- Ensuring all imports reference the package name directly (e.g., `from intuit.tools import ...`) rather than including the src path

Benefits:

- Prevents accidental imports from the source tree during development
- Ensures proper package installation and testing
- Resolves packaging conflicts (like the multi-package discovery issue we encountered)
- Makes the project structure more standard and maintainable
- Better separation between source code and other project files
