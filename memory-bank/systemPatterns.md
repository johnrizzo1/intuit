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