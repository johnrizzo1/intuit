# Product Context

This document describes the purpose, problems solved, user experience goals, and how the project should work from a product perspective.

## Purpose

Build a flexible, agentic personal assistant that centralizes access to digital tools and enables intuitive AI interaction.

## Problems Solved

- Fragmented access to web search, email, and weather information.
- Lack of a unified, extensible interface for interacting with AI and tools.
- Difficulty in real-time, voice-based, or command-line AI conversations.

## How it Should Work

- Users interact via CLI or a curses-like interface, or by passing queries as command arguments.
- Real-time, full-duplex voice conversation is supported.
- Tools (web search, Gmail, weather, etc.) are modular and easily extensible.
- The agent can enrich queries using a vector database built from the user's files.

## User Experience Goals

- Intuitive and responsive interaction.
- Seamless switching between CLI and voice.
- Cross-platform support (Linux, MacOS, Windows).
- Optional visual representation.
- Easy extensibility for new tools.