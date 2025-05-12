# Project Brief - Intuit: A personal assistant

## Overview

Today we are building an agentic assistant.  This assistant should be incredibly flexible.  We should be able to easily add new tools.  At a minimum the agent should be able to search the internet, read my gmail and tell me the weather.  The agent should support multiple interfaces.  For example, the agent should be able to be invoked on the cli and it will drop you into a curses like interface where you can converse with the agent.  Alternatively, you should also be able to pass the query to the agent via a simple command like argument.  Finally and possibly most importantly you should be able to converse with the agent in realtime by voice.  This voice interaction should work when invoked via the CLI and other interfaces to the agent.

## Core Features

- Uses multiple tools and MCP to interact with its environment, gather information and execute various tasks.
- Build a vector database of all the content of the files on my filesystem and use them to enrich my query
- Uses techniques such as multi-threading and multi-processing to help improve latency
- Full-Duplex Voice Communication
- Optional Visual Representation of the AI
- Full functionality via the CLI

## Target Users

Users of this application will be your general consumer.  The intent of this product is to provide intuitive interaction with the AI world.

## Technical Preferences (optional)

- Python
- Runs on Linux, MacOS, and Windows
- We use an agile test driven approach to development
- Uses devenv to manage dependencies
