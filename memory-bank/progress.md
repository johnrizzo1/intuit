# Progress

This document tracks what works, what's left to build, the current status, known issues, and the evolution of project decisions.

## What Works

- Core agent logic implemented.
- Initial tool plugins (web search, Gmail, weather, filesystem) added.
- CLI and voice user interfaces built and functional.
- File indexing function created for query enrichment.
- Local Calendar tool implemented (add, list, search, delete) using Pydantic/JSON.
- Local Notes tool implemented (add, list, search, delete) using Pydantic/JSON.
- Local Reminders tool implemented (add, list, search, delete) using Pydantic/JSON.
- CLI and voice integration for Calendar, Notes, and Reminders tools.
- Unit tests for Calendar, Notes, and Reminders tools core logic.
- Documentation for Calendar, Notes, and Reminders tools.
- Background process for reminder triggering implemented.
- MCP server functionality implemented to expose Intuit's tools as MCP resources.
- MCP client functionality implemented to connect to external MCP servers.
- Screenshot tool created as an example MCP implementation.
- Agent updated to support MCP tools.
- CLI commands added for MCP server management (start, list tools, status).
- Restart script created for the MCP server.

## What's Left to Build

- Expand tool/plugin ecosystem (beyond productivity tools).
- Enhance vector database capabilities.
- Optimize performance and latency.
- Ensure consistent error handling and user feedback across all tools and interfaces.
- Review and refactor code for maintainability and performance.

## Current Status

- Core system and productivity tools are fully operational.
- MCP integration has been successfully implemented, enabling both client and server functionality.
- The system can now expose its tools to external AI agents and connect to external MCP servers.
- The RAG pipeline has been fixed and is now functioning properly, enabling semantic search of indexed files.

## Known Issues

- No critical issues reported at this stage.

## Recently Fixed Issues

- Fixed the RAG pipeline by properly connecting the vector store to the FilesystemTool in both the main agent and MCP server.
- Fixed scope issues with imports in the create_agent function.
- Enhanced the VectorStore's search method to handle empty or wildcard queries to list all indexed files.
- Updated the FilesystemTool's search method to properly handle empty queries.
- Updated the agent's prompt to include instructions on how to handle queries about indexed files.
- Fixed indentation issues in the FilesystemTool's _search_files method.
- Added extensive logging throughout the vector store code to help diagnose RAG pipeline issues.
- Added missing CustomMCPTool for the filesystem_search tool in the agent.py file, allowing the agent to use the filesystem tool when asked to list all indexed files.
- Added a special case in the agent's process_input method to directly use the filesystem tool when the user asks to list all indexed files, bypassing the agent_executor.
- Fixed the special case to use the async version of the tool's run method (_arun) instead of the synchronous version (_run) to handle tools that only support async execution.

## Evolution of Project Decisions

- Shifted from initial setup to implementing core features and productivity tools.
- Adopted Pydantic and JSON for structured local data storage.
- Successfully completed the productivity tools milestone.
- Successfully implemented MCP integration, enabling both client and server functionality.
- Used the MCP protocol as a standardized way for AI models to interact with external tools and resources.
- Implemented a screenshot tool as a proof of concept for MCP integration.
- Added robust error handling and fallback mechanisms for MCP tool execution.
