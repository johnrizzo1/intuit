# Active Context

This document tracks the current work focus, recent changes, next steps, active decisions, important patterns, and learnings.

## Current Work Focus

Implementing Model Context Protocol (MCP) integration to enable Intuit to function as both an MCP client and server.

## Recent Changes

- Completed implementation of local productivity tools (Calendar, Notes, Reminders) with Pydantic models and JSON storage.
- Documented all productivity tools.
- Created a new implementation plan for MCP integration.
- Moved projectbrief.md into the memory-bank directory.

## Next Steps

- Create a base MCP server module (`src/intuit/mcp_server.py`).
- Implement FastMCP server initialization and configuration.
- Expose existing tools as MCP resources.
- Create a screenshot tool as an example MCP implementation.
- Update the agent to support MCP tools.

## Active Decisions and Considerations

- MCP will be used to expose Intuit's tools to external AI agents.
- MCP will also allow Intuit to connect to and use external MCP servers.
- The implementation will follow the pattern shown in the Python SDK examples.
- When the user runs a command, it should start the MCP server and then the decorated function will call the MCP server.

## Important Patterns and Preferences

- Modular design for tools with consistent interfaces.
- Unified backend logic for CLI and voice commands.
- MCP protocol for tool integration.
- Consistent error handling and user feedback.

## Learnings and Project Insights

- Using Pydantic and JSON provides a structured approach to local data storage.
- Integrating tools with the agent simplifies CLI and voice access.
- The MCP protocol provides a standardized way for AI models to interact with external tools and resources.
- Implementing both client and server MCP functionality will make Intuit more versatile and extensible.
