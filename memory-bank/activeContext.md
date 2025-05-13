# Active Context

This document tracks the current work focus, recent changes, next steps, active decisions, important patterns, and learnings.

## Current Work Focus

Refining and expanding the Model Context Protocol (MCP) integration that enables Intuit to function as both an MCP client and server.

## Recent Changes

- Completed implementation of local productivity tools (Calendar, Notes, Reminders) with Pydantic models and JSON storage.
- Documented all productivity tools.
- Created and implemented the MCP integration plan.
- Created a base MCP server module (`src/intuit/mcp_server.py`).
- Implemented FastMCP server initialization and configuration.
- Exposed existing tools as MCP resources.
- Created a screenshot tool as an example MCP implementation.
- Updated the agent to support MCP tools.
- Added CLI commands for MCP server management.
- Created a restart script for the MCP server.

## Next Steps

- Expand the tool/plugin ecosystem beyond the current set.
- Enhance vector database capabilities further (e.g., incremental indexing, better metadata extraction).
- Optimize performance and latency.
- Write unit tests for MCP server and client functionality.
- Document MCP integration in README.md.
- Create usage examples for MCP tools.

## Active Decisions and Considerations

- MCP is used to expose Intuit's tools to external AI agents.
- MCP also allows Intuit to connect to and use external MCP servers.
- The implementation follows the pattern shown in the Python SDK examples.
- The MCP server runs in a separate process to avoid blocking the main application.
- Tools are registered with the MCP server using decorators.
- The agent can discover and use tools from connected MCP servers.
- Robust error handling and fallback mechanisms are implemented for MCP tool execution.

## Important Patterns and Preferences

- Modular design for tools with consistent interfaces.
- Unified backend logic for CLI and voice commands.
- MCP protocol for tool integration.
- Consistent error handling and user feedback.

## Learnings and Project Insights

- Using Pydantic and JSON provides a structured approach to local data storage.
- Integrating tools with the agent simplifies CLI and voice access.
- The MCP protocol provides a standardized way for AI models to interact with external tools and resources.
- Implementing both client and server MCP functionality makes Intuit more versatile and extensible.
- Running the MCP server in a separate process improves stability and prevents blocking the main application.
- Dynamic tool discovery and registration enables a more flexible and extensible architecture.
- Providing fallback mechanisms for MCP tools ensures robustness even when servers are unavailable.
- Converting between JSON schema and Pydantic models requires careful handling of types and optional fields.
- The RAG pipeline requires proper connection between the vector store and the FilesystemTool to enable semantic search.
- When transitioning to MCP-based tools, it's important to ensure that existing functionality like the vector store is properly integrated.
- Proper scoping of imports in functions is crucial to avoid UnboundLocalError issues.
- Enhancing the agent's prompt with specific instructions for handling certain types of queries improves user experience.
- Handling edge cases like empty queries in search functionality provides a more intuitive user experience.
- The RAG pipeline supports different query types with different behaviors:
  * Empty string or '*' queries list all indexed files
  * Specific search terms perform semantic search to find relevant documents
  * Natural language queries find documents related to the query, even if they don't contain the exact words
- Clear instructions in the agent's prompt about how to use different query types are essential for effective RAG pipeline utilization.
- Detailed logging is crucial for diagnosing issues in complex systems like the RAG pipeline.
- Proper indentation in Python code is critical, especially in nested try-except blocks.
- When implementing search functionality, it's important to handle both the "no results found" case and the "error during search" case separately.
- When using a tool-based architecture with MCP, it's essential to register all tools properly with the agent to ensure they can be used when needed.
- The CustomMCPTool pattern provides a flexible way to expose tools to the agent, but requires explicit registration for each tool.
- Sometimes it's necessary to bypass the agent's normal execution flow for specific commands to ensure they work correctly, especially for critical functionality like the RAG pipeline.
- Adding special cases in the agent's process_input method allows for direct tool usage when needed, providing a more reliable user experience for important commands.
- When using tools directly, it's important to use the appropriate execution method (_run for synchronous execution or _arun for asynchronous execution) based on the tool's implementation.
- Some tools only support async execution, requiring the use of the _arun method instead of _run.
- Comprehensive logging is essential for debugging complex systems, especially when multiple components interact.
- Adding caller information to logs helps track the flow of execution through the system.
- Logging both the entry and exit points of methods provides visibility into the system's behavior.
