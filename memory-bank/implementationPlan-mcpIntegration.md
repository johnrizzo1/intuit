# Implementation Plan: MCP Integration

This document outlines the steps for implementing Model Context Protocol (MCP) integration in Intuit, allowing it to function as both an MCP client and server.

## Overview

The Model Context Protocol (MCP) enables communication between AI models and external tools/resources. By implementing MCP in Intuit, we'll enable:

1. **MCP Server Functionality**: Expose Intuit's tools as MCP resources that can be accessed by other AI agents
2. **MCP Client Functionality**: Allow Intuit to connect to and use external MCP servers

## Milestone: Implement MCP Integration

### MCP Server Implementation

- [x] Create a base MCP server module (`src/intuit/mcp_server.py`)
  - [x] Implement FastMCP server initialization
  - [x] Define server configuration options
  - [x] Create server start/stop functionality

- [x] Expose existing tools as MCP resources
  - [x] Create MCP tool wrappers for Calendar tool
  - [x] Create MCP tool wrappers for Notes tool
  - [x] Create MCP tool wrappers for Reminders tool
  - [x] Create MCP tool wrappers for Weather tool
  - [x] Create MCP tool wrappers for Web Search tool
  - [x] Create MCP tool wrappers for Filesystem tool

- [x] Implement server discovery mechanism
  - [x] Add server registration with the agent
  - [x] Create endpoint for listing available tools

- [x] Add CLI commands for MCP server management
  - [x] Add command to start the MCP server
  - [x] Add command to list available MCP tools
  - [x] Add command to stop the MCP server

### MCP Client Implementation

- [x] Create MCP client functionality in the agent
  - [x] Implement connection to external MCP servers
  - [x] Add support for discovering available tools on connected servers
  - [x] Create tool wrappers for external MCP tools

- [x] Add CLI commands for MCP client management
  - [x] Add command to connect to an external MCP server
  - [x] Add command to list connected MCP servers
  - [x] Add command to disconnect from an MCP server

### Integration with Agent

- [x] Update agent to support MCP tools
  - [x] Modify `_create_agent_executor` to handle MCP tools
  - [x] Update system prompt to include MCP tool usage instructions
  - [x] Add MCP tool execution logic

- [x] Implement automatic MCP server startup
  - [x] Start MCP server when agent is initialized (if configured)
  - [x] Register agent tools with the MCP server

### Example: Screenshot Tool Implementation

- [x] Create a screenshot tool using MCP
  - [x] Implement tool using pyautogui for capturing screenshots
  - [x] Expose the tool via MCP server
  - [x] Add CLI command for taking screenshots
  - [x] Add voice command support for taking screenshots

## General Tasks

- [x] Update `activeContext.md` and `progress.md` as milestones are completed
- [ ] Document MCP integration in README.md
- [ ] Create usage examples for MCP tools
- [ ] Write unit tests for MCP server and client functionality
- [x] Ensure proper error handling and user feedback
