#!/bin/bash

# Stop the MCP server if it's running
echo "Stopping MCP server..."
pkill -f "intuit mcp start-server" || echo "No MCP server process found"

# Wait a moment for the server to stop
sleep 2

# Start the MCP server in the background
echo "Starting MCP server..."
intuit mcp start-server &

# Wait for the server to start
sleep 5

# Run the chat command
echo "Testing Hacker News query..."
intuit chat "What are the top stories on Hacker News right now?"