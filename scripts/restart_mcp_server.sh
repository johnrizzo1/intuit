#!/bin/bash

# Stop the MCP server if it's running
echo "Stopping MCP server..."
pkill -f "intuit mcp start-server" || echo "No MCP server process found"

# Wait a moment for the server to stop
sleep 2

# Start the MCP server
echo "Starting MCP server..."
intuit mcp start-server &

# Wait for the server to start
sleep 5

echo "MCP server restarted. You can now use the Hacker News tool."
echo "Try: intuit chat \"What's the latest on Hacker News?\""