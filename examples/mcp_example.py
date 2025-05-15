#!/usr/bin/env python3
"""
Example script demonstrating how to use Intuit's MCP functionality programmatically.

This script shows how to:
1. Start an MCP server
2. Connect to an MCP server
3. List available MCP tools
4. Use MCP tools

Usage:
    python mcp_example.py server  # Start an MCP server
    python mcp_example.py client  # Connect to an MCP server and use its tools
"""

import os
import sys
import asyncio
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import intuit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intuit.mcp_server import MCPServerManager
from intuit.agent import Agent, AgentConfig
from intuit.tools.basetool import BaseTool


async def run_mcp_server():
    """Start an MCP server and keep it running."""
    logger.info("Starting MCP server...")
    
    # Create and start the server manager
    manager = MCPServerManager(host="localhost", port=8000)
    result = manager.start()
    logger.info(result)
    
    # Keep the server running until interrupted
    try:
        logger.info("MCP server is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping MCP server...")
        manager.stop()
        logger.info("MCP server stopped.")


async def run_mcp_client():
    """Connect to an MCP server and use its tools."""
    logger.info("Creating agent...")
    
    # Create an agent with default configuration
    agent = Agent(tools=[])
    
    # Connect to the MCP server
    logger.info("Connecting to MCP server...")
    result = await agent.connect_to_mcp_server("http://localhost:8000")
    logger.info(result)
    
    # Wait a moment for the connection to be established
    await asyncio.sleep(2)
    
    # List available MCP tools
    logger.info("Available MCP tools:")
    tools_info = agent.list_mcp_tools()
    logger.info(tools_info)
    
    # Use some MCP tools
    logger.info("Using MCP tools...")
    
    # Example 1: List calendar events
    logger.info("Listing calendar events...")
    calendar_result = await agent.process_input("List my calendar events")
    logger.info(f"Calendar result: {calendar_result}")
    
    # Example 2: Get weather information
    logger.info("Getting weather information...")
    weather_result = await agent.process_input("What's the weather like in New York?")
    logger.info(f"Weather result: {weather_result}")
    
    # Example 3: Take a screenshot
    logger.info("Taking a screenshot...")
    screenshot_result = await agent.process_input("Take a screenshot of my screen")
    logger.info(f"Screenshot result: {screenshot_result}")
    
    # Shutdown the agent
    logger.info("Shutting down agent...")
    await agent.shutdown_mcp_clients()
    logger.info("Agent shutdown complete.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] not in ["server", "client"]:
        print("Usage: python mcp_example.py [server|client]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        asyncio.run(run_mcp_server())
    elif mode == "client":
        asyncio.run(run_mcp_client())


if __name__ == "__main__":
    main()