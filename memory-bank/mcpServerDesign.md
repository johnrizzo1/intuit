# MCP Server Design Document

This document outlines the design and implementation details for the Model Context Protocol (MCP) server and client functionality in Intuit.

## Overview

The MCP integration will enable Intuit to:

1. Function as an MCP server, exposing its tools to other AI agents
2. Function as an MCP client, connecting to external MCP servers to use their tools

## MCP Server Implementation

### Core Components

#### 1. FastMCP Server

```python
# src/intuit/mcp_server.py
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
import io
import pyautogui

# Create server instance
mcp = FastMCP("Intuit Tools", dependencies=["pyautogui", "pydantic"])

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
```

#### 2. Tool Decorators

Tools will be exposed using the `@mcp.tool()` decorator:

```python
# Screenshot tool example
@mcp.tool()
def take_screenshot() -> Image:
    """
    Take a screenshot of the user's screen and return it as an image.
    
    Use this tool anytime the user wants you to look at something they're doing.
    """
    # Capture screenshot
    screenshot = pyautogui.screenshot()
    
    # Convert to bytes
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    
    # Return as Image type
    return Image(data=buffer.getvalue(), format="png")

# Calendar tool example
@mcp.tool()
def calendar_add(event: str) -> str:
    """
    Add a new calendar event.
    
    Args:
        event: Details of the calendar event to add
        
    Returns:
        Confirmation message with the event ID
    """
    from .tools.calendar import CalendarTool
    calendar_tool = CalendarTool()
    return calendar_tool.add_event(event)
```

#### 3. Server Management

```python
class MCPServerManager:
    """Manager for the MCP server instance."""
    
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.server = mcp
        self.is_running = False
        
    def start(self):
        """Start the MCP server."""
        if not self.is_running:
            # Start in a separate thread or process
            import threading
            self.server_thread = threading.Thread(
                target=self.server.run,
                kwargs={"host": self.host, "port": self.port}
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            return f"MCP server started at http://{self.host}:{self.port}"
        return "MCP server is already running"
        
    def stop(self):
        """Stop the MCP server."""
        if self.is_running:
            # Implement server shutdown logic
            self.is_running = False
            return "MCP server stopped"
        return "MCP server is not running"
```

## MCP Client Implementation

### Core Components

#### 1. MCP Client Connection

```python
# In agent.py
from mcp.client import Client as MCPClient

class Agent:
    # ... existing code ...
    
    def connect_to_mcp_server(self, url: str) -> str:
        """
        Connect to an external MCP server.
        
        Args:
            url: The URL of the MCP server to connect to
            
        Returns:
            Connection status message
        """
        try:
            client = MCPClient(url)
            self.mcp_clients[url] = client
            return f"Connected to MCP server at {url}"
        except Exception as e:
            return f"Failed to connect to MCP server: {str(e)}"
```

#### 2. MCP Tool Wrapper

```python
class MCPToolWrapper(BaseTool):
    """Wrapper for tools provided by external MCP servers."""
    
    def __init__(self, client, tool_name):
        self.client = client
        self.tool_name = tool_name
        self.name = f"mcp_{tool_name}"
        
        # Get tool schema and description
        tool_info = client.get_tool_info(tool_name)
        self.description = tool_info.get("description", f"MCP tool: {tool_name}")
        self.schema = tool_info.get("schema", {})
        
    async def _arun(self, **kwargs):
        """Execute the MCP tool."""
        try:
            result = await self.client.call_tool_async(self.tool_name, kwargs)
            return result
        except Exception as e:
            return f"Error executing MCP tool {self.tool_name}: {str(e)}"
```

## Integration with CLI

### CLI Commands for MCP Server

```python
# In main.py
mcp_app = typer.Typer(name="mcp", help="Manage MCP server and client")
app.add_typer(mcp_app)

@mcp_app.command()
def start_server(
    host: str = typer.Option("localhost", help="Server host"),
    port: int = typer.Option(8000, help="Server port")
):
    """Start the MCP server."""
    from .mcp_server import MCPServerManager
    manager = MCPServerManager(host=host, port=port)
    print(manager.start())
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping MCP server...")
        manager.stop()

@mcp_app.command()
def connect(url: str):
    """Connect to an external MCP server."""
    agent = asyncio.run(create_agent())
    print(agent.connect_to_mcp_server(url))
```

## Example Usage

### Starting the MCP Server

```bash
intuit mcp start-server --host localhost --port 8000
```

### Using MCP Tools in Agent

When the agent receives a request that requires an MCP tool:

1. The agent checks if it has the required tool locally
2. If not, it checks connected MCP servers for the tool
3. If found, it uses the MCPToolWrapper to execute the tool
4. If not found, it informs the user that the tool is not available

## Implementation Considerations

1. **Threading/Multiprocessing**: The MCP server should run in a separate thread or process to avoid blocking the main application.

2. **Error Handling**: Robust error handling for network issues, timeouts, and invalid responses.

3. **Authentication**: Consider adding authentication for MCP servers in production environments.

4. **Resource Management**: Implement proper cleanup of resources when shutting down the server or disconnecting clients.

5. **Tool Discovery**: Implement a mechanism for discovering and registering tools from MCP servers.

## Next Steps

1. Implement the basic MCP server module
2. Create wrappers for existing tools
3. Implement the screenshot tool as a proof of concept
4. Add CLI commands for server management
5. Update the agent to support MCP tools
6. Test with external MCP servers
