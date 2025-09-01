"""
MCP CLI commands.
"""

import asyncio
import time
import typer
import builtins
import logging
from .shared import create_agent_sync
from ..mcp_server import MCPServerManager, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT

# Create the MCP CLI app
mcp_cli_app = typer.Typer(
    name="mcp", help="Manage MCP server and client connections", no_args_is_help=True
)

logger = logging.getLogger(__name__)


@mcp_cli_app.command("start-server")
def start_mcp_server(
    host: str = typer.Option(DEFAULT_SERVER_HOST, help="Server host"),
    port: int = typer.Option(DEFAULT_SERVER_PORT, help="Server port"),
):
    """Start the Intuit MCP server."""
    logger.info(f"Attempting to start MCP server on {host}:{port}")
    manager = MCPServerManager(host=host, port=port)
    print(manager.start())
    if manager.is_running:
        print(f"MCP Server is running on http://{host}:{port}")
        # Keep server running interactively
        print("\nMCP Server running interactively. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping MCP server...")
            # Get tools info before stopping the server
            tools_info = manager.get_tools()
            manager.stop()
            print("MCP server stopped.")

            # Print tools info after server is stopped
            if isinstance(tools_info, list):
                for tool_data in tools_info:
                    print(
                        f"  - {tool_data.get('name')}: {tool_data.get('description')}"
                    )
            else:
                print(f"  Could not retrieve tool list: {tools_info}")
        print(
            "\nServer started successfully. Use 'uv run intuit mcp list-mcp-tools' to see available tools."
        )
        return  # Exit after starting server
    else:
        print("MCP Server failed to start. Check logs for details.")


@mcp_cli_app.command("list-mcp-tools")
def list_mcp_server_tools():
    """List available tools on the local MCP server in a human-readable format."""
    agent = None
    try:
        # Create a temporary agent to use its list_mcp_tools method
        agent = create_agent_sync()

        # Get tools from the MCP server
        if not agent.mcp_tools:
            # If no MCP tools are available in the agent, fall back to the static definition
            logger.info(
                "No MCP tools available in agent. Listing tools from global mcp_server instance definition."
            )
            from ..mcp_server import get_registered_tools

            tools_info = get_registered_tools()

            # Use builtins.list to reliably access the list type
            if not isinstance(tools_info, builtins.list):
                print(f"Could not retrieve tools: {tools_info}")
                return

            if not tools_info:
                print("No MCP tools found.")
                return

            # Format and print the tools using the same formatting as in the agent
            print("\nAvailable MCP Tools:")
            print("--------------------")

            # Group tools by prefix
            grouped_tools = {}
            for tool in tools_info:
                name = tool.get("name", "unknown")
                if isinstance(name, str):
                    prefix = name.split("_")[0] if "_" in name else "general"
                else:
                    prefix = "general"
                if prefix not in grouped_tools:
                    grouped_tools[prefix] = []
                grouped_tools[prefix].append(tool)

            # Print grouped tools
            for prefix in sorted(grouped_tools.keys()):
                if not grouped_tools[prefix]:
                    continue

                print(f"\n{prefix.capitalize()} Tools:")
                for tool in sorted(
                    grouped_tools[prefix], key=lambda x: x.get("name", "")
                ):
                    name = tool.get("name", "N/A")
                    description = (
                        tool.get("description") or "No description available."
                    ).strip()

                    # Format parameters
                    param_str = " (No parameters)"
                    if isinstance(
                        tool.get("parameters"), dict
                    ) and "properties" in tool.get("parameters", {}):
                        param_names = []
                        properties = tool.get("parameters", {}).get("properties", {})
                        if isinstance(properties, dict):
                            # Use a list comprehension instead of list(keys())
                            param_names = [key for key in properties.keys()]
                        if param_names:
                            param_str = f" (Parameters: {', '.join(param_names)})"

                    print(f"  - {name}{param_str}")
                    desc_lines = description.split("\n")
                    print(f"    > {desc_lines[0].strip()}")
                    for line in desc_lines[1:]:
                        stripped_line = line.strip()
                        if stripped_line:
                            print(f"      {stripped_line}")

            print("\n--------------------")
            return

        # If MCP tools are available in the agent, get the tools information
        tools_info = agent.mcp_tools

        if not tools_info:
            print("No MCP tools found.")
            return

        print("\nAvailable MCP Tools:")
        print("--------------------")

        # Group tools by prefix (e.g., 'calendar', 'notes')
        grouped_tools = {}
        for tool in tools_info:
            # Handle both dictionary-like objects and objects with attributes
            if hasattr(tool, "get") and callable(tool.get):
                # Dictionary-like object
                name = tool.get("name", "unknown")
            else:
                # Object with attributes
                name = getattr(tool, "name", "unknown")

            # Use the part before the first underscore as the group key
            if isinstance(name, str):
                prefix = name.split("_")[0] if "_" in name else "general"
            else:
                prefix = "general"
            if prefix not in grouped_tools:
                grouped_tools[prefix] = []
            grouped_tools[prefix].append(tool)

        # Sort groups alphabetically and tools within groups alphabetically
        for prefix in sorted(grouped_tools.keys()):
            # Skip empty groups if any somehow occur
            if not grouped_tools[prefix]:
                continue

            print(f"\n{prefix.capitalize()} Tools:")

            # Sort tools by name, handling both dict and object types
            def get_tool_name(tool):
                if hasattr(tool, "get") and callable(tool.get):
                    name = tool.get("name", "")
                else:
                    name = getattr(tool, "name", "")
                return str(name) if name else ""

            for tool in sorted(grouped_tools[prefix], key=get_tool_name):
                # Handle both dictionary-like objects and objects with attributes
                if hasattr(tool, "get") and callable(tool.get):
                    # Dictionary-like object
                    name = tool.get("name", "N/A")
                    # Clean up description, handle potential None and extra whitespace/newlines
                    desc = tool.get("description") or "No description available."
                    description = str(desc).strip()
                    params_schema = tool.get("parameters")  # This is the schema dict
                else:
                    # Object with attributes
                    name = getattr(tool, "name", "N/A")
                    # Clean up description, handle potential None and extra whitespace/newlines
                    desc = (
                        getattr(tool, "description", None)
                        or "No description available."
                    )
                    description = str(desc).strip()
                    params_schema = getattr(
                        tool, "schema", None
                    )  # This is the schema dict

                # Format parameters simply - just list names if available
                param_str = " (No parameters)"
                if isinstance(params_schema, dict) and "properties" in params_schema:
                    # Use builtins.list constructor to avoid name shadowing
                    param_names = builtins.list(params_schema["properties"].keys())
                    if param_names:
                        param_str = f" (Parameters: {', '.join(param_names)})"
                    # Handle case where 'properties' exists but is empty
                    elif not param_names:
                        param_str = " (No parameters)"
                # Handle case where tool has args_schema attribute (for CustomMCPTool objects)
                elif hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        # Try to get schema from args_schema
                        schema = tool.args_schema.schema()
                        if "properties" in schema and schema["properties"]:
                            param_names = builtins.list(schema["properties"].keys())
                            if param_names:
                                param_str = f" (Parameters: {', '.join(param_names)})"
                    except Exception as e:
                        logger.debug(f"Error getting parameters from args_schema: {e}")

                print(f"  - {name}{param_str}")
                # Indent description for readability, handle multi-line descriptions
                desc_lines = description.split("\n")
                # Print first line indented, stripping leading/trailing whitespace
                print(f"    > {desc_lines[0].strip()}")
                # Print subsequent lines further indented, stripping whitespace
                for line in desc_lines[1:]:
                    stripped_line = line.strip()
                    if (
                        stripped_line
                    ):  # Avoid printing empty lines from original formatting
                        print(f"      {stripped_line}")

        print("\n--------------------")

    except Exception as e:
        # Log at INFO level instead of ERROR so it only shows with -v flag
        logger.info(
            f"Error listing MCP tools: {e}", exc_info=True
        )  # Log full traceback
        print(f"An error occurred while listing MCP tools. Check logs for details.")
    finally:
        # Properly shut down MCP clients to avoid "unhandled errors in a TaskGroup" message
        if agent:
            asyncio.run(agent.shutdown_mcp_clients())
    return None
