"""
Unit tests for the MCP client functionality.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from typing import Dict, Any

from intuit.agent import Agent, MCPToolWrapper, AgentConfig
from intuit.tools.basetool import BaseTool


class TestMCPClient:
    """Tests for the MCP client functionality."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        # Mock the memory manager to avoid asyncio issues
        with patch("intuit.memory.manager.IntuitMemoryManager.start") as mock_start:
            agent = Agent(tools=[])
            return agent

    @patch("mcp.client.sse.sse_client")
    @pytest.mark.asyncio
    async def test_connect_to_mcp_server(self, mock_sse_client, agent):
        """Test connecting to an MCP server."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__.return_value = mock_client
        mock_sse_client.return_value = mock_client_context

        # Mock server info and tools
        mock_client.get_server_info.return_value = {
            "name": "Test Server",
            "version": "1.0.0",
        }
        mock_client.list_tools.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"},
                    },
                    "required": ["param1"],
                },
            }
        ]

        # Connect to the server
        result = await agent.connect_to_mcp_server("http://localhost:8000")

        # Wait a bit for the background task to run
        await asyncio.sleep(0.1)

        # Verify the connection was made
        assert "Connecting to MCP server" in result
        # sse_client is called twice: once for test connection, once for background task
        assert mock_sse_client.call_count == 2
        mock_client.get_server_info.assert_called_once()
        mock_client.list_tools.assert_called_once()

        # Verify that the tool was registered
        assert len(agent.mcp_tools) > 0
        assert any(tool.name == "mcp_test_tool" for tool in agent.mcp_tools)

    @patch("mcp.client.sse.sse_client")
    @pytest.mark.asyncio
    async def test_connect_to_mcp_server_error(self, mock_sse_client, agent):
        """Test error handling when connecting to an MCP server."""
        # Setup mock to raise an exception
        mock_sse_client.side_effect = Exception("Connection failed")

        # Connect to the server
        result = await agent.connect_to_mcp_server("http://localhost:8000")

        # Verify the error was handled
        assert "Failed to connect to MCP server" in result
        mock_sse_client.assert_called_once_with("http://localhost:8000")

        # Verify that fallback tools were created
        assert len(agent.mcp_tools) > 0

    def test_mcp_tool_wrapper_init(self):
        """Test initialization of MCPToolWrapper."""
        # Create a mock client
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000"

        # Create a tool info dictionary
        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "integer"},
                },
                "required": ["param1"],
            },
        }

        # Create the wrapper with name and description explicitly set
        wrapper = MCPToolWrapper(mock_client, "test_tool", tool_info)
        wrapper.name = "mcp_test_tool"
        wrapper.description = "A test tool"

        # Verify the wrapper was created correctly
        assert wrapper.name == "mcp_test_tool"
        assert wrapper.tool_name_on_server == "test_tool"
        assert wrapper.client == mock_client
        assert "A test tool" in wrapper.description
        assert wrapper.schema == tool_info["parameters"]
        assert wrapper.args_schema_pydantic is not None

    @pytest.mark.asyncio
    async def test_mcp_tool_wrapper_arun(self):
        """Test asynchronous execution of MCPToolWrapper."""
        # Create a mock client
        mock_client = MagicMock()
        mock_client.call_tool_async = AsyncMock(return_value="Tool result")

        # Create a tool info dictionary
        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"param1": {"type": "string"}},
            },
        }

        # Create the wrapper with name and description explicitly set
        wrapper = MCPToolWrapper(mock_client, "test_tool", tool_info)
        wrapper.name = "mcp_test_tool"
        wrapper.description = "A test tool"

        # Execute the tool
        result = await wrapper._arun(param1="test")

        # Verify the tool was executed correctly
        assert result == "Tool result"
        mock_client.call_tool_async.assert_called_once_with(
            tool_name="test_tool", arguments={"param1": "test"}
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_wrapper_arun_error(self):
        """Test error handling in MCPToolWrapper._arun."""
        # Create a mock client
        mock_client = MagicMock()
        mock_client.call_tool_async = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )

        # Create a tool info dictionary
        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        }

        # Create the wrapper with name and description explicitly set
        wrapper = MCPToolWrapper(mock_client, "test_tool", tool_info)
        wrapper.name = "mcp_test_tool"
        wrapper.description = "A test tool"

        # Execute the tool
        result = await wrapper._arun()

        # Verify the error was handled
        assert "Error during MCP tool" in result
        assert "Tool execution failed" in result
        mock_client.call_tool_async.assert_called_once()

    def test_mcp_tool_wrapper_run(self):
        """Test synchronous execution of MCPToolWrapper."""
        # Create a mock client
        mock_client = MagicMock()

        # Create a tool info dictionary
        tool_info = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        }

        # Create the wrapper with name and description explicitly set
        wrapper = MCPToolWrapper(mock_client, "test_tool", tool_info)
        wrapper.name = "mcp_test_tool"
        wrapper.description = "A test tool"

        # Patch the _arun method with an async mock that returns the expected result
        async def mock_arun_func(**kwargs):
            return "Tool result"

        with patch.object(wrapper, "_arun", side_effect=mock_arun_func) as mock_arun:
            # Execute the tool
            result = wrapper._run()

            # Verify the tool was executed correctly
            assert result == "Tool result"
            mock_arun.assert_called_once_with()

    def test_create_pydantic_schema_from_json(self):
        """Test creating a Pydantic schema from JSON schema."""
        # Create a mock client
        mock_client = MagicMock()

        # Create a wrapper with name and description explicitly set
        wrapper = MCPToolWrapper(mock_client, "test_tool", {"description": "Test"})
        wrapper.name = "mcp_test_tool"
        wrapper.description = "Test"

        # Test with a valid schema
        json_schema = {
            "type": "object",
            "properties": {
                "string_param": {"type": "string"},
                "int_param": {"type": "integer"},
                "float_param": {"type": "number"},
                "bool_param": {"type": "boolean"},
                "array_param": {"type": "array"},
                "object_param": {"type": "object"},
            },
            "required": ["string_param", "int_param"],
        }

        schema_class = wrapper._create_pydantic_schema_from_json(json_schema)

        # Verify the schema was created correctly
        assert schema_class is not None
        assert hasattr(schema_class, "__annotations__")
        annotations = schema_class.__annotations__
        assert "string_param" in annotations
        assert "int_param" in annotations
        assert "float_param" in annotations
        assert "bool_param" in annotations
        assert "array_param" in annotations
        assert "object_param" in annotations

        # Test with an empty schema
        empty_schema = {"type": "object", "properties": {}}
        empty_class = wrapper._create_pydantic_schema_from_json(empty_schema)
        assert empty_class is not None
        assert hasattr(empty_class, "__annotations__")
        assert len(empty_class.__annotations__) == 0

        # Test with a non-object schema
        non_object_schema = {"type": "string"}
        non_object_class = wrapper._create_pydantic_schema_from_json(non_object_schema)
        assert non_object_class is not None
        assert hasattr(non_object_class, "__annotations__")
        assert len(non_object_class.__annotations__) == 0
