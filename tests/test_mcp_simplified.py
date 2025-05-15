"""
Simplified tests for MCP functionality.
"""
import pytest
from unittest.mock import patch, MagicMock

from intuit.mcp_server import mcp_server, MCPServerManager, get_registered_tools


class TestMCPServer:
    """Tests for the MCP server functionality."""

    def test_mcp_server_initialization(self):
        """Test that the MCP server is initialized correctly."""
        assert mcp_server is not None
        assert mcp_server.name == "Intuit Tools"

    def test_get_registered_tools(self):
        """Test that registered tools can be retrieved."""
        tools = get_registered_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that each tool has the required fields
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)

    @patch("multiprocessing.Process")
    def test_server_manager_start(self, mock_process):
        """Test starting the MCP server manager."""
        # Setup mock
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.is_alive.return_value = True
        mock_process_instance.pid = 12345
        
        # Create server manager and start it
        manager = MCPServerManager(host="localhost", port=8000)
        result = manager.start()
        
        # Verify the server was started
        assert manager.is_running is True
        assert "MCP server started at http://localhost:8000" in result
        mock_process.assert_called_once()
        mock_process_instance.start.assert_called_once()

    @patch("multiprocessing.Process")
    def test_server_manager_stop(self, mock_process):
        """Test stopping the MCP server manager."""
        # Setup mock
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.is_alive.return_value = True
        mock_process_instance.pid = 12345
        
        # Create server manager and start it
        manager = MCPServerManager(host="localhost", port=8000)
        manager.start()
        
        # Stop the server
        result = manager.stop()
        
        # Verify the server was stopped
        assert manager.is_running is False
        assert "MCP server process stopped" in result
        mock_process_instance.terminate.assert_called_once()
        mock_process_instance.join.assert_called_once()

    def test_server_manager_get_tools(self):
        """Test getting tools from the MCP server manager."""
        manager = MCPServerManager(host="localhost", port=8000)
        tools = manager.get_tools()
        
        # When server is not started, get_tools returns a string message
        if not manager.is_running:
            assert isinstance(tools, str)
            assert "MCP Server not initialized" in tools
        else:
            # If server is running, it should return a list
            assert isinstance(tools, list)
            assert len(tools) > 0