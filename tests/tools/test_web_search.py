"""
Tests for the web search tool.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from intuit.tools.web_search import WebSearchTool


@pytest.mark.asyncio
async def test_web_search_tool_initialization(mock_env_vars):
    """Test that the web search tool initializes correctly."""
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "Search the web" in tool.description


@pytest.mark.asyncio
async def test_web_search_tool_run(mock_web_search_tool):
    """Test the web search tool's run method."""
    # Test the run method
    result = await mock_web_search_tool.arun("test query")

    # Verify the result structure
    assert result["query"] == "test query"
    assert isinstance(result["results"], list)
    assert result["total_results"] > 0


@pytest.mark.asyncio
async def test_web_search_tool_error_handling(mock_web_search_tool):
    """Test error handling in the web search tool."""
    # Mock an error
    mock_web_search_tool._search = AsyncMock(side_effect=Exception("API Error"))

    # Test the run method with error
    result = await mock_web_search_tool.arun("test query")

    # Verify error message is returned
    assert "error" in result
    assert "API Error" in result["error"]


@pytest.mark.asyncio
async def test_web_search_tool_no_results(mock_web_search_tool):
    """Test behavior when no results are found."""
    # Mock empty results
    mock_web_search_tool._search = AsyncMock(return_value=[])

    # Test the run method
    result = await mock_web_search_tool.arun("test query")

    # Verify appropriate response
    assert result["total_results"] == 0
    assert len(result["results"]) == 0
