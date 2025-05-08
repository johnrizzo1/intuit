"""
Tests for the CLI interface.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
import asyncio

from intuit.ui.cli import CLIInterface

class TestOutput(DummyOutput):
    """Test output class that captures output."""
    
    def __init__(self):
        super().__init__()
        self.written = []
    
    def write(self, data):
        self.written.append(data)
    
    def write_line(self, data):
        self.written.append(data + "\n")
    
    def get_output(self):
        return "".join(self.written)

class TestInput:
    """Test input class that provides predefined input."""
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.current = 0
    
    def read_text(self):
        if self.current < len(self.inputs):
            text = self.inputs[self.current]
            self.current += 1
            return text
        return ""

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = AsyncMock()
    agent.run.return_value = "Test response"
    return agent

@pytest.fixture
def cli_interface(mock_agent):
    """Create a CLI interface with a mock agent."""
    return CLIInterface(mock_agent)

@pytest.mark.asyncio
async def test_cli_interface_initialization(cli_interface, mock_agent):
    """Test that the CLI interface initializes correctly."""
    assert cli_interface.agent == mock_agent
    assert cli_interface.running is True

@pytest.mark.asyncio
async def test_cli_interface_run(cli_interface, mock_agent):
    """Test the run method of the CLI interface."""
    test_input = TestInput(["test query", "exit"])
    test_output = TestOutput()
    
    # Run the interface with a timeout
    try:
        await asyncio.wait_for(
            cli_interface.run(input_obj=test_input, output_obj=test_output),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        cli_interface.running = False  # Force stop the interface
        raise
    
    # Verify the agent was called correctly
    mock_agent.run.assert_called_once_with("test query")
    
    # Verify the output
    output = test_output.get_output()
    assert "Welcome to Intuit" in output
    assert "Assistant: Test response" in output
    assert "Goodbye" in output

@pytest.mark.asyncio
async def test_cli_interface_error_handling(cli_interface, mock_agent):
    """Test error handling in the CLI interface."""
    test_input = TestInput(["test query", "exit"])
    test_output = TestOutput()
    mock_agent.run.side_effect = Exception("Test error")
    
    # Run the interface with a timeout
    try:
        await asyncio.wait_for(
            cli_interface.run(input_obj=test_input, output_obj=test_output),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        cli_interface.running = False  # Force stop the interface
        raise
    
    # Verify error handling
    output = test_output.get_output()
    assert "Error: Test error" in output

@pytest.mark.asyncio
async def test_cli_interface_keyboard_interrupt(cli_interface, mock_agent):
    """Test handling of keyboard interrupt."""
    test_input = TestInput(["test query"])  # No exit command needed
    test_output = TestOutput()
    
    # Simulate keyboard interrupt after first input
    def side_effect(*args):
        raise KeyboardInterrupt()
    mock_agent.run.side_effect = side_effect
    
    # Run the interface with a timeout
    try:
        await asyncio.wait_for(
            cli_interface.run(input_obj=test_input, output_obj=test_output),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        cli_interface.running = False  # Force stop the interface
        raise
    
    # Verify keyboard interrupt handling
    output = test_output.get_output()
    assert "Goodbye" in output

@pytest.mark.asyncio
async def test_cli_interface_multiple_queries(cli_interface, mock_agent):
    """Test handling of multiple queries."""
    test_input = TestInput(["query 1", "query 2", "exit"])
    test_output = TestOutput()
    responses = ["Response 1", "Response 2"]
    mock_agent.run.side_effect = responses
    
    # Run the interface with a timeout
    try:
        await asyncio.wait_for(
            cli_interface.run(input_obj=test_input, output_obj=test_output),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        cli_interface.running = False  # Force stop the interface
        raise
    
    # Verify multiple queries were handled correctly
    assert mock_agent.run.call_count == 2
    output = test_output.get_output()
    assert "Assistant: Response 1" in output
    assert "Assistant: Response 2" in output 