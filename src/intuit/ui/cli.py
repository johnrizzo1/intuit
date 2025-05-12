"""
CLI interface for Intuit.
"""
import asyncio
from typing import Optional, Any
from prompt_toolkit import PromptSession
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..agent import Agent

class CLIInterface:
    """Command-line interface for Intuit."""
    
    def __init__(self, agent: Agent, quiet: bool = False):
        """Initialize the CLI interface."""
        self.agent = agent
        self.console = Console()
        self.prompt = PromptSession()
        self.running = True
        self.quiet = quiet
    
    async def _get_input(self, input_obj: Optional[Input] = None) -> str:
        """Get input from the user."""
        try:
            # For testing or non-interactive input
            if input_obj:
                text = input_obj.read_text()
                if not text:
                    self.running = False  # Stop if no more input
                return text
            
            # For real usage, use prompt_toolkit
            with patch_stdout():
                return await self.prompt.prompt_async("You: ")
        except (EOFError, KeyboardInterrupt):
            self.running = False
            return ""
        except Exception as e:
            if input_obj:  # For testing
                return ""
            raise  # Re-raise for real usage
    
    def _display_response(self, response: str, output_obj: Optional[Output] = None):
        """Display the agent's response."""
        if output_obj:
            output_obj.write_line(f"Assistant: {response}")
            output_obj.flush()
        else:
            self.console.print(Panel(
                Markdown(response),
                title="Assistant",
                border_style="blue"
            ))
    
    async def run(self, input_obj: Optional[Input] = None, output_obj: Optional[Output] = None) -> None:
        """
        Run the CLI interface.
        
        Args:
            input_obj: Optional input object for testing
            output_obj: Optional output object for testing
        """
        if not self.quiet:
            if output_obj:
                output_obj.write_line("Welcome to Intuit! Type 'exit' to quit.")
            else:
                self.console.print("Welcome to Intuit! Type 'exit' to quit.")
        
        while self.running:
            try:
                # Get user input
                user_input = await self._get_input(input_obj)
                if not user_input:  # Handle empty input in tests
                    continue
                
                # Check for exit command
                if user_input.lower().strip() in ('exit', 'quit'):
                    self.running = False
                    if output_obj:
                        output_obj.write_line("Goodbye!")
                    else:
                        self.console.print("Goodbye!")
                    break
                
                # Process input and get response
                response = await self.agent.run(user_input)
                
                # Display response
                self._display_response(response, output_obj)
                
            except KeyboardInterrupt:
                self.running = False
                if output_obj:
                    output_obj.write_line("\nGoodbye!")
                else:
                    self.console.print("\nGoodbye!")
                break
            except Exception as e:
                if output_obj:
                    output_obj.write_line(f"Error: {str(e)}")
                else:
                    self.console.print(f"Error: {str(e)}", style="red")
                continue

async def read_non_interactive_input() -> str:
    """Read input in non-interactive mode."""
    try:
        import sys
        return sys.stdin.read().strip()
    except (EOFError, KeyboardInterrupt):
        return ""

async def process_single_query(agent: Agent, query: str) -> None:
    """Process a single query without any terminal features."""
    try:
        response = await agent.run(query)
        # Print a simple formatted response without any terminal features
        print("\nIntuit:")
        print("─" * 80)
        print(response)
        print("─" * 80)
    except Exception as e:
        print(f"\nError: {str(e)}")

async def run_cli(agent: Agent, query: Optional[str] = None, quiet: bool = False) -> None:
    """Run the CLI interface."""
    try:
        if query:
            # Process single query without any terminal features
            await process_single_query(agent, query)
            return
        
        # Check if input is from a pipe
        import sys
        if not sys.stdin.isatty():
            # Read from pipe without any terminal features
            query = await read_non_interactive_input()
            if query:
                await process_single_query(agent, query)
            return
        
        # Run interactive interface
        interface = CLIInterface(agent, quiet=quiet)
        await interface.run()
    finally:
        # Properly shut down MCP clients to avoid "unhandled errors in a TaskGroup" message
        await agent.shutdown_mcp_clients()

def main() -> None:
    """CLI entry point."""
    # Initialize agent and run interface
    agent = Agent()  # Add configuration as needed
    asyncio.run(run_cli(agent)) 