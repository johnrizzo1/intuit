"""
Main entry point for Intuit.
"""
import asyncio
import typer
import sys
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .agent import Agent, AgentConfig
from .tools.web_search import WebSearchTool
from .tools.gmail import GmailTool
from .tools.weather import WeatherTool
from .vector_store.indexer import VectorStore
from .ui.cli import run_cli
from .ui.voice import run_voice

app = typer.Typer(no_args_is_help=True)

def create_agent(
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    index_filesystem: bool = False,
    filesystem_path: Optional[Path] = None,
    enable_gmail: bool = False,
    enable_weather: bool = False
) -> Agent:
    """
    Create and configure the Intuit agent with all tools.
    
    Args:
        model: Model to use
        temperature: Model temperature
        index_filesystem: Whether to index the filesystem
        filesystem_path: Path to index (defaults to home directory)
        enable_gmail: Whether to enable Gmail integration
        enable_weather: Whether to enable weather information
        
    Returns:
        Configured agent instance
    """
    # Initialize tools
    tools = [
        WebSearchTool(),
    ]
    
    # Add Gmail tool if enabled
    if enable_gmail:
        tools.append(GmailTool())
        
    # Add Weather tool if enabled
    if enable_weather:
        tools.append(WeatherTool())
    
    # Initialize vector store if requested
    if index_filesystem:
        vector_store = VectorStore()
        if filesystem_path:
            vector_store.index_directory(filesystem_path)
        else:
            vector_store.index_directory(Path.home())
    
    # Create agent configuration
    config = AgentConfig(
        model_name=model,
        temperature=temperature,
    )
    
    # Create and return agent
    return Agent(tools=tools, config=config)

@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to process"),
    model: str = typer.Option("gpt-4-turbo-preview", help="Model to use"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    no_index: bool = typer.Option(True, help="Don't index filesystem"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(False, help="Enable weather information"),
    quiet: bool = typer.Option(False, help="Suppress welcome message in non-interactive mode"),
):
    """Start the Intuit assistant in interactive mode or process a single query."""
    # Automatically set quiet mode if input is being piped
    if not sys.stdin.isatty():
        quiet = True

    agent = create_agent(
        model=model,
        temperature=temperature,
        index_filesystem=not no_index,
        filesystem_path=index_path,
        enable_gmail=enable_gmail,
        enable_weather=enable_weather
    )
    asyncio.run(run_cli(agent, query, quiet))

@app.command()
def voice(
    model: str = typer.Option("gpt-4-turbo-preview", help="Model to use"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    no_index: bool = typer.Option(False, help="Don't index filesystem"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(False, help="Enable weather information"),
):
    """Start the Intuit assistant in voice mode."""
    agent = create_agent(
        model=model,
        temperature=temperature,
        index_filesystem=not no_index,
        filesystem_path=index_path,
        enable_gmail=enable_gmail,
        enable_weather=enable_weather
    )
    try:
        asyncio.run(run_voice(agent))
    except KeyboardInterrupt:
        print("\nStopping voice interface...")

def main():
    """Main entry point."""
    app() 