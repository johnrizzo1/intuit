"""
Main entry point for Intuit - Refactored version.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import logging configuration
# Import agent factory
from .agent_factory import create_agent

# Import CLI modules
from .cli.cli_calendar import calendar_app
from .cli.cli_mcp import mcp_cli_app
from .cli.cli_memory import memory_app
from .cli.cli_notes import notes_app
from .cli.cli_reminders import reminders_app
from .logging_config import configure_logging

# Import UI modules
from .ui.cli import run_cli
from .ui.voice import run_voice


# Create the main app with verbosity callback
def set_verbosity(
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (can be used multiple times)",
    )
):
    """Set the verbosity level based on the number of -v flags."""
    configure_logging(verbose=verbose, quiet=(verbose == 0))
    return verbose


# Create main Typer app
app = typer.Typer(no_args_is_help=True, callback=set_verbosity)

# Mount tool CLI apps onto the main app
app.add_typer(calendar_app)
app.add_typer(notes_app)
app.add_typer(reminders_app)
app.add_typer(mcp_cli_app)
app.add_typer(memory_app)


@app.command()
def gui(
    model: str = typer.Option(
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        help="Model to use (can also be set via OPENAI_MODEL_NAME env var)",
    ),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(
        False, "--index/--no-index", help="Enable/disable filesystem indexing"
    ),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(
        True, "--weather/--no-weather", help="Enable/disable weather information"
    ),
    voice: bool = typer.Option(
        True, "--voice/--no-voice", help="Enable/disable voice interface"
    ),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(
        None, help="Type of OpenAI API (openai/azure)"
    ),
    openai_api_version: Optional[str] = typer.Option(
        None, help="API version for Azure OpenAI"
    ),
):
    """Start the Intuit assistant in GUI mode with the hockey puck interface."""
    try:
        # Check if PySide6 is installed
        try:
            import PySide6
        except ImportError:
            print(
                "Error: PySide6 is not installed. Please install it with: pip install PySide6"
            )
            sys.exit(1)

        # Create the agent
        print("Initializing Intuit agent...")
        agent = asyncio.run(
            create_agent(
                model=model,
                temperature=temperature,
                index_filesystem=index,
                filesystem_path=index_path,
                enable_gmail=enable_gmail,
                enable_weather=enable_weather,
                use_voice=voice,
                voice_language=voice_language,
                voice_slow=voice_slow,
                openai_api_base=openai_api_base,
                openai_api_type=openai_api_type,
                openai_api_version=openai_api_version,
            )
        )

        # Create a configuration dictionary
        config = {
            "voice_enabled": voice,
            "voice_language": voice_language,
            "voice_slow": voice_slow,
        }

        # Import the agent GUI module
        try:
            from intuit.ui.gui.agent_gui import run_agent_gui

            # Run the agent GUI
            print("Starting Intuit in GUI mode...")
            if voice:
                print("Voice interface enabled")

            run_agent_gui(agent=agent, config=config, block=True)

        except ImportError as e:
            print(f"Error importing GUI module: {e}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nExiting GUI mode...")
    except Exception as e:
        print(f"Error starting GUI: {e}", file=sys.stderr)
        sys.exit(1)


@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to process"),
    model: str = typer.Option(
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        help="Model to use (can also be set via OPENAI_MODEL_NAME env var)",
    ),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(
        False, "--index/--no-index", help="Enable/disable filesystem indexing"
    ),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(
        True, "--gmail/--no-gmail", help="Enable/disable Gmail integration"
    ),
    enable_weather: bool = typer.Option(
        True, "--weather/--no-weather", help="Enable/disable weather information"
    ),
    use_voice: bool = typer.Option(
        False, "--voice/--no-voice", help="Enable/disable voice output"
    ),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(
        None, help="Type of OpenAI API (openai/azure)"
    ),
    openai_api_version: Optional[str] = typer.Option(
        None, help="API version for Azure OpenAI"
    ),
    quiet: bool = typer.Option(
        False, help="Suppress welcome message in non-interactive mode"
    ),
):
    """Start the Intuit assistant in interactive mode or process a single query."""
    # Automatically set quiet mode if input is being piped
    if not sys.stdin.isatty():
        quiet = True

    agent = asyncio.run(
        create_agent(
            model=model,
            temperature=temperature,
            index_filesystem=index,
            filesystem_path=index_path,
            enable_gmail=enable_gmail,
            enable_weather=enable_weather,
            use_voice=use_voice,
            voice_language=voice_language,
            voice_slow=voice_slow,
            openai_api_base=openai_api_base,
            openai_api_type=openai_api_type,
            openai_api_version=openai_api_version,
        )
    )
    asyncio.run(run_cli(agent, query, quiet))


@app.command()
def voice(
    model: str = typer.Option(
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        help="Model to use (can also be set via OPENAI_MODEL_NAME env var)",
    ),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(
        False, "--index/--no-index", help="Enable/disable filesystem indexing"
    ),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(
        True, "--weather/--no-weather", help="Enable/disable weather information"
    ),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(
        None, help="Type of OpenAI API (openai/azure)"
    ),
    openai_api_version: Optional[str] = typer.Option(
        None, help="API version for Azure OpenAI"
    ),
):
    """Start the Intuit assistant in voice mode."""
    agent = asyncio.run(
        create_agent(
            model=model,
            temperature=temperature,
            index_filesystem=index,
            filesystem_path=index_path,
            enable_gmail=enable_gmail,
            enable_weather=enable_weather,
            use_voice=True,  # Always enable voice in voice mode
            voice_language=voice_language,
            voice_slow=voice_slow,
            openai_api_base=openai_api_base,
            openai_api_type=openai_api_type,
            openai_api_version=openai_api_version,
        )
    )
    try:
        # Reminder service is now started and stopped inside run_voice
        asyncio.run(run_voice(agent))
    except KeyboardInterrupt:
        print("\nStopping voice interface...")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
