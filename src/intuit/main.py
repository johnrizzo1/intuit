"""
Main entry point for Intuit.
"""
import asyncio
import typer
import sys
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime # Import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from .agent import Agent, AgentConfig
from .tools.web_search import WebSearchTool
from .tools.gmail import GmailTool
from .tools.weather import WeatherTool
from .tools.filesystem import FilesystemTool
from .tools.calendar import CalendarTool # Import CalendarTool
from .tools.notes import NotesTool # Import NotesTool
from .tools.reminders import RemindersTool # Import RemindersTool
from .vector_store.indexer import VectorStore
from .ui.cli import run_cli
from .ui.voice import run_voice
from .reminders_service import ReminderService # Import ReminderService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)

# Create Typer instances for each tool
calendar_app = typer.Typer(name="calendar", help="Manage calendar events")
notes_app = typer.Typer(name="notes", help="Manage notes")
reminders_app = typer.Typer(name="reminders", help="Manage reminders")

# Mount tool Typer instances onto the main app
app.add_typer(calendar_app)
app.add_typer(notes_app)
app.add_typer(reminders_app)

# Define commands for Calendar tool
@calendar_app.command()
def add(event: str):
    """Adds a new calendar event."""
    agent = asyncio.run(create_agent()) # Create agent with default config
    print(agent.tools[1].add_event(event)) # Assuming CalendarTool is the second tool

@calendar_app.command()
def list():
    """Lists all calendar events."""
    agent = asyncio.run(create_agent())
    print(agent.tools[1].list_events())

@calendar_app.command()
def search(keyword: str):
    """Searches calendar events for a keyword."""
    agent = asyncio.run(create_agent())
    print(agent.tools[1].search_events(keyword))

@calendar_app.command()
def delete(filename: str):
    """Deletes a calendar event by filename."""
    agent = asyncio.run(create_agent())
    print(agent.tools[1].delete_event(filename))

# Define commands for Notes tool
@notes_app.command()
def add(content: str):
    """Adds a new note."""
    agent = asyncio.run(create_agent()) # Create agent with default config
    print(agent.tools[2].add_note(content)) # Assuming NotesTool is the third tool

@notes_app.command()
def list():
    """Lists all notes."""
    agent = asyncio.run(create_agent())
    print(agent.tools[2].list_notes())

@notes_app.command()
def search(keyword: str):
    """Searches notes for a keyword."""
    agent = asyncio.run(create_agent())
    print(agent.tools[2].search_notes(keyword))

@notes_app.command()
def delete(id: str):
    """Deletes a note by ID."""
    agent = asyncio.run(create_agent())
    print(agent.tools[2].delete_note(id))

# Define commands for Reminders tool
@reminders_app.command()
def add(content: str, time: Optional[datetime] = typer.Option(None, help="Optional reminder time (ISO 8601 format)")):
    """Adds a new reminder."""
    agent = asyncio.run(create_agent()) # Create agent with default config
    print(agent.tools[3].add_reminder(content, time)) # Assuming RemindersTool is the fourth tool

@reminders_app.command()
def list():
    """Lists all reminders."""
    agent = asyncio.run(create_agent())
    print(agent.tools[3].list_reminders())

@reminders_app.command()
def search(keyword: str):
    """Searches reminders for a keyword."""
    agent = asyncio.run(create_agent())
    print(agent.tools[3].search_reminders(keyword))

@reminders_app.command()
def delete(id: str):
    """Deletes a reminder by ID."""
    agent = asyncio.run(create_agent())
    print(agent.tools[3].delete_reminder(id))


async def create_agent(
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature: float = 0.7,
    index_filesystem: bool = False,
    filesystem_path: Optional[Path] = None,
    enable_gmail: bool = True,
    enable_weather: bool = True,
    use_voice: bool = False,
    voice_language: str = "en",
    voice_slow: bool = False,
    openai_api_base: Optional[str] = None,
    openai_api_type: Optional[str] = None,
    openai_api_version: Optional[str] = None,
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
        use_voice: Whether to use voice output
        voice_language: Language for voice output
        voice_slow: Whether to speak slowly
        openai_api_base: Base URL for OpenAI API
        openai_api_type: Type of OpenAI API (openai/azure)
        openai_api_version: API version for Azure OpenAI
        
    Returns:
        Configured agent instance
    """
    logger.info("Creating agent with configuration:")
    logger.info("- Model: %s", model)
    logger.info("- Temperature: %f", temperature)
    logger.info("- Index filesystem: %s", index_filesystem)
    logger.info("- Filesystem path: %s", filesystem_path)
    logger.info("- Enable Gmail: %s", enable_gmail)
    logger.info("- Enable Weather: %s", enable_weather)
    logger.info("- Use Voice: %s", use_voice)
    logger.info("- Voice Language: %s", voice_language)
    logger.info("- Voice Slow: %s", voice_slow)
    logger.info("- OpenAI API Base: %s", openai_api_base)
    logger.info("- OpenAI API Type: %s", openai_api_type)
    logger.info("- OpenAI API Version: %s", openai_api_version)
    
    # Initialize tools
    tools = [
        WebSearchTool(),
        CalendarTool(), # Add CalendarTool
        NotesTool(), # Add NotesTool
        RemindersTool(), # Add RemindersTool
    ]
    
    # Add Gmail tool if enabled
    if enable_gmail:
        tools.append(GmailTool())
        
    # Add Weather tool if enabled
    if enable_weather:
        tools.append(WeatherTool())
    
    # Initialize vector store if requested
    vector_store = None
    if index_filesystem:
        logger.info("Initializing vector store")
        vector_store = VectorStore()
        if filesystem_path:
            logger.info("Indexing directory: %s", filesystem_path)
            await vector_store.index_directory(filesystem_path)
        else:
            logger.info("Indexing home directory")
            await vector_store.index_directory(Path.home())
    
    # Always add filesystem tool
    logger.info("Adding filesystem tool")
    tools.append(FilesystemTool(vector_store=vector_store))
    
    # Create agent configuration
    config = AgentConfig(
        model_name=model,
        temperature=temperature,
        use_voice=use_voice,
        voice_language=voice_language,
        voice_slow=voice_slow,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
        openai_api_version=openai_api_version,
    )
    
    # Create and return agent
    logger.info("Creating agent with %d tools", len(tools))

    # Initialize reminder service only if voice is enabled
    reminder_service = None
    if config.use_voice:
        reminders_tool_instance = next((tool for tool in tools if isinstance(tool, RemindersTool)), None)
        voice_output_instance = VoiceOutput(language=config.voice_language, slow=config.voice_slow)
        reminder_service = ReminderService(reminders_tool=reminders_tool_instance, voice_output=voice_output_instance)

    return Agent(tools=tools, config=config, reminder_service=reminder_service) # Pass reminder_service to Agent

@app.command()
@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to process"),
    model: str = typer.Option(os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), help="Model to use (can also be set via OPENAI_MODEL_NAME env var)"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(False, "--index/--no-index", help="Enable/disable filesystem indexing"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(True, "--gmail/--no-gmail", help="Enable/disable Gmail integration"),
    enable_weather: bool = typer.Option(True, "--weather/--no-weather", help="Enable/disable weather information"),
    use_voice: bool = typer.Option(False, "--voice/--no-voice", help="Enable/disable voice output"),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(None, help="Type of OpenAI API (openai/azure)"),
    openai_api_version: Optional[str] = typer.Option(None, help="API version for Azure OpenAI"),
    quiet: bool = typer.Option(False, help="Suppress welcome message in non-interactive mode"),
):
    """Start the Intuit assistant in interactive mode or process a single query."""
    # Automatically set quiet mode if input is being piped
    if not sys.stdin.isatty():
        quiet = True

    agent = asyncio.run(create_agent(
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
    ))
    asyncio.run(run_cli(agent, query, quiet))

@app.command()
def voice(
    model: str = typer.Option(os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), help="Model to use (can also be set via OPENAI_MODEL_NAME env var)"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(False, "--index/--no-index", help="Enable/disable filesystem indexing"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(True, "--weather/--no-weather", help="Enable/disable weather information"),
    voice_language: str = typer.Option("en", help="Language for voice output"),
    voice_slow: bool = typer.Option(False, "--slow/--no-slow", help="Speak slowly"),
    openai_api_base: Optional[str] = typer.Option(None, help="Base URL for OpenAI API"),
    openai_api_type: Optional[str] = typer.Option(None, help="Type of OpenAI API (openai/azure)"),
    openai_api_version: Optional[str] = typer.Option(None, help="API version for Azure OpenAI"),
):
    """Start the Intuit assistant in voice mode."""
    agent = asyncio.run(create_agent(
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
    ))
    try:
        if agent.reminder_service: # Start reminder service if initialized
            agent.reminder_service.start()
        asyncio.run(run_voice(agent))
    except KeyboardInterrupt:
        print("\nStopping voice interface...")
    finally:
        if agent.reminder_service: # Stop reminder service on exit
            agent.reminder_service.stop()

def main():
    """Main entry point."""
    app() 