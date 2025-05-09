"""
Main entry point for Intuit.
"""
import asyncio
import typer
import sys
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from .agent import Agent, AgentConfig
from .tools.web_search import WebSearchTool
from .tools.gmail import GmailTool
from .tools.weather import WeatherTool
from .tools.filesystem import FilesystemTool
from .vector_store.indexer import VectorStore
from .ui.cli import run_cli
from .ui.voice import run_voice

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)

async def create_agent(
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature: float = 0.7,
    index_filesystem: bool = False,
    filesystem_path: Optional[Path] = None,
    enable_gmail: bool = False,
    enable_weather: bool = False,
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
    ]
    
    # Add Gmail tool if enabled
    if enable_gmail:
        tools.append(GmailTool())
        
    # Add Weather tool if enabled
    if enable_weather:
        tools.append(WeatherTool())
    
    # Initialize vector store and filesystem tool if requested
    if index_filesystem:
        logger.info("Initializing vector store")
        vector_store = VectorStore()
        if filesystem_path:
            logger.info("Indexing directory: %s", filesystem_path)
            await vector_store.index_directory(filesystem_path)
        else:
            logger.info("Indexing home directory")
            await vector_store.index_directory(Path.home())
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
    return Agent(tools=tools, config=config)

@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="Query to process"),
    model: str = typer.Option(os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), help="Model to use (can also be set via OPENAI_MODEL_NAME env var)"),
    temperature: float = typer.Option(0.7, help="Model temperature"),
    index: bool = typer.Option(False, "--index/--no-index", help="Enable/disable filesystem indexing"),
    index_path: Optional[Path] = typer.Option(None, help="Path to index"),
    enable_gmail: bool = typer.Option(False, help="Enable Gmail integration"),
    enable_weather: bool = typer.Option(False, help="Enable weather information"),
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
    enable_weather: bool = typer.Option(False, help="Enable weather information"),
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
        asyncio.run(run_voice(agent))
    except KeyboardInterrupt:
        print("\nStopping voice interface...")

def main():
    """Main entry point."""
    app() 