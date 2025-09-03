# Intuit: A Personal Assistant

Intuit is a flexible, agentic personal assistant that can be accessed via CLI, voice, or direct command-line arguments. It features a vector database for filesystem content, internet search capabilities, Gmail integration, weather information, and local productivity tools (Calendar, Notes, Reminders).

## Features

- Vector database of filesystem content
- Internet search capabilities
- Gmail integration
- Weather information
- Local Productivity Tools: Calendar, Notes, and Reminders
- Persistent memory storage with ChromaDB
- Voice interface with real-time processing
- CLI interface with rich text formatting
- Multi-process support for improved latency
- Full-duplex communication
- MCP (Model Context Protocol) Server for exposing tools
- MCP Client for connecting to external tools

## Prerequisites

- Python 3.11 or higher
- Nix package manager (recommended for dependency management)
- OpenAI API key
- Google API credentials (for Gmail integration)
- Weather API key

## Download and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/johnrizzo1/intuit.git
   cd intuit
   ```

2. **Create a `.env` file with your API keys:**

   Copy the `.env.template` file and fill in your credentials.

   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

   For Gmail to work, you need to download your Google API credentials JSON file and place it in the project directory, or specify its path in the `.env` file.

3. **Install Python dependencies using uv:**

   ```bash
   uv sync
   ```

## Running Intuit

Intuit can be run in different modes:

### CLI Interactive Mode

Run the agent in an interactive command-line interface:

```bash
uv run intuit chat
```

### Single Query Mode

Pass a single query to the agent via the command line:

```bash
uv run intuit chat "What's the weather like in London?"
```

### Voice Mode

Start the agent in real-time voice interaction mode:

```bash
uv run intuit voice
```

### MCP (Model Context Protocol) Integration

Intuit supports the Model Context Protocol (MCP), which enables AI models to interact with external tools and resources in a standardized way. Intuit can function as both an MCP server (exposing its tools to other AI agents) and an MCP client (connecting to external MCP servers to use their tools).

#### MCP Server Mode

Run Intuit as an MCP server to expose its tools to other AI agents:

```bash
uv run intuit mcp start-server --host localhost --port 8000
```

You can list the tools available on the local MCP server:

```bash
uv run intuit mcp list-tools
```

The MCP server exposes the following tools:

- Calendar tools (add, list, search, delete)
- Notes tools (add, list, search, delete)
- Reminders tools (add, list, search, delete)
- Weather tools (get weather for a location)
- Web search tools (search the web)
- Filesystem tools (list, read, write, search)
- Screenshot tool (take a screenshot)
- Hacker News tools (get top, new, best stories)
- Memory tools (add, search, get, delete memories)

#### MCP Client Mode

Connect Intuit to an external MCP server to use its tools:

```bash
# In interactive mode
uv run intuit chat
> connect to MCP server at http://localhost:8000
```

```bash
# Or directly from the command line
uv run intuit mcp connect http://localhost:8000
```

List the tools available from connected MCP servers:

```bash
uv run intuit mcp list-mcp-tools
```

Once connected, you can use the tools from the external MCP server in your conversations with Intuit. The tools will be prefixed with "mcp\_" to distinguish them from local tools.

#### MCP Architecture

The MCP integration in Intuit follows a modular design:

1. **MCP Server**: Implemented in `src/intuit/mcp_server.py`, it uses FastMCP to expose Intuit's tools as MCP resources.
2. **MCP Client**: Implemented in `src/intuit/agent.py`, it connects to external MCP servers and wraps their tools for use by the agent.
3. **Tool Wrappers**: The `MCPToolWrapper` class in `src/intuit/agent.py` wraps external MCP tools to make them look like local tools to the agent.
4. **Error Handling**: Robust error handling and fallback mechanisms are implemented for MCP tool execution.

## Available Tool Commands

Intuit provides access to various tools via the CLI. Here are the available commands and examples:

### Calendar Tool (`intuit calendar`)

Manage your local calendar events.

- **Add an event:**

  ```bash
  uv run intuit calendar add "Meeting with the team on Friday at 2 PM"
  ```

- **List all events:**

  ```bash
  uv run intuit calendar list
  ```

- **Search for events:**

  ```bash
  uv run intuit calendar search "meeting"
  ```

- **Delete an event:**

  ```bash
  uv run intuit calendar delete [event_id]
  ```

  Replace `[event_id]` with the unique ID of the event.

### Notes Tool (`intuit notes`)

Manage your local notes.

- **Add a note:**

  ```bash
  uv run intuit notes add "Remember to buy groceries tomorrow."
  ```

- **List all notes:**

  ```bash
  uv run intuit notes list
  ```

- **Search for notes:**

  ```bash
  uv run intuit notes search "groceries"
  ```

- **Delete a note:**

  ```bash
  uv run intuit notes delete [note_id]
  ```

  Replace `[note_id]` with the unique ID of the note.

### Reminders Tool (`intuit reminders`)

Manage your local reminders.

- **Add a reminder:**

  ```bash
  uv run intuit reminders add "Call Mom" --time "2025-12-25T18:00:00"
  ```

  The `--time` argument is optional and should be in ISO 8601 format.

- **List all reminders:**

  ```bash
  uv run intuit reminders list
  ```

- **Search for reminders:**

  ```bash
  uv run intuit reminders search "call"
  ```

- **Delete a reminder:**

  ```bash
  uv run intuit reminders delete [reminder_id]
  ```

  Replace `[reminder_id]` with the unique ID of the reminder.

### Web Search Tool (via chat/voice)

Search the internet for information.

- **Example (CLI Single Query):**

  ```bash
  uv run intuit chat "What is the capital of France?"
  ```

- **Example (Voice):**
  "Search the web for the latest news on AI."

### Gmail Tool (via chat/voice - requires setup)

Manage your Gmail messages.

- **Example (CLI Single Query):**

  ```bash
  uv run intuit chat "How many unread emails do I have?"
  ```

- **Example (Voice):**
  "Show me my recent emails from John."

### Weather Tool (via chat/voice)

Get current weather information and forecasts.

- **Example (CLI Single Query):**

  ```bash
  uv run intuit chat "What's the weather like in New York City?"
  ```

- **Example (Voice):**
  "Tell me the weather forecast for tomorrow in Tokyo."

### Filesystem Tool (via chat/voice)

Search, read, and manage files on your filesystem.

- **Example (CLI Single Query):**

  ```bash
  uv run intuit chat "Find files about project planning in my documents folder."
  ```

- **Example (Voice):**
  "Read the content of the file named 'important_notes.txt'."

## Running Common Tasks

Here are some examples of how you can use Intuit for common tasks:

- **Get the weather and add a reminder:**

  ```bash
  uv run intuit chat "What's the weather today, and remind me to bring an umbrella if it's raining."
  ```

- **Search for a note and add a calendar event:**

  ```bash
  uv run intuit chat "Find my note about the meeting agenda, and add a calendar event for the meeting tomorrow at 10 AM."
  ```

- **Other interesting examples:**

  ```bash
  uv run intuit chat "Get the content of my clipboard."
  uv run intuit chat "Read my clipboard."
  # Take a screenshot
  uv run intuit chat "Take a screenshot of my screen."
  ```

### Memory Tool (`intuit memory`)

Intuit includes a persistent memory system powered by ChromaDB that allows the assistant to store and retrieve important information across conversations.

- **Add a memory:**

  ```bash
  uv run intuit memory add "User prefers dark mode" --importance 8 --tags preferences,ui
  ```

  The `--importance` parameter (1-10) helps prioritize memories, and `--tags` allows for categorization.

- **Search memories:**

  ```bash
  uv run intuit memory search "user preferences"
  ```

  This performs a semantic search to find relevant memories, even if they don't contain the exact search terms.

- **Get a specific memory by ID:**

  ```bash
  uv run intuit memory get <memory_id>
  ```

  Replace `<memory_id>` with the UUID of the memory you want to retrieve.

- **Delete a memory:**

  ```bash
  uv run intuit memory delete <memory_id>
  ```

- **Clear all memories:**

  ```bash
  uv run intuit memory clear
  ```

Memories are stored in the `.memory` directory using ChromaDB and persist between sessions, allowing Intuit to maintain context over time.

## MCP Usage Examples

Here are some detailed examples of how to use Intuit's MCP functionality:

### Starting the MCP Server

Start the MCP server to expose Intuit's tools to other AI agents:

```bash
# Start the server on the default host and port (localhost:8000)
uv run intuit mcp start-server

# Start the server on a custom host and port
uv run intuit mcp start-server --host 0.0.0.0 --port 9000
```

### Checking MCP Server Status

Check if the MCP server is running and see available tools:

```bash
uv run intuit mcp status
```

### Listing Available MCP Tools

List all tools available on the MCP server in a human-readable format:

```bash
uv run intuit mcp list-tools
```

### Connecting to an External MCP Server

Connect Intuit to an external MCP server to use its tools:

```bash
# In interactive mode
uv run intuit chat
> connect to MCP server at http://localhost:8000
```

### Using MCP Tools in Conversations

Once connected to an MCP server, you can use its tools in conversations:

```bash
# Using a calendar tool from an MCP server
uv run intuit chat "Add a meeting with John on Friday at 2 PM to my calendar"

# Using a weather tool from an MCP server
uv run intuit chat "What's the weather like in San Francisco?"

# Taking a screenshot using the MCP screenshot tool
uv run intuit chat "Take a screenshot of my screen"
```

### Using MCP Tools Programmatically

You can also use MCP tools programmatically in your Python code:

```python
import asyncio
from intuit.agent import Agent
from intuit.tools.basetool import BaseTool

async def main():
    # Create an agent with default tools
    agent = Agent(tools=[])

    # Connect to an MCP server
    await agent.connect_to_mcp_server("http://localhost:8000")

    # List available MCP tools
    print(agent.list_mcp_tools())

    # Use an MCP tool
    result = await agent.process_input("Take a screenshot of my screen")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Creating a Custom MCP Tool

You can create your own custom MCP tools and expose them via the MCP server:

```python
# In src/intuit/mcp_server.py
from mcp.server.fastmcp.utilities.types import Image as MCPImage
import io
import pyautogui

@mcp_server.tool()
def my_custom_tool(param1: str, param2: int = 10) -> str:
    """
    A custom tool that does something useful.

    Args:
        param1: The first parameter
        param2: The second parameter (default: 10)

    Returns:
        Result of the operation
    """
    # Your custom tool implementation here
    return f"Processed {param1} with parameter {param2}"
```

## Project Structure

```shell
intuit/
├── .env.template        # Template for environment variables
├── devenv.nix           # Nix development environment
├── pyproject.toml       # Python project configuration
├── README.md           # Project documentation
├── src/                 # Source code (src layout for proper packaging)
│   └── intuit/
│       ├── __init__.py
│       ├── main.py     # Typer CLI entry point and tool command definitions
│       ├── agent.py    # Core agent implementation and tool dispatch
│       ├── mcp_server.py # MCP Server implementation
│       ├── memory/     # Memory system implementation
│       │   ├── __init__.py
│       │   ├── chroma_store.py # ChromaDB-backed memory store
│       │   ├── manager.py # Memory manager
│       │   └── tools.py  # Memory tools
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── basetool.py
│       │   ├── web_search.py
│       │   ├── gmail.py
│       │   ├── weather.py
│       │   ├── filesystem.py
│       │   ├── calendar.py # Local Calendar Tool
│       │   ├── notes.py    # Local Notes Tool
│       │   ├── reminders.py # Local Reminders Tool
│       │   └── hackernews.py # HackerNews Tool
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── cli.py      # CLI interface
│       │   ├── voice.py    # Voice interface
│       │   └── gui/        # GUI components
│       │       ├── __init__.py
│       │       ├── main_gui.py
│       │       ├── standalone_gui.py
│       │       └── ...     # Other GUI files
│       ├── utils/          # Utility modules
│       │   ├── __init__.py
│       │   ├── progress.py
│       │   ├── spinner.py
│       │   └── voice_process.py
│       └── vector_store/
│           ├── __init__.py
│           ├── document.py
│           └── indexer.py
├── tests/               # Unit tests
│   ├── conftest.py
│   ├── tools/
│   │   ├── test_calendar.py # Calendar Tool tests
│   │   ├── test_notes.py    # Notes Tool tests
│   │   ├── test_reminders.py # Reminders Tool tests
│   │   └── ...         # Other tool tests
│   ├── memory/          # Memory system tests
│   ├── ui/              # UI tests
│   └── vector_store/    # Vector store tests
├── scripts/             # Utility scripts
├── examples/            # Example code and demos
├── docs/                # Documentation
├── memory-bank/         # Memory bank for project context
├── bak/                 # Backup files
└── data/                # Directory for local tool data (created on first use)
    ├── calendar/        # Calendar event JSON files
    ├── notes/           # Note JSON files
    └── reminders/       # Reminder JSON files
```

## Development

### Running Tests

```bash
uv run pytest
```

To run tests for a specific tool, e.g., the calendar tool:

```bash
uv run pytest tests/tools/test_calendar.py
```

### Code Formatting and Linting

```bash
uv run black .
uv run ruff check .
```

### Adding New Tools

1. Create a new tool class in `src/intuit/tools/` inheriting from `BaseTool`.
2. Implement the `run` method with your tool's logic.
3. Add `name` and `description` class attributes to your tool class.
4. Import your new tool in `src/intuit/main.py`.
5. Add your tool to the `tools` list in the `create_agent` function in `src/intuit/main.py`.
6. Add a new `typer.Typer` instance for your tool in `src/intuit/main.py` and mount it to the main `app`.
7. Define command functions for your tool's actions using `@<your_tool_app>.command()` in `src/intuit/main.py`.
8. Update the system prompt in `src/intuit/agent.py` to inform the agent about your new tool and its usage.
9. Add your tool's function definition and wrapped tool to the `functions` and `wrapped_tools` lists in the `_create_agent_executor` method in `src/intuit/agent.py`.
10. Write unit tests for your new tool in the `tests/tools/` directory.
11. Document your new tool in the `docs/` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- OpenAI for GPT-4 and Whisper models
- LangChain for the agent framework
- ChromaDB for vector storage
- Textual for the TUI framework
- Typer for the command-line interface
- uv for dependency management
