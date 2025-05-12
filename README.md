# Intuit: A Personal Assistant

Intuit is a flexible, agentic personal assistant that can be accessed via CLI, voice, or direct command-line arguments. It features a vector database for filesystem content, internet search capabilities, Gmail integration, weather information, and local productivity tools (Calendar, Notes, Reminders).

## Features

- Vector database of filesystem content
- Internet search capabilities
- Gmail integration
- Weather information
- Local Productivity Tools: Calendar, Notes, and Reminders
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

### MCP Server Mode

Run Intuit as an MCP server to expose its tools:

```bash
uv run intuit mcp start-server --host localhost --port 8000
```

You can list the tools available on the local MCP server:

```bash
uv run intuit mcp list-mcp-tools
```

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
  ```

## Project Structure

```shell
intuit/
├── .env.template        # Template for environment variables
├── devenv.nix           # Nix development environment
├── pyproject.toml       # Python project configuration
├── README.md           # Project documentation
├── src/
│   └── intuit/
│       ├── __init__.py
│       ├── main.py     # Typer CLI entry point and tool command definitions
│       ├── agent.py    # Core agent implementation and tool dispatch
│       ├── mcp_server.py # MCP Server implementation
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── web_search.py
│       │   ├── gmail.py
│       │   ├── weather.py
│       │   ├── filesystem.py
│       │   ├── calendar.py # Local Calendar Tool
│       │   ├── notes.py    # Local Notes Tool
│       │   └── reminders.py # Local Reminders Tool
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── cli.py      # (Older Argparse CLI - potentially deprecated)
│       │   └── voice.py
│       └── vector_store/
│           ├── __init__.py
│           └── indexer.py
├── tests/               # Unit tests
│   └── tools/
│       ├── test_calendar.py # Calendar Tool tests
│       ├── test_notes.py    # Notes Tool tests
│       └── test_reminders.py # Reminders Tool tests
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
