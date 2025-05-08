# Intuit: A Personal Assistant

Intuit is a flexible, agentic personal assistant that can be accessed via CLI, voice, or direct command-line arguments. It features a vector database for filesystem content, internet search capabilities, Gmail integration, and weather information.

## Features

- Vector database of filesystem content
- Internet search capabilities
- Gmail integration
- Weather information
- Voice interface with real-time processing
- CLI interface with rich text formatting
- Multi-process support for improved latency
- Full-duplex communication

## Prerequisites

- Python 3.11 or higher
- Nix package manager
- OpenAI API key
- Google API credentials (for Gmail integration)
- Weather API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intuit.git
   cd intuit
   ```

2. Set up the development environment:
   ```bash
   devenv up
   ```

3. Create a `.env` file with your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_key
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   WEATHER_API_KEY=your_weather_api_key
   ```

4. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### CLI Mode
```bash
intuit chat
```

### Single Query Mode
```bash
intuit chat "What's the weather like?"
```

### Voice Mode
```bash
intuit voice
```

## Project Structure

```
intuit/
├── devenv.nix           # Nix development environment
├── pyproject.toml       # Python project configuration
├── README.md           # Project documentation
├── src/
│   └── intuit/
│       ├── __init__.py
│       ├── main.py     # CLI entry point
│       ├── agent.py    # Core agent implementation
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── web_search.py
│       │   ├── gmail.py
│       │   ├── weather.py
│       │   └── filesystem.py
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   └── voice.py
│       └── vector_store/
│           ├── __init__.py
│           └── indexer.py
└── tests/
    └── __init__.py
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
ruff check .
```

### Adding New Tools

1. Create a new tool class in `src/intuit/tools/`
2. Inherit from `BaseTool`
3. Implement the `run` method
4. Add the tool to the agent's tool list

Example:
```python
from .base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Description of my tool"
    
    async def run(self, **kwargs):
        # Implement tool functionality
        return {"result": "tool output"}
```

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