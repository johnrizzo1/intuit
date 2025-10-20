# Local Models Migration Guide

## Overview

This document describes the migration from cloud-based models (OpenAI) to local models (Ollama) as the default configuration for the Intuit project.

## Changes Made

### 1. LLM Model Defaults

**Previous Default:** `gpt-4o-mini` (OpenAI cloud model)
**New Default:** `llama3.2:3b` (Ollama local model)

Files updated:
- [`src/intuit/agent.py`](../src/intuit/agent.py:39) - AgentConfig model_name default
- [`src/intuit/mcp_server.py`](../src/intuit/mcp_server.py:50) - MCP server default model
- [`src/intuit/main.py`](../src/intuit/main.py:408) - create_agent function default
- [`src/intuit/main.py`](../src/intuit/main.py:1055) - CLI query command default
- [`src/intuit/main.py`](../src/intuit/main.py:1112) - CLI voice command default
- [`src/intuit/memory/manager.py`](../src/intuit/memory/manager.py:18) - IntuitMemoryManager default
- [`src/intuit/memory/store.py`](../src/intuit/memory/store.py:15) - IntuitMemoryStore default

### 2. Embedding Model Defaults

**Previous Default:** `openai:text-embedding-3-small` (OpenAI cloud embeddings, 1536 dimensions)
**New Default:** `ollama:nomic-embed-text` (Ollama local embeddings, 384 dimensions)

Files updated:
- [`src/intuit/memory/store.py`](../src/intuit/memory/store.py:35) - Memory store embedding configuration
- [`src/intuit/memory/chroma_store.py`](../src/intuit/memory/chroma_store.py:24) - ChromaDB store default
- [`src/intuit/vector_store/indexer.py`](../src/intuit/vector_store/indexer.py:62) - Vector store embeddings
- [`test_memory.py`](../test_memory.py:6) - Test file example

### 3. Documentation Updates

- [`examples/voice_tui_demo.py`](../examples/voice_tui_demo.py:81) - Updated example commands
- [`.env.template`](../.env.template:76) - Added EMBEDDING_MODEL configuration

## Why Local Models?

### Benefits

1. **Privacy**: All data stays on your machine
2. **Cost**: No API costs for usage
3. **Offline**: Works without internet connection
4. **Speed**: Lower latency for local inference (with proper hardware)
5. **Control**: Full control over model versions and behavior

### Trade-offs

1. **Hardware Requirements**: Requires sufficient RAM and ideally GPU
2. **Model Quality**: Local models may not match GPT-4 quality (but llama3.2 is quite capable)
3. **Setup**: Requires Ollama installation and model downloads

## Setup Instructions

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai) and install Ollama for your platform.

### 2. Pull Required Models

```bash
# Pull the LLM model (3B parameters, ~2GB)
ollama pull llama3.2:3b

# Pull the embedding model (~274MB)
ollama pull nomic-embed-text
```

### 3. Verify Ollama is Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Or use the Ollama CLI
ollama list
```

### 4. Configure Environment Variables

Copy `.env.template` to `.env` and ensure these settings:

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Model Configuration
EMBEDDING_MODEL=ollama:nomic-embed-text
```

**Note:** The code automatically adds the `ollama/` provider prefix for LangMem compatibility when needed. You can use either `llama3.2:3b` or `ollama/llama3.2:3b` - both work correctly.

## Alternative Models

### LLM Models

You can use different Ollama models by setting `LLM_MODEL`:

```bash
# Smaller, faster (1.3B parameters)
LLM_MODEL=llama3.2:1b

# Larger, more capable (8B parameters)
LLM_MODEL=llama3.1:8b

# Alternative models
LLM_MODEL=mistral:7b
LLM_MODEL=phi3:mini
```

### Embedding Models

You can use different embedding models by setting `EMBEDDING_MODEL`:

```bash
# Default (recommended)
EMBEDDING_MODEL=ollama:nomic-embed-text

# Larger embeddings (1024 dimensions)
EMBEDDING_MODEL=ollama:mxbai-embed-large

# Smaller, faster (384 dimensions)
EMBEDDING_MODEL=ollama:all-minilm
```

## Reverting to OpenAI

If you need to use OpenAI models, you can override the defaults:

### Option 1: Environment Variables

```bash
# Set in .env file
OPENAI_MODEL_NAME=gpt-4o-mini
LLM_PROVIDER=openai
EMBEDDING_MODEL=openai:text-embedding-3-small
OPENAI_API_KEY=your-api-key-here
```

### Option 2: Command Line Arguments

```bash
# Use OpenAI for a specific command
devenv shell -- uv run intuit query "your question" --model gpt-4o-mini
```

### Option 3: Code Configuration

```python
from intuit.agent import Agent, AgentConfig

config = AgentConfig(
    model_name="gpt-4o-mini",
    openai_api_base="https://api.openai.com/v1"
)
agent = Agent(config=config)
```

## Performance Considerations

### Hardware Acceleration

The project automatically detects and uses available hardware:

- **NVIDIA GPU**: Uses CUDA for acceleration
- **Apple Silicon**: Uses CPU (MPS not supported by Whisper/Coqui)
- **CPU Only**: Falls back to CPU inference

### Memory Requirements

| Model | RAM Required | GPU VRAM (Optional) |
|-------|-------------|---------------------|
| llama3.2:1b | ~2GB | ~1GB |
| llama3.2:3b | ~4GB | ~2GB |
| llama3.1:8b | ~8GB | ~5GB |
| nomic-embed-text | ~512MB | ~256MB |

### Optimization Tips

1. **Use smaller models** for faster responses: `llama3.2:1b`
2. **Enable GPU acceleration** if available
3. **Adjust context window** via `LLM_MAX_TOKENS` for memory constraints
4. **Use quantized models** (Ollama does this automatically)

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama service
ollama serve

# Or on macOS/Linux with systemd
systemctl start ollama
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Connection Errors

Check that Ollama is accessible:

```bash
curl http://localhost:11434/api/tags
```

If using a different host/port, update `OLLAMA_BASE_URL` in `.env`.

### Memory Issues

If you encounter out-of-memory errors:

1. Use a smaller model: `llama3.2:1b`
2. Reduce `LLM_MAX_TOKENS` in `.env`
3. Close other applications
4. Consider using cloud models for complex tasks

## Migration Checklist

- [ ] Install Ollama
- [ ] Pull required models (`llama3.2:3b`, `nomic-embed-text`)
- [ ] Verify Ollama is running
- [ ] Update `.env` file with local model configuration
- [ ] Test basic query: `devenv shell -- uv run intuit query "Hello"`
- [ ] Test voice mode: `devenv shell -- uv run intuit voice`
- [ ] Verify memory/embedding functionality works

## Support

For issues or questions:

1. Check Ollama documentation: https://ollama.ai/docs
2. Review project logs with `-v` flag for verbose output
3. Ensure all dependencies are installed via `devenv shell`
4. Check hardware compatibility for your platform

## Future Improvements

Potential enhancements for local model support:

1. **Model auto-download**: Automatically pull models if not available
2. **Model selection UI**: Interactive model selection in CLI
3. **Performance profiling**: Built-in benchmarking tools
4. **Hybrid mode**: Use local for simple tasks, cloud for complex ones
5. **Model caching**: Optimize model loading and memory usage