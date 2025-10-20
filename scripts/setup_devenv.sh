#!/bin/bash
# Setup script for devenv environment with correct NumPy version

echo "=== Setting up Intuit with devenv + uv ==="
echo ""

# Rebuild devenv environment
echo "Step 1: Rebuilding devenv environment..."
devenv shell

echo ""
echo "Step 2: Installing Python dependencies with correct NumPy version..."
# Use uv to install with our constraints
uv pip install -e ".[gpu,tts-coqui]"

echo ""
echo "Step 3: Verifying installation..."
python -c "import numpy; print(f'✓ NumPy {numpy.__version__} installed')"
python -c "import numba; print(f'✓ Numba {numba.__version__} working')" 2>/dev/null || echo "⚠ Numba check failed (may not be needed)"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Make sure Ollama is running: ollama serve"
echo "2. Pull the model: ollama pull llama3.2:3b"
echo "3. Run the voice interface: intuit voice"