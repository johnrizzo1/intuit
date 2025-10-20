#!/bin/bash
# Fix NumPy version compatibility with Numba

echo "=== Fixing NumPy Version for Numba Compatibility ==="
echo ""
echo "Current NumPy version:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy import failed"
echo ""

echo "Uninstalling incompatible NumPy version..."
pip uninstall -y numpy

echo ""
echo "Installing NumPy <2.2 (compatible with Numba)..."
pip install "numpy>=1.24.0,<2.2"

echo ""
echo "Verifying installation..."
python -c "import numpy; print(f'✓ NumPy {numpy.__version__} installed successfully')"
python -c "import numba; print(f'✓ Numba {numba.__version__} can now import successfully')"

echo ""
echo "=== Fix Complete ==="
echo "You can now run: intuit voice"