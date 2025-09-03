[2025-01-03 12:58:00] - Migration to src/ layout to fix Nix packaging issue
**Problem**: The Nix build was failing with "Multiple top-level packages discovered in a flat-layout: ['bak', 'intuit']" error, preventing the development environment from loading properly.

**Decision**: Adopted the src/ layout pattern by:

1. Creating a `src/` directory
2. Moving the `intuit/` package into `src/intuit/`
3. Updating `pyproject.toml` with `[tool.setuptools.packages.find]` configuration to specify `where = ["src"]` and `include = ["intuit*"]`
4. Fixing all import references from `src.intuit.*` to `intuit.*` in scripts and modules
5. Updating the README.md project structure section to reflect the new layout

**Rationale**: The src layout is a Python packaging best practice that:

- Prevents accidental imports from the source tree during development
- Ensures proper package installation and distribution
- Resolves the multi-package discovery issue that was blocking the Nix build
- Makes the project structure more standard and maintainable

**Impact**: The package now builds successfully as a wheel without errors, and the Nix development environment loads properly. All imports have been updated to work with the new structure.
