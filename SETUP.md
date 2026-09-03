# Project Setup Guide

This guide explains how to set up the BondalyzerParaView project using `uv`, a fast Python package manager.

## Prerequisites

- **Python 3.10+** installed on your system
- **`uv`** installed (see [uv installation guide](https://github.com/astral-sh/uv#installation))

## Quick Start with `uv`

### Option 1: Using `uv venv` + `uv pip` (Recommended)

This approach creates a virtual environment and installs dependencies:

```bash
# Create a virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using `uv` with `pyproject.toml`

Install dependencies directly from the project configuration:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install core dependencies
uv pip install -e .

# (Optional) Install development dependencies
uv pip install -e ".[dev]"
```

### Option 3: Direct Install Without Separate venv

If you prefer to skip manual venv activation:

```bash
# Let uv manage the virtual environment automatically
uv run python trame_viewer.py ethene4.plt
```

## Verifying Installation

Check that all dependencies are installed correctly:

```bash
source .venv/bin/activate
python -c "import numpy, vtk, trame; print('All dependencies installed successfully!')"
```

## Running the Trame Viewer

Once dependencies are installed:

```bash
# With virtual environment activated
source .venv/bin/activate
python trame_viewer.py ethene4.plt

# Or use uv run directly
uv run python trame_viewer.py ethene4.plt
```

The application will start a web server (default: http://localhost:8080/)

## Files Reference

- **`pyproject.toml`**: Project metadata and dependency specifications
- **`requirements.txt`**: Pinned versions for reproducible installations
- **`.venv/`**: Virtual environment directory (created by `uv venv`)

## Dependency Version Pinning

The `requirements.txt` contains **exact versions** of all dependencies for reproducible builds. If you need to update dependencies:

```bash
# Update to latest compatible versions
uv pip install --upgrade -r requirements.txt

# Generate new pinned versions
uv pip freeze > requirements.txt
```

## Troubleshooting

### Port Already in Use

If port 8080 is busy, the server will fail. Check for existing processes:

```bash
lsof -i :8080
# Kill the process if needed (replace PID with actual process ID)
kill <PID>
```

### Missing Volume Data

The viewer will automatically generate `ethene4_zone0.vti` from the `.plt` file if it doesn't exist. This may take a moment on first run.

### Trame Import Errors

Ensure all trame packages are installed:

```bash
uv pip install trame trame-vuetify trame-vtk trame-server
```

## Environment Variables

You can customize the Trame server behavior:

```bash
# Run on a different port
TRAME_SERVER_PORT=9000 python trame_viewer.py ethene4.plt

# Enable debug output
DEBUG=1 python trame_viewer.py ethene4.plt
```

## For Contributors

To set up a development environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This installs the project in editable mode with dev tools (pytest, black, ruff).
