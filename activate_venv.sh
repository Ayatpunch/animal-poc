#!/bin/bash
# Script to activate virtual environment on Linux/Mac

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create a virtual environment first using: python -m venv venv"
    exit 1
fi

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Check if activation was successful
if [ -n "$VIRTUAL_ENV" ]; then
    echo "[SUCCESS] Virtual environment activated successfully!"
    echo "Location: $VIRTUAL_ENV"
    echo "Python version: $(python --version)"
else
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
