#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the start_system.py script
python "$SCRIPT_DIR/start_system.py"

# Keep the terminal open after execution
echo "Press any key to continue..."
read -n 1 -s