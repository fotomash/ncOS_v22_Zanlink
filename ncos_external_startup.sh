#!/bin/bash

# ncOS External Environment Startup Script

# Define the path to your external virtual environment
VENV_PATH="./ncos_env"

# --- Do not edit below this line ---

# Function to check if we are in the virtual environment
is_in_venv() {
    # Check if VIRTUAL_ENV is set and points to our target venv
    [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" == "$(cd "$VENV_PATH" && pwd)" ]
}

echo "--- ncOS Startup Sequence Initiated ---"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found at '$VENV_PATH'."
    echo "Please run the setup script to create it first."
    exit 1
fi

# Check if we are already in the correct virtual environment
if is_in_venv; then
    echo "[INFO] Already in the correct virtual environment."
else
    echo "[INFO] Activating virtual environment from: $VENV_PATH"
    # Activate the virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Verify activation
    if ! is_in_venv; then
        echo "[ERROR] Failed to activate the virtual environment."
        exit 1
    fi
fi

echo "[INFO] Virtual environment is active."
echo "[INFO] Starting the ncOS Ultimate Engine..."

# Run the main engine python script
python ncos_v5_ultimate_engine.py

# Deactivate on exit (optional, shell will close anyway)
# deactivate
echo "--- ncOS Engine has been shut down ---"