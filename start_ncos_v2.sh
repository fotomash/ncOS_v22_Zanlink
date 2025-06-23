#!/bin/bash

# ncOS Smart Startup Script (v2)
# This script checks for an active virtual environment before running.

echo "🚀 Starting ncOS Local Development Environment (v2)..."

# Check if a virtual environment is active by checking the VIRTUAL_ENV variable
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: No active virtual environment found."
    echo "Please activate your environment before running this script."
    echo "For example: source /path/to/your/ncos_env/bin/activate"
    exit 1
fi

echo "✅ Virtual environment '$VIRTUAL_ENV' is active."

# Ensure all dependencies from the requirements file are installed
echo "📦 Checking and installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Dependencies are up to date."

# Start the Python server
echo "▶️ Starting ncOS server..."
python3 ncos_engine_v3.py
