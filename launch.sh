#!/bin/bash
# ncOS v23 Launch Script

echo "🚀 Starting ncOS v23 Ultimate Trading System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directory if not exists
mkdir -p data

# Export environment variables
export NCOS_DATA_PATH="$(pwd)/data"
export PYTHONUNBUFFERED=1

# Start the API server
echo "Starting API server..."
python ncos_v23_api_server.py
