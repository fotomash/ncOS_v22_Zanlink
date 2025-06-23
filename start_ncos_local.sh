#!/bin/bash
# ncOS Local Development Startup Script

echo "Starting ncOS Local Development Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install flask flask-cors pandas python-dotenv

# Create necessary directories
mkdir -p data/ticks data/historical logs uploads

# Copy tick data if available
if [ -f "XAUUSD_TICKS_1days_20250623.csv" ]; then
    cp XAUUSD_TICKS_1days_20250623.csv data/ticks/
    echo "Copied tick data to data/ticks/"
fi

# Start the server
echo "Starting ncOS server on port 8000..."
echo "Your ngrok URL: https://emerging-tiger-fair.ngrok-free.app"
echo "Local URL: http://localhost:8000"
echo ""
echo "Server is starting without requiring any API keys..."
echo "Press Ctrl+C to stop the server"

python ncos_local_server.py
