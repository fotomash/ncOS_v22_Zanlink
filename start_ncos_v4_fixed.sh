#!/bin/bash

# ncOS v4 Smart Startup Script (Fixed)
# This script launches the full intelligence engine

echo "╔══════════════════════════════════════╗"
echo "║    ncOS v4 - Full Intelligence       ║"
echo "╚══════════════════════════════════════════╝"

# Check if a virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: No active virtual environment found."
    echo "Please activate your environment first:"
    echo "  source /Users/tom/Documents/GitHub/virtualenvs/ncos_env/bin/activate"
    exit 1
fi

echo "✅ Virtual environment is active: $VIRTUAL_ENV"

# Check for required files
echo "🔍 Checking required files..."

required_files=(
    "ncos_engine_v4.py"
    "ncos_theory_module_v2.py"
    "XAUUSD_TICKS_1days_20250623.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: Required file '$file' not found!"
        exit 1
    fi
done

echo "✅ All required files present"

# Install/update dependencies using python -m pip
echo "📦 Checking dependencies..."
python -m pip install flask flask-cors pandas python-dotenv numpy --quiet 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies might be missing. Attempting to continue..."
fi

# Launch the engine
echo ""
echo "🚀 Launching ncOS v4 Engine..."
echo "─────────────────────────────────"
echo "Your ngrok URL: https://emerging-tiger-fair.ngrok-free.app"
echo "Local URL: http://localhost:8000"
echo "─────────────────────────────────"
echo ""

# Run the engine with explicit python command
exec python ncos_engine_v4.py
