#!/bin/bash
# start_ncos_v5_ultimate.sh

echo "╔══════════════════════════════════════╗"
echo "║  ncOS v5 Ultimate - Full Power       ║"
echo "╚══════════════════════════════════════╝"

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: No active virtual environment"
    echo "Please activate: source ~/virtualenvs/ncos_env/bin/activate"
    exit 1
fi

echo "✅ Virtual environment active: $VIRTUAL_ENV"

# Install dependencies
echo "📦 Installing dependencies..."
python -m pip install flask flask-cors pandas numpy python-dotenv --quiet

# Configuration
export NCOS_DATA_PATH="/Volumes/[C] Windows 11/Users/tom/AppData/Roaming/MetaQuotes/Terminal/81A933A9AFC5DE3C23B15CAB19C63850/MQL5/Files/PandasExports"

echo ""
echo "🚀 Launching ncOS v5 Ultimate..."
echo "─────────────────────────────────"
echo "Data Path: $NCOS_DATA_PATH"
echo "Your ngrok URL: https://emerging-tiger-fair.ngrok-free.app"
echo "Local URL: http://localhost:8000"
echo "─────────────────────────────────"
echo ""
echo "📋 Interactive Menu Commands:"
echo "  1-8: Menu options"
echo "  9-13: Quick pair selection"
echo ""

exec python ncos_v5_ultimate_engine.py
```[object Object]