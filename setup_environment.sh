#!/bin/bash
# ncOS Environment Setup Script

echo "🚀 Setting up ncOS v22 Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ncos_env

# Activate virtual environment
source ncos_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs config backups cache

# Set up environment file if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp ncos_ngrok.env .env
    echo "⚠️  Please edit .env file with your API keys"
fi

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source ncos_env/bin/activate"
