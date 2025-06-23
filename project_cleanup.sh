#!/bin/bash
# =================================================================
# ncOS Project Cleanup and Consolidation Script
# =================================================================
# This script will:
# 1. Create a safe archive directory.
# 2. Move old, duplicate, and conflicting files into the archive.
# 3. Remove redundant Python virtual environments.
# 4. Standardize the project to use the local Flask server.
# =================================================================

echo "🚀 Starting ncOS Project Cleanup..."

# 1. Create a safe archive directory for old files
ARCHIVE_DIR="_archive_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"
echo "✅ Created archive directory: $ARCHIVE_DIR"

# Function to move a file/directory if it exists
move_to_archive() {
    if [ -e "$1" ]; then
        echo "  -> Archiving $1"
        mv "$1" "$ARCHIVE_DIR/"
    fi
}

# 2. Archive old and conflicting files
echo "\nArchiving old and conflicting files..."
move_to_archive "app.py" # Conflicting FastAPI entry point
move_to_archive "main.py"
move_to_archive "main_v23.py"
move_to_archive "v24_1.py"
move_to_archive "v24.md"
move_to_archive "validate.py" # Old validator
move_to_archive "validate_system.py" # We will replace this logic
move_to_archive "pack.py"
move_to_archive "tf.py"

# Archive duplicate files
move_to_archive "docker-compose (1).yml"
move_to_archive "cleanup_ncos (1).sh"
move_to_archive "ncos_launcher (1).py"
move_to_archive "ncos_ngrok (1).env"

# Archive old config/setup files we are replacing
move_to_archive "ncos_config_ngrok.json"
move_to_archive "ncos_config_zanlink.json"
move_to_archive "ncos_ngrok.env"
move_to_archive "requirements_local.txt"
move_to_archive "requirements_3.13.txt"
move_to_archive "setup_missing_files.py"
move_to_archive "deploy_zanlink.sh"
move_to_archive "ncos_zanlink_bridge.py"
move_to_archive "ncOS_v22_fixes"

# 3. Remove redundant virtual environments
echo "\nRemoving redundant virtual environments..."
if [ -d "venv" ]; then
    echo "  -> Removing 'venv' directory..."
    rm -rf "venv"
fi
if [ -d "archive0" ]; then
    echo "  -> Removing 'archive0' directory..."
    rm -rf "archive0"
fi
echo "✅ Kept 'ncos_env' as the primary environment."

# 4. Standardize Configuration and Requirements
echo "\nStandardizing configuration for local development..."

# Create a single, definitive .env file for local use
cat > .env << EOL
# Main .env file for ncOS Local Development
# This file is used by ncos_local_server.py

# Server Configuration
NCOS_ENV=development
FLASK_APP=ncos_local_server.py
FLASK_RUN_PORT=8000
FLASK_RUN_HOST=0.0.0.0
FLASK_DEBUG=1

# Ngrok URL (Update if it changes)
NGROK_URL=https://emerging-tiger-fair.ngrok-free.app

# API Keys (Optional, not needed for local server)
OPENAI_API_KEY="your_openai_key_here"
# Add other keys as needed
EOL
echo "✅ Created a standardized .env file."

# Create a single, definitive requirements.txt file
cat > requirements.txt << EOL
# Main requirements.txt for ncOS Local Development
# Run 'pip install -r requirements.txt' after activating your environment.

# Server
flask
flask-cors
python-dotenv

# Data Handling
pandas
numpy

# HTTP Requests
requests
EOL
echo "✅ Created a standardized requirements.txt file."

# 5. Update the Startup Script
echo "\nUpdating the startup script..."
cat > start_ncos_local.sh << EOL
#!/bin/bash
# ncOS Local Development Startup Script (Updated)

echo "Starting ncOS Local Development Environment..."

# Ensure we are using the correct virtual environment
if [ -d "ncos_env" ]; then
    echo "Activating 'ncos_env' virtual environment..."
    source ncos_env/bin/activate
else
    echo "ERROR: 'ncos_env' not found. Please create it first:"
    echo "python3 -m venv ncos_env"
    exit 1
fi

# Upgrade core tools and install dependencies
echo "\nUpdating build tools and installing dependencies from requirements.txt..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Start the local Flask server
echo "\n=================================================="
echo "🚀 Launching ncOS Local Server..."
echo "Your ngrok URL should point to http://localhost:8000"
echo "Press Ctrl+C to stop the server."
echo "=================================================="
flask run
EOL
chmod +x start_ncos_local.sh
echo "✅ Updated start_ncos_local.sh to be the single, reliable way to start the server."

echo "\n🎉 Cleanup and consolidation complete!"
echo "\nYour project is now clean. Please follow these steps:"
echo "1. Close and reopen your terminal."
echo "2. Activate the environment: source ncos_env/bin/activate"
echo "3. Run the server: ./start_ncos_local.sh"
