#!/bin/bash
# ncOS v22 Cleanup Script

echo "Starting ncOS v22 cleanup..."

# Create archive directory
mkdir -p archive0

# Move virtual environment
if [ -d "ncos_env" ]; then
    echo "Moving virtual environment to archive..."
    mv ncos_env archive0/
fi

# Remove Mac OS metadata files
echo "Removing Mac OS metadata files..."
find . -name "._*" -type f -delete

# Remove Python cache
echo "Removing Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Move old log files
echo "Archiving old log files..."
find . -name "*.log" -type f -exec mv {} archive0/ \;

echo "Cleanup complete!"
