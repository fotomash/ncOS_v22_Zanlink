#!/usr/bin/env python3
"""
Quick launcher for ncOS Real Data Engine
"""

import subprocess
import sys
import os

def main():
    # Check for required packages
    required = ['fastapi', 'uvicorn', 'pandas', 'numpy']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

    # Run the server
    print("\n🚀 Starting ncOS Real Data Engine...\n")
    subprocess.run([sys.executable, '-m', 'uvicorn', 'ncos_v24_real_api:app', '--host', '0.0.0.0', '--port', '8000'])

if __name__ == "__main__":
    main()
