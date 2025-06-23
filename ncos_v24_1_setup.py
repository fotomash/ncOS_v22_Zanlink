#!/usr/bin/env python3
# NCOS v24.1 Setup Script

import os
import json
from pathlib import Path

def create_directories():
    dirs = ["config", "core", "data", "journal", "logs"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("✓ Directories created")

def create_configs():
    configs = {
        "system_config.json": {
            "system": {"name": "NCOS v24.1", "version": "24.1.0"}
        },
        "strategy_rules.json": {
            "Inv": [{"source": "structure", "metric": "choch_detected", "condition": "equals", "value": True}],
            "MAZ2": [{"source": "smc", "metric": "fvg_quality", "condition": ">=", "value": 0.7}],
            "TMC": [{"source": "structure", "metric": "bos_detected", "condition": "equals", "value": True}],
            "Mentfx": [{"source": "confluence", "metric": "rsi_confluence", "condition": ">=", "value": 0.7}]
        },
        "risk_config.json": {
            "default_risk": {"account_size": 10000, "risk_percent": 1.0, "tp_rr": 3.0}
        }
    }

    for filename, config in configs.items():
        with open(f"config/{filename}", 'w') as f:
            json.dump(config, f, indent=2)
    print("✓ Configuration files created")

def main():
    print("🚀 NCOS v24.1 Setup Starting...")
    create_directories()
    create_configs()
    print("✅ Setup complete!")
    print("Next: Copy core modules to core/ directory")

if __name__ == "__main__":
    main()
