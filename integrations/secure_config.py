# Secure Configuration Loader
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SecureConfig:
    """Secure configuration management - never expose API keys in code"""

    def __init__(self, config_file: str = 'config/secure_config.json'):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._override_with_env()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration structure"""
        return {
            "api_keys": {
                "finnhub": None,  # Load from environment
                "openai": None,
                "anthropic": None,
                "binance": None,
                "alpaca": None
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "ncos_trading"
            },
            "redis": {
                "url": "redis://localhost:6379/0"
            },
            "ngrok": {
                "url": os.getenv("NCOS_API_URL", "https://emerging-tiger-fair.ngrok-free.app")
            }
        }

    def _override_with_env(self):
        """Override config with environment variables"""
        # API Keys - NEVER hardcode these
        self.config["api_keys"]["finnhub"] = os.getenv("FINNHUB_API_KEY")
        self.config["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
        self.config["api_keys"]["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
        self.config["api_keys"]["binance"] = os.getenv("BINANCE_API_KEY")
        self.config["api_keys"]["alpaca"] = os.getenv("ALPACA_API_KEY")

        # Database
        self.config["database"]["host"] = os.getenv("NCOS_DB_HOST", self.config["database"]["host"])
        self.config["database"]["port"] = int(os.getenv("NCOS_DB_PORT", self.config["database"]["port"]))
        self.config["database"]["user"] = os.getenv("NCOS_DB_USER", "ncos_user")
        self.config["database"]["password"] = os.getenv("NCOS_DB_PASSWORD")

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service (returns None if not set)"""
        key = self.config["api_keys"].get(service)
        if not key:
            print(f"⚠️  Warning: {service.upper()}_API_KEY not found in environment")
        return key

    def get(self, path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global config instance
config = SecureConfig()

# Example usage
if __name__ == "__main__":
    # This will load from environment, not hardcoded
    finnhub_key = config.get_api_key("finnhub")
    if finnhub_key:
        print("✅ Finnhub API key loaded from environment")
    else:
        print("❌ Finnhub API key not found - set FINNHUB_API_KEY in .env")
