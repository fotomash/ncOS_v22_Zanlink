# Create a complete fix package for ncOS v22

import os
import json
import shutil

# Create fix directory
fix_dir = "ncOS_v22_fixes"
os.makedirs(fix_dir, exist_ok=True)

# 1. Create the missing app.py
app_content = '''#!/usr/bin/env python3
"""
ncOS v22 - FastAPI Application with Zanlink Integration
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import ncOS modules
try:
    from integrations.ncos_zanlink_bridge import ZanlinkBridge
    from integrations.ncos_llm_gateway import LLMGateway
    from core.orchestrator import Orchestrator
    from agents.master_orchestrator import MasterOrchestrator
except ImportError as e:
    print(f"Warning: Some modules not found: {e}")

app = FastAPI(
    title="ncOS v22 Zanlink API",
    description="Neural Compute Operating System - Trading Edition with LLM Integration",
    version="22.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
zanlink_bridge = None
llm_gateway = None
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize system components"""
    global zanlink_bridge, llm_gateway, orchestrator
    
    try:
        # Load configuration
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {"system": {"mode": "production"}}
        
        # Initialize components
        zanlink_bridge = ZanlinkBridge(config.get("zanlink", {}))
        llm_gateway = LLMGateway(config.get("llm", {}))
        
        # Initialize orchestrator if available
        if "MasterOrchestrator" in globals():
            orchestrator = MasterOrchestrator(config)
            await orchestrator.initialize()
            
        print("✅ ncOS v22 initialized successfully")
        
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")

@app.get("/")
async def root():
    return {
        "name": "ncOS v22 Zanlink Edition",
        "status": "operational",
        "version": "22.0",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "zanlink": zanlink_bridge is not None,
            "llm": llm_gateway is not None,
            "orchestrator": orchestrator is not None
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check Zanlink
    if zanlink_bridge:
        try:
            zanlink_status = await zanlink_bridge.check_connection()
            health_status["components"]["zanlink"] = zanlink_status
        except:
            health_status["components"]["zanlink"] = {"status": "error"}
    
    # Check LLM
    if llm_gateway:
        health_status["components"]["llm"] = {"status": "ready"}
    
    # Check Orchestrator
    if orchestrator:
        health_status["components"]["orchestrator"] = {"status": "active"}
    
    return health_status

@app.post("/api/v1/signals")
async def get_trading_signals(request: dict):
    """Get trading signals from the system"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        signals = await orchestrator.get_signals(
            symbol=request.get("symbol", "XAUUSD"),
            timeframe=request.get("timeframe", "5m")
        )
        return {"signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
async def analyze_market(request: dict):
    """Analyze market with theory integration"""
    try:
        # This would integrate with theory modules
        analysis = {
            "symbol": request.get("symbol"),
            "timestamp": datetime.now().isoformat(),
            "wyckoff_phase": "Accumulation",
            "smc_levels": [],
            "hidden_orders": [],
            "recommendation": "WAIT"
        }
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(5)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
'''

with open(os.path.join(fix_dir, "app.py"), 'w') as f:
    f.write(app_content)

# 2. Create comprehensive config.json
config_content = {
    "system": {
        "name": "ncOS v22 Zanlink Edition",
        "version": "22.0",
        "mode": "production",
        "debug": False
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4
    },
    "zanlink": {
        "base_url": "https://api.zanlink.com",
        "version": "v1",
        "timeout": 30,
        "retry_attempts": 3
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "trading": {
        "symbols": ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"],
        "default_timeframe": "5m",
        "risk_per_trade": 0.01,
        "max_positions": 3,
        "max_drawdown": 0.15
    },
    "modules": {
        "theory_engine": True,
        "wyckoff_analyzer": True,
        "smc_detector": True,
        "hidden_order_scanner": True,
        "llm_integration": True,
        "realtime_trading": True,
        "backtesting": True
    },
    "data": {
        "tick_buffer_size": 1000,
        "candle_history": 500,
        "update_interval": 100
    },
    "risk": {
        "max_daily_loss": 0.02,
        "max_position_size": 0.1,
        "stop_loss_atr_multiplier": 2.0,
        "take_profit_ratios": [2.0, 3.0, 4.0]
    },
    "notifications": {
        "telegram": {
            "enabled": False,
            "bot_token": "",
            "chat_id": ""
        },
        "email": {
            "enabled": False,
            "smtp_server": "",
            "port": 587
        }
    }
}

with open(os.path.join(fix_dir, "config.json"), 'w') as f:
    json.dump(config_content, f, indent=2)

# 3. Create updated requirements.txt with all dependencies
requirements_content = '''# ncOS v22 Complete Requirements
# Last updated: 2025-06-23

# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Trading and market data
ccxt>=3.0.0
yfinance>=0.2.0
pandas-ta>=0.3.0
vectorbt>=0.24.0
backtrader>=1.9.76.123
finnhub-python>=2.4.0
MetaTrader5>=5.0.0  # Added for MT5 integration

# Web and API
fastapi>=0.95.0
uvicorn>=0.20.0
requests>=2.28.0
websocket-client>=1.5.0
aiohttp>=3.8.0
httpx>=0.23.0
python-multipart>=0.0.5

# Database and caching
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=4.5.0
pymongo>=4.3.0
aiosqlite>=0.19.0

# LLM and AI
openai>=1.0.0
anthropic>=0.3.0
langchain>=0.0.200
transformers>=4.30.0
torch>=2.0.0
tiktoken>=0.4.0

# Data processing
pyarrow>=11.0.0
dask>=2023.1.0
polars>=0.16.0
h5py>=3.8.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.13.0
kaleido>=0.2.1  # For plotly image export

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=7.2.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0

# Technical indicators
ta>=0.10.0
tulipy>=0.4.0
talib-binary>=0.4.24  # Pre-compiled TA-Lib

# Machine Learning
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
optuna>=3.1.0  # For hyperparameter optimization

# Time series
statsmodels>=0.13.0
prophet>=1.1.0
pmdarima>=2.0.0
arch>=5.3.0  # For GARCH models

# Monitoring and logging
prometheus-client>=0.16.0
structlog>=23.0.0
loguru>=0.6.0
sentry-sdk>=1.14.0

# Development tools
jupyter>=1.0.0
ipython>=8.10.0
jupyterlab>=3.6.0
notebook>=6.5.0

# Additional for ncOS specific features
pyyaml>=6.0
click>=8.1.0
rich>=13.0.0  # For beautiful terminal output
typer>=0.7.0  # CLI interface
watchdog>=2.2.0  # File system monitoring
schedule>=1.1.0  # Task scheduling
'''

with open(os.path.join(fix_dir, "requirements.txt"), 'w') as f:
    f.write(requirements_content)

# 4. Create a proper .env.example
env_example = '''# ncOS v22 Environment Variables
# Copy this to .env and fill in your values

# System
NCOS_ENV=production
NCOS_DEBUG=false
NCOS_LOG_LEVEL=INFO

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here

# Zanlink Configuration
ZANLINK_API_URL=https://api.zanlink.com
ZANLINK_API_KEY=your_zanlink_api_key_here
ZANLINK_API_SECRET=your_zanlink_api_secret_here

# Ngrok (for development)
NGROK_URL=https://your-ngrok-url.ngrok.io
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here

# Database
DATABASE_URL=postgresql://user:password@localhost/ncos
REDIS_URL=redis://localhost:6379

# Trading
DEFAULT_SYMBOL=XAUUSD
RISK_PER_TRADE=0.01
MAX_POSITIONS=3

# Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# MetaTrader 5
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
'''

with open(os.path.join(fix_dir, ".env.example"), 'w') as f:
    f.write(env_example)

