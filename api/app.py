#!/usr/bin/env python3
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
