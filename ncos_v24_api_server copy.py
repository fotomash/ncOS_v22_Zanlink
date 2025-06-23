# ncos_v5_fixed_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path

# Import menu system
from ncos_v5_ultimate_menu import UltimateMenuSystem

app = FastAPI(
    title="ncOS Trading System v5",
    version="5.0",
    description="Advanced Trading Analysis System"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)

# Global state
SYSTEM_STATE = {
    "menu": UltimateMenuSystem(),
    "cache": {},
    "sessions": {}
}

class ActionRequest(BaseModel):
    action: str
    params: Optional[Dict[str, Any]] = None

class ActionResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    menu: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """System info endpoint"""
    return {
        "system": "ncOS Trading System v5",
        "status": "online",
        "version": "5.0",
        "endpoints": {
            "health": "/health",
            "menu": "/menu",
            "action": "/action",
            "data": "/data/scan"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "5.0"
    }

@app.get("/menu")
async def get_menu():
    """Get current menu"""
    menu = SYSTEM_STATE["menu"]
    current_menu = menu.get_main_menu()
    
    return ActionResponse(
        status="success",
        menu=current_menu
    )

@app.post("/action")
async def process_action(request: ActionRequest):
    """Main action processor"""
    try:
        action = request.action
        params = request.params or {}
        
        # Menu selection
        if action == "menu_select":
            choice = params.get("choice", "")
            result = SYSTEM_STATE["menu"].process_selection(choice)
            
            if "error" in result:
                return ActionResponse(
                    status="error",
                    error=result["error"]
                )
            
            if result.get("action") == "execute_analysis":
                # Execute analysis
                analysis_data = await execute_analysis(result["params"])
                return ActionResponse(
                    status="success",
                    data=analysis_data,
                    message="Analysis complete!"
                )
            
            elif result.get("action") == "scan_data":
                # Scan available data
                files = scan_data_files()
                return ActionResponse(
                    status="success",
                    data={"files": files},
                    message=f"Found {len(files)} data files"
                )
            
            elif result.get("action") == "quick_analysis":
                # Quick analysis
                pair = result["params"]["pair"]
                analysis = await quick_analysis(pair)
                return ActionResponse(
                    status="success",
                    data=analysis,
                    message=result.get("message")
                )
            
            # Return menu
            return ActionResponse(
                status="success",
                menu=result.get("menu"),
                message=result.get("message")
            )
        
        # Direct actions
        elif action == "reset":
            SYSTEM_STATE["menu"] = UltimateMenuSystem()
            return ActionResponse(
                status="success",
                menu=SYSTEM_STATE["menu"].get_main_menu(),
                message="Menu reset to main"
            )
        
        else:
            return ActionResponse(
                status="error",
                error=f"Unknown action: {action}"
            )
            
    except Exception as e:
        return ActionResponse(
            status="error",
            error=str(e)
        )

def scan_data_files() -> List[Dict]:
    """Scan for available data files"""
    files = []
    
    # Check multiple locations
    locations = [".", "data", "uploads", "data/ticks", "data/historical"]
    
    for location in locations:
        path = Path(location)
        if path.exists():
            for file in path.glob("*.csv"):
                files.append({
                    "name": file.name,
                    "path": str(file),
                    "size": f"{file.stat().st_size / 1024:.1f} KB",
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
    
    return files

async def quick_analysis(pair: str) -> Dict:
    """Quick analysis for a trading pair"""
    # Look for data file
    data_file = None
    for file in Path(".").glob(f"*{pair}*.csv"):
        data_file = file
        break
    
    if not data_file:
        return {"error": f"No data found for {pair}"}
    
    try:
        # Load data
        df = pd.read_csv(data_file, nrows=1000)  # Limit rows for quick analysis
        
        result = {
            "pair": pair,
            "data_points": len(df),
            "timespan": {}
        }
        
        # Analyze based on data type
        if 'bid' in df.columns and 'ask' in df.columns:
            # Tick data
            result["data_type"] = "tick"
            result["latest_bid"] = float(df['bid'].iloc[-1])
            result["latest_ask"] = float(df['ask'].iloc[-1])
            result["average_spread"] = float((df['ask'] - df['bid']).mean())
            result["price_range"] = {
                "high": float(df['bid'].max()),
                "low": float(df['bid'].min())
            }
            result["volatility"] = float(df['bid'].std())
            
        elif 'close' in df.columns:
            # OHLC data
            result["data_type"] = "ohlc"
            result["latest_close"] = float(df['close'].iloc[-1])
            result["price_range"] = {
                "high": float(df['high'].max()),
                "low": float(df['low'].min())
            }
            
        if 'timestamp' in df.columns:
            result["timespan"] = {
                "start": str(df['timestamp'].iloc[0]),
                "end": str(df['timestamp'].iloc[-1])
            }
            
        return result
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

async def execute_analysis(params: Dict) -> Dict:
    """Execute full analysis"""
    pair = params.get("pair")
    timeframe = params.get("timeframe")
    analysis_type = params.get("analysis_type")
    
    # Mock advanced analysis
    analysis = {
        "pair": pair,
        "timeframe": timeframe,
        "type": analysis_type,
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    if analysis_type == "smc":
        analysis["results"]["smart_money"] = {
            "order_blocks": [
                {"level": 3385.50, "type": "bullish", "strength": "high"},
                {"level": 3378.20, "type": "bearish", "strength": "medium"}
            ],
            "liquidity_zones": [
                {"zone": [3390.00, 3392.00], "side": "buy"},
                {"zone": [3375.00, 3377.00], "side": "sell"}
            ]
        }
    
    elif analysis_type == "structure":
        analysis["results"]["market_structure"] = {
            "trend": "neutral",
            "key_levels": [3380.00, 3385.00, 3390.00],
            "swing_points": {
                "highs": [3387.50, 3391.20],
                "lows": [3378.30, 3382.10]
            }
        }
    
    return analysis

@app.get("/data/scan")
async def data_scan_endpoint():
    """Scan available data files"""
    files = scan_data_files()
    return {
        "status": "success",
        "count": len(files),
        "files": files
    }

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Suppress logs for clean output
    logging.getLogger("uvicorn").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    
    print("\n" + "="*50)
    print("🚀 ncOS Trading System v5 Starting...")
    print("="*50)
    print(f"📡 Local: http://localhost:8000")
    print(f"🌐 Ngrok: https://4529-213-205-193-19.ngrok-free.app")
    print("="*50)
    print("📋 Ready for connections!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")