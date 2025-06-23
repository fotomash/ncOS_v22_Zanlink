"""
ncOS Integration Bridge - Zanlink Edition
Connects existing ncOS system with LLM-optimized components via Zanlink
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from pathlib import Path
import httpx

# Configuration for Zanlink
ZANLINK_CONFIG = {
    "base_url": "https://zanlink.com/api/v1",
    "endpoints": {
        "analyze": "https://zanlink.com/api/v1/analyze",
        "quick_status": "https://zanlink.com/api/v1/quick/status",
        "patterns": "https://zanlink.com/api/v1/patterns/detect",
        "bridge": "https://zanlink.com/api/v1/bridge/process",
        "journal": "https://zanlink.com/api/v1/journal",
        "trade": "https://zanlink.com/api/v1/trade"
    },
    "timeout": 30,
    "retry_attempts": 3
}

class ZanlinkIntegrationBridge:
    """
    Main integration bridge for Zanlink-hosted ncOS
    Provides seamless connection between LLMs and trading system
    """

    def __init__(self, api_key: Optional[str] = None):
        self.config = ZANLINK_CONFIG
        self.api_key = api_key or os.getenv("ZANLINK_API_KEY")
        self.client = httpx.AsyncClient(
            timeout=self.config["timeout"],
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        )
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def analyze_market(self, symbol: str = "XAUUSD", 
                           timeframe: str = "H1", 
                           analysis_type: str = "market") -> Dict[str, Any]:
        """
        Analyze market with pre-processed insights

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            analysis_type: Type of analysis (market, patterns, session, performance)

        Returns:
            Pre-processed analysis ready for LLM consumption
        """
        try:
            response = await self.client.post(
                self.config["endpoints"]["analyze"],
                json={
                    "type": analysis_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "context": {
                        "timestamp": datetime.now().isoformat(),
                        "source": "llm_bridge"
                    }
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "summary": "Analysis unavailable",
                "insights": ["Error occurred during analysis"],
                "recommendations": ["Please try again later"]
            }

    async def get_quick_status(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """
        Get ultra-fast market status
        Perfect for quick ChatGPT queries
        """
        try:
            response = await self.client.get(
                self.config["endpoints"]["quick_status"],
                params={"symbol": symbol}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "symbol": symbol,
                "one_line_summary": f"Unable to fetch status for {symbol}",
                "error": str(e)
            }

    async def detect_patterns(self, symbol: str = "XAUUSD") -> Dict[str, Any]:
        """
        Detect trading patterns using Wyckoff and SMC
        """
        try:
            response = await self.client.get(
                self.config["endpoints"]["patterns"],
                params={"symbol": symbol}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "patterns": [],
                "summary": "Pattern detection unavailable",
                "bias": "neutral",
                "error": str(e)
            }

    async def process_complex_request(self, action: str, 
                                    context: Dict[str, Any], 
                                    format: str = "chatgpt") -> Dict[str, Any]:
        """
        Process complex requests through the bridge
        """
        try:
            response = await self.client.post(
                self.config["endpoints"]["bridge"],
                json={
                    "action": action,
                    "context": context,
                    "format": format
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "response": f"Error processing request: {str(e)}",
                "error": True
            }

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Utility functions for easy integration
async def quick_market_check(symbol: str = "XAUUSD") -> str:
    """
    One-line market check for ChatGPT

    Example:
        status = await quick_market_check("XAUUSD")
        # Returns: "XAUUSD bullish at 1948.75, watch 1950.50 resistance"
    """
    bridge = ZanlinkIntegrationBridge()
    try:
        result = await bridge.get_quick_status(symbol)
        return result.get("one_line_summary", "Status unavailable")
    finally:
        await bridge.close()

async def get_trading_signals(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    Get trading signals with entry/exit recommendations
    """
    bridge = ZanlinkIntegrationBridge()
    try:
        # Get patterns
        patterns = await bridge.detect_patterns(symbol)

        # Get market analysis
        analysis = await bridge.analyze_market(symbol, analysis_type="market")

        # Combine into trading signals
        return {
            "symbol": symbol,
            "bias": patterns.get("bias", "neutral"),
            "patterns_detected": len(patterns.get("patterns", [])),
            "entry_zones": analysis.get("data", {}).get("entry_zones", []),
            "stop_loss": analysis.get("data", {}).get("stop_loss", None),
            "take_profit": analysis.get("data", {}).get("take_profit", []),
            "confidence": analysis.get("data", {}).get("confidence", 0),
            "recommendation": analysis.get("recommendations", ["No clear signal"])[0]
        }
    finally:
        await bridge.close()

# FastAPI endpoints for local testing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ncOS Zanlink Bridge - Local")

class MarketRequest(BaseModel):
    symbol: str = "XAUUSD"
    timeframe: str = "H1"

class SignalRequest(BaseModel):
    symbol: str = "XAUUSD"

@app.get("/")
async def root():
    return {
        "service": "ncOS Zanlink Integration Bridge",
        "version": "2.0",
        "endpoints": {
            "market_check": "/market/quick",
            "signals": "/signals",
            "analyze": "/analyze"
        }
    }

@app.get("/market/quick")
async def quick_check(symbol: str = "XAUUSD"):
    """Quick market status check"""
    status = await quick_market_check(symbol)
    return {"status": status}

@app.post("/signals")
async def get_signals(request: SignalRequest):
    """Get trading signals"""
    signals = await get_trading_signals(request.symbol)
    return signals

@app.post("/analyze")
async def analyze(request: MarketRequest):
    """Full market analysis"""
    bridge = ZanlinkIntegrationBridge()
    try:
        result = await bridge.analyze_market(
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        return result
    finally:
        await bridge.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
