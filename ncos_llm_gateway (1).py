
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime

class LLMRequest(BaseModel):
    """Simplified request format for LLM"""
    action: str  # analyze, summarize, detect, predict
    context: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = {}

class LLMResponse(BaseModel):
    """Pre-digested response for LLM"""
    summary: str
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    visualizations: Optional[List[Dict[str, Any]]] = []

class NCOSLLMGateway:
    """Unified gateway for LLM interactions"""

    def __init__(self):
        self.processors = {
            "analyze_session": self._analyze_session,
            "market_overview": self._market_overview,
            "pattern_detection": self._pattern_detection,
            "trade_recommendation": self._trade_recommendation,
            "performance_summary": self._performance_summary
        }

    async def process_llm_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request and return pre-digested data"""
        if request.action not in self.processors:
            raise HTTPException(400, f"Unknown action: {request.action}")

        return await self.processors[request.action](request.context, request.filters)

    async def _analyze_session(self, context: Dict, filters: Dict) -> LLMResponse:
        """Analyze trading session with pre-processed insights"""
        session_id = context.get("session_id")

        # Simulate data processing
        analysis = {
            "trades": 15,
            "win_rate": 0.73,
            "patterns_detected": ["Wyckoff Spring", "SMC MSS", "Liquidity Sweep"],
            "key_levels": [1950.50, 1945.30, 1940.00],
            "bias": "bullish",
            "strength": 7.5
        }

        summary = f"Session {session_id}: Strong bullish bias (7.5/10) with 73% win rate across 15 trades. Key patterns: Wyckoff Spring at 1945.30, followed by SMC MSS confirming upward momentum."

        insights = [
            "Liquidity sweep below 1940 created optimal long entry",
            "Volume profile shows institutional accumulation",
            "Price respecting 1950.50 resistance - potential breakout zone"
        ]

        recommendations = [
            "Wait for retest of 1945.30 for long entries",
            "Set alerts at 1950.50 for breakout confirmation",
            "Reduce position size if price breaks below 1940"
        ]

        return LLMResponse(
            summary=summary,
            data=analysis,
            insights=insights,
            recommendations=recommendations
        )

    async def _market_overview(self, context: Dict, filters: Dict) -> LLMResponse:
        """Provide market overview in LLM-friendly format"""
        symbol = context.get("symbol", "XAUUSD")

        overview = {
            "symbol": symbol,
            "current_price": 1948.75,
            "daily_change": 0.82,
            "volume_profile": "above_average",
            "market_structure": "bullish",
            "key_zones": {
                "resistance": [1950.50, 1955.00, 1960.00],
                "support": [1945.30, 1940.00, 1935.50]
            },
            "sentiment": {
                "retail": "bearish",
                "institutional": "accumulating"
            }
        }

        summary = f"{symbol} showing bullish structure at 1948.75 (+0.82%). Institutional accumulation detected while retail remains bearish - classic smart money divergence."

        return LLMResponse(
            summary=summary,
            data=overview,
            insights=[
                "Smart money accumulating during retail bearishness",
                "Volume spike at 1945.30 suggests strong support",
                "Break above 1950.50 likely to trigger short squeeze"
            ],
            recommendations=[
                "Long bias preferred with stops below 1940",
                "Scale in positions between 1945-1948",
                "Target 1955 and 1960 for partial profits"
            ]
        )

# FastAPI app
app = FastAPI(title="ncOS LLM Gateway")
gateway = NCOSLLMGateway()

@app.post("/llm/process", response_model=LLMResponse)
async def process_llm_request(request: LLMRequest):
    """Single endpoint for all LLM interactions"""
    return await gateway.process_llm_request(request)

@app.get("/llm/actions")
async def get_available_actions():
    """List all available LLM actions"""
    return {
        "actions": list(gateway.processors.keys()),
        "description": "Use these actions in your requests"
    }
