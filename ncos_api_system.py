
"""
ncOS Professional Trading API System
Advanced REST API with WebSocket support for real-time trading analysis
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TickData(BaseModel):
    timestamp: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

class SpreadAnalysisResponse(BaseModel):
    symbol: str
    timestamp: str
    spread_bps: float
    spread_percentage: float
    spread_quality: str
    historical_percentile: Optional[float] = None

class ManipulationAnalysis(BaseModel):
    overall_score: float = Field(..., ge=0, le=1, description="Manipulation probability (0-1)")
    risk_level: str = Field(..., description="HIGH/MEDIUM/LOW")
    detected_patterns: List[str]
    anomaly_flags: List[str]
    confidence_factors: Dict[str, float]
    regulatory_notes: str
    recommended_actions: List[str]

class MarketMicrostructureMetrics(BaseModel):
    price_impact: float
    volatility: float
    order_flow_toxicity: float
    liquidity_score: float
    market_efficiency: float

class ComprehensiveTickAnalysis(BaseModel):
    tick_data: TickData
    spread_analysis: SpreadAnalysisResponse
    manipulation_analysis: ManipulationAnalysis
    microstructure_metrics: MarketMicrostructureMetrics
    timestamp_processed: str
    processing_latency_ms: float

class PatternRecognitionRequest(BaseModel):
    data: List[Dict[str, Any]]
    timeframe: str = "1h"
    pattern_types: List[str] = ["wyckoff", "smc", "harmonic"]

class BacktestRequest(BaseModel):
    strategy_name: str
    data_source: str
    start_date: str
    end_date: str
    initial_capital: float = 10000
    parameters: Dict[str, Any] = {}

# Initialize FastAPI app
app = FastAPI(
    title="ncOS Professional Trading API",
    description="Advanced trading analysis API with real-time manipulation detection",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
class APIState:
    def __init__(self):
        self.tick_processors = {}
        self.active_streams = {}
        self.analysis_cache = {}
        self.websocket_connections = set()

api_state = APIState()

class AdvancedTickProcessor:
    """Enhanced tick processor for professional trading analysis"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tick_buffer = []
        self.spread_history = []
        self.volume_profile = {}
        self.manipulation_indicators = {
            'spoofing_score': 0.0,
            'layering_score': 0.0,
            'quote_stuffing_score': 0.0,
            'wash_trading_score': 0.0,
            'momentum_ignition_score': 0.0
        }

    async def process_tick(self, tick: TickData) -> ComprehensiveTickAnalysis:
        """Process tick with comprehensive analysis"""
        start_time = datetime.now()

        # Update internal state
        self.tick_buffer.append(tick.dict())
        if len(self.tick_buffer) > 1000:
            self.tick_buffer.pop(0)

        # Calculate spread metrics
        spread_analysis = self._analyze_spread(tick)

        # Detect manipulation patterns
        manipulation_analysis = await self._detect_manipulation(tick)

        # Calculate microstructure metrics
        microstructure = self._calculate_microstructure(tick)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ComprehensiveTickAnalysis(
            tick_data=tick,
            spread_analysis=spread_analysis,
            manipulation_analysis=manipulation_analysis,
            microstructure_metrics=microstructure,
            timestamp_processed=datetime.now().isoformat(),
            processing_latency_ms=processing_time
        )

    def _analyze_spread(self, tick: TickData) -> SpreadAnalysisResponse:
        """Analyze bid-ask spread with historical context"""
        spread_bps = ((tick.ask - tick.bid) / ((tick.ask + tick.bid) / 2)) * 10000
        spread_pct = ((tick.ask - tick.bid) / tick.last) * 100

        # Store spread history
        self.spread_history.append(spread_bps)
        if len(self.spread_history) > 100:
            self.spread_history.pop(0)

        # Calculate quality metrics
        if spread_bps < 1:
            quality = "EXCELLENT"
        elif spread_bps < 3:
            quality = "GOOD"
        elif spread_bps < 8:
            quality = "FAIR"
        else:
            quality = "POOR"

        # Historical percentile
        percentile = None
        if len(self.spread_history) > 10:
            percentile = (sum(1 for s in self.spread_history if s <= spread_bps) / len(self.spread_history)) * 100

        return SpreadAnalysisResponse(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            spread_bps=spread_bps,
            spread_percentage=spread_pct,
            spread_quality=quality,
            historical_percentile=percentile
        )

    async def _detect_manipulation(self, tick: TickData) -> ManipulationAnalysis:
        """Advanced manipulation detection"""
        detected_patterns = []
        anomaly_flags = []
        confidence_factors = {}

        # Spoofing detection
        spoofing_score = self._detect_spoofing(tick)
        if spoofing_score > 0.6:
            detected_patterns.append("SPOOFING")
            confidence_factors["spoofing"] = spoofing_score

        # Quote stuffing detection
        quote_stuffing_score = self._detect_quote_stuffing(tick)
        if quote_stuffing_score > 0.7:
            detected_patterns.append("QUOTE_STUFFING")
            confidence_factors["quote_stuffing"] = quote_stuffing_score

        # Price manipulation detection
        price_manip_score = self._detect_price_manipulation(tick)
        if price_manip_score > 0.5:
            detected_patterns.append("PRICE_MANIPULATION")
            confidence_factors["price_manipulation"] = price_manip_score

        # Volume anomalies
        volume_anomaly = self._detect_volume_anomalies(tick)
        if volume_anomaly:
            anomaly_flags.extend(volume_anomaly)

        # Calculate overall score
        scores = [spoofing_score, quote_stuffing_score, price_manip_score]
        overall_score = np.mean(scores)

        # Risk level
        if overall_score > 0.7:
            risk_level = "HIGH"
        elif overall_score > 0.4:
            risk_level = "MEDIUM" 
        else:
            risk_level = "LOW"

        # Regulatory notes
        regulatory_notes = self._generate_regulatory_assessment(detected_patterns, overall_score)

        # Recommended actions
        actions = self._recommend_actions(detected_patterns, risk_level)

        return ManipulationAnalysis(
            overall_score=overall_score,
            risk_level=risk_level,
            detected_patterns=detected_patterns,
            anomaly_flags=anomaly_flags,
            confidence_factors=confidence_factors,
            regulatory_notes=regulatory_notes,
            recommended_actions=actions
        )

    def _detect_spoofing(self, tick: TickData) -> float:
        """Detect spoofing patterns"""
        if len(self.tick_buffer) < 5:
            return 0.0

        score = 0.0

        # Large quote sizes without execution
        if tick.bid_size and tick.ask_size:
            recent_volumes = [t.get('volume', 0) for t in self.tick_buffer[-5:]]
            avg_volume = np.mean(recent_volumes) if recent_volumes else 1

            total_quote_size = tick.bid_size + tick.ask_size
            if total_quote_size > avg_volume * 15:  # Unusually large quotes
                score += 0.4

            # Quick cancellations (would need order book data)
            # This is a simplified heuristic
            if total_quote_size > avg_volume * 20:
                score += 0.3

        return min(score, 1.0)

    def _detect_quote_stuffing(self, tick: TickData) -> float:
        """Detect excessive quote activity"""
        if len(self.tick_buffer) < 20:
            return 0.0

        # Count quotes in last 1 second
        current_time = datetime.fromisoformat(tick.timestamp)
        recent_ticks = []

        for t in reversed(self.tick_buffer):
            tick_time = datetime.fromisoformat(t['timestamp'])
            if (current_time - tick_time).total_seconds() <= 1.0:
                recent_ticks.append(t)
            else:
                break

        quote_rate = len(recent_ticks)

        # More than 100 quotes per second is highly suspicious
        if quote_rate > 100:
            return 1.0
        elif quote_rate > 50:
            return 0.8
        elif quote_rate > 25:
            return 0.5
        else:
            return 0.0

    def _detect_price_manipulation(self, tick: TickData) -> float:
        """Detect price manipulation schemes"""
        if len(self.tick_buffer) < 10:
            return 0.0

        prices = [t['last'] for t in self.tick_buffer[-10:]]
        volumes = [t['volume'] for t in self.tick_buffer[-10:]]

        # Look for pump and dump patterns
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_spike = volumes[-1] / (np.mean(volumes[:-1]) + 1e-9)

        score = 0.0

        # Rapid price increase with volume spike
        if price_change > 0.02 and volume_spike > 3:  # 2% price jump + 3x volume
            score += 0.6

        # Price volatility without fundamental justification
        price_volatility = np.std(prices) / np.mean(prices)
        if price_volatility > 0.01:  # 1% volatility
            score += 0.3

        return min(score, 1.0)

    def _detect_volume_anomalies(self, tick: TickData) -> List[str]:
        """Detect volume-based anomalies"""
        anomalies = []

        if len(self.tick_buffer) < 5:
            return anomalies

        volumes = [t['volume'] for t in self.tick_buffer[-5:]]
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]

        # Volume spikes
        if tick.volume > avg_volume * 5:
            anomalies.append("EXTREME_VOLUME_SPIKE")
        elif tick.volume > avg_volume * 2:
            anomalies.append("HIGH_VOLUME")

        # Unusual low volume
        if tick.volume < avg_volume * 0.1 and avg_volume > 0:
            anomalies.append("UNUSUALLY_LOW_VOLUME")

        return anomalies

    def _calculate_microstructure(self, tick: TickData) -> MarketMicrostructureMetrics:
        """Calculate market microstructure metrics"""
        if len(self.tick_buffer) < 10:
            return MarketMicrostructureMetrics(
                price_impact=0.0,
                volatility=0.0,
                order_flow_toxicity=0.0,
                liquidity_score=0.5,
                market_efficiency=0.5
            )

        recent_data = self.tick_buffer[-10:]

        # Price impact
        volumes = [t['volume'] for t in recent_data]
        prices = [t['last'] for t in recent_data]

        price_changes = [abs(prices[i] - prices[i-1])/prices[i-1] 
                        for i in range(1, len(prices))]
        avg_price_impact = np.mean(price_changes) / (np.mean(volumes) + 1e-9)

        # Volatility
        returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(returns) if returns else 0.0

        # Order flow toxicity (simplified)
        spreads = [((t['ask'] - t['bid']) / ((t['ask'] + t['bid'])/2)) for t in recent_data[-5:]]
        toxicity = np.mean(spreads) if spreads else 0.0

        # Liquidity score (inverse of spread)
        current_spread = (tick.ask - tick.bid) / tick.last
        liquidity_score = max(0, 1 - current_spread * 100)

        # Market efficiency (price discovery speed)
        price_autocorr = np.corrcoef(prices[:-1], prices[1:])[0,1] if len(prices) > 2 else 0
        efficiency = max(0, 1 - abs(price_autocorr))

        return MarketMicrostructureMetrics(
            price_impact=avg_price_impact,
            volatility=volatility,
            order_flow_toxicity=toxicity,
            liquidity_score=liquidity_score,
            market_efficiency=efficiency
        )

    def _generate_regulatory_assessment(self, patterns: List[str], score: float) -> str:
        """Generate regulatory compliance assessment"""
        notes = []

        if score > 0.8:
            notes.append("CRITICAL: High probability market manipulation detected.")
            notes.append("Immediate regulatory reporting recommended under MAR Article 16.")

        if "SPOOFING" in patterns:
            notes.append("Spoofing patterns detected - potential violation of Dodd-Frank Act Section 747.")

        if "QUOTE_STUFFING" in patterns:
            notes.append("Excessive quote activity may constitute market disruption under MiFID II.")

        if "PRICE_MANIPULATION" in patterns:
            notes.append("Price manipulation indicators present - review under Securities Exchange Act Section 9.")

        if not notes:
            notes.append("No immediate regulatory concerns identified based on current analysis.")

        return " ".join(notes)

    def _recommend_actions(self, patterns: List[str], risk_level: str) -> List[str]:
        """Recommend specific actions based on analysis"""
        actions = []

        if risk_level == "HIGH":
            actions.extend([
                "Halt automated trading systems",
                "Contact compliance department immediately", 
                "Document all evidence for regulatory submission",
                "Consider market making pause"
            ])
        elif risk_level == "MEDIUM":
            actions.extend([
                "Increase monitoring frequency to real-time",
                "Alert risk management team",
                "Review recent order flow patterns"
            ])
        else:
            actions.extend([
                "Continue standard monitoring",
                "Log analysis for audit trail"
            ])

        if "SPOOFING" in patterns:
            actions.append("Analyze order book depth and cancellation rates")

        if "QUOTE_STUFFING" in patterns:
            actions.append("Implement quote rate limiting")

        return actions

# API Routes
@app.post("/api/v1/analyze-tick", response_model=ComprehensiveTickAnalysis)
async def analyze_tick(tick_data: TickData):
    """
    Analyze individual tick for spread, manipulation, and microstructure metrics
    """
    try:
        # Get or create processor for symbol
        if tick_data.symbol not in api_state.tick_processors:
            api_state.tick_processors[tick_data.symbol] = AdvancedTickProcessor(tick_data.symbol)

        processor = api_state.tick_processors[tick_data.symbol]

        # Process tick
        analysis = await processor.process_tick(tick_data)

        # Broadcast to WebSocket clients
        await broadcast_to_websockets(analysis.dict())

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing tick: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-analyze")
async def batch_analyze_ticks(ticks: List[TickData]):
    """
    Analyze multiple ticks in batch for efficiency
    """
    try:
        results = []

        for tick in ticks:
            if tick.symbol not in api_state.tick_processors:
                api_state.tick_processors[tick.symbol] = AdvancedTickProcessor(tick.symbol)

            processor = api_state.tick_processors[tick.symbol]
            analysis = await processor.process_tick(tick)
            results.append(analysis)

        return {"analyses": results, "total_processed": len(results)}

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/manipulation-report/{symbol}")
async def get_manipulation_report(symbol: str, hours: int = 24):
    """
    Generate comprehensive manipulation report for symbol
    """
    try:
        if symbol not in api_state.tick_processors:
            raise HTTPException(status_code=404, detail="No data available for symbol")

        processor = api_state.tick_processors[symbol]

        # Generate comprehensive report
        report = {
            "symbol": symbol,
            "report_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "total_ticks_analyzed": len(processor.tick_buffer),
            "manipulation_indicators": processor.manipulation_indicators,
            "spread_statistics": {
                "current_spread_bps": processor.spread_history[-1] if processor.spread_history else 0,
                "average_spread_bps": np.mean(processor.spread_history) if processor.spread_history else 0,
                "spread_volatility": np.std(processor.spread_history) if len(processor.spread_history) > 1 else 0
            },
            "regulatory_status": "COMPLIANT" if np.mean(list(processor.manipulation_indicators.values())) < 0.3 else "REVIEW_REQUIRED",
            "next_review_due": (datetime.now() + timedelta(hours=8)).isoformat()
        }

        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/live-analysis/{symbol}")
async def websocket_live_analysis(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for live analysis streaming
    """
    await websocket.accept()
    api_state.websocket_connections.add(websocket)

    try:
        while True:
            # Keep connection alive and send heartbeat
            await asyncio.sleep(1)
            await websocket.send_json({
                "type": "heartbeat", 
                "timestamp": datetime.now().isoformat()
            })

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        api_state.websocket_connections.discard(websocket)

async def broadcast_to_websockets(data: dict):
    """Broadcast analysis results to all connected WebSocket clients"""
    if api_state.websocket_connections:
        message = {"type": "analysis", "data": data}
        disconnected = set()

        for websocket in api_state.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)

        # Remove disconnected clients
        api_state.websocket_connections -= disconnected

@app.get("/api/v1/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_processors": len(api_state.tick_processors),
        "websocket_connections": len(api_state.websocket_connections)
    }

@app.get("/api/v1/system-status")
async def system_status():
    """Detailed system status"""
    return {
        "api_version": "1.0.0",
        "uptime": "system_uptime_here",
        "memory_usage": "memory_stats_here",
        "active_symbols": list(api_state.tick_processors.keys()),
        "total_ticks_processed": sum(len(p.tick_buffer) for p in api_state.tick_processors.values()),
        "cache_size": len(api_state.analysis_cache)
    }

if __name__ == "__main__":
    uvicorn.run(
        "ncos_api:app",
        host="0.0.0.0", 
        port=8080,
        reload=True,
        log_level="info"
    )
