# ncos_server_no_api_keys.py
"""
ncOS Server - No API Keys Required
Uses local analysis and mock services
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
from mock_zanlink import mock_zanlink

app = Flask(__name__)
CORS(app)

# Global state
state = {
    "current_pair": "XAUUSD",
    "current_timeframe": "M5",
    "data_loaded": True,
    "zanlink": mock_zanlink
}

@app.route('/status', methods=['GET'])
def get_status():
    """System status"""
    return jsonify({
        "status": "online",
        "version": "6.0-nokeys",
        "current_pair": state["current_pair"],
        "current_timeframe": state["current_timeframe"],
        "data_loaded": state["data_loaded"],
        "mode": "local_analysis"
    })

@app.route('/set_instrument', methods=['POST'])
def set_instrument():
    """Set trading instrument"""
    data = request.json
    state["current_pair"] = data.get("pair", "XAUUSD")
    state["current_timeframe"] = data.get("timeframe", "M5")
    
    return jsonify({
        "status": "success",
        "message": f"Instrument set to {state['current_pair']} {state['current_timeframe']}"
    })

@app.route('/analyze', methods=['POST'])
def analyze_market():
    """Perform market analysis"""
    data = request.json
    pair = data.get("pair", state["current_pair"])
    timeframe = data.get("timeframe", state["current_timeframe"])
    analysis_type = data.get("analysis_type", "full")
    
    # Use mock Zanlink for analysis
    analysis = state["zanlink"].analyze_market(pair, timeframe)
    
    # Add analysis type specific data
    if analysis_type == "smc":
        analysis["smc_levels"] = {
            "order_blocks": [
                {"level": analysis["analysis"]["support_levels"][0], "type": "bullish"},
                {"level": analysis["analysis"]["resistance_levels"][0], "type": "bearish"}
            ],
            "fvg_zones": [
                {"high": analysis["analysis"]["current_price"] * 1.002, 
                 "low": analysis["analysis"]["current_price"] * 1.001}
            ]
        }
    
    return jsonify({
        "pair": pair,
        "timeframe": timeframe,
        "analysis_type": analysis_type,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/signals', methods=['POST'])
def get_signals():
    """Get trading signals"""
    data = request.json
    pair = data.get("pair", state["current_pair"])
    strategy = data.get("strategy", "moderate")
    
    # Use mock Zanlink for signals
    signals = state["zanlink"].get_signals(pair, strategy)
    
    return jsonify({
        "pair": pair,
        "strategy": strategy,
        "signals": signals,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute trade (demo mode)"""
    data = request.json
    
    # Mock trade execution
    trade_id = f"trade_{datetime.now().timestamp()}"
    
    return jsonify({
        "trade_id": trade_id,
        "status": "executed",
        "message": f"Demo trade executed: {data['action']} {data['volume']} {data['pair']}",
        "execution_price": 2050.50,  # Mock price
        "mode": "demo"
    })

@app.route('/chart', methods=['GET'])
def get_chart():
    """Get chart data"""
    # Generate mock OHLC data
    timestamps = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    base_price = 2050.0
    
    candles = []
    for i, ts in enumerate(timestamps):
        volatility = np.random.uniform(0.0005, 0.002)
        open_price = base_price * (1 + np.random.uniform(-volatility, volatility))
        high = open_price * (1 + np.random.uniform(0, volatility))
        low = open_price * (1 - np.random.uniform(0, volatility))
        close = np.random.uniform(low, high)
        
        candles.append({
            "timestamp": ts.isoformat(),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2)
        })
        
        base_price = close
    
    # Add some mock patterns
    markings = [
        {
            "type": "FVG_Bullish",
            "time": timestamps[50].isoformat(),
            "price_level_start": candles[50]["low"],
            "price_level_end": candles[50]["high"],
            "description": "Fair Value Gap (Bullish)"
        },
        {
            "type": "OrderBlock_Bearish",
            "time": timestamps[70].isoformat(),
            "price_level_start": candles[70]["low"],
            "price_level_end": candles[70]["high"],
            "description": "Bearish Order Block"
        }
    ]
    
    return jsonify({
        "pair": state["current_pair"],
        "timeframe": state["current_timeframe"],
        "candles": candles,
        "markings": markings
    })

if __name__ == '__main__':
    print("🚀 ncOS Server (No API Keys Required)")
    print("📡 Running on http://localhost:5005")
    print("🔧 Using local analysis - no external APIs")
    app.run(host='0.0.0.0', port=5005, debug=False)