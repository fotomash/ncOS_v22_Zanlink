#!/usr/bin/env python3
"""
ncOS Local Server - Works with ngrok without external API dependencies
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path('ncos_config_local.json')
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {"api": {"local_port": 8000}}

# Create necessary directories
os.makedirs('./data/ticks', exist_ok=True)
os.makedirs('./data/historical', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./uploads', exist_ok=True)

# In-memory storage for demo
demo_data = {
    "market_data": [],
    "signals": [],
    "trades": [],
    "backtests": []
}

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "status": "online",
        "service": "ncOS Local Server",
        "version": "22.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/",
            "/health",
            "/data",
            "/upload-trace",
            "/api/trading",
            "/api/market",
            "/api/signals",
            "/api/backtest"
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    })

@app.route('/data', methods=['GET', 'POST'])
def data_endpoint():
    """Handle data requests"""
    if request.method == 'GET':
        # Return available data files
        tick_files = list(Path('./data/ticks').glob('*.csv'))
        historical_files = list(Path('./data/historical').glob('*.parquet'))

        return jsonify({
            "tick_files": [f.name for f in tick_files],
            "historical_files": [f.name for f in historical_files],
            "demo_data_available": True
        })

    elif request.method == 'POST':
        # Store incoming data
        data = request.get_json()
        if data:
            demo_data["market_data"].append({
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
            return jsonify({"status": "success", "message": "Data stored"})
        return jsonify({"status": "error", "message": "No data provided"}), 400

@app.route('/upload-trace', methods=['POST'])
def upload_trace():
    """Handle trace uploads"""
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file:
                filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                filepath = os.path.join('./uploads', filename)
                file.save(filepath)
                logger.info(f"Trace file saved: {filename}")
                return jsonify({
                    "status": "success",
                    "filename": filename,
                    "message": "Trace uploaded successfully"
                })

        # Handle JSON trace data
        data = request.get_json()
        if data:
            filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('./uploads', filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return jsonify({
                "status": "success",
                "filename": filename,
                "message": "Trace data saved"
            })

    except Exception as e:
        logger.error(f"Error uploading trace: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "No trace data provided"}), 400

@app.route('/api/trading', methods=['GET', 'POST'])
def trading_api():
    """Trading API endpoint"""
    if request.method == 'GET':
        return jsonify({
            "trades": demo_data["trades"][-10:],  # Last 10 trades
            "demo_mode": True
        })

    elif request.method == 'POST':
        trade = request.get_json()
        if trade:
            trade["timestamp"] = datetime.now().isoformat()
            trade["demo"] = True
            demo_data["trades"].append(trade)
            return jsonify({
                "status": "success",
                "trade_id": len(demo_data["trades"]),
                "message": "Demo trade executed"
            })
        return jsonify({"status": "error", "message": "Invalid trade data"}), 400

@app.route('/api/market', methods=['GET'])
def market_data():
    """Market data endpoint"""
    # Return demo market data or load from files
    try:
        # Check for XAUUSD tick data
        tick_file = Path('./data/ticks/XAUUSD_TICKS_1days_20250623.csv')
        if tick_file.exists():
            df = pd.read_csv(tick_file, sep='\t', nrows=100)  # Last 100 ticks
            return jsonify({
                "symbol": "XAUUSD",
                "data": df.to_dict(orient='records'),
                "source": "file"
            })
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")

    # Return demo data
    return jsonify({
        "symbol": "XAUUSD",
        "bid": 3357.50,
        "ask": 3357.81,
        "spread": 0.31,
        "timestamp": datetime.now().isoformat(),
        "source": "demo"
    })

@app.route('/api/signals', methods=['GET', 'POST'])
def signals_api():
    """Trading signals endpoint"""
    if request.method == 'GET':
        return jsonify({
            "signals": demo_data["signals"][-10:],  # Last 10 signals
            "active_signals": len([s for s in demo_data["signals"] if s.get("active", False)])
        })

    elif request.method == 'POST':
        signal = request.get_json()
        if signal:
            signal["timestamp"] = datetime.now().isoformat()
            signal["id"] = len(demo_data["signals"]) + 1
            demo_data["signals"].append(signal)
            return jsonify({
                "status": "success",
                "signal_id": signal["id"],
                "message": "Signal recorded"
            })
        return jsonify({"status": "error", "message": "Invalid signal data"}), 400

@app.route('/api/backtest', methods=['POST'])
def backtest_api():
    """Backtesting endpoint"""
    params = request.get_json()
    if not params:
        return jsonify({"status": "error", "message": "No backtest parameters provided"}), 400

    # Simple demo backtest result
    result = {
        "id": len(demo_data["backtests"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "parameters": params,
        "results": {
            "total_trades": 100,
            "winning_trades": 55,
            "losing_trades": 45,
            "win_rate": 0.55,
            "profit_factor": 1.25,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25
        },
        "status": "completed"
    }

    demo_data["backtests"].append(result)

    return jsonify(result)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == '__main__':
    port = config.get('api', {}).get('local_port', 8000)
    logger.info(f"Starting ncOS Local Server on port {port}")
    logger.info("Server is configured to work with ngrok")
    logger.info("No external API keys required")

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )
