import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, g
from werkzeug.serving import make_server
import threading
import logging
from typing import Optional, Dict, Any, List
import requests # --- NEW ---

# --- Configuration ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- System Configuration (MODIFIED) ---
SYSTEM_CONFIG = {
    "remote_api_base_url": "https://emerging-tiger-fair.ngrok-free.app", # --- NEW ---
    "default_pair": "XAUUSD",
    "default_timeframe": "M5",
    "max_candles_render": 250,
    "server_host": "0.0.0.0",
    "server_port": 5005
}

# --- Confluence Scoring Engine (NEW) ---
WEIGHTS = dict(
    smc_structure=0.30,
    wyckoff_phase=0.20,
    harmonic_alignment=0.15,
    candle_patterns=0.15,
    rsi_state=0.10,
    trend_bias=0.10
)

class ConfluenceAggregator:
    def __init__(self, weights=WEIGHTS):
        self.w = weights
        self.history = []

    def score(self, analysis: dict) -> Tuple[float, Dict[str, Any]]:
        """Calculates a confluence score from a structured analysis dictionary."""
        score = 0
        details = {}

        try:
            # Safely access nested dictionary keys
            s = analysis.get("smc", {}).get("structure", "neutral")
            w = analysis.get("wyckoff", {}).get("current_phase", "no_phase")
            h = analysis.get("harmonic", {}).get("patterns_detected", [])
            rsi = analysis.get("indicators", {}).get("rsi", 50)
            trend = analysis.get("trend_analysis", {}).get("direction", "sideways")
            patterns = analysis.get("patterns", {}).get("counts", {})

            # Scoring logic
            smc_score = self.w["smc_structure"] * (1 if s in ("bullish", "strong_bullish") else -1 if s in ("bearish", "strong_bearish") else 0)
            wyckoff_score = self.w["wyckoff_phase"] * (1 if w == "accumulation" else -1 if w == "distribution" else 0)
            harmonic_score = self.w["harmonic_alignment"] * (1 if any(p in ["abcd", "gartley"] for p in h) else 0) # Simplified for now
            rsi_score = self.w["rsi_state"] * (1 if 50 < rsi < 70 else -1 if 30 < rsi < 50 else 0)
            trend_score = self.w["trend_bias"] * (1 if trend.startswith("strong_bullish") else -1 if trend.startswith("strong_bearish") else 0)
            
            score = smc_score + wyckoff_score + harmonic_score + rsi_score + trend_score
            details = {
                "smc_score": round(smc_score, 2),
                "wyckoff_score": round(wyckoff_score, 2),
                "harmonic_score": round(harmonic_score, 2),
                "rsi_score": round(rsi_score, 2),
                "trend_score": round(trend_score, 2)
            }

        except Exception as e:
            print(f"Error during scoring: {e}")
            return 0.0, {"error": str(e)}

        final_score = round(score, 3)
        self.history.append((pd.Timestamp.now(), final_score, details))
        return final_score, details

# Instantiate the aggregator globally
aggregator = ConfluenceAggregator()

# --- Session State ---
def get_session():
    if 'session' not in g:
        g.session = {
            "current_pair": None,
            "current_timeframe": None,
            "data_loaded": False,
            "df": None,
            "markings": None, # --- NEW ---
            "last_error": None
        }
    return g.session

# --- Data Loading and Processing (MODIFIED) ---
def fetch_remote_chart(pair: str, tf: str) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
    """Fetches candles and markings from the live remote analysis server."""
    session = get_session()
    base_url = SYSTEM_CONFIG["remote_api_base_url"]
    chart_url = f"{base_url}/chart"
    instrument_url = f"{base_url}/set_instrument"
    
    try:
        # 1. Set the instrument remotely
        req_body = {"pair": pair, "timeframe": tf}
        headers = {'ngrok-skip-browser-warning': 'true'} # Add this header
        
        response_set = requests.post(instrument_url, json=req_body, timeout=10, headers=headers)
        response_set.raise_for_status()

        # 2. Pull the full JSON
        response_get = requests.get(chart_url, timeout=15, headers=headers)
        response_get.raise_for_status()
        data = response_get.json()

        # 3. Convert candles to DataFrame
        if not data.get("candles"):
            session['last_error'] = "Remote API returned no candles."
            return None, None
            
        candles_df = pd.DataFrame(data["candles"])
        candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"])
        candles_df.set_index('timestamp', inplace=True)
        
        markings = data.get("markings", [])
        return candles_df, markings

    except requests.exceptions.RequestException as e:
        session['last_error'] = f"Failed to connect to remote API at {base_url}: {e}"
        return None, None
    except Exception as e:
        session['last_error'] = f"An error occurred while fetching remote data: {e}"
        return None, None

# --- API Endpoints ---
@app.route('/status', methods=['GET'])
def get_status():
    session = get_session()
    return jsonify({
        "status": "running",
        "engine_version": "v6 (Live)",
        "remote_api": SYSTEM_CONFIG['remote_api_base_url'],
        "current_pair": session.get("current_pair"),
        "current_timeframe": session.get("current_timeframe"),
        "data_loaded": session.get("data_loaded"),
        "last_error": session.get("last_error")
    })

@app.route('/set_instrument', methods=['POST'])
def set_instrument():
    """Sets the instrument and fetches live data from the remote source."""
    session = get_session()
    data = request.json
    pair = data.get('pair')
    timeframe = data.get('timeframe')

    if not pair or not timeframe:
        return jsonify({"status": "error", "message": "Missing 'pair' or 'timeframe'."}), 400

    df, markings = fetch_remote_chart(pair, timeframe)

    if df is not None:
        session.update({
            "current_pair": pair,
            "current_timeframe": timeframe,
            "df": df.tail(SYSTEM_CONFIG["max_candles_render"]),
            "markings": markings,
            "data_loaded": True,
            "last_error": None
        })
        return jsonify({"status": "success", "message": f"Live data for {pair}/{timeframe} loaded successfully."})
    else:
        session["data_loaded"] = False
        return jsonify({"status": "error", "message": session['last_error']}), 500

@app.route('/chart', methods=['GET'])
def get_chart_data():
    """Returns the currently loaded OHLC data and the raw markings."""
    session = get_session()
    if not session.get('data_loaded'):
        return jsonify({"status": "error", "message": "No data loaded. Set an instrument first."}), 400

    chart_data = {
        "pair": session['current_pair'],
        "timeframe": session['current_timeframe'],
        "candles": session['df'].reset_index().to_dict(orient='records'),
        "markings": session['markings']
    }
    return jsonify(chart_data)

@app.route('/analysis', methods=['GET'])
def get_analysis():
    """Performs confluence analysis on the loaded data and returns the score."""
    session = get_session()
    if not session.get('data_loaded') or session.get('markings') is None:
        return jsonify({"status": "error", "message": "No analysis data loaded. Set an instrument first."}), 400

    # The 'markings' from your new CLI are already the full analysis object
    # This is a huge simplification!
    full_analysis_from_remote = session['markings']
    
    # We pass this directly to the scorer
    confluence_score, score_details = aggregator.score(full_analysis_from_remote)

    return jsonify({
        "pair": session['current_pair'],
        "timeframe": session['current_timeframe'],
        "confluence_score": confluence_score,
        "score_breakdown": score_details,
        "full_analysis": full_analysis_from_remote
    })

# --- Server Thread ---
class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f" * ncOS v6 Live Engine running on http://{SYSTEM_CONFIG['server_host']}:{SYSTEM_CONFIG['server_port']}")
        print(f" * Connected to remote analysis API: {SYSTEM_CONFIG['remote_api_base_url']}")
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

if __name__ == '__main__':
    server_thread = ServerThread(app, SYSTEM_CONFIG['server_host'], SYSTEM_CONFIG['server_port'])
    try:
        server_thread.start()
    except KeyboardInterrupt:
        print("\n * Shutting down server...")
        server_thread.shutdown()