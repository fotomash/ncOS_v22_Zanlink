import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, g
from werkzeug.serving import make_server
import threading
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

# --- Configuration ---
# Suppress startup messages for a cleaner console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- System Configuration ---
SYSTEM_CONFIG = {
    "data_path": "/Volumes/[C] Windows 11/Users/tom/AppData/Roaming/MetaQuotes/Terminal/81A933A9AFC5DE3C23B15CAB19C63850/MQL5/Files/PandasExports",
    "default_pair": "XAUUSD",
    "default_timeframe": "M1",
    "ohlc_timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
    "max_candles_render": 250, # Limit candles for performance
    "server_host": "0.0.0.0",
    "server_port": 5005
}

# --- Session State ---
def get_session():
    """Get the current session state, creating one if it doesn't exist."""
    if 'session' not in g:
        g.session = {
            "current_pair": SYSTEM_CONFIG["default_pair"],
            "current_timeframe": SYSTEM_CONFIG["default_timeframe"],
            "data_loaded": False,
            "df": None,
            "last_error": None
        }
    return g.session

# --- Data Loading and Processing ---
def find_and_load_data(pair: str) -> Optional[pd.DataFrame]:
    """Finds the latest CSV for a pair and loads it."""
    session = get_session()
    data_path = SYSTEM_CONFIG["data_path"]
    try:
        files = [f for f in os.listdir(data_path) if f.lower().startswith(pair.lower()) and f.lower().endswith('.csv')]
        if not files:
            session['last_error'] = f"No CSV data file found for {pair} in {data_path}"
            return None

        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_path, f)))
        df = pd.read_csv(os.path.join(data_path, latest_file))
        
        # --- Data Validation and Formatting ---
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            session['last_error'] = "Data file missing 'timestamp' column."
            return None

        # Handle Tick vs OHLC data
        if 'bid' in df.columns and 'ask' in df.columns:
            # It's tick data, resample to OHLC
            df['price'] = (df['bid'] + df['ask']) / 2
            ohlc = df['price'].resample('1T').ohlc()
            ohlc.columns = ['open', 'high', 'low', 'close']
            return ohlc.dropna()
        elif all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # It's already OHLC data
            return df
        else:
            session['last_error'] = "Data file is not in a recognized OHLC or Tick format."
            return None

    except Exception as e:
        session['last_error'] = f"Error loading data for {pair}: {str(e)}"
        return None

def resample_data(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """Resamples the base M1 dataframe to the target timeframe."""
    if df is None or df.empty:
        return None
    try:
        rule = timeframe.replace('M', 'T') # Convert M1 to 1T, M5 to 5T etc.
        resampled_df = df['close'].resample(rule).last().to_frame()
        resampled_df['open'] = df['open'].resample(rule).first()
        resampled_df['high'] = df['high'].resample(rule).max()
        resampled_df['low'] = df['low'].resample(rule).min()
        return resampled_df[['open', 'high', 'low', 'close']].dropna()
    except Exception as e:
        get_session()['last_error'] = f"Error resampling data to {timeframe}: {str(e)}"
        return None

# --- Chart Marking and Analysis ---
def mark_smc_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identifies basic SMC patterns like Order Blocks and FVGs."""
    markings = []
    if df is None or len(df) < 3:
        return markings

    # Simplified FVG detection
    for i in range(1, len(df) - 1):
        prev_high = df['high'].iloc[i-1]
        next_low = df['low'].iloc[i+1]
        if prev_high < next_low:
            markings.append({
                "type": "FVG_Bullish",
                "from_time": df.index[i-1].isoformat(),
                "to_time": df.index[i+1].isoformat(),
                "price_level_start": prev_high,
                "price_level_end": next_low,
                "description": "Fair Value Gap (Bullish)"
            })

        prev_low = df['low'].iloc[i-1]
        next_high = df['high'].iloc[i+1]
        if prev_low > next_high:
            markings.append({
                "type": "FVG_Bearish",
                "from_time": df.index[i-1].isoformat(),
                "to_time": df.index[i+1].isoformat(),
                "price_level_start": next_high,
                "price_level_end": prev_low,
                "description": "Fair Value Gap (Bearish)"
            })
            
    # Simplified Order Block detection (last down-close before up-move)
    for i in range(1, len(df) - 1):
        if df['close'].iloc[i-1] > df['open'].iloc[i-1] and \
           df['close'].iloc[i] < df['open'].iloc[i] and \
           df['close'].iloc[i+1] > df['open'].iloc[i+1] and \
           df['high'].iloc[i+1] > df['high'].iloc[i]:
            markings.append({
                "type": "OrderBlock_Bullish",
                "time": df.index[i].isoformat(),
                "price_level_start": df['low'].iloc[i],
                "price_level_end": df['high'].iloc[i],
                "description": "Potential Bullish Order Block"
            })

    return markings

# --- API Endpoints ---
@app.route('/status', methods=['GET'])
def get_status():
    """Returns the current status of the ncOS engine."""
    session = get_session()
    return jsonify({
        "status": "running",
        "current_pair": session.get("current_pair"),
        "current_timeframe": session.get("current_timeframe"),
        "data_loaded": session.get("data_loaded"),
        "last_error": session.get("last_error")
    })

@app.route('/set_instrument', methods=['POST'])
def set_instrument():
    """Sets the trading pair and timeframe for analysis."""
    session = get_session()
    data = request.json
    pair = data.get('pair', session['current_pair'])
    timeframe = data.get('timeframe', session['current_timeframe'])

    session['current_pair'] = pair
    session['current_timeframe'] = timeframe
    
    # Load data for the new instrument
    base_df = find_and_load_data(pair)
    if base_df is not None:
        session['df_m1'] = base_df # Store the base M1 data
        session['data_loaded'] = True
        session['last_error'] = None
        return jsonify({"status": "success", "message": f"Instrument set to {pair} on {timeframe}. Data loaded."})
    else:
        session['data_loaded'] = False
        return jsonify({"status": "error", "message": session['last_error']}), 400

@app.route('/chart', methods=['GET'])
def get_chart_data():
    """Returns OHLC data with optional SMC markings."""
    session = get_session()
    if not session.get('data_loaded'):
        return jsonify({"status": "error", "message": "No data loaded. Please set an instrument first."}), 400

    df_m1 = session.get('df_m1')
    target_timeframe = session.get('current_timeframe')
    
    # Resample if necessary
    if target_timeframe == 'M1':
        df_resampled = df_m1
    else:
        df_resampled = resample_data(df_m1, target_timeframe)

    if df_resampled is None or df_resampled.empty:
        return jsonify({"status": "error", "message": session.get('last_error', "Failed to generate chart data.")}), 500

    # Limit the output
    df_limited = df_resampled.tail(SYSTEM_CONFIG['max_candles_render'])
    
    # Generate markings
    smc_markings = mark_smc_patterns(df_limited)

    # Format data for JSON response
    chart_data = {
        "pair": session['current_pair'],
        "timeframe": session['current_timeframe'],
        "candles": df_limited.reset_index().to_dict(orient='records'),
        "markings": smc_markings
    }
    return jsonify(chart_data)

# --- Server Thread ---
class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        threading.Thread.__init__(self)
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        print(f" * ncOS v5 Ultimate Engine running on http://{SYSTEM_CONFIG['server_host']}:{SYSTEM_CONFIG['server_port']}")
        print(" * Use Ctrl+C to stop the server.")
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

def start_server():
    global server_thread
    server_thread = ServerThread(app, SYSTEM_CONFIG['server_host'], SYSTEM_CONFIG['server_port'])
    server_thread.start()

def stop_server():
    global server_thread
    if server_thread:
        server_thread.shutdown()
        server_thread.join()

if __name__ == '__main__':
    try:
        start_server()
    except KeyboardInterrupt:
        print(" * Shutting down server...")
        stop_server()