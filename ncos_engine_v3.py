
import os
import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd

# --- IMPORT THE TRADING LOGIC ---
# This is the key step that connects the server to the real logic.
try:
    from ncos_theory_module import get_final_signal_from_data
    LOGIC_IMPORTED = True
except ImportError:
    LOGIC_IMPORTED = False

# --- SETUP ---
load_dotenv()
app = Flask(__name__)
CORS(app)

DATA_FILE = "XAUUSD_TICKS_1days_20250623.csv"

# --- THE ORCHESTRATOR FUNCTION ---
def generate_trading_signal():
    """
    This function orchestrates the process of generating a real signal.
    """
    if not LOGIC_IMPORTED:
        return {"signal": "HOLD", "confidence": 0, "reason": "Server Error: Trading logic module could not be imported."}

    try:
        # 1. Load the data
        df = pd.read_csv(DATA_FILE, sep='\t', engine='python')

        # 2. Pass data to the logic module to get the signal
        signal_result = get_final_signal_from_data(df)

        # 3. Add a timestamp and return
        signal_result["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        return signal_result

    except FileNotFoundError:
        return {"signal": "HOLD", "confidence": 0, "reason": f"Server Error: Data file '{DATA_FILE}' not found."}
    except Exception as e:
        return {"signal": "HOLD", "confidence": 0, "reason": f"Server Error: An unexpected error occurred: {str(e)}"}


# --- API ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def api_status():
    """Provides the current status of the server."""
    return jsonify({
        "status": "ok",
        "message": "ncOS Engine v3 is running. Logic module loaded.",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

@app.route('/api/signal', methods=['GET'])
def api_signal():
    """
    This endpoint now triggers the REAL trading logic instead of a placeholder.
    """
    # Call the orchestrator function to get a live signal
    live_signal = generate_trading_signal()
    return jsonify(live_signal)

# --- SERVER STARTUP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("--- ncOS Full Engine v3 ---")
    print(f"Attempting to load data from: {DATA_FILE}")
    print("Starting server...")
    app.run(host='0.0.0.0', port=port, debug=False) # Debug set to False for cleaner production-like output
