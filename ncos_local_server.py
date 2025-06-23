
import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the GPT to connect

# --- API Endpoints ---

@app.route('/')
def index():
    """A simple base route to confirm the server is running."""
    return "<h1>ncOS Local Server is running</h1><p>Use the /api/status or /api/signal endpoints.</p>"

@app.route('/api/status', methods=['GET'])
def api_status():
    """
    This is the endpoint for the 'getStatus' action.
    It provides the current status of the server, matching the openapi_v2.yaml schema.
    """
    status_data = {
        "status": "ok",
        "message": "ncOS server is running and the API endpoint is connected.",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return jsonify(status_data)

@app.route('/api/signal', methods=['GET'])
def api_signal():
    """
    This is the endpoint for the 'getSignal' action.
    It provides a placeholder trading signal, matching the openapi_v2.yaml schema.
    Later, we will replace the placeholder data with real logic.
    """
    signal_data = {
        "signal": "HOLD",
        "confidence": 0.78,
        "reason": "Placeholder Signal: Market is consolidating. Awaiting clearer pattern from the engine.",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return jsonify(signal_data)

# --- Server Startup ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print("--- ncOS Local Server v2 ---")
    print(f"Starting server on port {port}...")
    print("API Endpoints are now active:")
    print("-> http://127.0.0.1:{}/api/status".format(port))
    print("-> http://127.0.0.1:{}/api/signal".format(port))
    print("----------------------------")
    # The 'debug=True' allows the server to auto-reload when you save changes to this file.
    app.run(host='0.0.0.0', port=port, debug=True)
