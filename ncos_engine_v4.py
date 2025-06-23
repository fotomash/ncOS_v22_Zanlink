import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime
import glob

# Import the modules we created
from ncos_v5_unified_strategy import UnifiedStrategy
from ncos_v5_chart_marker import ChartMarker

# --- System Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the API

# Load the main configuration
try:
    with open('ncos_v5_complete_config.json', 'r') as f:
        CONFIG = json.load(f).get('ncos_v5_complete_config', {})
except FileNotFoundError:
    print("FATAL: ncos_v5_complete_config.json not found. Exiting.")
    exit()

# Instantiate our core logic modules
strategy_analyzer = UnifiedStrategy()
chart_marker = ChartMarker()

# --- System State Management ---
# This dictionary holds the live state of the application
SystemState = {
    "selected_pair": "XAUUSD",
    "data_loaded": False,
    "df": None,
    "analysis_results": None,
    "chart_markings": None,
    "signals": [],
    "data_path": CONFIG.get('data_config', {}).get('primary_path', './'),
    "last_error": None
}


# --- Helper Functions ---
def find_and_load_data(pair: str, base_path: str) -> pd.DataFrame | None:
    """Finds the most relevant CSV for a pair and loads it."""
    print(f"Searching for data for {pair} in {base_path}...")

    # Search for files like XAUUSD*.csv, EURUSD*.csv etc.
    search_pattern = os.path.join(base_path, f"{pair}*.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"No CSV files found for {pair} at {search_pattern}")
        SystemState['last_error'] = f"No data file found for {pair}."
        return None

    # Simple logic: use the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    print(f"Found latest file: {latest_file}")

    try:
        # Attempt to load with various common settings
        df = pd.read_csv(latest_file, sep=r'\s*,\s*|\t', engine='python', on_bad_lines='skip')

        # Find timestamp column
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if not time_col:
            SystemState['last_error'] = "Could not find a timestamp column in the data."
            return None

        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]

        print(f"Successfully loaded {len(df)} rows of data for {pair}.")
        SystemState['last_error'] = None
        return df
    except Exception as e:
        print(f"Error loading data from {latest_file}: {e}")
        SystemState['last_error'] = f"Failed to parse data file: {e}"
        return None


# --- API Endpoints ---

@app.route('/api/menu', methods=['GET'])
def get_menu():
    """Endpoint to provide the interactive menu and current system state."""
    menu_config = CONFIG.get('menu_features', {})

    menu_data = {
        "menu": {
            "title": "ncOS v5 Ultimate - Interactive Menu",
            "options": [
                {"key": "1", "label": "Select Pair", "description": "Change the active trading pair."},
                {"key": "2", "label": "Load Data", "description": "Load the latest data for the selected pair."},
                {"key": "3", "label": "Run Full Analysis", "description": "Execute all unified strategies."},
                {"key": "4", "label": "Generate Chart Markings", "description": "Create visual chart analysis."},
                {"key": "5", "label": "View Active Signals",
                 "description": "Show tradeable setups from the last analysis."},
                {"key": "6", "label": "Perform Risk Analysis",
                 "description": "Get entry/exit levels for a top signal."},
                {"key": "7", "label": "Change Data Path", "description": "Update the folder to search for data files."},
                {"key": "8", "label": "System Status", "description": "View the current system state."}
            ],
            "quick_pairs": [
                {"key": str(9 + i), "pair": pair} for i, pair in enumerate(menu_config.get('quick_pairs', []))
            ]
        },
        "current_state": {
            "selected_pair": SystemState['selected_pair'],
            "data_loaded": SystemState['data_loaded'],
            "last_analysis": SystemState['analysis_results']['metadata']['generated_at'] if SystemState.get(
                'analysis_results') else None,
            "active_signals": len(SystemState['signals']),
            "data_path": SystemState['data_path']
        }
    }
    return jsonify(menu_data)


@app.route('/api/action', methods=['POST'])
def execute_action():
    """Main endpoint to handle all user actions from the menu."""
    data = request.json
    action = data.get('action')
    params = data.get('params', {})

    if not action:
        return jsonify({"success": False, "error": "No action specified."}), 400

    result = {"message": "Action completed successfully."}

    # --- Action Logic ---
    if action == 'select_pair':
        pair = params.get('pair')
        if pair:
            SystemState['selected_pair'] = pair.upper()
            SystemState['data_loaded'] = False  # Reset state on pair change
            SystemState['df'] = None
            result['message'] = f"Selected pair set to {SystemState['selected_pair']}. Please load data."
        else:
            return jsonify({"success": False, "error": "No pair provided for selection."}), 400

    elif action == 'load_data':
        df = find_and_load_data(SystemState['selected_pair'], SystemState['data_path'])
        if df is not None:
            SystemState['df'] = df
            SystemState['data_loaded'] = True
            result['message'] = f"Loaded {len(df)} data points for {SystemState['selected_pair']}."
        else:
            return jsonify({"success": False, "error": SystemState['last_error']}), 500

    elif action == 'analyze':
        if not SystemState['data_loaded']:
            return jsonify({"success": False, "error": "Data not loaded. Please load data first."}), 400

        context = {}  # Future use for passing more context
        analysis = strategy_analyzer.analyze_all_strategies(SystemState['df'], context)
        SystemState['analysis_results'] = analysis
        SystemState['signals'] = [s for s in analysis if s.get('tradeable')]
        result['message'] = f"Analysis complete. Found {len(SystemState['signals'])} tradeable signals."
        result['data'] = analysis

    elif action == 'generate_chart':
        if not SystemState['data_loaded']:
            return jsonify({"success": False, "error": "Data not loaded. Please load data first."}), 400

        markings = chart_marker.generate_full_markup(SystemState['df'])
        SystemState['chart_markings'] = markings
        result['message'] = "Chart markings generated."
        result['data'] = markings

    elif action == 'view_signals':
        if not SystemState['analysis_results']:
            return jsonify({"success": False, "error": "No analysis has been run yet."}), 400
        result['message'] = f"Displaying {len(SystemState['signals'])} active signals."
        result['data'] = SystemState['signals']

    elif action == 'risk_analysis':
        if not SystemState['signals']:
            return jsonify({"success": False, "error": "No active signals to analyze."}), 400

        top_signal = SystemState['signals'][0]  # Analyze the highest confidence signal
        levels = strategy_analyzer.get_entry_exit_levels(top_signal['strategy_type'], SystemState['df'])
        result['message'] = f"Risk analysis for top signal: {top_signal['strategy_type']}"
        result['data'] = {
            "signal": top_signal,
            "trade_levels": levels
        }

    else:
        return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

    return jsonify({"success": True, "action": action, "result": result})


@app.route('/api/quick_analysis/<string:pair>', methods=['GET'])
def quick_analysis(pair):
    """Shortcut endpoint for a fast, combined analysis."""
    df = find_and_load_data(pair.upper(), SystemState['data_path'])
    if df is None:
        return jsonify({"error": SystemState['last_error']}), 500

    context = {}
    analysis = strategy_analyzer.analyze_all_strategies(df, context)
    signals = [s for s in analysis if s.get('tradeable')]

    response = {
        "pair": pair.upper(),
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "trend": "detecting...",  # Placeholder
            "momentum": "detecting...",  # Placeholder
            "volatility": df['close'].pct_change().std() * 100,
            "strategies_detected": signals,
            "risk_assessment": {
                "risk_level": "Medium" if signals else "Low",
                "factors": ["Market open", "Recent news"]
            }
        }
    }
    return jsonify(response)


# --- Main Execution ---
if __name__ == '__main__':
    print("╔══════════════════════════════════════╗")
    print("║  ncOS v5 Ultimate - All Intelligence ║")
    print("╚══════════════════════════════════════╝")
    print(f"Starting on port 8000...")
    print(f"Data path: {SystemState['data_path']}")
    # Use host='0.0.0.0' to make it accessible on your network and via ngrok
    app.run(host='0.0.0.0', port=8000, debug=False)