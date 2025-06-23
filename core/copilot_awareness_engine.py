# copilot_awareness_engine.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   The core awareness module for Zanzibar Copilot Autonomous Mode.
#   It processes incoming data (charts/CSVs), detects the asset,
#   fetches macro context, analyzes indicators, builds a situation report,
#   and triggers the strategy matching engine.

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import traceback
from pathlib import Path
import json

# --- Import necessary sub-modules ---
try: from core.macro_enrichment_engine import fetch_macro_context, detect_asset_class
except ImportError: print("[ERROR][AwarenessEngine] Cannot import macro_enrichment_engine."); fetch_macro_context = None; detect_asset_class = None
try: from core.strategy_match_engine import match_strategy
except ImportError: print("[ERROR][AwarenessEngine] Cannot import strategy_match_engine."); match_strategy = None
# Import indicator and SMC enrichment engines (assuming they are used here for snapshot)
try: from core.indicator_enrichment_engine import calculate_standard_indicators
except ImportError: print("[WARN][AwarenessEngine] Indicator engine not found."); calculate_standard_indicators = None
try: from core.smc_enrichment_engine import tag_smc_zones
except ImportError: print("[WARN][AwarenessEngine] SMC enrichment engine not found."); tag_smc_zones = None

# --- Placeholder for Asset Detection from Chart Data ---
# This would need significant implementation if processing images/chart objects
def detect_asset_from_chart(chart_data: Any) -> str:
    """ Placeholder: Detects asset symbol from chart data (image, object, etc.). """
    print("[WARN][AwarenessEngine] detect_asset_from_chart is a placeholder.")
    # If chart_data is a dataframe, try to get from a known location or filename convention
    if isinstance(chart_data, pd.DataFrame) and hasattr(chart_data, 'name'):
        # Simple guess based on potential name attribute
        return str(chart_data.name).split('_')[0] if chart_data.name else "UNKNOWN_ASSET"
    # Add logic for image analysis or specific chart object parsing if needed
    return "UNKNOWN_ASSET" # Default

def detect_asset_from_csv(csv_path: Path) -> str:
    """ Detects asset symbol from CSV filename convention. """
    # Example conventions: OANDA_EUR_USD_M1..., EURUSD_M1..., BTCUSDT_Ticks...
    filename = csv_path.stem.upper()
    parts = filename.split('_')
    # Look for known patterns
    if len(parts) > 1 and ":" in parts[0]: return parts[0] # e.g., OANDA:EUR_USD
    if len(parts) > 0:
        # Check for common symbols
        common_symbols = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSDT", "ETHUSDT", "SPX500", "NSX100"] # Add more
        if parts[0] in common_symbols: return parts[0]
        # Add more sophisticated detection if needed
    return parts[0] if parts else "UNKNOWN_ASSET"


# --- Placeholder for Indicator Analysis ---
def analyze_indicators_snapshot(enriched_data: Dict[str, pd.DataFrame], tf_list: List[str]) -> Dict:
    """
    Placeholder: Extracts key states/values from enriched indicator data across relevant TFs.
    Needs actual logic to summarize indicator states (e.g., RSI level, MACD cross, EMA alignment).
    """
    print("[INFO][AwarenessEngine] Analyzing indicators snapshot (Placeholder)...")
    snapshot = {}
    # Example: Get latest RSI value from M15
    try:
        if 'm15' in enriched_data and not enriched_data['m15'].empty:
             rsi_col = next((col for col in enriched_data['m15'].columns if 'RSI_' in col), None)
             if rsi_col:
                  snapshot['RSI_14_M15'] = { # Example structure
                       "value": round(enriched_data['m15'][rsi_col].iloc[-1], 1),
                       "is_overbought": enriched_data['m15'][rsi_col].iloc[-1] >= 70,
                       "is_oversold": enriched_data['m15'][rsi_col].iloc[-1] <= 30
                  }
        # Example: Get EMA alignment from H1
        if 'h1' in enriched_data and not enriched_data['h1'].empty:
             ema_fast_col = next((col for col in enriched_data['h1'].columns if 'EMA_' in col and int(col.split('_')[-1]) < 100), None) # Find a fast EMA
             ema_slow_col = next((col for col in enriched_data['h1'].columns if 'EMA_' in col and int(col.split('_')[-1]) >= 100), None) # Find a slow EMA
             if ema_fast_col and ema_slow_col:
                  snapshot['EMA_H1'] = {
                       "fast_above_slow": enriched_data['h1'][ema_fast_col].iloc[-1] > enriched_data['h1'][ema_slow_col].iloc[-1]
                  }
        # Add logic to extract states for BB, ATR, MACD, Stoch, SMC tags etc. across specified TFs
        # This needs to align with the rules defined in strategy_rules.json
    except Exception as e:
         print(f"[ERROR][AwarenessEngine] Error analyzing indicator snapshot: {e}")
         traceback.print_exc()

    print(f"[DEBUG][AwarenessEngine] Indicator Snapshot (Partial): {snapshot}")
    return snapshot

# --- Main Awareness Processing Function ---
def process_incoming_data(data_input: Any, data_type: str = 'unknown') -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Processes incoming data (chart object, CSV path, DataFrame), builds situation report,
    and triggers strategy matching.

    Args:
        data_input: The input data (e.g., Path object for CSV, DataFrame, chart object).
        data_type (str): Type hint ('csv', 'dataframe', 'chart_object').

    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: (situation_report, matched_strategy_result)
                                               Returns (None, None) on critical error.
    """
    print(f"[INFO][AwarenessEngine] Processing incoming data (type: {data_type})...")
    asset_symbol = "UNKNOWN"
    enriched_tf_data = None # This should hold the multi-TF dataframes AFTER enrichment

    # --- Step 1: Detect Asset & Load/Prepare Data ---
    try:
        if data_type == 'csv' and isinstance(data_input, Path):
            asset_symbol = detect_asset_from_csv(data_input)
            print(f"[INFO][AwarenessEngine] Detected asset from CSV: {asset_symbol}")
            # TODO: Implement loading & resampling CSV data here
            # This might call parts of the DataPipeline or a dedicated CSV processor
            # For now, assume it results in 'enriched_tf_data' dictionary
            print("[WARN][AwarenessEngine] CSV data loading/processing not implemented yet.")
            enriched_tf_data = {} # Placeholder
        elif data_type == 'dataframe' and isinstance(data_input, pd.DataFrame):
            # Assume single TF dataframe is provided, need to detect asset & potentially aggregate/enrich
            asset_symbol = detect_asset_from_chart(data_input) # Use placeholder chart detection
            print(f"[INFO][AwarenessEngine] Detected asset from DataFrame: {asset_symbol}")
            # TODO: Need logic to handle single TF df - maybe run enrichment on it?
            # Or assume it's already enriched M1 data needing aggregation?
            print("[WARN][AwarenessEngine] DataFrame input handling needs refinement.")
            # For testing, let's assume it's enriched M15 data
            enriched_tf_data = {'m15': data_input} if 'm15' not in enriched_tf_data else enriched_tf_data # Simple assignment
        elif data_type == 'chart_object': # Requires specific logic based on chart object type
             asset_symbol = detect_asset_from_chart(data_input)
             print(f"[INFO][AwarenessEngine] Detected asset from chart object: {asset_symbol}")
             print("[WARN][AwarenessEngine] Chart object processing not implemented yet.")
             enriched_tf_data = {} # Placeholder
        else:
            print(f"[ERROR][AwarenessEngine] Unknown data input type: {data_type}")
            return None, None

        if asset_symbol == "UNKNOWN" or not enriched_tf_data:
             print("[ERROR][AwarenessEngine] Could not determine asset or process data.")
             return None, None

    except Exception as e:
        print(f"[ERROR][AwarenessEngine] Failed during asset detection or data prep: {e}")
        traceback.print_exc()
        return None, None

    # --- Step 2: Fetch Macro Context ---
    macro_context = {}
    if fetch_macro_context:
        try:
            macro_context = fetch_macro_context(asset_symbol)
        except Exception as e:
            print(f"[ERROR][AwarenessEngine] Failed fetching macro context: {e}")
            # Continue without macro context? Or return error? Let's continue with warning.
            macro_context = {"error": str(e), "risk_state": "Neutral", "macro_data": {}}
    else:
        print("[WARN][AwarenessEngine] Macro enrichment engine unavailable.")


    # --- Step 3: Analyze Indicators Snapshot ---
    # This step assumes enrichment happened *before* this engine was called,
    # OR we need to run enrichment engines here on the loaded data.
    # Let's assume 'enriched_tf_data' holds the data after enrichment pipeline.
    indicators_snapshot = analyze_indicators_snapshot(enriched_tf_data, list(enriched_tf_data.keys()))


    # --- Step 4: Build Full Situation Awareness Report ---
    # This report structure needs to align with what strategy_match_engine expects
    situation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": asset_symbol,
        "macro_context": macro_context,
        "indicators": indicators_snapshot, # Simplified structure for now
        # Add SMC tags summary if available from enrichment
        # "smc_tags": extract_smc_summary(enriched_tf_data),
        "enriched_tf_data": enriched_tf_data # Pass the actual data for potential use in matching/analysis
    }
    print(f"[INFO][AwarenessEngine] Situation Report Compiled for {asset_symbol}.")
    # print(f"[DEBUG][AwarenessEngine] Report Snippet: {json.dumps(situation_report, default=str, indent=2)}") # Optional full debug


    # --- Step 5: Pass to Strategy Matching Engine ---
    matched_strategy_result = {"strategy": None, "confidence": 0.0}
    if match_strategy:
        try:
            # Load confidence threshold from config
            copilot_conf_path = Path("config/copilot_config.json")
            confidence_threshold = 0.75 # Default
            if copilot_conf_path.is_file():
                 with open(copilot_conf_path, 'r') as f: conf_data = json.load(f)
                 confidence_threshold = conf_data.get("confidence_threshold", 0.75)

            matched_strategy_result = match_strategy(situation_report, confidence_threshold)
        except Exception as e:
            print(f"[ERROR][AwarenessEngine] Strategy matching failed: {e}")
            traceback.print_exc()
            matched_strategy_result["error"] = str(e)
    else:
        print("[WARN][AwarenessEngine] Strategy match engine unavailable.")


    return situation_report, matched_strategy_result

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Copilot Awareness Engine ---")

    # Create a dummy DataFrame as input (e.g., simulating enriched M15 data)
    index = pd.date_range(start='2024-04-28 12:00', periods=50, freq='15T', tz='UTC')
    dummy_m15_df = pd.DataFrame({
        'Open': np.random.rand(50) + 1.1,
        'High': np.random.rand(50)/2 + 1.105,
        'Low': np.random.rand(50)/2 + 1.095,
        'Close': np.random.rand(50) + 1.1,
        'Volume': np.random.randint(100,1000,50),
        'RSI_14': np.linspace(30, 70, 50), # Simulate RSI for snapshot test
        'EMA_48': np.linspace(1.098, 1.102, 50)
    }, index=index)
    dummy_m15_df.name = "OANDA:EUR_USD_M15" # Add name for asset detection

    # Simulate the engine receiving this DataFrame
    situation, match = process_incoming_data(dummy_m15_df, data_type='dataframe')

    print("\n--- Awareness Engine Output ---")
    if situation:
        print("Situation Report (Partial):")
        print(f"  Asset: {situation.get('asset')}")
        print(f"  Macro: {situation.get('macro_context')}")
        print(f"  Indicators: {situation.get('indicators')}")
    else:
        print("Situation Report: None (Error during processing)")

    if match:
        print("\nStrategy Match Result:")
        print(json.dumps(match, indent=2))
    else:
        print("\nStrategy Match Result: None (Error or no match)")

    print("\n--- Test Complete ---")
