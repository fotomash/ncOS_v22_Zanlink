# autonomous_chart_reporting.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   Contains engines for automatically generating charts and reports
#   based on the results of the autonomous top-down analysis.

import pandas as pd
from typing import Dict, Optional, Any
import traceback
import json # For potential report saving

# --- Import Dependencies ---
# Charting function from main orchestrator
try:
    from copilot_orchestrator import generate_analysis_chart_json, load_strategy_profile
    CHARTING_AVAILABLE = True
except ImportError:
    print("[ERROR][AutoChartReport] Cannot import generate_analysis_chart_json/load_strategy_profile from copilot_orchestrator.")
    generate_analysis_chart_json = None
    load_strategy_profile = None
    CHARTING_AVAILABLE = False

# Journaling function (assuming it's now in advanced_smc_orchestrator)
try:
    from core.advanced_smc_orchestrator import log_trade_summary # Use the enhanced logger
    JOURNALING_AVAILABLE = True
except ImportError:
    print("[ERROR][AutoChartReport] Cannot import log_trade_summary from advanced_smc_orchestrator.")
    log_trade_summary = None
    JOURNALING_AVAILABLE = False


# --- Autonomous Chart Generator ---
def auto_generate_chart(
    asset: str,
    analysis_result: Dict,
    enriched_tf_data: Dict[str, pd.DataFrame],
    chart_tf: str = "M15" # Default TF to chart
    ):
    """
    Automatically generates and potentially saves/displays an annotated chart.

    Args:
        asset (str): The asset symbol.
        analysis_result (Dict): The full output from the analysis engine
                                (e.g., run_advanced_smc_strategy).
        enriched_tf_data (Dict): Dictionary of enriched DataFrames by timeframe.
        chart_tf (str): The primary timeframe to display on the chart (e.g., "M15", "H1").
    """
    if not CHARTING_AVAILABLE or not generate_analysis_chart_json:
        print("[WARN][AutoChart] Charting function unavailable. Skipping auto-chart generation.")
        return

    print(f"🖼️ [AutoChart] Copilot: Auto-generating annotated chart for {asset} ({chart_tf})...")

    try:
        # Extract necessary components for the charting function
        strategy_variant = analysis_result.get("variant") # Get variant used
        price_df = enriched_tf_data.get(chart_tf.lower()) # Get data for the target TF

        if price_df is None or price_df.empty:
            print(f"[WARN][AutoChart] No data found for chart timeframe {chart_tf}. Trying H1 fallback.")
            chart_tf = "H1" # Fallback TF
            price_df = enriched_tf_data.get(chart_tf.lower())
            if price_df is None or price_df.empty:
                 print(f"[ERROR][AutoChart] No data found for fallback chart timeframe {chart_tf}. Cannot generate chart.")
                 return

        # Load chart options from strategy profile
        profile = load_strategy_profile(strategy_variant) if load_strategy_profile and strategy_variant else {}
        chart_options = profile.get("chart_options", {})

        # Call the main charting function (ensure all needed keys are passed)
        chart_json_str = generate_analysis_chart_json(
            price_df=price_df.tail(200), # Limit candles for performance
            chart_tf=chart_tf.upper(),
            pair=asset,
            target_time=analysis_result.get("target_timestamp", ""), # Use timestamp from result
            structure_data=analysis_result.get("analysis", {}).get("smc_structure"),
            inducement_result=analysis_result.get("analysis", {}).get("liquidity_result"), # Adjust key if needed
            poi_tap_result=analysis_result.get("poi_tap_result"), # Assuming this key exists
            phase_result=analysis_result.get("analysis", {}).get("phase_result"),
            confirmation_result=analysis_result.get("confirmation_result"), # Assuming this key exists
            entry_result=analysis_result.get("final_entry_result"), # Use the combined entry result
            variant_name=strategy_variant,
            chart_options=chart_options
        )

        if chart_json_str:
            print(f"[INFO][AutoChart] Chart JSON generated successfully for {asset}.")
            # TODO: Implement saving chart JSON to file or displaying it
            # Example: save_chart_json(chart_json_str, asset, strategy_variant)
            # Example: display_chart_from_json(chart_json_str)
        else:
            print(f"[WARN][AutoChart] Chart JSON generation failed for {asset}.")

    except Exception as e:
        print(f"[ERROR][AutoChart] Failed during auto-chart generation for {asset}: {e}")
        traceback.print_exc()


# --- Autonomous Reporting Engine ---
def auto_generate_report(
    asset: str,
    analysis_result: Dict,
    matched_strategy_result: Dict
    ):
    """
    Automatically logs a summary report of the detected setup to the trade journal.
    Uses the enhanced log_trade_summary function.

    Args:
        asset (str): The asset symbol.
        analysis_result (Dict): The full output from the analysis engine.
        matched_strategy_result (Dict): Output from the strategy matcher.
    """
    if not JOURNALING_AVAILABLE or not log_trade_summary:
        print("[WARN][AutoReport] Journaling function unavailable. Skipping auto-report.")
        return

    print(f"📜 [AutoReport] Copilot: Auto-logging strategy summary for {asset}...")

    try:
        # Prepare the dictionary for log_trade_summary
        # Extract details from analysis_result and matched_strategy_result
        entry_details = analysis_result.get("final_entry_result", {}) # This now contains SL/Risk info
        strategy_name = matched_strategy_result.get("strategy", "Unknown")
        confidence = matched_strategy_result.get("confidence", 0.0)

        # Build the dictionary matching the expected keys for log_trade_summary
        report_data = {
            "timestamp": entry_details.get("entry_time") or analysis_result.get("target_timestamp", datetime.now(timezone.utc).isoformat()),
            "pair": asset,
            "strategy_variant": strategy_name,
            "trade_type": entry_details.get("direction"),
            "conviction_score": entry_details.get("conviction_score"), # From SL/Risk engine run
            "entry_price": entry_details.get("entry_price"),
            "structural_sl": entry_details.get("sl_details", {}).get("structural_sl_raw"),
            "atr_sl": entry_details.get("sl_details", {}).get("atr_sl_raw"),
            "final_sl": entry_details.get("final_sl_price"),
            "tp1": entry_details.get("tp1"), # Assuming TP is set in entry_details
            "tp2": entry_details.get("tp2"),
            "risk_percent": entry_details.get("risk_percent"),
            "lot_size": entry_details.get("final_lot_size"),
            "risk_amount_usd": entry_details.get("risk_amount_usd"),
            "sl_distance_pips": entry_details.get("sl_details", {}).get("sl_distance_pips"),
            "pip_point_value": entry_details.get("sl_details", {}).get("pip_point_value"),
            "sl_choice_reason": entry_details.get("sl_details", {}).get("sl_choice_reason"),
            "status": "DETECTED" if not entry_details.get("entry_confirmed") else "CONFIRMED_ENTRY", # Indicate if it's just a detected setup or a confirmed entry
            "comments": f"Auto-detected setup. Confidence: {confidence:.2f}. Bias: {analysis_result.get('analysis',{}).get('smc_structure',{}).get('htf_bias')}",
            "error": analysis_result.get("error") or entry_details.get("error")
        }

        # Call the enhanced logger
        log_trade_summary(report_data)

    except Exception as e:
        print(f"[ERROR][AutoReport] Failed during auto-report generation for {asset}: {e}")
        traceback.print_exc()


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Autonomous Charting & Reporting Engines ---")

    # Create dummy inputs
    dummy_asset = "OANDA:EUR_USD"
    dummy_match = {"strategy": "Trend Continuation Long", "confidence": 0.85}
    dummy_analysis = {
        "variant": "Trend Continuation Long",
        "symbol": dummy_asset,
        "target_timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": [{"step": "Mock Analysis", "status": "Success"}],
        "analysis": {
            "smc_structure": {"htf_bias": "Bullish"},
            "phase_result": {"phase": "Impulse"}
        },
        "final_entry_result": { # Simulate a confirmed entry with SL/Risk data
            "entry_confirmed": True,
            "entry_price": 1.1050,
            "direction": "buy",
            "entry_candle_timestamp": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            "tp1": 1.1080,
            "final_sl_price": 1.1035,
            "final_lot_size": 0.55,
            "risk_percent": 0.75,
            "risk_amount_usd": 75.0,
            "conviction_score": 4,
            "sl_details": {
                "structural_sl_raw": 1.1038,
                "atr_sl_raw": 1.1035,
                "final_sl": 1.1035,
                "sl_distance_pips": 15.0,
                "pip_point_value": 10.0,
                "sl_choice_reason": "Wider (Min of Struct=1.10380, ATR=1.10350)"
            },
            "symbol": dummy_asset,
            "variant": "Trend Continuation Long",
            "error": None
        },
        "error": None,
        "status": "completed",
         "chart_json": "{}" # Assume chart was generated
    }
    dummy_enriched = {
        "m15": pd.DataFrame({'Close': [1.1, 1.1050]}, index=pd.date_range(end=datetime.now(timezone.utc), periods=2, freq='15T'))
    }

    print("\nTesting Auto Chart Generation...")
    auto_generate_chart(dummy_asset, dummy_analysis, dummy_enriched, chart_tf="M15")

    print("\nTesting Auto Report Generation...")
    auto_generate_report(dummy_asset, dummy_analysis, dummy_match)

    print("\n--- Check log file: journal/trade_log.csv for auto-report ---")
    print("\n--- Test Complete ---")
