import pandas as pd
from typing import Dict, Optional, List
import json
from pathlib import Path
import traceback # For detailed error logging
import numpy as np  # Needed for example usage random operations

# Version: 5.1.9 (Fully Prop Desk Compliant)

# --- Import necessary analysis modules ---
# Core analysis modules
try: from market_structure_analyzer_smc import analyze_market_structure
except ImportError: analyze_market_structure = None; print("[WARN][AdvSMCOrch] market_structure_analyzer_smc not found.")
try: from liquidity_engine_smc import detect_inducement_from_structure # Or other relevant liquidity functions
except ImportError: detect_inducement_from_structure = None; print("[WARN][AdvSMCOrch] liquidity_engine_smc not found.")
try: from poi_manager_smc import find_and_validate_smc_pois # Or function to get relevant POIs
except ImportError: find_and_validate_smc_pois = None; print("[WARN][AdvSMCOrch] poi_manager_smc not found.")
try: from confluence_engine import ConfluenceEngine # If needed for specific variants
except ImportError: ConfluenceEngine = None; print("[WARN][AdvSMCOrch] confluence_engine not found.")

# --- Import the newly drafted modules ---
try: from impulse_correction_detector import detect_impulse_correction_phase
except ImportError: detect_impulse_correction_phase = None; print("[ERROR][AdvSMCOrch] impulse_correction_detector.py not found!")
try: from confirmation_engine_smc import confirm_smc_entry
except ImportError: confirm_smc_entry = None; print("[ERROR][AdvSMCOrch] confirmation_engine_smc.py not found!")
try: from entry_executor_smc import execute_smc_entry
except ImportError: execute_smc_entry = None; print("[ERROR][AdvSMCOrch] entry_executor_smc.py not found!")

# --- Import the POI Tap Checker --- ### MODIFICATION START ###
try: from poi_hit_watcher_smc import check_poi_tap_smc
except ImportError: check_poi_tap_smc = None; print("[ERROR][AdvSMCOrch] poi_hit_watcher_smc.py not found!")
# --- MODIFICATION END --- ###

# --- Import Liquidity VWAP Detector ---
try: from liquidity_vwap_detector import LiquidityVWAPDetector
except ImportError: LiquidityVWAPDetector = None; print("[WARN][AdvSMCOrch] liquidity_vwap_detector not found.")

import core
import pkgutil
import importlib

# Dynamically import every module in core/

for finder, module_name, is_pkg in pkgutil.iter_modules(core.__path__):
    try:
        importlib.import_module(f"core.{module_name}")
        print(f"[INFO][AdvSMCOrch] Imported core.{module_name}")
    except Exception as e:
        print(f"[WARN][AdvSMCOrch] Failed to import core.{module_name}: {e}")

#––– Wyckoff Top-Down Analysis Setup –––
try:
    from core.event_detector import find_initial_wyckoff_events
    from core.state_machine   import WyckoffStateMachine
    from core.phase_detector_wyckoff_v1 import detect_wyckoff_multi_tf
    print("[INFO][AdvSMCOrch] Loaded Wyckoff event detector, state machine, and multi-TF Wyckoff detector")
except ImportError as e:
    print(f"[WARN][AdvSMCOrch] Wyckoff detector/state machine or multi-TF detector not found: {e}")

# --- Signal dispatching ---
def dispatch_pine_payload(entry_result: Dict):
    """ Sends the trade signal (actual implementation expected in prod). """
    if entry_result and entry_result.get("pine_payload"):
        payload = entry_result["pine_payload"]
        print(f"[INFO][AdvSMCOrch] === DISPATCHING TRADE SIGNAL ===")
        print(json.dumps(payload, indent=2))
        # TODO: Implement actual dispatch logic
        print(f"[INFO][AdvSMCOrch] === SIGNAL DISPATCH COMPLETE ===")
    else:
        print("[WARN][AdvSMCOrch] No valid Pine payload found in entry result to dispatch.")

# --- Function to load strategy profiles ---
def load_strategy_profile(variant_name: str) -> Dict:
    """ Loads specific configuration for a strategy variant from strategy_profiles.json. Fallbacks to default if not found. """
    config_path = Path("strategy_profiles.json")
    if not config_path.exists():
        print(f"[ERROR][AdvSMCOrch] strategy_profiles.json not found at {config_path.resolve()}.")
        return {}
    try:
        with open(config_path, "r") as f:
            all_profiles = json.load(f)
        profile = None
        if variant_name in all_profiles:
            profile = all_profiles[variant_name]
            print(f"[INFO][AdvSMCOrch] Loaded strategy profile for variant: {variant_name}")
        elif "default" in all_profiles:
            print(f"[WARN][AdvSMCOrch] Strategy variant '{variant_name}' not found in profiles. Falling back to 'default'.")
            profile = all_profiles["default"]
        else:
            print(f"[WARN][AdvSMCOrch] Strategy variant '{variant_name}' not found and no 'default' profile present. Using empty config.")
            profile = {}
        return profile
    except Exception as e:
        print(f"[ERROR][AdvSMCOrch] Failed to load or parse strategy_profiles.json: {e}")
        return {}

# --- Main Orchestration Function ---
def run_advanced_smc_strategy(
    all_tf_data: Dict[str, pd.DataFrame],
    strategy_variant: str,
    target_timestamp: pd.Timestamp, # Note: This timestamp might be less relevant now, focus is on latest data + POI taps
    symbol: str = "XAUUSD"
) -> Dict:
    """
    Orchestrates the Advanced SMC strategy flow using specialized modules,
    incorporating POI tap checks and data slicing.
    """
    print(f"\n--- Running Advanced SMC Orchestrator for Variant: {strategy_variant} ---")
    orchestration_result = {
        "variant": strategy_variant,
        "symbol": symbol,
        "target_timestamp": target_timestamp.strftime('%Y-%m-%d %H:%M:%S'), # Keep for reference
        "steps": [],
        "final_entry_result": None,
        "error": None
    }

    # --- 1. Load Strategy Variant Configuration ---
    variant_config = load_strategy_profile(strategy_variant)
    phase_config = variant_config.get("phase_detection_config", {})
    confirmation_config = variant_config.get("confirmation_config", {})
    execution_config = variant_config.get("execution_config", {})
    risk_config = variant_config.get("risk_model_config", {})
    poi_tap_config = variant_config.get("poi_tap_config", {}) # Config for POI tap checker
    htf_structure_tf = variant_config.get("structure_timeframe", "h4")
    poi_tf = variant_config.get("poi_timeframe", "h1") # TF where POIs are identified
    poi_tap_check_tf = variant_config.get("poi_tap_check_timeframe", "m15") # TF to check for POI tap
    confirmation_tf = variant_config.get("confirmation_timeframe", "m5")
    execution_tf = variant_config.get("execution_timeframe", "m1")

    required_tfs = {htf_structure_tf, poi_tf, poi_tap_check_tf, confirmation_tf, execution_tf}
    for tf in required_tfs:
        if tf not in all_tf_data or all_tf_data[tf].empty:
            error_msg = f"Missing or empty data for required timeframe: {tf}"
            print(f"[ERROR][AdvSMCOrch] {error_msg}")
            orchestration_result["error"] = error_msg
            return orchestration_result

    # --- 2. Perform Initial HTF Analysis (Structure, POI) ---
    print(f"[INFO][AdvSMCOrch] Performing HTF Analysis (Structure TF: {htf_structure_tf}, POI TF: {poi_tf})...")
    structure_data = None
    if analyze_market_structure:
        try:
            structure_data = analyze_market_structure(all_tf_data[htf_structure_tf])
            orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Success", "bias": structure_data.get('htf_bias')})
        except Exception as e:
             print(f"[ERROR][AdvSMCOrch] Structure Analysis failed: {e}")
             orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Failed", "error": str(e)})
    else:
         orchestration_result["steps"].append({"step": "HTF Structure Analysis", "status": "Skipped", "reason": "Module not available"})

    # --- Get Relevant HTF POIs ---
    relevant_htf_pois = []
    if find_and_validate_smc_pois and structure_data:
         try:
             # *** Stub POI for testing flow ***
             htf_poi_type = structure_data.get('htf_bias', 'Unknown') if structure_data else 'Unknown'
             # Make stub POI relative to latest price for better testing chance
             latest_price = all_tf_data[poi_tap_check_tf]['Close'].iloc[-1]
             if htf_poi_type == 'Bullish':
                  poi_low = latest_price * 0.999
                  poi_high = latest_price * 0.9995
                  relevant_htf_pois.append({'range': [poi_low, poi_high], 'type': 'Bullish', 'timestamp': pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=4), 'source_tf': poi_tf})
             elif htf_poi_type == 'Bearish':
                  poi_low = latest_price * 1.0005
                  poi_high = latest_price * 1.001
                  relevant_htf_pois.append({'range': [poi_low, poi_high], 'type': 'Bearish', 'timestamp': pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=4), 'source_tf': poi_tf})
             print(f"[INFO][AdvSMCOrch] Found {len(relevant_htf_pois)} relevant HTF POI(s) (Stubbed near {latest_price:.5f}).")
             orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Success (Stubbed)", "count": len(relevant_htf_pois)})
         except Exception as e:
             print(f"[ERROR][AdvSMCOrch] POI Identification failed: {e}")
             orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Failed", "error": str(e)})
    else:
        orchestration_result["steps"].append({"step": "HTF POI Identification", "status": "Skipped", "reason": "Module or structure data not available"})

    # --- 3. Check Price Interaction with POI(s) ---
    entry_found = False
    if not relevant_htf_pois:
        print(f"[INFO][AdvSMCOrch] No POIs tapped or matched.")
        orchestration_result["steps"].append({"step": "POI Tap", "status": "No POIs tapped or matched"})

    for htf_poi in relevant_htf_pois:
        if entry_found: break

        print(f"\n[INFO][AdvSMCOrch] Evaluating HTF POI: Type={htf_poi['type']}, Range={htf_poi['range']}")
        orchestration_result["steps"].append({"step": "POI Evaluation Start", "poi_range": htf_poi['range'], "poi_type": htf_poi['type']})

        # --- Call POI Tap Checker ---
        poi_tap_result = None
        tap_time = None
        if check_poi_tap_smc:
            print(f"[INFO][AdvSMCOrch] Checking POI tap (Check TF: {poi_tap_check_tf})...")
            try:
                poi_tap_result = check_poi_tap_smc(
                    price_data=all_tf_data[poi_tap_check_tf], # Use specific TF for tap check
                    poi_range=htf_poi['range'],
                    config=poi_tap_config # Pass tap-specific config
                )
                orchestration_result["steps"].append({"step": "POI Tap Check", "result": poi_tap_result})
                if not poi_tap_result or not poi_tap_result.get("is_tapped"):
                    print("[INFO][AdvSMCOrch] POI was not tapped. Skipping.")
                    continue # Move to next POI if not tapped
                else:
                    tap_time = poi_tap_result.get("tap_time") # Store the tap time
                    print(f"[INFO][AdvSMCOrch] POI Tap CONFIRMED at {tap_time}.")
                    # --- Bias Injection Logging (future module connection) ---
                    bias_injection_applicable = variant_config.get("enable_bias_injection", False)
                    if bias_injection_applicable:
                        print("[INFO][AdvSMCOrch] Bias injection module is enabled and will be applied if connected.")
                    else:
                        print("[INFO][AdvSMCOrch] Bias injection not applicable for this variant or not enabled.")
            except Exception as e:
                print(f"[ERROR][AdvSMCOrch] POI Tap Check failed: {e}")
                orchestration_result["steps"].append({"step": "POI Tap Check", "status": "Failed", "error": str(e)})
                continue # Move to next POI if check fails
        else:
            print("[WARN][AdvSMCOrch] POI Tap Check skipped: Module not available.")
            orchestration_result["steps"].append({"step": "POI Tap Check", "status": "Skipped", "reason": "Module not available"})
            continue


        # --- VWAP Sweep Deviation Check ---
        if LiquidityVWAPDetector:
            try:
                vwap_detector = LiquidityVWAPDetector(window=20, deviation_threshold=1.2)
                sweep_df = vwap_detector.detect_sweeps(all_tf_data[confirmation_tf])
                latest_sweep = sweep_df.iloc[-1]["VWAP_Sweep"]
                orchestration_result["steps"].append({"step": "VWAP Sweep Check", "latest": latest_sweep})
                print(f"[INFO][AdvSMCOrch] VWAP sweep detected: {latest_sweep}")
            except Exception as e:
                print(f"[ERROR][AdvSMCOrch] VWAP sweep detection failed: {e}")
                orchestration_result["steps"].append({"step": "VWAP Sweep Check", "status": "Failed", "error": str(e)})
        else:
            orchestration_result["steps"].append({"step": "VWAP Sweep Check", "status": "Skipped", "reason": "Module not available"})

        # --- 4. Check Impulse/Correction Phase (if required) ---
        phase_check_passed = True
        if detect_impulse_correction_phase:
            require_phase = variant_config.get("require_phase", None)
            if require_phase:
                print(f"[INFO][AdvSMCOrch] Checking market phase (Confirmation TF: {confirmation_tf}). Required: {require_phase}")
                try:
                    # Check phase on data *up to the tap time*? Or slightly after? Let's use up to tap time.
                    phase_check_data = all_tf_data[confirmation_tf][all_tf_data[confirmation_tf].index <= tap_time]
                    if phase_check_data.empty: raise ValueError("No data available up to tap time for phase check.")

                    phase_result = detect_impulse_correction_phase(
                        price_data=phase_check_data,
                        structure_data=structure_data,
                        config=phase_config
                    )
                    orchestration_result["steps"].append({"step": "Phase Detection", "result": phase_result})
                    current_phase = phase_result.get("phase", "Unknown")
                    # Check if phase matches requirement OR if valid_for_entry is true? Let's use require_phase match.
                    if current_phase == require_phase: # and phase_result.get("valid_for_entry", False):
                        phase_check_passed = True
                        print(f"[INFO][AdvSMCOrch] Phase check passed. Current phase: {current_phase}")
                    else:
                        phase_check_passed = False
                        print(f"[INFO][AdvSMCOrch] Phase check failed. Current phase: {current_phase}, Required: {require_phase}. Skipping POI.")
                except Exception as e:
                    phase_check_passed = False
                    print(f"[ERROR][AdvSMCOrch] Phase Detection failed: {e}")
                    orchestration_result["steps"].append({"step": "Phase Detection", "status": "Failed", "error": str(e)})
            else:
                 orchestration_result["steps"].append({"step": "Phase Detection", "status": "Skipped", "reason": "Not required by variant"})
        else:
             orchestration_result["steps"].append({"step": "Phase Detection", "status": "Skipped", "reason": "Module not available"})

        if not phase_check_passed:
            continue


        # --- 5. Run Confirmation Engine (using sliced data) --- ### MODIFICATION START ###
        confirmation_result = None
        ltf_poi_timestamp = None # Timestamp when the LTF POI for entry was formed
        if confirm_smc_entry and tap_time:
             print(f"[INFO][AdvSMCOrch] Running Confirmation Engine (Conf TF: {confirmation_tf}, Data After: {tap_time})...")
             try:
                 # Slice confirmation data: Only candles STRICTLY AFTER the tap time
                 confirmation_data_slice = all_tf_data[confirmation_tf][all_tf_data[confirmation_tf].index > tap_time]

                 if confirmation_data_slice.empty or len(confirmation_data_slice) < confirmation_config.get('confirmation_lookback', 30): # Check if enough data after tap
                      print("[WARN][AdvSMCOrch] Not enough data after POI tap for confirmation lookback. Skipping.")
                      orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Skipped", "reason": "Insufficient data after tap"})
                      continue
                 else:
                      print(f"[DEBUG][AdvSMCOrch] Confirmation using {len(confirmation_data_slice)} candles from {confirmation_data_slice.index.min()} onwards.")
                      confirmation_result = confirm_smc_entry(
                         htf_poi=htf_poi,
                         ltf_data=confirmation_data_slice, # Pass the SLICED data
                         strategy_variant=strategy_variant,
                         config=confirmation_config
                         # Pass confluence/liquidity data if available and needed
                      )
                      orchestration_result["steps"].append({"step": "Confirmation Engine", "result": confirmation_result})

                      if not confirmation_result or not confirmation_result.get("confirmation_status"):
                         print("[INFO][AdvSMCOrch] Confirmation failed or pattern not found.")
                         continue
                      else:
                         print("[INFO][AdvSMCOrch] Confirmation SUCCESS.")
                         # Store the timestamp of the identified LTF POI for the next slice
                         if confirmation_result.get("ltf_poi_timestamp"):
                              try:
                                   ltf_poi_timestamp = pd.Timestamp(confirmation_result["ltf_poi_timestamp"], tz='UTC')
                              except Exception as ts_parse_err:
                                   print(f"[WARN][AdvSMCOrch] Could not parse LTF POI timestamp: {ts_parse_err}")
                                   ltf_poi_timestamp = None # Ensure it's None if parsing fails


             except Exception as e:
                 print(f"[ERROR][AdvSMCOrch] Confirmation Engine failed: {e}\n{traceback.format_exc()}")
                 orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Failed", "error": str(e)})
                 continue
        else:
             orchestration_result["steps"].append({"step": "Confirmation Engine", "status": "Skipped", "reason": "Module not available or POI not tapped"})
             continue
        # --- MODIFICATION END --- ###


        # --- 6. Run Entry Executor (using sliced data) --- ### MODIFICATION START ###
        final_entry_result = None
        if execute_smc_entry and confirmation_result and confirmation_result.get("confirmation_status") and ltf_poi_timestamp:
             print(f"[INFO][AdvSMCOrch] Running Entry Executor (Exec TF: {execution_tf}, Data After: {ltf_poi_timestamp})...")
             try:
                 # Slice execution data: Only candles STRICTLY AFTER the LTF POI formation time
                 execution_data_slice = all_tf_data[execution_tf][all_tf_data[execution_tf].index > ltf_poi_timestamp]

                 if execution_data_slice.empty:
                     print("[WARN][AdvSMCOrch] No execution timeframe data available after LTF POI timestamp. Skipping entry.")
                     orchestration_result["steps"].append({"step": "Entry Execution", "status": "Skipped", "reason": "No data after LTF POI formation"})
                     continue
                 else:
                      print(f"[DEBUG][AdvSMCOrch] Execution using {len(execution_data_slice)} candles from {execution_data_slice.index.min()} onwards.")
                      final_entry_result = execute_smc_entry(
                         ltf_data=execution_data_slice, # Pass the SLICED data
                         confirmation_data=confirmation_result,
                         strategy_variant=strategy_variant,
                         risk_model_config=risk_config,
                         spread_points=variant_config.get("spread_points", 0),
                         symbol=symbol,
                         wyckoff_results=variant_config.get('wyckoff_results', {}) # NEW PARAM
                         # Pass confluence/execution params if available/needed
                      )
                      orchestration_result["steps"].append({"step": "Entry Execution", "result": final_entry_result})

                      if final_entry_result and final_entry_result.get("entry_confirmed"):
                         print("[INFO][AdvSMCOrch] Entry Execution SUCCESS.")
                         orchestration_result["final_entry_result"] = final_entry_result
                         entry_found = True
                         # --- 7. Dispatch Signal ---
                         dispatch_pine_payload(final_entry_result)
                      else:
                         print("[INFO][AdvSMCOrch] Entry execution failed or conditions not met.")

             except Exception as e:
                 print(f"[ERROR][AdvSMCOrch] Entry Executor failed: {e}\n{traceback.format_exc()}")
                 orchestration_result["steps"].append({"step": "Entry Execution", "status": "Failed", "error": str(e)})

        else:
             reason = "Module not available"
             if not (confirmation_result and confirmation_result.get("confirmation_status")): reason = "Confirmation failed"
             if not ltf_poi_timestamp: reason = "LTF POI timestamp missing"
             orchestration_result["steps"].append({"step": "Entry Execution", "status": "Skipped", "reason": reason})
             continue
        # --- MODIFICATION END --- ###

    # --- End of POI Loop ---
    if not entry_found:
        print(f"\n[INFO][AdvSMCOrch] No confirmed entry found for variant {strategy_variant}.")
        orchestration_result["summary"] = f"Orchestration complete: No entry found (FAILURE)."
    else:
        orchestration_result["summary"] = f"Orchestration complete: Entry successfully found and executed (SUCCESS)."

    print(f"--- Advanced SMC Orchestrator Complete for Variant: {strategy_variant} ---")
    print(f"[SUMMARY][AdvSMCOrch] {orchestration_result['summary']}")

    # --- 4b. Multi-Framework Wyckoff & Microstructure Analysis ---
    try:
        phase_config = variant_config.get('phase_detection_config', {})
        indicator_profiles = variant_config.get('indicator_profiles', {})
        multi_wyckoff_results = detect_wyckoff_multi_tf(
            all_tf_data,
            config=phase_config,
            indicator_profiles=indicator_profiles
        )
        variant_config['wyckoff_results'] = multi_wyckoff_results
        print("[INFO][AdvSMCOrch] Completed multi-timeframe Wyckoff analysis.")
    except Exception as e:
        print(f"[ERROR][AdvSMCOrch] Multi-timeframe Wyckoff analysis failed: {e}")
        variant_config['wyckoff_results'] = {}

    # --- Optional Markdown Export ---
    try:
        from pathlib import Path
        output_dir = Path("journal")
        output_dir.mkdir(exist_ok=True)
        md_path = output_dir / f"summary_{strategy_variant}_{symbol}_{target_timestamp.strftime('%Y%m%d_%H%M')}.md"
        with open(md_path, "w") as f:
            f.write(f"# SMC Orchestration Summary\n\n**Variant**: {strategy_variant}\n\n**Symbol**: {symbol}\n\n**Timestamp**: {target_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n\n## Steps Executed\n")
            for step in orchestration_result["steps"]:
                f.write(f"- **{step.get('step', 'Unknown')}**: {step.get('status', '✓')}\n")
            if orchestration_result.get("final_entry_result"):
                f.write(f"\n## Entry Summary\n\n- **Entry Confirmed**: {orchestration_result['final_entry_result'].get('entry_confirmed')}\n")
                f.write(f"- **Entry Price**: {orchestration_result['final_entry_result'].get('entry_price')}\n")
                f.write(f"- **SL**: {orchestration_result['final_entry_result'].get('sl')}\n")
                f.write(f"- **TP1**: {orchestration_result['final_entry_result'].get('tp1')}\n")
                f.write(f"- **RR**: {orchestration_result['final_entry_result'].get('rr_ratio')}\n")
            f.write(f"\n---\nOrchestration Status: {orchestration_result['summary']}\n")
        print(f"[EXPORT][AdvSMCOrch] Markdown summary saved to: {md_path}")
        try:
            from telegram_alert_engine import send_simple_summary_alert
            alert_summary = orchestration_result.get("summary", "SMC orchestration completed.")
            send_simple_summary_alert(f"[ADV SMC] {alert_summary}")
        except Exception as te:
            print(f"[WARN][AdvSMCOrch] Telegram alert failed: {te}")
    except Exception as e:
        print(f"[ERROR][AdvSMCOrch] Failed to write markdown summary: {e}")

    return orchestration_result


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    import re
    from pathlib import Path
    print("--- Testing Advanced SMC Orchestrator with real CSV data ---")

    # --- Load CSV files named like SYMBOL_TF_*.csv ---
    all_tf_data = {}
    symbol = None
    pattern = re.compile(r"([A-Z]+)_([A-Za-z0-9]+)_.*\.csv")
    for path in Path().glob("*.csv"):
        m = pattern.match(path.name)
        if not m:
            print(f"[WARN] Skipping unrecognized file {path.name}")
            continue
        sym, tf_raw = m.groups()
        if symbol is None:
            symbol = sym
        elif symbol != sym:
            print(f"[ERROR] Multiple symbols detected: {symbol} vs {sym}. Exiting.")
            exit(1)
        tf = tf_raw.lower()
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            all_tf_data[tf] = df
            print(f"[INFO] Loaded {tf.upper()} data for {symbol} from {path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {path.name}: {e}")

    if not all_tf_data:
        print("[ERROR] No valid CSV data found. Exiting.")
        exit(1)

    test_variant = "Inv"
    test_symbol  = symbol
    test_timestamp = max(df.index.max() for df in all_tf_data.values())

    # --- Run the orchestrator ---
    result = run_advanced_smc_strategy(
        all_tf_data      = all_tf_data,
        strategy_variant = test_variant,
        target_timestamp = test_timestamp,
        symbol           = test_symbol
    )
    # Downstream usage: Example of accessing H1 Wyckoff result
    wyckoff_results = load_strategy_profile(test_variant).get('wyckoff_results', {})
    h1_wyckoff = wyckoff_results.get('h1', {}).get('wyckoff', {})
    # Print H1 Wyckoff if present
    if h1_wyckoff:
        print("[INFO][AdvSMCOrch] H1 Wyckoff phase:", h1_wyckoff)
    print(json.dumps(result, indent=2, default=str))


