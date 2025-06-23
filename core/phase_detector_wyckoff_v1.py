# phase_detector_wyckoff_v1.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0 (Wyckoff Accumulation Phase Detector - Basic)
# Description:
#   Detects Wyckoff Accumulation schematics from HTF OHLCV data (Simplified V1).
#   Outputs potential phase points (SC, AR, ST, Spring, Test, LPS, SOS)
#   and phase classification (A/B/C/D/E). Includes basic PnF target estimation.

# --- ZANALYTICS AI MICRO-WYCKOFF INTEGRATION ---
#
# This module integrates HTF phase detection with LTF microstructure interpretation
# using AI-validated logic defined in the Micro Wyckoff Detection Protocol.
#
# - Detects HTF events: SC, AR, ST, Spring, Test, LPS, SOS
# - Classifies phases: A → E
# - Links to LTF signal states: CHoCH, BOS, Micro-Accumulation, Micro-Distribution
#
# Used by copilot_orchestrator and downstream AI agents for macro-to-micro continuity.

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import traceback
# --- Multi-TF integration imports ---
from core.indicators import add_indicators_multi_tf
from core.micro_wyckoff_phase_engine import detect_micro_wyckoff_phase

# --- Wyckoff Event Detection (Simplified V1) ---
def _detect_wyckoff_events(ohlcv: pd.DataFrame, config: Dict) -> Dict:
    """
    Detect Wyckoff schematic key events (SC, AR, ST, Spring, Test, LPS, SOS).
    NOTE: This is a simplified V1 implementation using basic min/max and volume checks.
          Does not perform full VSA or detailed schematic sequence validation.
    """
    events = {}
    if ohlcv.empty or len(ohlcv) < 10: # Need some data
        print("[WARN][WyckoffDetect] Insufficient data for event detection.")
        return events

    # --- Configuration ---
    pivot_lookback = config.get("pivot_lookback", 20)
    min_volume_multiplier = config.get("min_volume_surge_multiplier", 1.8) # Default multiplier for climax volume
    spring_test_lookback = config.get("spring_test_lookback", 5) # Bars after Spring/ST to look for Test/LPS

    try:
        # --- Ensure Volume & Calculate Avg Volume ---
        if 'Volume' in ohlcv.columns:
            # Calculate rolling average volume, ensuring enough periods
            min_vol_periods = max(1, int(pivot_lookback / 2))
            ohlcv['Avg_Volume'] = ohlcv['Volume'].rolling(window=pivot_lookback, min_periods=min_vol_periods).mean()
            # Fill initial NaNs in Avg_Volume with the first valid calculation or 1
            first_valid_avg_vol = ohlcv['Avg_Volume'].dropna().iloc[0] if not ohlcv['Avg_Volume'].dropna().empty else 1
            ohlcv['Avg_Volume'] = ohlcv['Avg_Volume'].fillna(first_valid_avg_vol).fillna(1) # Fill NaNs
        else:
            print("[WARN][WyckoffDetect] Volume data missing. Event detection accuracy reduced.")
            ohlcv['Avg_Volume'] = 1 # Assign dummy value to avoid errors

        # --- SC (Selling Climax) Detection ---
        # Find overall low point in the dataset (potential SC)
        sc_idx = ohlcv['Low'].idxmin()
        sc_value = ohlcv.loc[sc_idx, 'Low'] if pd.notna(sc_idx) else np.nan

        if pd.notna(sc_idx):
            sc_volume = ohlcv.loc[sc_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
            avg_volume_at_sc = ohlcv.loc[sc_idx, 'Avg_Volume']

            # Condition: Significant low + high volume climax
            if sc_volume >= min_volume_multiplier * avg_volume_at_sc:
                events['SC'] = {"price": sc_value, "time": sc_idx, "volume": sc_volume}
                print(f"[DEBUG][WyckoffDetect] Potential SC detected at {sc_idx} (Price: {sc_value:.4f}, Vol: {sc_volume})")
            else:
                 # Still mark the low, maybe Preliminary Support (PS) or just a low
                 events['LowPoint1'] = {"price": sc_value, "time": sc_idx, "volume": sc_volume}
                 print(f"[DEBUG][WyckoffDetect] Low point detected at {sc_idx} (Price: {sc_value:.4f}, Vol: {sc_volume}) - Volume below SC threshold.")
        else:
            print("[WARN][WyckoffDetect] Could not find minimum low for SC detection.")
            return events # Cannot proceed without initial low

        # Use the identified low point (SC or LowPoint1) as the reference
        ref_low_event = events.get('SC') or events.get('LowPoint1')
        if not ref_low_event: return events # Exit if no reference low found

        # --- AR (Automatic Rally) Detection ---
        # Find highest high after the reference low (SC or initial low)
        post_sc_df = ohlcv.loc[ref_low_event['time']:]
        if not post_sc_df.empty and len(post_sc_df) > 1: # Need more than just the SC candle
            ar_idx = post_sc_df['High'].iloc[1:].idxmax() # Exclude SC candle itself for AR high
            ar_value = post_sc_df.loc[ar_idx, 'High'] if pd.notna(ar_idx) else np.nan
            if pd.notna(ar_idx):
                events['AR'] = {"price": ar_value, "time": ar_idx}
                print(f"[DEBUG][WyckoffDetect] Potential AR detected at {ar_idx} (Price: {ar_value:.4f})")
            else:
                 print("[WARN][WyckoffDetect] Could not find max high for AR detection after SC.")
                 return events # Cannot proceed without AR
        else: return events

        # --- ST (Secondary Test) Detection ---
        # Find lowest low after the AR
        post_ar_df = post_sc_df.loc[events['AR']['time']:]
        if not post_ar_df.empty and len(post_ar_df) > 1:
            st_idx = post_ar_df['Low'].iloc[1:].idxmin() # Exclude AR candle itself
            st_value = post_ar_df.loc[st_idx, 'Low'] if pd.notna(st_idx) else np.nan
            if pd.notna(st_idx):
                 # Condition: ST should ideally be near or slightly above SC level, on lower volume
                 st_volume = ohlcv.loc[st_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
                 events['ST'] = {"price": st_value, "time": st_idx, "volume": st_volume}
                 print(f"[DEBUG][WyckoffDetect] Potential ST detected at {st_idx} (Price: {st_value:.4f}, Vol: {st_volume})")
            else:
                 print("[WARN][WyckoffDetect] Could not find min low for ST detection after AR.")
                 # Can potentially continue without ST, but phase B/C are less clear
        else: return events # Cannot proceed easily without post-AR data

        # --- Spring / Shakeout (Phase C) Detection ---
        # Look for a low that penetrates below the SC/ST support level
        st_event = events.get('ST')
        if st_event: # Requires ST as reference
            support_level = min(ref_low_event['price'], st_event['price'])
            post_st_df = post_ar_df.loc[st_event['time']:]
            if not post_st_df.empty and len(post_st_df) > 1:
                 # Find lows after ST that break the support level
                 possible_spring_df = post_st_df.iloc[1:][post_st_df['Low'].iloc[1:] < support_level]
                 if not possible_spring_df.empty:
                      # Find the lowest point among those that broke support
                      spring_idx = possible_spring_df['Low'].idxmin()
                      spring_value = possible_spring_df.loc[spring_idx, 'Low']
                      spring_volume = ohlcv.loc[spring_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
                      # Optional: Check if volume on spring is lower than SC/ST
                      events['Spring'] = {"price": spring_value, "time": spring_idx, "volume": spring_volume}
                      print(f"[DEBUG][WyckoffDetect] Potential Spring detected at {spring_idx} (Price: {spring_value:.4f}, Vol: {spring_volume}) below support {support_level:.4f}")

                      # --- Test of Spring Detection ---
                      # Look for a higher low shortly after the Spring
                      post_spring_df = post_st_df.loc[spring_idx:]
                      if len(post_spring_df) > 1:
                           # Find lowest low within N bars after the spring candle
                           test_candidates = post_spring_df.iloc[1:spring_test_lookback+1]
                           if not test_candidates.empty:
                                test_idx = test_candidates['Low'].idxmin()
                                test_value = test_candidates.loc[test_idx, 'Low']
                                # Condition: Test low must be higher than Spring low
                                if test_value > spring_value:
                                     test_volume = ohlcv.loc[test_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
                                     events['Test'] = {"price": test_value, "time": test_idx, "volume": test_volume}
                                     print(f"[DEBUG][WyckoffDetect] Potential Test of Spring detected at {test_idx} (Price: {test_value:.4f}, Vol: {test_volume})")


        # --- LPS (Last Point of Support) Detection ---
        # Simplified: Look for higher lows after Spring/Test or after ST if no Spring
        ref_point_for_lps = events.get('Test') or events.get('Spring') or events.get('ST')
        if ref_point_for_lps:
            post_ref_df = ohlcv.loc[ref_point_for_lps['time']:]
            if len(post_ref_df) > spring_test_lookback: # Need enough bars
                 # Look for lows in the window after the reference point
                 lps_candidates_df = post_ref_df.iloc[1:spring_test_lookback+1]
                 if not lps_candidates_df.empty:
                      # Check if the lowest low in this window is higher than the reference low
                      min_low_after_ref = lps_candidates_df['Low'].min()
                      if min_low_after_ref > ref_point_for_lps['price']:
                           # Find the index of that lowest low (potential LPS)
                           lps_idx = lps_candidates_df['Low'].idxmin()
                           lps_value = min_low_after_ref
                           lps_volume = ohlcv.loc[lps_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
                           events['LPS'] = {"price": lps_value, "time": lps_idx, "volume": lps_volume}
                           print(f"[DEBUG][WyckoffDetect] Potential LPS detected at {lps_idx} (Price: {lps_value:.4f}, Vol: {lps_volume})")


        # --- SOS (Sign of Strength) Detection ---
        # Simplified: Look for a high that breaks above the AR resistance level after Phase C attempt
        phase_c_event = events.get('Test') or events.get('Spring') or events.get('LPS') # Use LPS if available
        ar_event = events.get('AR')
        if ar_event and phase_c_event:
             resistance_level = ar_event['price']
             post_phase_c_df = ohlcv.loc[phase_c_event['time']:]
             if not post_phase_c_df.empty and len(post_phase_c_df) > 1:
                  # Check candles closing above resistance
                  possible_sos_df = post_phase_c_df.iloc[1:][post_phase_c_df['Close'].iloc[1:] > resistance_level]
                  if not possible_sos_df.empty:
                       # Find the first candle that broke resistance
                       sos_idx = possible_sos_df.index[0]
                       sos_value = possible_sos_df.loc[sos_idx, 'Close'] # Use Close for confirmation
                       sos_volume = ohlcv.loc[sos_idx, 'Volume'] if 'Volume' in ohlcv.columns else 0
                       # Optional: Check for increased volume on SOS break
                       events['SOS'] = {"price": sos_value, "time": sos_idx, "volume": sos_volume}
                       print(f"[DEBUG][WyckoffDetect] Potential SOS detected at {sos_idx} (Price: {sos_value:.4f}, Vol: {sos_volume}) breaking resistance {resistance_level:.4f}")


    except Exception as e:
        print(f"[ERROR][WyckoffDetect] Error during Wyckoff event detection: {e}")
        traceback.print_exc()
        events['error'] = str(e)

    # Clean up helper column
    if 'Avg_Volume' in ohlcv.columns:
         ohlcv.drop(columns=['Avg_Volume'], inplace=True, errors='ignore')

    return events


def _classify_phases(events: Dict, config: Dict) -> Dict:
    """
    Classify Wyckoff Schematic phases A/B/C/D/E based on detected events.
    Returns a dictionary mapping phase names to start/end times.
    """
    phases = {}
    last_event_time = None

    # Phase A: Stopping the prior trend (SC to AR)
    sc_event = events.get('SC') or events.get('LowPoint1') # Use either SC or first low
    ar_event = events.get('AR')
    if sc_event and ar_event:
        phases['A'] = {'start': sc_event['time'], 'end': ar_event['time'], 'description': 'Stopping Action'}
        last_event_time = ar_event['time']
    else: return phases # Cannot define phases without initial range

    # Phase B: Building the cause (AR to ST)
    st_event = events.get('ST')
    if st_event and last_event_time and st_event['time'] > last_event_time:
        phases['B'] = {'start': last_event_time, 'end': st_event['time'], 'description': 'Building Cause'}
        last_event_time = st_event['time']
    else:
        # If ST is missing, Phase B might extend longer, ending implicitly before Phase C starts
        # We need a Phase C event to define the end of B in this case
        pass # Continue to check for Phase C events

    # Phase C: The test (Spring or LPS if no Spring)
    spring_event = events.get('Spring')
    test_event = events.get('Test') # Test of Spring
    lps_event_after_b = events.get('LPS') if events.get('LPS') and last_event_time and events['LPS']['time'] > last_event_time else None

    phase_c_start_time = last_event_time # Phase C starts after B ends (or after A if B is skipped)
    phase_c_end_time = None

    if spring_event and spring_event['time'] > phase_c_start_time:
        # If Spring exists, Phase C ends with the Spring or its Test
        phase_c_end_event = test_event if test_event and test_event['time'] > spring_event['time'] else spring_event
        phases['C'] = {'start': phase_c_start_time, 'end': phase_c_end_event['time'], 'description': 'Test (Spring/Test)'}
        phase_c_end_time = phase_c_end_event['time']
    elif lps_event_after_b:
        # If no Spring, but an LPS occurs after Phase B, consider it the Phase C test
        phases['C'] = {'start': phase_c_start_time, 'end': lps_event_after_b['time'], 'description': 'Test (LPS)'}
        phase_c_end_time = lps_event_after_b['time']
    # else: Phase C might be absent or very brief if price moves directly to SOS after ST

    last_event_time = phase_c_end_time if phase_c_end_time else last_event_time # Update last known time

    # Phase D: Breaking out of the range (SOS or LPS after Phase C)
    sos_event = events.get('SOS')
    lps_event_phase_d = events.get('LPS') # LPS can also occur in Phase D

    phase_d_start_time = last_event_time # Starts after C (or B if C was skipped)
    phase_d_end_time = None

    if sos_event and phase_d_start_time and sos_event['time'] > phase_d_start_time:
        # Phase D leads up to and includes the SOS
        phases['D'] = {'start': phase_d_start_time, 'end': sos_event['time'], 'description': 'Markup within Range / SOS'}
        phase_d_end_time = sos_event['time']
    elif lps_event_phase_d and phase_d_start_time and lps_event_phase_d['time'] > phase_d_start_time:
         # If SOS hasn't happened yet, but we have LPS after C/B, we might be in early D
         # Check if this LPS is later than the one potentially used for Phase C end
         if phase_c_end_time is None or lps_event_phase_d['time'] > phase_c_end_time:
              phases['D'] = {'start': phase_d_start_time, 'end': lps_event_phase_d['time'], 'description': 'Markup within Range (LPS)'}
              phase_d_end_time = lps_event_phase_d['time']
    # else: Phase D might not be clearly defined yet

    last_event_time = phase_d_end_time if phase_d_end_time else last_event_time

    # Phase E: Markup outside the range
    if phases.get('D') and last_event_time:
        # Phase E starts after Phase D ends
        phases['E'] = {'start': last_event_time, 'end': None, 'description': 'Markup Trend'} # End is ongoing

    return phases


def _generate_pnf_target(events: Dict, config: Dict) -> Dict:
    """
    Simple PnF target projection from Spring or LPS (Placeholder Calculation).
    Requires actual PnF chart construction for accurate counts.
    """
    # --- Placeholder PnF Logic ---
    box_size = config.get("pnf_box_size", 1.0) # Example: $1 box size
    reversal = config.get("pnf_reversal", 3) # Example: 3 box reversal
    pnf_count_base = config.get("pnf_count_base", 10) # *** PLACEHOLDER BOX COUNT ***

    target_conservative = np.nan
    target_aggressive = np.nan
    base_low_price = np.nan

    # Use LPS or Spring as the base low for projection
    lps_event = events.get('LPS')
    spring_event = events.get('Spring')
    st_event = events.get('ST') # Fallback base

    # Prioritize LPS, then Spring, then ST as the base
    base_event = lps_event or spring_event or st_event

    if base_event:
        base_low_price = base_event.get('price', np.nan)
        if pd.notna(base_low_price):
            target_conservative = base_low_price + (pnf_count_base * box_size * reversal)
            # Aggressive target might use a wider count or different base (e.g., SC)
            # For simplicity V1, let's just add a multiplier to the count
            aggressive_count = pnf_count_base * config.get("pnf_aggressive_multiplier", 1.5)
            target_aggressive = base_low_price + (aggressive_count * box_size * reversal)


    return {
        "target_conservative": round(target_conservative, 2) if pd.notna(target_conservative) else None,
        "target_aggressive": round(target_aggressive, 2) if pd.notna(target_aggressive) else None,
        "pnf_count_base": pnf_count_base, # Log the placeholder count used
        "box_size": box_size,
        "reversal": reversal,
        "projection_base_price": round(base_low_price, 2) if pd.notna(base_low_price) else None,
        "calculation_note": "Placeholder PnF count used. Requires real PnF chart analysis."
    }

# --- Main Detection Function ---
# Renamed for consistency with other enrichment engines
def detect_wyckoff_phases_and_events(
    df: pd.DataFrame,
    timeframe: str, # Added timeframe for context
    config: Optional[Dict] = None
    ) -> Dict[str, Any]:
    """
    Analyzes HTF price action to assign a probable Wyckoff phase and detect key events.
    Focuses on Accumulation for V1.

    Args:
        df (pd.DataFrame): HTF OHLCV dataframe with DatetimeIndex. Requires 'High', 'Low', 'Close', 'Volume'.
        timeframe (str): Identifier for the timeframe being analyzed (e.g., 'H1', 'D1').
        config (Dict, optional): Configuration parameters for event detection, PnF, etc.
                                 e.g., {'pivot_lookback': 20, 'min_volume_surge_multiplier': 1.8, ...}

    Returns:
        Dict: {
                  'detected_events': Dict of detected Wyckoff events (SC, AR, ST, Spring, Test, LPS, SOS) with time/price/volume.
                  'phase_classification': Dict mapping phase names (A-E) to start/end times.
                  'current_phase': str (A/B/C/D/E/Unknown) - The estimated phase of the *latest* data point.
                  'pnf_targets': Dict | None - Estimated PnF targets if calculated.
                  'error': str | None - Error message if processing failed.
              }
    """
    print(f"[INFO][WyckoffEngineV1] Running Wyckoff Phase Detection for TF={timeframe}...")
    result = {
        'detected_events': {},
        'phase_classification': {},
        'current_phase': 'Unknown',
        'pnf_targets': None,
        'error': None
    }
    if df is None or df.empty or len(df) < 20: # Need sufficient data
        result['error'] = "Insufficient data for Wyckoff analysis."
        print(f"[WARN][WyckoffEngineV1] {result['error']}")
        return result
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
         result['error'] = "Missing required OHLCV columns."
         print(f"[WARN][WyckoffEngineV1] {result['error']}")
         return result
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
         # Attempt to localize if naive, assuming UTC
         if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
              print(f"[WARN][WyckoffEngineV1] Input DataFrame index is timezone-naive. Assuming UTC.")
              try:
                   df = df.tz_localize('UTC')
              except Exception as tz_err:
                   result['error'] = f"Failed to localize DataFrame index to UTC: {tz_err}"
                   print(f"[ERROR][WyckoffEngineV1] {result['error']}")
                   return result
         else:
              result['error'] = "Input DataFrame must have a DatetimeIndex."
              print(f"[WARN][WyckoffEngineV1] {result['error']}")
              return result


    if config is None: config = {} # Use default parameters if none provided

    try:
        # 1. Detect Key Events
        # Pass a copy to avoid modifying the original DataFrame passed into this function
        events = _detect_wyckoff_events(df.copy(), config)
        # Convert Timestamps to ISO strings *before* storing in the final result
        result['detected_events'] = {
            k: {**v, 'time': v['time'].isoformat()}
            for k, v in events.items()
            if k != 'error' and isinstance(v.get('time'), pd.Timestamp)
        }
        if 'error' in events: result['error'] = f"Event detection error: {events['error']}"

        # 2. Classify Phases based on events (use original events dict with Timestamps)
        phases = _classify_phases(events, config)
        # Convert Timestamps to ISO strings for the final result
        result['phase_classification'] = {
            k: {**v, 'start': v['start'].isoformat(),
                'end': v['end'].isoformat() if isinstance(v.get('end'), pd.Timestamp) else None}
            for k, v in phases.items()
            if isinstance(v.get('start'), pd.Timestamp) # Ensure start time is valid
        }

        # 3. Determine Current Phase
        if not phases:
            result['current_phase'] = 'Unknown (No Phases)'
        else:
            last_timestamp = df.index[-1]
            current_phase = 'Unknown'
            # Iterate through phases to find where the last timestamp falls
            # Use the 'phases' dict which still has Timestamp objects for comparison
            valid_phases = {k:v for k,v in phases.items() if isinstance(v.get('start'), pd.Timestamp)}
            sorted_phases = sorted(valid_phases.items(), key=lambda item: item[1]['start']) # Sort by start time

            for phase_name, phase_info in sorted_phases:
                start_time = phase_info['start']
                end_time = phase_info.get('end') # This is already a Timestamp or None
                if last_timestamp >= start_time:
                    if end_time is None or last_timestamp <= end_time:
                        current_phase = phase_name
                        # Don't break, let the last matching phase be assigned (e.g., E overrides D)
                    elif end_time and last_timestamp > end_time:
                         # If last timestamp is after the end of this phase, continue check
                         continue
            # If still Unknown, assign the last known phase name
            result['current_phase'] = current_phase if current_phase != 'Unknown' else (sorted_phases[-1][0] if sorted_phases else 'Unknown')


        # 4. Generate PnF Targets (Optional)
        if config.get("enable_pnf_projection", True) and ('Spring' in events or 'LPS' in events or 'ST' in events):
            pnf_targets = _generate_pnf_target(events, config) # Pass original events
            result['pnf_targets'] = pnf_targets

        print(f"[INFO][WyckoffEngineV1] Wyckoff Analysis Complete. Current Estimated Phase: {result['current_phase']}")

    except Exception as e:
        result['error'] = f"Error during Wyckoff phase analysis: {e}"
        print(f"[ERROR][WyckoffEngineV1] {result['error']}")
        traceback.print_exc()

    return result


# --- Multi-TF Wyckoff and Microstructure Wrapper ---
def detect_wyckoff_multi_tf(
    all_tf_data: Dict[str, pd.DataFrame],
    config: Optional[Dict] = None,
    indicator_profiles: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run Wyckoff phase detection across multiple timeframes.
    Steps:
      1. Enrich each TF with indicators.
      2. Run HTF Wyckoff detection per TF.
      3. Run microstructure phase engine on LTFs (e.g., M1, M5) if available.
    Returns a dict mapping TF -> detection result.
    """
    if config is None:
        config = {}
    if indicator_profiles is None:
        indicator_profiles = config.get('indicator_profiles', {})
    # 1. Enrich all TFs
    enriched = add_indicators_multi_tf(all_tf_data, indicator_profiles)
    results = {}
    for tf, df in enriched.items():
        try:
            # 2. High-level Wyckoff detection
            htf_result = detect_wyckoff_phases_and_events(df, timeframe=tf, config=config)
            # 3. Microstructure detection for lower TFs
            # --- Micro Wyckoff AI protocol integration ---
            if tf.lower() in ['m1', 'm5', 'm15']:
                micro_result = detect_micro_wyckoff_phase(df)
                if isinstance(htf_result, dict):
                    htf_result['micro_context'] = micro_result
            results[tf] = {
                'wyckoff': htf_result,
                # 'micro': micro_result   # (micro now linked in htf_result['micro_context'] if applicable)
            }
        except Exception as e:
            results[tf] = {'error': str(e)}
    return results

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Wyckoff Phase Detector v1 ---")

    # Create dummy data simulating accumulation
    periods = 100
    index = pd.date_range(start='2024-01-01', periods=periods, freq='D', tz='UTC')
    # Phase A: Downtrend -> SC -> AR -> ST
    price = np.linspace(110, 100, 20) # Downtrend
    price = np.append(price, [98, 97]) # SC
    price = np.append(price, np.linspace(97.5, 105, 8)) # AR
    price = np.append(price, np.linspace(104, 100, 10)) # ST
    # Phase B: Range bound
    price = np.append(price, np.random.uniform(100, 105, 20))
    # Phase C: Spring + Test
    price = np.append(price, [96.5]) # Spring below SC/ST (index 59)
    price = np.append(price, [99, 101]) # Test > Spring (indices 60, 61)
    # Phase D: LPS + SOS
    price = np.append(price, np.linspace(100.5, 104, 10)) # Higher lows (LPS area) (indices 62-71)
    price = np.append(price, [106, 107]) # SOS breaks AR level (105) (indices 72, 73)
    # Phase E: Markup
    price = np.append(price, np.linspace(106.5, 115, periods - len(price))) # indices 74 onwards

    volume = np.random.randint(100, 500, periods).astype(float)
    volume[20:22] *= 3 # High volume on SC
    volume[22:30] *= 1.5 # Moderate volume on AR
    volume[30:40] *= 0.8 # Lower volume on ST
    volume[59] *= 1.2 # Moderate volume on Spring
    volume[60:62] *= 0.7 # Low volume on Test
    volume[72:74] *= 2.5 # High volume on SOS

    dummy_df = pd.DataFrame(index=index)
    dummy_df['Close'] = price
    # Generate Open/High/Low relative to Close
    dummy_df['Open'] = dummy_df['Close'] - np.random.uniform(-0.5, 0.5, periods)
    dummy_df['High'] = np.maximum(dummy_df['Open'], dummy_df['Close']) + np.random.uniform(0, 0.5, periods)
    dummy_df['Low'] = np.minimum(dummy_df['Open'], dummy_df['Close']) - np.random.uniform(0, 0.5, periods)
    dummy_df['Volume'] = volume

    print("Dummy Data Head:")
    print(dummy_df.head())

    test_config = {
        "pivot_lookback": 15,
        "min_volume_surge_multiplier": 1.7,
        "spring_test_lookback": 5, # How many bars after spring to look for test
        "enable_pnf_projection": True,
        "pnf_box_size": 0.5,
        "pnf_reversal": 3,
        "pnf_count_base": 12 # Placeholder count
    }

    print("\nRunning Wyckoff Phase Detection...")
    wyckoff_result = detect_wyckoff_phases_and_events(dummy_df, timeframe="D1", config=test_config)

    print("\n--- Wyckoff Analysis Result ---")
    # Print events nicely
    print("Detected Events:")
    if wyckoff_result.get('detected_events'):
        # Sort events by time for clarity
        sorted_events = sorted(wyckoff_result['detected_events'].items(), key=lambda item: item[1]['time'])
        for key, val in sorted_events:
            print(f"  - {key:<10}: Price={val.get('price'):<8.2f} Time={val.get('time')} Volume={val.get('volume', 'N/A'):<8.0f}")
    else: print("  None")
    # Print phases nicely
    print("\nPhase Classification:")
    if wyckoff_result.get('phase_classification'):
        # Sort phases by start time
        sorted_phases = sorted(wyckoff_result['phase_classification'].items(), key=lambda item: item[1]['start'])
        for key, val in sorted_phases:
            print(f"  - Phase {key}: Start={val.get('start')}, End={val.get('end')}, Desc={val.get('description')}")
    else: print("  None")
    print(f"\nCurrent Estimated Phase: {wyckoff_result.get('current_phase')}")
    # Print PnF
    print("\nPnF Targets:")
    print(f"  {wyckoff_result.get('pnf_targets')}")
    # Print errors
    if wyckoff_result.get('error'):
        print(f"\nError: {wyckoff_result['error']}")

    print("\n--- Test Complete ---")
