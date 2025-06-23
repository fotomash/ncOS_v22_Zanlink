# --- Constants Class ---
class ConfirmationConstants:
    CHOCH = "CHoCH"
    BOS = "BOS"
    LTF = "LTF"

# confirmation_engine_smc.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   Confirms potential SMC trade entries by analyzing Lower Timeframe (LTF)
#   structure (CHoCH/BOS) after price mitigates a Higher Timeframe (HTF) POI.
#   Identifies the refined LTF POI responsible for the structural break.

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import traceback

# Assuming helper functions might be needed from other modules
# We might need swing point detection logic similar to other engines.
# Let's define a local version for clarity within this module's context.

def _find_ltf_swing_points(series: pd.Series, n: int = 2) -> pd.Series:
    """
    Identifies local swing points (highs/lows) in the LTF data slice. n=candles each side.

    Args:
        series (pd.Series): Price series (High or Low).
        n (int): Number of bars on each side to define a swing point.

    Returns:
        pd.Series: Series with swing point prices at their index, NaN otherwise.
    """
    if not isinstance(series, pd.Series) or series.empty or len(series) < 2*n+1:
        return pd.Series(dtype=float) # Return empty series if input invalid
    # Use rolling window centered on the point being evaluated
    # Find points that are the max/min within the window (2*n+1 size)
    local_max = series.rolling(window=2*n+1, center=True, min_periods=n+1).max()
    local_min = series.rolling(window=2*n+1, center=True, min_periods=n+1).min()
    # Create NaN series to store results
    swing_points = pd.Series(np.nan, index=series.index)
    # Mark swing highs where the series value equals the rolling max
    swing_points[series == local_max] = series[series == local_max] # Tag highs
    # Mark swing lows where the series value equals the rolling min
    swing_points[series == local_min] = series[series == local_min] # Tag lows
    return swing_points

def _find_ltf_poi_candle(df_slice: pd.DataFrame, break_candle_index: int, is_bullish_break: bool) -> Optional[Tuple[pd.Timestamp, List[float], str]]:
    """
    Finds the LTF POI (e.g., Order Block candle) that likely caused the break.
    Looks backwards from the candle *before* the break candle.

    Args:
        df_slice (pd.DataFrame): The LTF data slice being analyzed.
        break_candle_index (int): The integer index (.iloc) of the candle that broke structure.
        is_bullish_break (bool): True if looking for a bullish break (caused by demand), False for bearish.

    Returns:
        Optional[Tuple[pd.Timestamp, List[float], str]]: (timestamp, [low, high], type) of the POI candle, or None.
                                                        Type could be 'OB' or potentially 'FVG' if refined.
    """
    if break_candle_index <= 0 or break_candle_index >= len(df_slice):
        print("[WARN][ConfirmEngine] Invalid break_candle_index provided to _find_ltf_poi_candle.")
        return None

    # Look backwards from the candle *before* the one that broke structure
    lookback_limit = 10 # How many candles back to search for the origin candle
    search_end_idx = break_candle_index - 1
    search_start_idx = max(0, search_end_idx - lookback_limit)

    origin_candle = None
    origin_candle_iloc = -1

    # Iterate backwards to find the last relevant opposing candle
    for i in range(search_end_idx, search_start_idx - 1, -1):
        # Ensure index is valid before accessing iloc
        if i < 0 or i >= len(df_slice):
             continue
        try:
            candle = df_slice.iloc[i]
            # Check for required columns and NaN values
            if not all(col in candle.index for col in ['Close', 'Open']) or candle[['Close', 'Open']].isnull().any():
                continue
            is_bullish_candle = candle['Close'] > candle['Open']
            is_bearish_candle = candle['Close'] < candle['Open']

            # If looking for origin of bullish break, find the last bearish candle
            if is_bullish_break and is_bearish_candle:
                origin_candle = candle
                origin_candle_iloc = i
                break # Found the most recent one
            # If looking for origin of bearish break, find the last bullish candle
            elif not is_bullish_break and is_bullish_candle:
                origin_candle = candle
                origin_candle_iloc = i
                break # Found the most recent one
        except IndexError:
             print(f"[WARN][ConfirmEngine] IndexError accessing candle at iloc {i} in _find_ltf_poi_candle.")
             continue # Skip this index if out of bounds


    if origin_candle is not None and not origin_candle.isnull().all():
        poi_timestamp = origin_candle.name # Get the timestamp (index)
        # Define POI range based on the candle body or full range (configurable?)
        # Let's use the full candle range (High/Low) for now as the POI
        if 'Low' in origin_candle.index and 'High' in origin_candle.index and pd.notna(origin_candle['Low']) and pd.notna(origin_candle['High']):
             poi_range = [origin_candle['Low'], origin_candle['High']]
             poi_type = "OB" # Basic Order Block type
             print(f"[DEBUG][ConfirmEngine] Found potential LTF POI candle ({poi_type}) at index {origin_candle_iloc} ({poi_timestamp}) Range: {poi_range}")
             # Ensure poi_timestamp is a Timestamp object
             if not isinstance(poi_timestamp, pd.Timestamp):
                 poi_timestamp = pd.Timestamp(poi_timestamp) # Convert if necessary
             return poi_timestamp, poi_range, poi_type
        else:
             print(f"[WARN][ConfirmEngine] Identified POI candle at {origin_candle_iloc} has missing High/Low values.")
             return None
    else:
        print(f"[DEBUG][ConfirmEngine] No clear origin POI candle found within lookback before break at index {break_candle_index}")
        return None


# --- Main Confirmation Function ---
def confirm_smc_entry(
    htf_poi: Dict,
    ltf_data: pd.DataFrame,
    strategy_variant: str,
    config: Optional[Dict] = None,
    structure_context: Optional[Dict] = None # Optional HTF context
    ) -> Dict[str, Any]:
    """
    Confirms a potential trade entry based on LTF structure shift after HTF POI mitigation.

    Args:
        htf_poi (Dict): Dictionary describing the HTF POI that was mitigated.
                        Expected keys: 'type' ('Bullish'/'Bearish'), potentially 'range'.
        ltf_data (pd.DataFrame): The enriched DataFrame for the confirmation timeframe (e.g., M15)
                                 containing data *after* the HTF POI tap time.
                                 Requires 'High', 'Low', 'Close' columns and DatetimeIndex.
        strategy_variant (str): Name of the strategy variant.
        config (Dict, optional): Configuration parameters. Keys like:
                                 'confirmation_lookback': How many bars to analyze in ltf_data.
                                 'swing_n': Lookback/forward for swing point detection.
                                 'require_bos': If True, requires BOS, otherwise CHoCH is sufficient.
                                 'min_break_distance_atr': Min distance (in ATR multiples) for a valid break. (TODO)
                                 'confluence_checks': List of checks to perform (e.g., ['rsi_div', 'volume_spike']). (TODO)
        structure_context (Dict, optional): Output from HTF structure analysis for context.

    Returns:
        Dict: Containing confirmation status and details. Keys:
              'confirmation_status': bool
              'confirmation_type': str | None (e.g., 'M15_CHoCH', 'M15_BOS')
              'choch_details': Dict | None ({'timestamp', 'price', 'type', 'broken_swing_timestamp'})
              'bos_details': Dict | None ({'timestamp', 'price', 'type', 'broken_swing_timestamp'})
              'ltf_poi_range': List[float] | None # Range of the identified LTF POI causing the break
              'ltf_poi_timestamp': str | None # ISO format timestamp of the LTF POI candle
              'ltf_poi_type': str | None # Type of LTF POI (e.g., 'OB')
              'conviction_score': int # Placeholder score (1-5)
              'error': str | None
    """
    print(f"[INFO][ConfirmEngine] Running Confirmation Engine for HTF POI Type: {htf_poi.get('type')}...")
    result: Dict[str, Any] = {
        "confirmation_status": False,
        "confirmation_type": None,
        "choch_details": None,
        "bos_details": None,
        "ltf_poi_range": None,
        "ltf_poi_timestamp": None, # Store as ISO string
        "ltf_poi_type": None,
        "conviction_score": 3, # Default placeholder score
        "error": None
    }

    # --- Input Validation ---
    if ltf_data is None or ltf_data.empty:
        result["error"] = "LTF data for confirmation is missing or empty."
        print(f"[ERROR][ConfirmEngine] {result['error']}")
        return result
    if not isinstance(ltf_data.index, pd.DatetimeIndex):
         result["error"] = "LTF data index must be a DatetimeIndex."
         print(f"[ERROR][ConfirmEngine] {result['error']}")
         return result
    if not all(col in ltf_data.columns for col in ['High', 'Low', 'Close']):
         result["error"] = "LTF data missing required High/Low/Close columns."
         print(f"[ERROR][ConfirmEngine] {result['error']}")
         return result

    htf_poi_type = htf_poi.get('type')
    if htf_poi_type not in ['Bullish', 'Bearish']:
         result["error"] = f"Invalid HTF POI type: {htf_poi_type}"
         print(f"[ERROR][ConfirmEngine] {result['error']}")
         return result

    # --- Configuration ---
    if config is None: config = {}
    confirmation_lookback = config.get('confirmation_lookback', 30) # How many LTF bars to analyze
    swing_n = config.get('swing_n', 2) # Lookback for LTF swings (default 2 bars each side)
    require_bos = config.get('require_bos', False) # Default: CHoCH is sufficient
    # min_break_dist = config.get('min_break_distance_atr', 0.5) # TODO: Use ATR for break validation

    # Limit analysis to the lookback period
    analysis_df = ltf_data.tail(confirmation_lookback).copy()
    if len(analysis_df) < (2 * swing_n + 3): # Need enough data for swings and break check
        result["error"] = f"Insufficient LTF data ({len(analysis_df)} bars) for confirmation lookback ({confirmation_lookback}) and swing detection (n={swing_n})."
        print(f"[WARN][ConfirmEngine] {result['error']}")
        return result

    print(f"[DEBUG][ConfirmEngine] Analyzing last {len(analysis_df)} bars of LTF data.")

    try:
        # --- Identify Local Swings within the Analysis Window ---
        local_highs = _find_ltf_swing_points(analysis_df['High'], n=swing_n)
        local_lows = _find_ltf_swing_points(analysis_df['Low'], n=swing_n)

        # Combine valid swings into a sorted list with their iloc indices
        swings = []
        for idx, price in local_highs.dropna().items():
            try:
                loc = analysis_df.index.get_loc(idx)
                swings.append({'timestamp': idx, 'price': price, 'type': 'High', 'iloc': loc})
            except KeyError:
                print(f"[WARN][ConfirmEngine] Could not get iloc for swing high at {idx}")
        for idx, price in local_lows.dropna().items():
            try:
                loc = analysis_df.index.get_loc(idx)
                swings.append({'timestamp': idx, 'price': price, 'type': 'Low', 'iloc': loc})
            except KeyError:
                print(f"[WARN][ConfirmEngine] Could not get iloc for swing low at {idx}")

        swings.sort(key=lambda x: x['timestamp']) # Sort by timestamp

        if len(swings) < 2: # Need at least two swings to define structure
            result["error"] = "Not enough swing points identified in the LTF confirmation window."
            print(f"[INFO][ConfirmEngine] {result['error']}")
            return result

        print(f"[DEBUG][ConfirmEngine] Identified {len(swings)} local swing points in confirmation window.")

        # --- Helper for Structure Break Detection ---
        def detect_structure_break(direction: str, swings: List[dict], analysis_df: pd.DataFrame):
            """
            direction: 'Bullish' or 'Bearish'
            Returns (break_found, break_type, break_details, ltf_poi_details)
            """
            break_found = False
            break_type = None
            break_details = {}
            ltf_poi_details = None
            if direction == 'Bullish':
                last_high = next((s for s in reversed(swings) if s['type'] == 'High'), None)
                if last_high is None:
                    return (False, None, {}, None)
                target_break_level = last_high['price']
                target_break_time = last_high['timestamp']
                break_check_df = analysis_df[analysis_df.index > target_break_time]
                if not break_check_df.empty:
                    break_candle_mask = break_check_df['Close'] > target_break_level
                    if break_candle_mask.any():
                        first_break_candle_timestamp = break_check_df[break_candle_mask].index[0]
                        first_break_candle_price = break_check_df.loc[first_break_candle_timestamp, 'Close']
                        break_found = True
                        break_type = ConfirmationConstants.CHOCH # Default to CHoCH for first break
                        break_details = {
                            'timestamp': first_break_candle_timestamp.isoformat(),
                            'price': first_break_candle_price,
                            'type': 'Bullish',
                            'broken_swing_timestamp': target_break_time.isoformat()
                        }
                        print(f"[INFO][ConfirmEngine] Bullish {break_type} confirmed at {break_details['timestamp']} (Price: {break_details['price']:.5f}), broke high at {target_break_time}")
                        try:
                            break_candle_iloc = analysis_df.index.get_loc(first_break_candle_timestamp)
                            ltf_poi_details = _find_ltf_poi_candle(analysis_df, break_candle_iloc, is_bullish_break=True)
                        except KeyError:
                            print(f"[WARN][ConfirmEngine] Could not get iloc for break candle at {first_break_candle_timestamp} to find POI.")
            elif direction == 'Bearish':
                last_low = next((s for s in reversed(swings) if s['type'] == 'Low'), None)
                if last_low is None:
                    return (False, None, {}, None)
                target_break_level = last_low['price']
                target_break_time = last_low['timestamp']
                break_check_df = analysis_df[analysis_df.index > target_break_time]
                if not break_check_df.empty:
                    break_candle_mask = break_check_df['Close'] < target_break_level
                    if break_candle_mask.any():
                        first_break_candle_timestamp = break_check_df[break_candle_mask].index[0]
                        first_break_candle_price = break_check_df.loc[first_break_candle_timestamp, 'Close']
                        break_found = True
                        break_type = ConfirmationConstants.CHOCH # Default to CHoCH for first break
                        break_details = {
                            'timestamp': first_break_candle_timestamp.isoformat(),
                            'price': first_break_candle_price,
                            'type': 'Bearish',
                            'broken_swing_timestamp': target_break_time.isoformat()
                        }
                        print(f"[INFO][ConfirmEngine] Bearish {break_type} confirmed at {break_details['timestamp']} (Price: {break_details['price']:.5f}), broke low at {target_break_time}")
                        try:
                            break_candle_iloc = analysis_df.index.get_loc(first_break_candle_timestamp)
                            ltf_poi_details = _find_ltf_poi_candle(analysis_df, break_candle_iloc, is_bullish_break=False)
                        except KeyError:
                            print(f"[WARN][ConfirmEngine] Could not get iloc for break candle at {first_break_candle_timestamp} to find POI.")
            return (break_found, break_type, break_details, ltf_poi_details)

        # --- Detect CHoCH / BOS (via helper) ---
        break_found, break_type, break_details, ltf_poi_details = detect_structure_break(htf_poi_type, swings, analysis_df)

        # --- Update Result ---
        if break_found:
            # Determine TF prefix dynamically if possible
            tf_prefix = ConfirmationConstants.LTF # Default prefix
            if hasattr(analysis_df.index, 'freqstr') and analysis_df.index.freqstr:
                # Convert pandas freq string (e.g., '15T') to 'M15' format
                freq = analysis_df.index.freqstr.upper()
                freq_num = ''.join(filter(str.isdigit, freq))
                freq_unit = ''.join(filter(str.isalpha, freq))
                if freq_unit in ['T', 'MIN']:
                    tf_prefix = f"M{freq_num}"
                elif freq_unit == 'H':
                    tf_prefix = f"H{freq_num}"
                elif freq_unit == 'D':
                    tf_prefix = f"D{freq_num}"
                # Add more conversions if needed
            result["confirmation_status"] = True
            result["confirmation_type"] = f"{tf_prefix}_{break_type}"
            if break_type == ConfirmationConstants.CHOCH:
                result["choch_details"] = break_details
            elif break_type == ConfirmationConstants.BOS:
                result["bos_details"] = break_details
            else:
                # Default fallback
                result["choch_details"] = break_details
            if ltf_poi_details:
                # Store timestamp as ISO format string for JSON compatibility
                result["ltf_poi_timestamp"] = ltf_poi_details[0].isoformat() if isinstance(ltf_poi_details[0], pd.Timestamp) else None
                result["ltf_poi_range"] = ltf_poi_details[1]
                result["ltf_poi_type"] = ltf_poi_details[2]
            else:
                print("[WARN][ConfirmEngine] Confirmation break found, but failed to identify originating LTF POI candle.")
        else:
            print("[INFO][ConfirmEngine] No confirming LTF structure break (CHoCH/BOS) found in the lookback window.")
            result["confirmation_status"] = False

    except Exception as e:
        result["error"] = f"Error during confirmation analysis: {e}"
        print(f"[ERROR][ConfirmEngine] {result['error']}")
        traceback.print_exc()
        result["confirmation_status"] = False

    print(f"[INFO][ConfirmEngine] Confirmation Status: {result['confirmation_status']}")
    return result

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Confirmation Engine (SMC) ---")

    # Create dummy HTF POI
    htf_poi_bullish = {'type': 'Bullish', 'range': [1.0990, 1.0995]}
    htf_poi_bearish = {'type': 'Bearish', 'range': [1.1055, 1.1060]}

    # Create dummy LTF (M15) data *after* a hypothetical tap
    base_time = pd.Timestamp('2024-04-28 14:00:00', tz='UTC')
    periods = 40
    timestamps = pd.date_range(start=base_time, periods=periods, freq='15T') # Explicitly M15

    # Scenario 1: Bullish Confirmation (Pullback then CHoCH up)
    data1 = {'Open': np.linspace(1.1005, 1.0998, periods), 'Close': np.linspace(1.1003, 1.0996, periods)}
    data1['High'] = np.maximum(data1['Open'], data1['Close']) + 0.0003
    data1['Low'] = np.minimum(data1['Open'], data1['Close']) - 0.0003
    ltf_df1 = pd.DataFrame(data1, index=timestamps)
    # Create a low, then a lower high, then break the high
    swing_low_iloc = 10
    swing_high_iloc = 20
    break_iloc = 30
    ltf_df1.iloc[swing_low_iloc, ltf_df1.columns.get_loc('Low')] = 1.0992 # Low point
    ltf_df1.iloc[swing_high_iloc, ltf_df1.columns.get_loc('High')] = 1.1002 # Lower high point
    ltf_df1.iloc[break_iloc, ltf_df1.columns.get_loc('Close')] = 1.1005 # Close above the high
    # Add potential OB candle before break
    ltf_df1.iloc[break_iloc-1, ltf_df1.columns.get_loc('Open')] = ltf_df1.iloc[break_iloc-1]['Close'] + 0.0001 # Make it bearish
    ltf_df1.iloc[break_iloc-1, ltf_df1.columns.get_loc('Low')] = ltf_df1.iloc[break_iloc-1]['Close'] - 0.0002 # Ensure valid candle
    ltf_df1.iloc[break_iloc-1, ltf_df1.columns.get_loc('High')] = ltf_df1.iloc[break_iloc-1]['Open'] + 0.00005 # Ensure valid candle


    print("\n--- Testing Bullish Confirmation ---")
    result1 = confirm_smc_entry(htf_poi_bullish, ltf_df1, "Inv")
    print(json.dumps(result1, indent=2, default=str))

    # Scenario 2: Bearish Confirmation (Rally then CHoCH down)
    data2 = {'Open': np.linspace(1.1045, 1.1052, periods), 'Close': np.linspace(1.1047, 1.1054, periods)}
    data2['High'] = np.maximum(data2['Open'], data2['Close']) + 0.0003
    data2['Low'] = np.minimum(data2['Open'], data2['Close']) - 0.0003
    ltf_df2 = pd.DataFrame(data2, index=timestamps)
    # Create a high, then a higher low, then break the low
    swing_high_iloc = 10
    swing_low_iloc = 20
    break_iloc = 30
    ltf_df2.iloc[swing_high_iloc, ltf_df2.columns.get_loc('High')] = 1.1058 # High point
    ltf_df2.iloc[swing_low_iloc, ltf_df2.columns.get_loc('Low')] = 1.1048 # Higher low point
    ltf_df2.iloc[break_iloc, ltf_df2.columns.get_loc('Close')] = 1.1045 # Close below the low
     # Add potential OB candle before break
    ltf_df2.iloc[break_iloc-1, ltf_df2.columns.get_loc('Open')] = ltf_df2.iloc[break_iloc-1]['Close'] - 0.0001 # Make it bullish
    ltf_df2.iloc[break_iloc-1, ltf_df2.columns.get_loc('High')] = ltf_df2.iloc[break_iloc-1]['Open'] + 0.0002 # Ensure valid candle
    ltf_df2.iloc[break_iloc-1, ltf_df2.columns.get_loc('Low')] = ltf_df2.iloc[break_iloc-1]['Open'] - 0.00005 # Ensure valid candle


    print("\n--- Testing Bearish Confirmation ---")
    result2 = confirm_smc_entry(htf_poi_bearish, ltf_df2, "Inv")
    print(json.dumps(result2, indent=2, default=str))

    # Scenario 3: No Confirmation
    data3 = {'Open': np.linspace(1.1000, 1.0990, periods), 'Close': np.linspace(1.0998, 1.0988, periods)} # Trending down
    data3['High'] = np.maximum(data3['Open'], data3['Close']) + 0.0002
    data3['Low'] = np.minimum(data3['Open'], data3['Close']) - 0.0002
    ltf_df3 = pd.DataFrame(data3, index=timestamps)

    print("\n--- Testing No Confirmation ---")
    result3 = confirm_smc_entry(htf_poi_bullish, ltf_df3, "Inv")
    print(json.dumps(result3, indent=2, default=str))


    print("\n--- Test Complete ---")
