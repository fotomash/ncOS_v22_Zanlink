
# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: poi_manager_smc.py
# Description: Identifies, validates, and scores Points of Interest (POIs) based on Smart Money Concepts (SMC).
# llm_trader_v1_4_2/poi_manager_smc.py
# Module responsible for identifying, validating, and potentially scoring
# Points of Interest (POIs) based on SMC principles.
# Includes implemented logic for FVG, basic OB, Wick, and Breaker identification.
# Mitigation logic updated to check for close beyond midpoint.

"""
Phase 1 Enhancement (via pms.md)
--------------------------------
This version incorporates POI scoring logic based on:
- Mitigation status
- Fibonacci zone alignment
- Prior tap memory
- Wyckoff phase matching (if available)
- Volume-based context

Scores are calculated and tagged via `poi_score`, and reasons stored in `poi_reason`.
Scoring config is aligned with strategy_profiles.json > poi_scoring.
"""

import pandas as pd
import numpy as np
import inspect # For logging helper
from datetime import datetime, timezone # For timestamp comparisons

# --- Confluence Engine Import ---
from confluence_engine import compute_confluence_indicators

# --- Import necessary components ---
try:
    # Import the filter function we created
    from fibonacci_filter import apply_fibonacci_filter
    FIB_FILTER_LOADED = True
    # print("INFO (poi_manager_smc): Imported apply_fibonacci_filter.") # Less verbose
except ImportError:
    FIB_FILTER_LOADED = False
    print("ERROR (poi_manager_smc): Could not import apply_fibonacci_filter from fibonacci_filter.py.")
    # Define dummy if needed for script structure integrity
    def apply_fibonacci_filter(*args, **kwargs):
        print("WARN (poi_manager_smc): Using dummy apply_fibonacci_filter.")
        # Default to pass if filter unavailable, or could default to fail
        return {'is_valid_poi': True, 'filter_reason': 'Fib Filter Module Missing - Default Pass'}

# --- Logging Helper ---
def log_info(message, level="INFO"):
    """Prepends module name to log messages."""
    # Basic logger
    module_name = "POI_MANAGER_SMC"
    print(f"[{level}][{module_name}] {message}")

# --- POI Scoring Helper ---
def score_poi(poi, config_weights=None):
    """
    Scores a POI based on multiple confirmation layers.
    Accepts optional config weights dict. Returns (score, reasons).
    """
    if config_weights is None:
        config_weights = {
            "mitigated": 0.3,
            "fib_zone": 0.25,
            "tap_memory": 0.15,
            "volume_ok": 0.1,
            "wyckoff_match": 0.2
        }
    score = 0.0
    reasons = []

    # Inject confluence-based scoring (DSS, VWAP, BB)
    confluence = poi.get("confluence", {})
    dss_slope = confluence.get("dss_slope", 0)
    vwap_dev = abs(confluence.get("vwap_deviation", 0))
    bb_width = confluence.get("bb_width", 0)

    if dss_slope > 0.1:
        score += 0.05
        reasons.append("dss_slope_up")

    if vwap_dev < 2.0:
        score += 0.05
        reasons.append("vwap_near")

    if bb_width < 0.025:
        score += 0.03
        reasons.append("bb_squeeze")

    if poi.get("mitigation_status") == "valid":
        score += config_weights["mitigated"]
        reasons.append("mitigated")
    if poi.get("fib_valid") == True:
        score += config_weights["fib_zone"]
        reasons.append("fib_zone")
    if poi.get("tap_count", 0) > 0:
        score += config_weights["tap_memory"]
        reasons.append("tap_memory")
    if poi.get("volume", 0) > 0:
        score += config_weights["volume_ok"]
        reasons.append("volume")

    # Enhanced logic: volume signature and POI shape factors
    volume_class = poi.get("volume_signature", "normal")
    if volume_class == "climax":
        score += 0.1
        reasons.append("volume_climax")
    elif volume_class == "absorption":
        score += 0.05
        reasons.append("volume_absorption")

    # Add shape/width logic: narrow range POIs often more reactive
    width = abs(poi.get("poi_level_top", 0) - poi.get("poi_level_bottom", 0))
    if 0 < width < 0.0025:
        score += 0.05
        reasons.append("narrow_zone")

    # Add reward factor for M15 or H1 OBs
    tf_bonus = poi.get("source_tf", "").lower()
    if tf_bonus in ["m15", "h1"]:
        score += 0.05
        reasons.append(f"tf_{tf_bonus}_bonus")

    # Enhanced Wyckoff tag logic
    wyckoff_phase = poi.get("wyckoff_phase", "").lower()
    if any(kw in wyckoff_phase for kw in ["spring", "phase c", "phase d"]):
        score += config_weights["wyckoff_match"]
        reasons.append("wyckoff_phase_conf")

    # Ensure poi_score and poi_reason are set before return (if used in context)
    poi["poi_score"] = round(score, 3)
    poi["poi_reason"] = ", ".join(reasons)

    return round(score, 3), reasons

# --- POI Identification Helper Functions ---
# (find_order_blocks, find_imbalances, _check_bos_after, find_breaker_blocks, find_rejection_wicks remain unchanged from previous version)
def find_order_blocks(df, bias, tf_string):
    """
    Identifies potential Order Blocks (basic implementation).
    Looks for the last opposite candle before a larger move (displacement)
    in the direction that aligns with the identified OB type (opposite to bias).
    Args:
        df (pd.DataFrame): Price data (e.g., H4, H1). Needs OHLC columns.
        bias (str): 'Bullish' or 'Bearish'. Helps identify relevant OBs.
        tf_string (str): The timeframe string (e.g., 'H4') for logging/tagging.
    Returns:
        list: List of potential POI dictionaries.
    """
    log_info(f"Task: Identifying potential Order Blocks on {tf_string}...")
    potential_obs = []
    min_move_factor = 1.5
    continuation_check = 1

    if len(df) < (2 + continuation_check):
        log_info("Result: Not enough candles to find OBs.", "WARN")
        return []

    opens = df['Open'].to_numpy(dtype=float, na_value=np.nan)
    highs = df['High'].to_numpy(dtype=float, na_value=np.nan)
    lows = df['Low'].to_numpy(dtype=float, na_value=np.nan)
    closes = df['Close'].to_numpy(dtype=float, na_value=np.nan)

    for i in range(len(df) - (1 + continuation_check)):
        try:
            ob_open = opens[i]; ob_high = highs[i]; ob_low = lows[i]; ob_close = closes[i]
            move_open = opens[i+1]; move_high = highs[i+1]; move_low = lows[i+1]; move_close = closes[i+1]

            if np.isnan([ob_open, ob_high, ob_low, ob_close, move_open, move_high, move_low, move_close]).any(): continue

            ob_is_bullish = ob_close > ob_open
            ob_is_bearish = ob_close < ob_open
            move_is_bullish = move_close > move_open
            move_is_bearish = move_close < move_open

            ob_range = abs(ob_high - ob_low)
            move_range = abs(move_high - move_low)
            is_significant_move = move_range > (ob_range * min_move_factor) if ob_range > 1e-9 else move_range > 0.0001

            move_continued = False
            if is_significant_move:
                cont_start_idx = i + 2; cont_end_idx = i + 2 + continuation_check
                if cont_end_idx <= len(df):
                    if np.isnan(highs[cont_start_idx : cont_end_idx]).any() or np.isnan(lows[cont_start_idx : cont_end_idx]).any(): continue
                    if move_is_bullish:
                        if np.nanmax(highs[cont_start_idx : cont_end_idx]) > move_high: move_continued = True
                    elif move_is_bearish:
                        if np.nanmin(lows[cont_start_idx : cont_end_idx]) < move_low: move_continued = True

            poi_type = None
            if ob_is_bearish and move_is_bullish and is_significant_move and move_continued: poi_type = 'Bullish OB'
            elif ob_is_bullish and move_is_bearish and is_significant_move and move_continued: poi_type = 'Bearish OB'

            if poi_type:
                potential_obs.append({'timestamp': df.index[i], 'poi_level_top': float(ob_high), 'poi_level_bottom': float(ob_low), 'type': poi_type, 'source_tf': tf_string})
        except IndexError: log_info(f"WARN: Index error during OB check near index {i} on {tf_string}.", "WARN"); continue
        except Exception as e: log_info(f"ERROR: Error checking OB at index {i} on {tf_string}: {e}", "ERROR"); continue

    log_info(f"Result: Found {len(potential_obs)} potential OB(s) on {tf_string}.")
    return potential_obs

def find_imbalances(df, tf_string):
    """
    Identifies Fair Value Gaps (FVGs) / Imbalances using standard 3-candle pattern.
    Args:
        df (pd.DataFrame): Price data. Needs High, Low columns.
        tf_string (str): The timeframe string (e.g., 'H1') for logging/tagging.
    Returns:
        list: List of potential FVG POI dictionaries.
    """
    log_info(f"Task: Identifying Imbalances (FVGs) on {tf_string}...")
    potential_fvgs = []
    if len(df) < 3: log_info("Result: Not enough candles to find FVGs.", "WARN"); return []

    highs = df['High'].to_numpy(dtype=float, na_value=np.nan)
    lows = df['Low'].to_numpy(dtype=float, na_value=np.nan)

    for i in range(1, len(df) - 1):
        try:
            prev_high = highs[i-1]; prev_low = lows[i-1]
            next_high = highs[i+1]; next_low = lows[i+1]
            if np.isnan([prev_high, prev_low, next_high, next_low]).any(): continue

            # Bullish FVG: Low of next candle is higher than High of previous candle
            if next_low > prev_high and (next_low - prev_high) > 1e-9:
                potential_fvgs.append({'timestamp': df.index[i], 'poi_level_top': float(next_low), 'poi_level_bottom': float(prev_high), 'type': 'Bullish FVG', 'source_tf': tf_string})
            # Bearish FVG: High of next candle is lower than Low of previous candle
            elif next_high < prev_low and (prev_low - next_high) > 1e-9:
                 potential_fvgs.append({'timestamp': df.index[i], 'poi_level_top': float(prev_low), 'poi_level_bottom': float(next_high), 'type': 'Bearish FVG', 'source_tf': tf_string})
        except IndexError: log_info(f"WARN: Index error during FVG check near index {i} on {tf_string}.", "WARN"); continue
        except Exception as e: log_info(f"ERROR: Error checking FVG at index {i} on {tf_string}: {e}", "ERROR"); continue

    log_info(f"Result: Found {len(potential_fvgs)} potential FVG(s) on {tf_string}.")
    return potential_fvgs

def _check_bos_after(df, break_level, start_check_idx, look_forward, check_above=True):
    """ Checks for a candle body close break after a specific index. """
    if start_check_idx >= len(df) or break_level is None: return False
    end_check_idx = min(start_check_idx + look_forward, len(df))
    if 'Close' not in df.columns: return False
    closes = pd.to_numeric(df['Close'], errors='coerce').values

    for i in range(start_check_idx, end_check_idx):
        try:
            if i >= len(closes): break
            close_price = closes[i]
            if np.isnan(close_price): continue
            if check_above and close_price > break_level: return True
            if not check_above and close_price < break_level: return True
        except IndexError: break
        except Exception as e: log_info(f"ERROR: Error in _check_bos_after at index {i}: {e}", "ERROR"); continue
    return False

def find_breaker_blocks(df, structure_points, tf_string):
    """
    Identifies potential Breaker Blocks based on failed ('Weak') structure points.
    Args:
        df (pd.DataFrame): Price data (OHLC) for the timeframe to find breakers on.
        structure_points (list): Output from market structure analyzer containing points
                                 with 'type' like 'Weak High', 'Weak Low', and 'timestamp'.
        tf_string (str): The timeframe string (e.g., 'H4') for logging/tagging.
    Returns:
        list: List of potential Breaker POI dictionaries.
    """
    log_info(f"Task: Identifying Breaker Blocks on {tf_string}...")
    potential_breakers = []
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not structure_points or df.empty or not isinstance(df.index, pd.DatetimeIndex) or not all(col in df.columns for col in required_cols):
        log_info("Result: Missing structure points, price data, DatetimeIndex, or OHLC columns for Breaker check.", "WARN")
        return []

    df_tz = df.index.tz
    if df_tz is None:
        log_info(f"WARN: DataFrame for {tf_string} is timezone-naive. Assuming UTC.", "WARN")
        try: df = df.tz_localize('UTC') if df.index.tz is None else df.tz_convert('UTC'); df_tz = df.index.tz
        except Exception as e: log_info(f"ERROR: Failed to localize DataFrame index for Breaker check: {e}", "ERROR"); return []

    try:
        tz_aware_structure_points = []
        for p in structure_points:
             p_copy = p.copy(); ts = p_copy.get('timestamp')
             if isinstance(ts, pd.Timestamp):
                 if ts.tzinfo is None: p_copy['timestamp'] = ts.tz_localize(df_tz)
                 elif ts.tzinfo != df_tz: p_copy['timestamp'] = ts.tz_convert(df_tz)
                 tz_aware_structure_points.append(p_copy)
             else: log_info(f"WARN: Invalid timestamp in structure point: {p}", "WARN")
        structure_points = tz_aware_structure_points
    except Exception as tz_err: log_info(f"WARN: Error converting structure point timezones: {tz_err}.", "WARN")

    weak_highs = [p for p in structure_points if isinstance(p, dict) and p.get('type') == 'Weak High']
    weak_lows = [p for p in structure_points if isinstance(p, dict) and p.get('type') == 'Weak Low']

    # --- Find Bearish Breakers (Acts as Support after break UP) ---
    for wh in weak_highs:
        wh_ts = wh.get('timestamp'); wh_price = wh.get('price')
        if not isinstance(wh_ts, pd.Timestamp) or wh_price is None: continue
        if wh_ts not in df.index: continue
        try: wh_idx = df.index.get_loc(wh_ts)
        except KeyError: log_info(f"WARN: Weak High timestamp {wh_ts} not found in {tf_string} index.", "WARN"); continue

        bos_up_confirmed = _check_bos_after(df, wh_price, wh_idx + 1, look_forward=20, check_above=True)
        if not bos_up_confirmed: continue
        log_info(f"Found Weak High at {wh_ts} broken upwards (potential Bearish Breaker setup).")

        breaker_candle_data = None
        for j in range(wh_idx - 1, max(-1, wh_idx - 10), -1):
             if j < 0: break
             try:
                 candle = df.iloc[j]
                 if pd.notna(candle['Close']) and pd.notna(candle['Open']) and candle['Close'] < candle['Open']:
                     breaker_candle_data = {'timestamp': df.index[j], 'poi_level_top': float(candle['High']), 'poi_level_bottom': float(candle['Low'])}
                     log_info(f"  -> Found origin down candle before Weak High at {df.index[j]}")
                     break
             except Exception as e: log_info(f"Error checking candle at index {j} for breaker: {e}", "ERROR"); continue

        if breaker_candle_data is not None:
            potential_breakers.append({**breaker_candle_data, 'type': 'Bearish Breaker', 'source_tf': tf_string, 'origin_weak_point_ts': wh_ts})
            log_info(f"Identified potential Bearish Breaker POI from candle at {breaker_candle_data['timestamp']}")

    # --- Find Bullish Breakers (Acts as Resistance after break DOWN) ---
    for wl in weak_lows:
        wl_ts = wl.get('timestamp'); wl_price = wl.get('price')
        if not isinstance(wl_ts, pd.Timestamp) or wl_price is None: continue
        if wl_ts not in df.index: continue
        try: wl_idx = df.index.get_loc(wl_ts)
        except KeyError: log_info(f"WARN: Weak Low timestamp {wl_ts} not found in {tf_string} index.", "WARN"); continue

        bos_down_confirmed = _check_bos_after(df, wl_price, wl_idx + 1, look_forward=20, check_above=False)
        if not bos_down_confirmed: continue
        log_info(f"Found Weak Low at {wl_ts} broken downwards (potential Bullish Breaker setup).")

        breaker_candle_data = None
        for j in range(wl_idx - 1, max(-1, wl_idx - 10), -1):
             if j < 0: break
             try:
                 candle = df.iloc[j]
                 if pd.notna(candle['Close']) and pd.notna(candle['Open']) and candle['Close'] > candle['Open']:
                     breaker_candle_data = {'timestamp': df.index[j], 'poi_level_top': float(candle['High']), 'poi_level_bottom': float(candle['Low'])}
                     log_info(f"  -> Found origin up candle before Weak Low at {df.index[j]}")
                     break
             except Exception as e: log_info(f"Error checking candle at index {j} for breaker: {e}", "ERROR"); continue

        if breaker_candle_data is not None:
            potential_breakers.append({**breaker_candle_data, 'type': 'Bullish Breaker', 'source_tf': tf_string, 'origin_weak_point_ts': wl_ts})
            log_info(f"Identified potential Bullish Breaker POI from candle at {breaker_candle_data['timestamp']}")

    log_info(f"Result: Found {len(potential_breakers)} potential Breaker Block(s) on {tf_string}.")
    return potential_breakers


def find_rejection_wicks(df, tf_string, min_wick_ratio=0.5, min_body_multiple=2.0):
    """
    Identifies potential Rejection Wicks based on wick vs body/range size
    and protrusion beyond the previous candle.
    Args:
        df (pd.DataFrame): Price data (likely HTF like D1/H4). Needs OHLC.
        tf_string (str): The timeframe string (e.g., 'D1') for logging/tagging.
        min_wick_ratio (float): Minimum wick size as a fraction of total candle range.
        min_body_multiple (float): Minimum wick size as a multiple of body size (if body > 0).
    Returns:
        list: List of potential Wick POI dictionaries.
    """
    log_info(f"Task: Identifying Rejection Wicks on {tf_string}...")
    potential_wicks = []
    if len(df) < 2: log_info("Result: Not enough candles to find Rejection Wicks.", "WARN"); return []

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        log_info(f"ERROR: Missing required columns for wick detection: {required_cols}", "ERROR")
        return []

    high = pd.to_numeric(df['High'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    low = pd.to_numeric(df['Low'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    open_ = pd.to_numeric(df['Open'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)
    close = pd.to_numeric(df['Close'], errors='coerce').to_numpy(dtype=float, na_value=np.nan)

    prev_high = np.roll(high, 1); prev_high[0] = np.nan
    prev_low = np.roll(low, 1); prev_low[0] = np.nan

    body_size = np.abs(close - open_)
    total_range = high - low
    upper_wick = np.where(np.isnan(high) | np.isnan(open_) | np.isnan(close), 0.0, high - np.maximum(open_, close))
    lower_wick = np.where(np.isnan(low) | np.isnan(open_) | np.isnan(close), 0.0, np.minimum(open_, close) - low)

    body_size_safe = np.where(np.isnan(body_size) | (body_size < 1e-9), 1e-9, body_size)
    total_range_safe = np.where(np.isnan(total_range) | (total_range < 1e-9), 1e-9, total_range)
    upper_wick = np.nan_to_num(upper_wick, nan=0.0)
    lower_wick = np.nan_to_num(lower_wick, nan=0.0)

    with np.errstate(invalid='ignore'):
        significant_uw = (upper_wick > (total_range_safe * min_wick_ratio)) & \
                         (upper_wick > (body_size_safe * min_body_multiple)) & \
                         (high > prev_high)
        significant_lw = (lower_wick > (total_range_safe * min_wick_ratio)) & \
                         (lower_wick > (body_size_safe * min_body_multiple)) & \
                         (low < prev_low)

    indices_uw = np.where(significant_uw)[0]
    indices_lw = np.where(significant_lw)[0]

    for i in indices_uw:
        if not np.isnan(high[i]) and not np.isnan(open_[i]) and not np.isnan(close[i]):
            potential_wicks.append({'timestamp': df.index[i], 'poi_level_top': float(high[i]), 'poi_level_bottom': float(max(open_[i], close[i])), 'type': 'Bearish Rejection Wick', 'source_tf': tf_string})

    for i in indices_lw:
         if not np.isnan(low[i]) and not np.isnan(open_[i]) and not np.isnan(close[i]):
             potential_wicks.append({'timestamp': df.index[i], 'poi_level_top': float(min(open_[i], close[i])), 'poi_level_bottom': float(low[i]), 'type': 'Bullish Rejection Wick', 'source_tf': tf_string})

    log_info(f"Result: Found {len(potential_wicks)} potential Wick(s) on {tf_string}.")
    return potential_wicks


# --- Validation Helper Functions ---

def is_mitigated(poi, df_subsequent):
    """
    Checks if a POI zone has been mitigated.
    Refined Logic: Checks if any subsequent candle CLOSES beyond the 50% midpoint.
    """
    if df_subsequent.empty: return False
    try:
        poi_top = float(poi['poi_level_top']); poi_bottom = float(poi['poi_level_bottom'])
        poi_type = poi.get('type', '') # Get POI type for direction
        if poi_top < poi_bottom: poi_top, poi_bottom = poi_bottom, poi_top
        if poi_top == poi_bottom: poi_mid = poi_top
        else: poi_mid = poi_bottom + (poi_top - poi_bottom) / 2.0

        closes_sub = pd.to_numeric(df_subsequent['Close'], errors='coerce')
        if np.isnan(poi_mid): return True # Treat as mitigated if POI levels were bad

        valid_closes = closes_sub.dropna()
        if valid_closes.empty: return False # No valid subsequent closes to check

        # Check for close beyond midpoint based on POI type
        if 'Bullish' in poi_type: # For Bullish POI, mitigation = close below midpoint
            mitigated = (valid_closes <= poi_mid).any()
        elif 'Bearish' in poi_type: # For Bearish POI, mitigation = close above midpoint
            mitigated = (valid_closes >= poi_mid).any()
        else: # Default/Unknown POI type - use original logic? Or assume not mitigated? Let's be conservative.
             lows_sub = pd.to_numeric(df_subsequent['Low'], errors='coerce')
             highs_sub = pd.to_numeric(df_subsequent['High'], errors='coerce')
             valid_subsequent = ~np.isnan(lows_sub) & ~np.isnan(highs_sub)
             if not valid_subsequent.any(): return False
             mitigated = ((lows_sub[valid_subsequent] <= poi_mid) & (highs_sub[valid_subsequent] >= poi_mid)).any()

        return mitigated
    except Exception as e: log_info(f"ERROR: Exception during mitigation check: {e}", "ERROR"); return True


# --- Main POI Manager Function ---

def find_and_validate_smc_pois(all_tf_data, structure_data, inducement_result, parameters=None):
    """
    Identifies potential SMC POIs and validates them based on structure,
    liquidity (inducement), and Fibonacci confluence.
    """
    log_info("--- Running POI Manager SMC ---")
    validated_pois = []
    default_poi_params = {
        'poi_identification_tf': ['h4', 'h1'], 'require_inducement': True,
        'require_fib_filter': True, 'fib_filter_params': None
    }
    if parameters is None: parameters = default_poi_params
    else:
        for key, value in default_poi_params.items(): parameters.setdefault(key, value)
        parameters['poi_identification_tf'] = [tf.lower() for tf in parameters['poi_identification_tf']]

    log_info(f"Parameters: POI TFs={[tf.upper() for tf in parameters['poi_identification_tf']]}, Req Inducement={parameters['require_inducement']}, Req FibFilter={parameters['require_fib_filter']}")

    if structure_data is None: structure_data = {}; log_info("WARN: Structure data is None.", "WARN")
    if inducement_result is None: inducement_result = {}; log_info("WARN: Inducement result is None.", "WARN")
    if all_tf_data is None: all_tf_data = {}; log_info("WARN: All TF data is None.", "WARN")

    htf_bias = structure_data.get('htf_bias', 'Uncertain')
    valid_trading_range_data = structure_data.get('valid_trading_range')
    structure_points = structure_data.get('structure_points', []) # Used for Breakers

    log_info(f"Context: HTF Bias = {htf_bias}")
    if valid_trading_range_data and isinstance(valid_trading_range_data.get('start'), dict) and isinstance(valid_trading_range_data.get('end'), dict):
         log_info(f"Context: Valid Trading Range Found (Type: {valid_trading_range_data.get('type')})")
    else:
         log_info("Context: No Valid Trading Range Found in structure data."); valid_trading_range_data = None

    inducement_swept = inducement_result.get('status', False)
    inducement_sweep_time = inducement_result.get('sweep_candle', {}).get('timestamp') if inducement_swept else None
    log_info(f"Context: Inducement Status = {inducement_swept}" + (f" at {inducement_sweep_time}" if inducement_swept else ""))

    if parameters['require_inducement'] and not inducement_swept:
        log_info("Requirement: Inducement required but not detected. No POIs will be validated.")
        log_info("--- POI Manager Finished: Found 0 valid POI(s) ---")
        return []

    potential_pois = []
    poi_tf_list = parameters['poi_identification_tf']
    log_info(f"Task: Identifying potential POIs on TFs: {[tf.upper() for tf in poi_tf_list]}")
    for tf_key in poi_tf_list:
        if tf_key in all_tf_data and isinstance(all_tf_data[tf_key], pd.DataFrame) and not all_tf_data[tf_key].empty:
            df_tf = all_tf_data[tf_key]
            tf_string = tf_key.upper()
            potential_pois.extend(find_order_blocks(df_tf, htf_bias, tf_string))
            potential_pois.extend(find_imbalances(df_tf, tf_string))
            potential_pois.extend(find_breaker_blocks(df_tf, structure_points, tf_string))
            potential_pois.extend(find_rejection_wicks(df_tf, tf_string))
        else: log_info(f"Skipping POI identification on {tf_key.upper()}: Data not available or empty.", "WARN")

    log_info(f"Identified {len(potential_pois)} total potential POIs across specified TFs.")
    if not potential_pois: log_info("--- POI Manager Finished: Found 0 valid POI(s) ---"); return []

    potential_pois = [p for p in potential_pois if isinstance(p.get('timestamp'), pd.Timestamp) and not np.isnan(p.get('poi_level_top', np.nan)) and not np.isnan(p.get('poi_level_bottom', np.nan))]
    valid_timestamp_pois = [p for p in potential_pois if pd.notna(p.get('timestamp'))]
    invalid_timestamp_pois = [p for p in potential_pois if pd.isna(p.get('timestamp'))]
    if invalid_timestamp_pois: log_info(f"WARN: Found {len(invalid_timestamp_pois)} POIs with invalid timestamps.", "WARN")
    valid_timestamp_pois.sort(key=lambda x: x['timestamp'], reverse=True)
    potential_pois = valid_timestamp_pois

    log_info("Task: Validating potential POIs...")
    final_valid_pois = []
    processed_poi_zones = set()

    for poi in potential_pois:
        if not isinstance(poi, dict) or not all(k in poi for k in ('timestamp', 'poi_level_top', 'poi_level_bottom', 'type', 'source_tf')):
            log_info(f"Skipping invalid POI structure: {poi}", "WARN"); continue
        try: poi_key = f"{poi.get('type')}_{float(poi.get('poi_level_bottom')):.5f}_{float(poi.get('poi_level_top')):.5f}"
        except: poi_key = str(poi)
        if poi_key in processed_poi_zones: continue
        processed_poi_zones.add(poi_key)

        poi_timestamp = poi.get('timestamp'); poi_type = poi.get('type', 'Unknown POI'); poi_tf = poi.get('source_tf', 'Unknown')
        if not isinstance(poi_timestamp, pd.Timestamp) or pd.isna(poi_timestamp):
            log_info(f"Skipping validation for POI with invalid timestamp: {poi}", "WARN")
            continue

        log_info(f"--- Validating POI: {poi_type} ({poi_tf}) at {poi_timestamp} [{poi.get('poi_level_bottom'):.5f}-{poi.get('poi_level_top'):.5f}] ---")
        validation_passed = True; validation_reasons = []

        # 1. Inducement Timing Check
        if inducement_swept and isinstance(inducement_sweep_time, pd.Timestamp):
            if poi_timestamp.tzinfo is None and inducement_sweep_time.tzinfo is not None: poi_timestamp = poi_timestamp.tz_localize(inducement_sweep_time.tzinfo)
            elif inducement_sweep_time.tzinfo is None and poi_timestamp.tzinfo is not None: inducement_sweep_time = inducement_sweep_time.tz_localize(poi_timestamp.tzinfo)
            elif poi_timestamp.tzinfo is None and inducement_sweep_time.tzinfo is None: poi_timestamp = poi_timestamp.tz_localize('UTC'); inducement_sweep_time = inducement_sweep_time.tz_localize('UTC')
            elif poi_timestamp.tzinfo != inducement_sweep_time.tzinfo:
                try: inducement_sweep_time = inducement_sweep_time.tz_convert(poi_timestamp.tzinfo)
                except Exception as tz_err: log_info(f"WARN: Timezone conversion failed for inducement check: {tz_err}", "WARN"); validation_passed = False; validation_reasons.append("TZ Conversion Error")
            if validation_passed and poi_timestamp <= inducement_sweep_time:
                log_info("Validation FAIL: POI formed before or during inducement sweep."); validation_reasons.append("POI Before Inducement"); validation_passed = False
            elif validation_passed: log_info("Validation PASS: POI formed after inducement sweep.")
        elif inducement_swept: log_info("WARN: Cannot perform inducement timing check - POI timestamp invalid.", "WARN")

        # 2. Mitigation Check (Uses refined logic checking CLOSE beyond midpoint)
        if validation_passed:
            poi_source_tf_key = poi_tf.lower()
            if poi_source_tf_key and poi_source_tf_key in all_tf_data:
                df_poi_tf = all_tf_data[poi_source_tf_key]
                if isinstance(poi_timestamp, pd.Timestamp) and poi_timestamp in df_poi_tf.index:
                    poi_idx_loc = df_poi_tf.index.get_loc(poi_timestamp)
                    df_subsequent = df_poi_tf.iloc[poi_idx_loc + 1:]
                    if is_mitigated(poi, df_subsequent): # Calls updated is_mitigated
                        log_info("Validation FAIL: POI appears mitigated (close beyond 50%)."); validation_passed = False; validation_reasons.append("Mitigated")
                    else: log_info("Validation PASS: POI is unmitigated (close beyond 50%).")
                else: log_info(f"WARN: Could not find POI timestamp {poi_timestamp} in source TF {poi_tf} for mitigation check.")
            else: log_info(f"WARN: Could not perform mitigation check - POI source TF '{poi_tf}' data unavailable.")

        # 3. Bias Alignment Check
        if validation_passed:
            if htf_bias != 'Uncertain':
                bullish_poi_types = ['Bullish OB', 'Bullish FVG', 'Bullish Breaker', 'Demand', 'Bullish Rejection Wick']
                bearish_poi_types = ['Bearish OB', 'Bearish FVG', 'Bearish Breaker', 'Supply', 'Bearish Rejection Wick']
                is_bullish_poi_type = any(ptype in poi_type for ptype in bullish_poi_types)
                is_bearish_poi_type = any(ptype in poi_type for ptype in bearish_poi_types)
                if (htf_bias == 'Bullish' and is_bearish_poi_type) or (htf_bias == 'Bearish' and is_bullish_poi_type):
                    log_info(f"Validation FAIL: POI type '{poi_type}' contradicts HTF Bias '{htf_bias}'."); validation_passed = False; validation_reasons.append("Bias Conflict")
                elif (htf_bias == 'Bullish' and not is_bullish_poi_type) or (htf_bias == 'Bearish' and not is_bearish_poi_type):
                    if 'FVG' not in poi_type: log_info(f"Validation FAIL: POI type '{poi_type}' does not align with expected types for HTF Bias '{htf_bias}'."); validation_passed = False; validation_reasons.append("Bias Type Mismatch")
                    else: log_info(f"Validation NOTE: POI type '{poi_type}' is neutral, bias check passed by default.")
                else: log_info(f"Validation PASS: POI type '{poi_type}' aligns with HTF Bias '{htf_bias}'.")
            else: log_info("Skipping Bias Alignment check: HTF Bias is Uncertain.")

        # 4. Fibonacci Filter Check
        if validation_passed and parameters['require_fib_filter'] and FIB_FILTER_LOADED:
            if valid_trading_range_data:
                try:
                    start_price = float(valid_trading_range_data['start']['price']); end_price = float(valid_trading_range_data['end']['price'])
                    swing_range_input = {'swing_high_price': max(start_price, end_price), 'swing_low_price': min(start_price, end_price), 'source_timeframe': valid_trading_range_data.get('start',{}).get('source_tf', 'HTF')}
                    poi_input_for_fib = {'poi_level_top': float(poi['poi_level_top']), 'poi_level_bottom': float(poi['poi_level_bottom'])}
                    fib_result = apply_fibonacci_filter(swing_range_input, poi_input_for_fib, htf_bias, parameters.get('fib_filter_params'))
                    if not fib_result['is_valid_poi']:
                        log_info(f"Validation FAIL: POI failed Fibonacci Filter ({fib_result['filter_reason']})."); validation_passed = False; validation_reasons.append(f"FibFilter Fail ({fib_result['filter_reason']})")
                except Exception as fib_err: log_info(f"ERROR: Fibonacci Filter execution failed: {fib_err}", "ERROR"); validation_passed = False; validation_reasons.append("FibFilter Error")
            else:
                log_info("Skipping Fibonacci Filter: No valid trading range found.", "WARN")
                if parameters['require_fib_filter']: validation_passed = False; validation_reasons.append("FibFilter Skipped (No Range)")
        elif validation_passed and parameters['require_fib_filter'] and not FIB_FILTER_LOADED:
            log_info("Skipping Fibonacci Filter: Module not loaded.", "WARN")
            if parameters['require_fib_filter']: validation_passed = False; validation_reasons.append("FibFilter Skipped (Not Loaded)")

        # Enhanced scoring logic
        score, reasons = score_poi(poi)
        if score < 0.5:
            log_info(f"POI scoring below threshold (score={score}, reasons={reasons}). Skipping.")
            continue
        # --- Store Validated POI ---
        poi['is_valid'] = validation_passed
        poi['validation_reason'] = "Pass" if validation_passed else ", ".join(validation_reasons)
        if validation_passed:
            log_info(f"Outcome: POI at {poi_timestamp} ({poi_type}) is VALID.")
            poi['score'] = 1.0 # Placeholder score
            poi['poi_score'] = score
            poi['poi_reason'] = ", ".join(reasons)
            final_valid_pois.append(poi)
        else:
            log_info(f"Outcome: POI at {poi_timestamp} ({poi_type}) is INVALID ({poi['validation_reason']}).")

    final_valid_pois.sort(key=lambda x: x.get('timestamp', pd.Timestamp.min.replace(tzinfo=timezone.utc)), reverse=True)
    log_info(f"--- POI Manager Finished: Found {len(final_valid_pois)} valid POI(s) ---")
    return final_valid_pois

# --- Example Usage Block ---
if __name__ == '__main__':
    log_info("--- Testing POI Manager SMC (Implemented Identification) ---")
    start_time = pd.Timestamp('2024-01-10 00:00', tz='UTC')
    # Use lowercase 'h' for frequency strings
    h4_index = pd.date_range(start_time, periods=50, freq='4h')
    h1_index = pd.date_range(start_time, periods=50*4, freq='h')
    dummy_h4 = pd.DataFrame({'Open': 105,'High': 106,'Low': 104,'Close': 105.5,'Volume': 1000}, index=h4_index)
    dummy_h1 = pd.DataFrame({'Open': 105,'High': 105.5,'Low': 104.5,'Close': 105.2,'Volume': 250}, index=h1_index)
    dummy_h4['Close'] += np.random.randn(len(dummy_h4)) * 0.5; dummy_h4['High'] = dummy_h4[['Open', 'Close']].max(axis=1) + np.random.rand(len(dummy_h4)) * 0.5; dummy_h4['Low'] = dummy_h4[['Open', 'Close']].min(axis=1) - np.random.rand(len(dummy_h4)) * 0.5
    dummy_h1['Close'] += np.random.randn(len(dummy_h1)) * 0.2; dummy_h1['High'] = dummy_h1[['Open', 'Close']].max(axis=1) + np.random.rand(len(dummy_h1)) * 0.2; dummy_h1['Low'] = dummy_h1[['Open', 'Close']].min(axis=1) - np.random.rand(len(dummy_h1)) * 0.2
    dummy_all_tf_data = {'h4': dummy_h4, 'h1': dummy_h1}
    dummy_structure = { 'htf_bias': 'Bullish', 'valid_trading_range': { 'start': {'price': 100.0, 'timestamp': pd.Timestamp('2024-01-10 04:00', tz='UTC'), 'source_tf': 'H4'}, 'end': {'price': 108.0, 'timestamp': pd.Timestamp('2024-01-11 12:00', tz='UTC'), 'source_tf': 'H4'}, 'type': 'Uptrend'}, 'structure_points': [{'timestamp': pd.Timestamp('2024-01-10 04:00', tz='UTC'), 'price': 100.0, 'type': 'Strong Low'}], 'erl_targets': [], 'discount_premium': {'midpoint': 104.0, 'discount_max': 104.0, 'premium_min': 104.0} }
    dummy_inducement = {'status': True, 'sweep_candle': {'timestamp': pd.Timestamp('2024-01-11 08:00', tz='UTC')}}
    poi_params = {'poi_identification_tf': ['h1'], 'require_inducement': True, 'require_fib_filter': False}

    validated_pois = find_and_validate_smc_pois(dummy_all_tf_data, dummy_structure, dummy_inducement, poi_params)
    print("\n--- POI Manager Results ---")
    if validated_pois:
        print(f"Found {len(validated_pois)} valid POI(s):")
        print(json.dumps(validated_pois, indent=2, default=str))
    else: print("No valid POIs found (Using implemented identification logic).")
    print("\n--- POI Manager Test Complete ---")

