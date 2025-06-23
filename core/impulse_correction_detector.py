import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Attempt to import TA-Lib, but make it optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN] TA-Lib library not found. EMA/RSI calculations will use pandas equivalents or be skipped.")

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates EMA using pandas if TA-Lib is not available."""
    if TALIB_AVAILABLE:
        return talib.EMA(series, timeperiod=period)
    else:
        # print(f"[DEBUG] Calculating EMA {period} using pandas.")
        return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculates RSI using pandas if TA-Lib is not available."""
    if TALIB_AVAILABLE:
        return talib.RSI(series, timeperiod=period)
    else:
        # print(f"[DEBUG] Calculating RSI {period} using pandas.")
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def detect_impulse_correction_phase(
    price_data: pd.DataFrame,
    structure_data: Optional[Dict] = None,
    liquidity_sweeps: Optional[Dict] = None, # Placeholder for future use
    config: Optional[Dict] = None
) -> Dict:
    """
    Detects whether the market is likely in an Impulse or Correction phase
    based on price action, structure, and optional indicators.

    Args:
        price_data: pd.DataFrame with OHLCV data (e.g., M1, M5, M15). Index must be DatetimeIndex.
        structure_data: Optional output from market_structure_analyzer_smc.py.
                        Expected keys like 'htf_bias', 'structure_points', 'discount_premium'.
        liquidity_sweeps: Optional output from liquidity_engine_smc.py (e.g., recent sweeps).
        config: Optional dictionary for thresholds and parameters:
                'ema_period': int (default 48) - Period for EMA slope calculation.
                'rsi_period': int (default 14) - Period for RSI calculation.
                'impulse_candle_count': int (default 3) - Min consecutive strong candles for impulse.
                'impulse_body_ratio': float (default 0.6) - Min body/range ratio for strong candle.
                'ema_slope_threshold': float (default 0.0001) - Min slope to indicate impulse.
                'correction_rsi_threshold': tuple (default (40, 60)) - RSI range suggesting correction.
                'lookback_candles': int (default 20) - How many recent candles to analyze.

    Returns:
        Dictionary:
        {
          "phase": "Impulse" | "Correction" | "Unknown",
          "basis": List[str] - Reasons for the phase determination.
          "valid_for_entry": bool - True if the phase is suitable for entry (e.g., Correction for Mentfx).
          "error": str | None - Error message if processing failed.
        }
    """
    # Initialize result
    result = {
        "phase": "Unknown",
        "basis": [],
        "valid_for_entry": False, # Default to False
        "error": None
    }

    # --- Input Validation ---
    if price_data is None or price_data.empty or len(price_data) < 10: # Need sufficient data
        result["error"] = "Insufficient price data provided."
        return result
    if not all(col in price_data.columns for col in ['Open', 'High', 'Low', 'Close']):
         result["error"] = "Price data missing required OHLC columns."
         return result
    if not isinstance(price_data.index, pd.DatetimeIndex):
         result["error"] = "Price data index must be a DatetimeIndex."
         return result

    # --- Configuration ---
    if config is None: config = {}
    ema_period = config.get('ema_period', 48)
    rsi_period = config.get('rsi_period', 14)
    impulse_candle_count = config.get('impulse_candle_count', 3)
    impulse_body_ratio = config.get('impulse_body_ratio', 0.6)
    ema_slope_threshold = config.get('ema_slope_threshold', 0.0001) # Needs tuning based on asset/TF
    correction_rsi_low, correction_rsi_high = config.get('correction_rsi_threshold', (40, 60))
    lookback_candles = config.get('lookback_candles', 20)

    # --- Data Preparation ---
    try:
        # Use only the required lookback period for efficiency
        analysis_data = price_data.iloc[-lookback_candles:].copy()
        if len(analysis_data) < max(ema_period, rsi_period, impulse_candle_count, 5): # Ensure enough data for calculations
            result["error"] = f"Lookback ({lookback_candles}) too short for calculations (min needed: {max(ema_period, rsi_period, 5)})."
            return result

        analysis_data['body_size'] = abs(analysis_data['Close'] - analysis_data['Open'])
        analysis_data['range_size'] = analysis_data['High'] - analysis_data['Low']
        analysis_data['body_ratio'] = analysis_data['body_size'] / analysis_data['range_size'].replace(0, np.nan) # Avoid division by zero
        analysis_data['is_bullish'] = analysis_data['Close'] > analysis_data['Open']
        analysis_data['is_bearish'] = analysis_data['Close'] < analysis_data['Open']

        # Calculate Indicators
        analysis_data[f'ema_{ema_period}'] = calculate_ema(analysis_data['Close'], ema_period)
        analysis_data['ema_slope'] = analysis_data[f'ema_{ema_period}'].diff() # Simple slope calculation

        analysis_data[f'rsi_{rsi_period}'] = calculate_rsi(analysis_data['Close'], rsi_period)

    except Exception as e:
        result["error"] = f"Error during data preparation or indicator calculation: {e}"
        return result

    # --- Phase Detection Logic ---
    try:
        # Get data for the most recent candle(s)
        last_candle = analysis_data.iloc[-1]
        prev_candle = analysis_data.iloc[-2] if len(analysis_data) >= 2 else None

        # Get HTF bias if available
        htf_bias = structure_data.get('htf_bias', 'Unknown') if structure_data else 'Unknown'

        # --- Impulse Checks ---
        is_impulsive = False
        impulse_reasons = []

        # 1. Strong Candle Bodies Check
        recent_candles = analysis_data.tail(impulse_candle_count)
        strong_bullish_streak = (recent_candles['is_bullish'] & (recent_candles['body_ratio'] >= impulse_body_ratio)).all()
        strong_bearish_streak = (recent_candles['is_bearish'] & (recent_candles['body_ratio'] >= impulse_body_ratio)).all()

        if strong_bullish_streak:
            impulse_reasons.append(f"{impulse_candle_count}+ strong bullish candles")
            if htf_bias in ['Bullish', 'Unknown']: is_impulsive = True # Align with bias or unknown
        if strong_bearish_streak:
            impulse_reasons.append(f"{impulse_candle_count}+ strong bearish candles")
            if htf_bias in ['Bearish', 'Unknown']: is_impulsive = True # Align with bias or unknown

        # 2. EMA Slope Check
        last_ema_slope = last_candle['ema_slope']
        if not pd.isna(last_ema_slope):
            if last_ema_slope > ema_slope_threshold:
                impulse_reasons.append(f"EMA{ema_period} slope positive > {ema_slope_threshold}")
                if htf_bias in ['Bullish', 'Unknown']: is_impulsive = True
            elif last_ema_slope < -ema_slope_threshold:
                impulse_reasons.append(f"EMA{ema_period} slope negative < {-ema_slope_threshold}")
                if htf_bias in ['Bearish', 'Unknown']: is_impulsive = True

        # 3. Structure Break Check (Requires structure_data)
        # TODO: Check if the recent move broke a significant swing high/low from structure_data
        # Example: Check if last_candle['High'] > last_weak_high from structure_data
        # if structure_data and recent_break_occurred:
        #     impulse_reasons.append("Recent structure break (BOS/CHoCH)")
        #     is_impulsive = True

        # --- Correction Checks ---
        is_corrective = False
        correction_reasons = []

        # 1. Small Candle Bodies / Low Momentum
        recent_body_ratios = analysis_data['body_ratio'].tail(5).mean() # Avg ratio over last 5 bars
        if not pd.isna(recent_body_ratios) and recent_body_ratios < (impulse_body_ratio / 2): # Significantly smaller bodies
             correction_reasons.append("Small avg body ratio recently")
             is_corrective = True

        # 2. Price in Discount/Premium or HTF POI (Requires structure_data)
        if structure_data and 'discount_premium' in structure_data and structure_data['discount_premium']:
            dp_info = structure_data['discount_premium']
            current_price = last_candle['Close']
            if dp_info.get('midpoint'):
                 if htf_bias == 'Bullish' and current_price < dp_info['midpoint']:
                     correction_reasons.append("Price in Discount zone")
                     is_corrective = True
                 elif htf_bias == 'Bearish' and current_price > dp_info['midpoint']:
                     correction_reasons.append("Price in Premium zone")
                     is_corrective = True
        # TODO: Add check if price is inside a known HTF POI range

        # 3. RSI indicating consolidation/ranging
        last_rsi = last_candle[f'rsi_{rsi_period}']
        if not pd.isna(last_rsi) and correction_rsi_low <= last_rsi <= correction_rsi_high:
             correction_reasons.append(f"RSI ({last_rsi:.1f}) between {correction_rsi_low}-{correction_rsi_high}")
             is_corrective = True
        # TODO: Add RSI divergence check? Requires more historical data/logic.

        # 4. Time Decay (Simple Placeholder)
        # TODO: Implement time decay logic (e.g., track time since last impulse peak/trough)
        # time_since_last_impulse = ...
        # if time_since_last_impulse > pd.Timedelta(minutes=10):
        #     correction_reasons.append("Time decay since last impulse")
        #     is_corrective = True


        # --- Determine Final Phase ---
        if is_impulsive and not is_corrective:
            result["phase"] = "Impulse"
            result["basis"] = impulse_reasons
            # Typically don't want to enter during strong impulse unless specific breakout strategy
            result["valid_for_entry"] = False
        elif is_corrective and not is_impulsive:
            result["phase"] = "Correction"
            result["basis"] = correction_reasons
            # Correction phase is often preferred for retracement entries (Mentfx, TMC)
            result["valid_for_entry"] = True
        elif is_impulsive and is_corrective:
            # Conflicting signals - phase is uncertain
            result["phase"] = "Unknown"
            result["basis"] = ["Conflicting Impulse/Correction signals"] + impulse_reasons + correction_reasons
            result["valid_for_entry"] = False
        else:
            # No strong signals either way
            result["phase"] = "Unknown"
            result["basis"] = ["No clear Impulse or Correction signals"]
            result["valid_for_entry"] = False

        # Override valid_for_entry based on specific strategy needs if necessary
        # Example: if strategy == 'Breakout' and result['phase'] == 'Impulse': result['valid_for_entry'] = True


    except Exception as e:
        import traceback
        print(f"[CRITICAL] Exception in detect_impulse_correction_phase: {e}\n{traceback.format_exc()}")
        result["error"] = f"Runtime error: {e}"
        result["phase"] = "Unknown"
        result["basis"] = []
        result["valid_for_entry"] = False


    return result


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing Impulse/Correction Detector ---")

    # Create dummy data
    timestamps = pd.date_range(start='2023-10-27 09:00:00', periods=100, freq='T', tz='UTC')
    base_price = 1.1000
    data = {
        'Open': base_price + np.random.normal(0, 0.0002, 100),
        'High': base_price + np.random.normal(0.0003, 0.0002, 100),
        'Low': base_price + np.random.normal(-0.0003, 0.0002, 100),
        'Close': base_price + np.random.normal(0, 0.0002, 100),
        'Volume': np.random.randint(100, 1000, 100)
    }
    dummy_df = pd.DataFrame(data, index=timestamps)
    # Ensure High >= Open/Close and Low <= Open/Close
    dummy_df['High'] = dummy_df[['High', 'Open', 'Close']].max(axis=1)
    dummy_df['Low'] = dummy_df[['Low', 'Open', 'Close']].min(axis=1)

    # --- Simulate Impulse ---
    impulse_start_idx = 50
    impulse_end_idx = 60
    for i in range(impulse_start_idx, impulse_end_idx):
        dummy_df.loc[timestamps[i], 'Open'] = dummy_df.loc[timestamps[i-1], 'Close']
        dummy_df.loc[timestamps[i], 'Close'] = dummy_df.loc[timestamps[i], 'Open'] + 0.0005 # Strong bullish close
        dummy_df.loc[timestamps[i], 'High'] = dummy_df.loc[timestamps[i], 'Close'] + 0.0001
        dummy_df.loc[timestamps[i], 'Low'] = dummy_df.loc[timestamps[i], 'Open'] - 0.0001

    # --- Simulate Correction ---
    correction_start_idx = 60
    correction_end_idx = 75
    for i in range(correction_start_idx, correction_end_idx):
         dummy_df.loc[timestamps[i], 'Open'] = dummy_df.loc[timestamps[i-1], 'Close']
         dummy_df.loc[timestamps[i], 'Close'] = dummy_df.loc[timestamps[i], 'Open'] - 0.0001 # Small bearish close (pullback)
         dummy_df.loc[timestamps[i], 'High'] = dummy_df.loc[timestamps[i], 'Open'] + 0.0001
         dummy_df.loc[timestamps[i], 'Low'] = dummy_df.loc[timestamps[i], 'Close'] - 0.0001


    # Dummy structure data
    dummy_structure = {
        'htf_bias': 'Bullish',
        'discount_premium': {'midpoint': 1.1020} # Example midpoint
    }

    print("\nTesting during simulated Impulse:")
    impulse_test_data = dummy_df.iloc[:impulse_end_idx]
    impulse_result = detect_impulse_correction_phase(impulse_test_data, dummy_structure)
    import json
    print(json.dumps(impulse_result, indent=2))

    print("\nTesting during simulated Correction:")
    correction_test_data = dummy_df.iloc[:correction_end_idx]
    correction_result = detect_impulse_correction_phase(correction_test_data, dummy_structure)
    print(json.dumps(correction_result, indent=2))

    # Optional: push alert
    try:
        from telegram_alert_engine import send_simple_summary_alert
        summary = f"[TEST] Phase: {correction_result['phase']} | Entry OK: {correction_result['valid_for_entry']} | Basis: {', '.join(correction_result['basis'])}"
        send_simple_summary_alert(summary)
    except Exception as e:
        print(f"[TELEGRAM ALERT ERROR] {e}")

    print("\nTesting with insufficient data:")
    short_data = dummy_df.iloc[:10]
    short_result = detect_impulse_correction_phase(short_data, dummy_structure)
    print(json.dumps(short_result, indent=2))


    print("\n--- Testing Complete ---")
