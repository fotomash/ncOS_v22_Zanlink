import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def check_poi_tap_smc(
    price_data: pd.DataFrame,
    poi_range: List[float],
    config: Optional[Dict] = None
) -> Dict:
    """
    Checks if price has tapped (mitigated) a given Point of Interest (POI) range.

    Args:
        price_data: pd.DataFrame with OHLC data, indexed by Timestamp.
                    Should contain recent candles relevant to the POI timeframe.
        poi_range: List or Tuple [poi_low, poi_high].
        config: Optional dictionary for parameters:
                'poi_padding_pct': float (default 0.0) - Percentage to expand POI range.
                                   e.g., 0.2 means expand by 0.2% on each side.
                'lookback_candles': int (default 10) - How many recent candles to check.

    Returns:
        Dictionary:
        {
          "is_tapped": bool - True if POI was tapped within the lookback.
          "tap_time": pd.Timestamp | None - Timestamp of the first candle that tapped.
          "tap_type": str | None - Currently 'Overlap', can be refined (e.g., 'Wick', 'Body').
          "candle_index": int | None - Index of the tapping candle in the input DataFrame.
          "padded_poi_range": List[float] | None - The POI range after padding.
          "error": str | None - Error message if processing failed.
        }
    """
    # Initialize result
    result = {
        "is_tapped": False,
        "tap_time": None,
        "tap_type": None,
        "candle_index": None,
        "padded_poi_range": None,
        "error": None
    }

    # --- Input Validation ---
    if price_data is None or price_data.empty:
        result["error"] = "Price data is missing or empty."
        return result
    if not isinstance(poi_range, (list, tuple)) or len(poi_range) != 2:
        result["error"] = "Invalid POI range format. Expected [low, high]."
        return result
    if not all(col in price_data.columns for col in ['High', 'Low']):
         result["error"] = "Price data missing required High/Low columns."
         return result
    if not isinstance(price_data.index, pd.DatetimeIndex):
         result["error"] = "Price data index must be a DatetimeIndex."
         return result

    # --- Configuration ---
    if config is None: config = {}
    poi_padding_pct = config.get('poi_padding_pct', 0.0)
    lookback_candles = config.get('lookback_candles', 10)

    # Ensure lookback doesn't exceed available data
    lookback_candles = min(lookback_candles, len(price_data))
    if lookback_candles <= 0:
        result["error"] = "Lookback period is zero or negative."
        return result

    # --- Logic ---
    try:
        poi_low, poi_high = min(poi_range), max(poi_range)

        # Apply padding if specified
        if poi_padding_pct > 0:
            poi_size = poi_high - poi_low
            padding_amount = poi_size * (poi_padding_pct / 100.0)
            padded_low = poi_low - padding_amount
            padded_high = poi_high + padding_amount
            result["padded_poi_range"] = [round(padded_low, 5), round(padded_high, 5)] # Use 5 decimals typical for forex
            print(f"[DEBUG][POITap] Original POI: [{poi_low:.5f}, {poi_high:.5f}], Padded POI ({poi_padding_pct}%): [{padded_low:.5f}, {padded_high:.5f}]")
        else:
            padded_low = poi_low
            padded_high = poi_high
            result["padded_poi_range"] = [round(padded_low, 5), round(padded_high, 5)]


        # Check the most recent 'lookback_candles'
        relevant_data = price_data.iloc[-lookback_candles:]

        # Find candles where the candle's range overlaps with the padded POI range
        # Overlap = Candle Low <= POI High AND Candle High >= POI Low
        overlap_mask = (relevant_data['Low'] <= padded_high) & (relevant_data['High'] >= padded_low)

        tapping_candles = relevant_data[overlap_mask]

        if not tapping_candles.empty:
            first_tap_candle = tapping_candles.iloc[0] # The first candle in the lookback that tapped
            first_tap_time = first_tap_candle.name # Timestamp is the index
            # Find the index relative to the original input dataframe
            try:
                 # Get index location within the sliced dataframe first
                 slice_loc = tapping_candles.index.get_loc(first_tap_time)
                 # Map back to the original dataframe's index location
                 # This assumes relevant_data is a direct slice from the end
                 original_loc = len(price_data) - lookback_candles + slice_loc
                 # Alternative: Use get_loc on the original index if timestamps are unique
                 # original_loc = price_data.index.get_loc(first_tap_time) # Might fail if timestamps aren't unique
            except Exception as idx_err:
                 print(f"[WARN][POITap] Could not reliably determine original candle index: {idx_err}")
                 original_loc = None


            result["is_tapped"] = True
            result["tap_time"] = first_tap_time
            result["tap_type"] = "Overlap" # Can be refined later (Wick vs Body)
            result["candle_index"] = original_loc # Store the index if found
            print(f"[INFO][POITap] POI tapped at {first_tap_time} by candle index ~{original_loc}. Type: {result['tap_type']}")

        else:
            print(f"[INFO][POITap] POI range [{padded_low:.5f}, {padded_high:.5f}] not tapped in the last {lookback_candles} candles.")
            result["is_tapped"] = False


    except Exception as e:
        import traceback
        print(f"[ERROR][POITap] Exception in check_poi_tap_smc: {e}\n{traceback.format_exc()}")
        result["error"] = f"Runtime error: {e}"
        result["is_tapped"] = False

    return result

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing POI Hit Watcher (SMC) ---")

    # Create dummy price data
    timestamps = pd.date_range(start='2023-10-27 10:00:00', periods=20, freq='T', tz='UTC')
    data = {
        'Open': np.linspace(1.1020, 1.1030, 20),
        'High': np.linspace(1.1022, 1.1032, 20),
        'Low': np.linspace(1.1018, 1.1028, 20),
        'Close': np.linspace(1.1019, 1.1029, 20),
    }
    dummy_price_df = pd.DataFrame(data, index=timestamps)

    # --- Simulate a tap ---
    tap_index = 15 # Candle at index 15 will tap
    poi_to_tap = [1.1010, 1.1015] # POI below current price
    dummy_price_df.loc[timestamps[tap_index], 'Low'] = 1.1014 # Wick into the POI high
    dummy_price_df.loc[timestamps[tap_index], 'High'] = 1.1025 # Keep high above POI

    print("\nTesting POI Tap (Wick):")
    tap_config = {'lookback_candles': 10, 'poi_padding_pct': 0.0}
    tap_result = check_poi_tap_smc(dummy_price_df, poi_to_tap, config=tap_config)
    print(json.dumps(tap_result, indent=2, default=str)) # Use default=str for Timestamp

    print("\nTesting POI Tap (With Padding):")
    padding_config = {'lookback_candles': 10, 'poi_padding_pct': 0.1} # 0.1% padding
    padding_result = check_poi_tap_smc(dummy_price_df, poi_to_tap, config=padding_config)
    print(json.dumps(padding_result, indent=2, default=str))


    print("\nTesting POI Miss:")
    miss_poi = [1.0900, 1.0910] # POI far below price
    miss_config = {'lookback_candles': 10, 'poi_padding_pct': 0.0}
    miss_result = check_poi_tap_smc(dummy_price_df, miss_poi, config=miss_config)
    print(json.dumps(miss_result, indent=2, default=str))

    print("\n--- Testing Complete ---")

