# core/liquidity_sweep_detector.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-29
# Version: 1.0.0 (Fractal Liquidity Sweeps)
# Description:
#   Identifies candles that sweep liquidity above recent fractal highs
#   or below recent fractal lows. Adds columns indicating the price level swept.

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import traceback

# --- Helper Function for Fractals ---
def _tag_fractals(high: pd.Series, low: pd.Series, n: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Identifies simple fractal highs and lows using a rolling window approach.
    A fractal high requires the high to be higher than 'n' bars on both sides.
    A fractal low requires the low to be lower than 'n' bars on both sides.

    Args:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        n (int): Number of bars on each side to define a fractal (default=2 means 5-bar fractal).

    Returns:
        Tuple[pd.Series, pd.Series]: (fractal_highs, fractal_lows)
                                     Series contain the price at the fractal point, NaN otherwise.
    """
    nan_series = pd.Series(np.nan, index=high.index)
    if high.isnull().all() or low.isnull().all() or len(high) < 2*n+1:
        print("[WARN][_tag_fractals] Insufficient data or NaN series for fractal calculation.")
        return nan_series, nan_series

    fractal_high = pd.Series(np.nan, index=high.index)
    fractal_low = pd.Series(np.nan, index=low.index)

    # Check for N bars left and N bars right using rolling window comparisons
    # High Fractals
    is_highest_in_window = high == high.rolling(window=2*n+1, center=True, min_periods=n+1).max()
    fractal_high[is_highest_in_window] = high[is_highest_in_window]

    # Low Fractals
    is_lowest_in_window = low == low.rolling(window=2*n+1, center=True, min_periods=n+1).min()
    fractal_low[is_lowest_in_window] = low[is_lowest_in_window]

    # Refinement: Ensure a point isn't both a high and low fractal simultaneously
    both_fractal = fractal_high.notna() & fractal_low.notna()
    fractal_high[both_fractal] = np.nan
    fractal_low[both_fractal] = np.nan

    return fractal_high, fractal_low

# --- Main Sweep Tagging Function ---
def tag_liquidity_sweeps(
    df: pd.DataFrame,
    tf: str = "Unknown",
    config: Optional[Dict] = None
    ) -> pd.DataFrame:
    """
    Identifies and tags liquidity sweeps of recent fractal highs/lows.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame with DatetimeIndex. Requires 'High', 'Low', 'Close'.
        tf (str): Timeframe string (for logging).
        config (Dict, optional): Configuration dictionary. Keys:
                                 'fractal_n': Bars on each side for fractal ID (default 2).

    Returns:
        pd.DataFrame: Original DataFrame with added columns:
            - 'liq_sweep_fractal_high': Price of the fractal high swept (NaN if no sweep).
            - 'liq_sweep_fractal_low': Price of the fractal low swept (NaN if no sweep).
            # Future: Add 'liq_sweep_high_ext', 'liq_sweep_low_int' etc. based on structure context
    """
    df_out = df.copy()
    log_prefix = f"[LiqSweepDetect][{tf}]" # Add TF to log prefix
    print(f"{log_prefix} Running Liquidity Sweep detection...")

    required_cols = ['High', 'Low', 'Close']
    if not all(col in df_out.columns for col in required_cols):
        print(f"{log_prefix} WARN: Missing required columns (HLC). Skipping.")
        df_out['liq_sweep_fractal_high'] = np.nan
        df_out['liq_sweep_fractal_low'] = np.nan
        return df_out
    if df_out.empty:
        print(f"{log_prefix} WARN: Input DataFrame empty. Skipping.")
        return df_out

    # --- Configuration ---
    if config is None: config = {}
    fractal_n = config.get('fractal_n', 2) # Default to 5-bar fractal (n=2)

    # --- Initialize Columns ---
    df_out['liq_sweep_fractal_high'] = np.nan
    df_out['liq_sweep_fractal_low'] = np.nan

    try:
        # 1. Identify Fractals
        fractal_highs, fractal_lows = _tag_fractals(df_out['High'], df_out['Low'], n=fractal_n)

        # 2. Find the most recent fractal high/low *before* the current bar
        last_fractal_high = fractal_highs.ffill().shift(1)
        last_fractal_low = fractal_lows.ffill().shift(1)

        # 3. Detect Sweeps
        # Ensure last fractals are numeric before comparison
        last_fractal_high_numeric = pd.to_numeric(last_fractal_high, errors='coerce')
        last_fractal_low_numeric = pd.to_numeric(last_fractal_low, errors='coerce')

        # Sweep High: Current High > Previous Fractal High AND Current Close < Previous Fractal High
        sweep_high_mask = (df_out['High'] > last_fractal_high_numeric) & \
                          (df_out['Close'] < last_fractal_high_numeric)
        df_out.loc[sweep_high_mask, 'liq_sweep_fractal_high'] = last_fractal_high_numeric[sweep_high_mask]

        # Sweep Low: Current Low < Previous Fractal Low AND Current Close > Previous Fractal Low
        sweep_low_mask = (df_out['Low'] < last_fractal_low_numeric) & \
                         (df_out['Close'] > last_fractal_low_numeric)
        df_out.loc[sweep_low_mask, 'liq_sweep_fractal_low'] = last_fractal_low_numeric[sweep_low_mask]

        sweep_high_count = sweep_high_mask.sum()
        sweep_low_count = sweep_low_mask.sum()
        print(f"{log_prefix} Completed. Found {sweep_high_count} High Sweeps, {sweep_low_count} Low Sweeps.")

    except Exception as e:
        print(f"{log_prefix} ERROR: Failed during Liquidity Sweep detection: {e}")
        traceback.print_exc()
        # Ensure columns exist even if error occurs
        if 'liq_sweep_fractal_high' not in df_out: df_out['liq_sweep_fractal_high'] = np.nan
        if 'liq_sweep_fractal_low' not in df_out: df_out['liq_sweep_fractal_low'] = np.nan

    return df_out

# --- Example Usage ---
# Removed for operational code
