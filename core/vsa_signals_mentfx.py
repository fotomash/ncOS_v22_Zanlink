# MENTFX-style Volume Spread Analysis (VSA) signal engine
# Includes bar stats, No Supply/Demand, SoS/SoW detection, and structural climax logic.

import pandas as pd
import numpy as np

def calculate_bar_stats(df: pd.DataFrame, index: int, lookback: int = 50) -> dict:
    if index < 1 or index < lookback:
        return {'spread': np.nan, 'close_pos': np.nan, 'spread_percentile': np.nan,
                'volume_percentile': np.nan, 'avg_volume': np.nan}
    current_bar = df.iloc[index]
    lookback_start_idx = max(0, index - lookback)
    spread = current_bar['High'] - current_bar['Low']
    close_pos = (current_bar['Close'] - current_bar['Low']) / spread if spread > 0 else 0.5
    lookback_df = df.iloc[lookback_start_idx : index + 1]
    ranges = lookback_df['High'] - lookback_df['Low']
    volumes = lookback_df['Volume']
    spread_percentile = ranges.rank(pct=True).iloc[-1] * 100
    volume_percentile = volumes.rank(pct=True).iloc[-1] * 100
    avg_volume = df['Volume'].iloc[lookback_start_idx : index].mean()
    return {
        'spread': spread, 'close_pos': close_pos,
        'spread_percentile': spread_percentile,
        'volume_percentile': volume_percentile,
        'avg_volume': avg_volume
    }

def identify_no_supply_bar(df, i, narrow_spread_pctle=30.0, close_pos_threshold=0.4) -> bool:
    if i < 2: return False
    current, prev1, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
    is_down_bar = current['Close'] < current['Open']
    stats = calculate_bar_stats(df, i)
    if pd.isna(stats['spread']): return False
    is_narrow = stats['spread_percentile'] <= narrow_spread_pctle
    closes_high = stats['close_pos'] >= close_pos_threshold
    lower_vol = current['Volume'] < prev1['Volume'] and current['Volume'] < prev2['Volume']
    return is_down_bar and is_narrow and closes_high and lower_vol

def identify_no_demand_bar(df, i, narrow_spread_pctle=30.0, close_pos_threshold=0.6) -> bool:
    if i < 2: return False
    current, prev1, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
    is_up_bar = current['Close'] > current['Open']
    stats = calculate_bar_stats(df, i)
    if pd.isna(stats['spread']): return False
    is_narrow = stats['spread_percentile'] <= narrow_spread_pctle
    closes_low = stats['close_pos'] <= close_pos_threshold
    lower_vol = current['Volume'] < prev1['Volume'] and current['Volume'] < prev2['Volume']
    return is_up_bar and is_narrow and closes_low and lower_vol

def identify_sos_bar(df, i, wide_spread_pctle=70.0, high_close_pos=0.7, high_volume_pctle=70.0) -> bool:
    if i < 1: return False
    current = df.iloc[i]
    is_up_bar = current['Close'] > current['Open']
    stats = calculate_bar_stats(df, i)
    if pd.isna(stats['spread']): return False
    return is_up_bar and stats['spread_percentile'] >= wide_spread_pctle            and stats['close_pos'] >= high_close_pos and stats['volume_percentile'] >= high_volume_pctle

def identify_sow_bar(df, i, wide_spread_pctle=70.0, low_close_pos=0.3, high_volume_pctle=70.0) -> bool:
    if i < 1: return False
    current = df.iloc[i]
    is_down_bar = current['Close'] < current['Open']
    stats = calculate_bar_stats(df, i)
    if pd.isna(stats['spread']): return False
    return is_down_bar and stats['spread_percentile'] >= wide_spread_pctle            and stats['close_pos'] <= low_close_pos and stats['volume_percentile'] >= high_volume_pctle

def find_potential_climaxes(df, lookback_period=50, volume_multiplier=2.5, wide_spread_pctle=80.0) -> list:
    peaks = []
    if len(df) < lookback_period + 1: return peaks
    for i in range(len(df) - 1, len(df) - lookback_period - 1, -1):
        if i < 1: continue
        stats = calculate_bar_stats(df, i, lookback=lookback_period)
        if pd.isna(stats['avg_volume']): continue
        vol_thresh = stats['avg_volume'] * volume_multiplier
        if df['Volume'].iloc[i] > vol_thresh and stats['spread_percentile'] >= wide_spread_pctle:
            peaks.append(df.index[i])
    return peaks

def is_testing_level(df, index, level, tolerance_pct=0.1) -> bool:
    if index < 0 or index >= len(df): return False
    bar = df.iloc[index]
    tol = level * (tolerance_pct / 100.0)
    return level - tol <= bar['Low'] <= level + tol or level - tol <= bar['High'] <= level + tol
