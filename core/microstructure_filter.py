# microstructure_filter.py
# Ultra-Tight Scalping Microstructure Filter (ZANALYTICS v5.1.9+)

import pandas as pd


def evaluate_microstructure(ticks_df, window=5, min_drift_pips=1.0, max_spread=1.8, min_rr=1.8, rr=None):
    """
    Ultra-tight evaluation of tick structure for scalping validity.
    Includes spread, directional slope, volatility, tick rate, flip-count, RR gate.
    """
    if len(ticks_df) < window:
        return {"micro_confirmed": False, "reason": "Not enough tick data"}

    recent_ticks = ticks_df.tail(window).copy()

    recent_ticks['Mid'] = (recent_ticks['<BID>'] + recent_ticks['<ASK>']) / 2
    recent_ticks['Spread'] = (recent_ticks['<ASK>'] - recent_ticks['<BID>']) * 100  # pips

    # Spread Stats
    avg_spread = recent_ticks['Spread'].mean()
    spread_std = recent_ticks['Spread'].std()
    if avg_spread > max_spread:
        return {"micro_confirmed": False, "reason": f"Spread too wide ({avg_spread:.2f} pips)"}
    if spread_std > 0.3:
        return {"micro_confirmed": False, "reason": "Spread volatility too high"}

    # Drift Analysis
    drift = abs(recent_ticks['Mid'].iloc[-1] - recent_ticks['Mid'].iloc[0]) * 100
    if drift < min_drift_pips:
        return {"micro_confirmed": False, "reason": f"Drift too small ({drift:.2f} pips)"}

    # Directional Slope Consistency
    diffs = recent_ticks['Mid'].diff().fillna(0)
    direction = 1 if diffs.sum() > 0 else -1
    consistent_ticks = (diffs * direction > 0).sum()
    if consistent_ticks < int(window * 0.6):
        return {"micro_confirmed": False, "reason": f"Inconsistent direction ({consistent_ticks}/{window})"}

    # Flip Count (Chop Rejection)
    flips = (diffs.diff().fillna(0) * diffs.shift().fillna(0) < 0).sum()
    if flips > 2:
        return {"micro_confirmed": False, "reason": f"Tick direction unstable (flips={flips})"}

    # Tick Arrival Rate
    recent_ticks['Datetime'] = pd.to_datetime(recent_ticks['<DATE>'] + ' ' + recent_ticks['<TIME>'])
    tick_rate = (recent_ticks['Datetime'].iloc[-1] - recent_ticks['Datetime'].iloc[0]).total_seconds()
    if tick_rate > 1.5:
        return {"micro_confirmed": False, "reason": f"Tick rate too slow ({tick_rate:.2f}s)"}

    # RR Validation
    if rr is not None and rr < min_rr:
        return {"micro_confirmed": False, "reason": f"RR too low ({rr:.2f})"}

    return {
        "micro_confirmed": True,
        "tick_drift_pips": round(drift, 2),
        "spread": round(avg_spread, 2),
        "spread_std": round(spread_std, 2),
        "ticks_consistent": int(consistent_ticks),
        "flips": int(flips),
        "tick_rate_s": round(tick_rate, 2),
        "rr": rr if rr else "N/A"
    }


def get_last_m5_pivot(m5_structure) -> float:
    """
    Returns the price level of the last M5 CHoCH low or order-block pivot.
    m5_structure: dict-like object with a 'last_choch_low' entry containing an event with a .price attribute.
    """
    event = m5_structure.get('last_choch_low')
    if event is None:
        raise ValueError("No M5 pivot available yet in microstructure filter.")
    return event.price
