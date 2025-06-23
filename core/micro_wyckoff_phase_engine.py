# micro_wyckoff_phase_engine.py
# Ultra-Tight Scalping Microstructure Filter (ZANALYTICS v5.1.9+)
# Module repurposed for M1 or tick-based Wyckoff phase inference, including Phase C logic

import pandas as pd
from typing import Optional, Dict, Any


def detect_micro_wyckoff_phase(
    ticks_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect Micro Wyckoff Phase C (Spring + reclaim), LPS, and optional SOS.
    Uses tick or M1 data to infer Wyckoff phases.
    Config can supply 'micro_window', 'micro_buffer_pips', and 'pip_size'.

    -- STRUCTURED MICRO WYCKOFF LOGIC (ZANALYTICS AI PROTOCOL) --

    Signal Types:
      - CHoCH: Higher high + lower low vs previous bar
      - BOS: Current high > rolling max
      - Phase Tag: Close > Open → Accumulation | Close < Open → Distribution

    Interpretation Matrix:
      - CHoCH + no BOS → Liquidity trap
      - CHoCH + BOS → Structural shift
      - Repeated Accum → Pre-breakout load
      - Distribution cluster → Breakdown risk

    Triggers:
      - CHoCH in last 2 bars
      - Spread < 0.3 and tick volume rising
      - BOS absence → trap entry
      - BOS confirm → momentum entry

    Timeframes:
      - M1, M5, M15
    """
    if config is None:
        config = {}
    window = config.get('micro_window', 5)
    buffer = config.get('micro_buffer_pips', 5)

    if len(ticks_df) < window:
        return {"phase": None, "reason": "Not enough tick data"}

    recent_ticks = ticks_df.tail(window).copy()

    # Calculate Mid price and Spread in pips
    recent_ticks['Mid'] = (recent_ticks['<BID>'] + recent_ticks['<ASK>']) / 2
    recent_ticks['Spread'] = (recent_ticks['<ASK>'] - recent_ticks['<BID>']) * 100  # pips

    # Tick Arrival Rate
    recent_ticks['Datetime'] = pd.to_datetime(recent_ticks['<DATE>'] + ' ' + recent_ticks['<TIME>'])
    tick_rate = (recent_ticks['Datetime'].iloc[-1] - recent_ticks['Datetime'].iloc[0]).total_seconds()

    # Placeholder logic for Phase C detection (Spring + reclaim)
    # and LPS detection (higher low + reclaim after Spring)
    # Optional SOS detection (BOS with follow-through) can be added here

    # Example placeholders for swing lows and entry zone
    last_swing_low = recent_ticks['Mid'].min()
    entry_zone = [
        last_swing_low,
        last_swing_low + buffer * config.get('pip_size', 0.0001)
    ]

    atr_col = next((col for col in ticks_df.columns if col.startswith('ATR_')), None)
    atr_val = recent_ticks[atr_col].iloc[-1] if atr_col else None

    # Placeholder return for detected Phase C microstructure
    return {
        "phase": "C",
        "trigger": "micro_spring",
        "confidence": 0.92,
        "entry_zone": entry_zone,
        "mid_last": recent_ticks['Mid'].iloc[-1],
        "spread_avg": recent_ticks['Spread'].mean(),
        "tick_rate_s": round(tick_rate, 2),
        "atr": atr_val,
        "config_window": window,
        "config_buffer": buffer
    }