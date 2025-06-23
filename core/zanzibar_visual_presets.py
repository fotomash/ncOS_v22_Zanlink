
from datetime import datetime

def base_metadata(source):
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "source": source
    }

def get_preset_mentfx_fib(symbol="XAUUSD", tf="M15"):
    return {
        "symbol": symbol,
        "timeframe": tf,
        "poi_zones": [
            {
                "type": "Demand",
                "start_time": "2025-04-16T10:00:00",
                "end_time": "2025-04-16T10:30:00",
                "low": 3172.2,
                "high": 3174.5,
                "source": tf,
                "traits": {
                    "fractal_low": True,
                    "dss_bullish_cross": True,
                    "momentum_bias": "rising",
                    "ema48_touch": True,
                    "bb_lower_tap": True,
                    "fib_validated": True
                },
                "fib_data": {
                    "swing_high": 3192.0,
                    "swing_low": 3171.9,
                    "fib_zone_start": 3176.8,
                    "fib_zone_end": 3173.6
                },
                "entry_level": 3173.5,
                "sl_level": 3171.5,
                "tp_level": 3184.0,
                "label": "Mentfx ICI + Fib Validated",
                "structure_model": "Reaccumulation",
                "poi_confidence": "HIGH"
            }
        ],
        "custom_drawings": [],
        "metadata": base_metadata("mentfx_fib_template")
    }

def get_preset_maz2(symbol="XAUUSD", tf="M15"):
    return {
        "symbol": symbol,
        "timeframe": tf,
        "poi_zones": [
            {
                "type": "Demand",
                "start_time": "2025-04-16T11:00:00",
                "end_time": "2025-04-16T11:30:00",
                "low": 3162.0,
                "high": 3164.0,
                "source": tf,
                "traits": {
                    "ema48_touch": True,
                    "fib_validated": True,
                    "imbalance_below": True,
                    "liquidity_sweep": True
                },
                "fib_data": {
                    "swing_high": 3185.0,
                    "swing_low": 3161.5,
                    "fib_zone_start": 3166.8,
                    "fib_zone_end": 3163.5
                },
                "entry_level": 3164.0,
                "sl_level": 3161.5,
                "tp_level": 3175.5,
                "label": "MAZ2 POI with Imbalance & Fib",
                "structure_model": "Discount Reentry",
                "poi_confidence": "HIGH"
            }
        ],
        "custom_drawings": [],
        "metadata": base_metadata("maz2_template")
    }

def get_preset_tmc(symbol="XAUUSD", tf="M15"):
    return {
        "symbol": symbol,
        "timeframe": tf,
        "poi_zones": [
            {
                "type": "Supply",
                "start_time": "2025-04-16T12:00:00",
                "end_time": "2025-04-16T12:30:00",
                "low": 3190.0,
                "high": 3192.5,
                "source": tf,
                "traits": {
                    "compression_rally": True,
                    "equal_highs_above": True,
                    "bb_upper_tap": True,
                    "fractal_high": True
                },
                "entry_level": 3192.0,
                "sl_level": 3194.0,
                "tp_level": 3178.0,
                "label": "TMC Compression Trap",
                "structure_model": "Distribution",
                "poi_confidence": "MEDIUM"
            }
        ],
        "custom_drawings": [],
        "metadata": base_metadata("tmc_template")
    }

def get_preset_mentfx_reversal(symbol="XAUUSD", tf="M15"):
    return {
        "symbol": symbol,
        "timeframe": tf,
        "poi_zones": [
            {
                "type": "Supply",
                "start_time": "2025-04-16T13:00:00",
                "end_time": "2025-04-16T13:30:00",
                "low": 3186.0,
                "high": 3188.5,
                "source": tf,
                "traits": {
                    "fractal_high": True,
                    "dss_bearish_cross": True,
                    "ema48_touch": True,
                    "bb_upper_tap": True,
                    "trendline_break_retap": True
                },
                "entry_level": 3188.0,
                "sl_level": 3191.0,
                "tp_level": 3175.0,
                "label": "Mentfx M15 Reversal with Trendline Sweep",
                "structure_model": "Reversal Trap",
                "poi_confidence": "HIGH"
            }
        ],
        "custom_drawings": [],
        "metadata": base_metadata("mentfx_reversal_template")
    }
