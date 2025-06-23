# pnf_v4_ingestor.py
# Ingests V4-style Wyckoff P&F prompt input and returns structured schema object with SL auto-calculation logic

from typing import Dict, Any
from datetime import datetime

def parse_v4_prompt(raw_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts V4 Wyckoff + P&F setup prompt input into structured ingestible form.
    Automatically calculates SL using Wyckoff ATI model if required fields are present.
    """
    # Extract values needed for SL calculation
    entry_price = raw_input.get("M1 Refined POI Level Selected for Entry")
    wick_origin = raw_input.get("Wick Trap Origin Level")
    spread_value = raw_input.get("Spread Value (Ticks)")
    buffer = raw_input.get("SL Buffer", 0.10)

    # Attempt SL calculation only if all required fields are present
    try:
        entry_price = float(entry_price)
        wick_origin = float(wick_origin)
        spread_value = float(spread_value)
        buffer = float(buffer)
        sl_value = wick_origin - spread_value - buffer
        sl_confidence = "Auto-computed using Wyckoff ATI"
    except (TypeError, ValueError):
        sl_value = None
        sl_confidence = "Insufficient data — manual review required"

    output = {
        "metadata": {
            "symbol": raw_input.get("Asset Symbol"),
            "analysis_time": raw_input.get("Analysis Date/Time", str(datetime.utcnow())),
            "htf_tf": raw_input.get("Analysis Timeframe (HTF)"),
            "intermediate_tf": raw_input.get("P&F Timeframe (Intermediate)"),
            "ltf_tf": raw_input.get("Entry Timeframe (LTF)"),
            "baseline_risk": float(raw_input.get("Baseline Risk Per Trade (%)", 0.25)),
            "min_rr": float(raw_input.get("Minimum Acceptable R:R (TP1)", 3.0)),
            "max_dd": float(raw_input.get("Max Daily Drawdown Limit (%)", 1.0)),
            "no_overnight": raw_input.get("No Overnight Hold Constraint Applies?", "Yes") == "Yes"
        },
        "execution": {
            "entry_price": entry_price,
            "sl": sl_value,
            "sl_confidence": sl_confidence,
            "tp1": raw_input.get("Proposed TP1 Level"),
            "tp2": raw_input.get("Proposed TP2 Level"),
            "rr_tp1": float(raw_input.get("Calculated R:R (TP1)", 0)),
            "conviction_score": int(raw_input.get("Overall Conviction Score (1-5)", 0)),
            "risk_adjusted": float(raw_input.get("Adjusted Risk % for this Trade", 0.25)),
            "position_size": float(raw_input.get("Calculated Position Size", 0.0)),
            "go": raw_input.get("GO / NO-GO Decision") == "GO"
        }
    }
    return output
