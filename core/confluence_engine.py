
"""
confluence_engine.py
Phase 3: Ultra-Fine Confluence Indicator Extraction for Entry Logic

Provides centralized logic for DSS slope, VWAP deviation, and Bollinger Band width.
Used by POI scoring and entry trigger validators.
"""

import pandas as pd
import numpy as np


def compute_confluence_indicators(df_ltf: pd.DataFrame, df_htf: pd.DataFrame = None) -> dict:
    """
    Computes confluence indicators:
    - DSS slope
    - VWAP deviation
    - Bollinger Band width
    Returns dictionary of values for entry scoring and filtering.
    """
    result = {
        "dss_slope": 0.0,
        "vwap_deviation": 0.0,
        "bb_width": 0.0,
        "entry_grade": "neutral"
    }

    if df_ltf is None or len(df_ltf) < 10:
        return result

    try:
        # DSS slope: use last 10 DSS values
        dss = df_ltf["dss"].tail(10).values
        if len(dss) >= 2:
            slope = dss[-1] - dss[-4]  # 4-period slope
            result["dss_slope"] = round(slope, 4)

        # VWAP deviation: price - vwap / std dev
        close = df_ltf["close"]
        vwap = df_ltf["vwap"]
        if "vwap" in df_ltf.columns:
            vwap_dev = (close - vwap) / (close.rolling(20).std())
            result["vwap_deviation"] = round(vwap_dev.iloc[-1], 4)

        # BB width
        if "bb_upper" in df_ltf.columns and "bb_lower" in df_ltf.columns:
            bb_width = df_ltf["bb_upper"] - df_ltf["bb_lower"]
            result["bb_width"] = round(bb_width.iloc[-1], 4)

        # Entry grade
        if result["dss_slope"] > 0.1 and abs(result["vwap_deviation"]) < 2.0 and result["bb_width"] < 0.02:
            result["entry_grade"] = "high"
        elif result["dss_slope"] > 0 and abs(result["vwap_deviation"]) < 2.5:
            result["entry_grade"] = "medium"
        else:
            result["entry_grade"] = "low"

    except Exception as e:
        print(f"[confluence_engine] Error computing indicators: {e}")

    return result