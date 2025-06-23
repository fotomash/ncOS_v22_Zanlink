import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def compute_spread_instability(
    df: pd.DataFrame,
    window_size: int = 25,
    high_vol_baseline: float = 0.0008,
    winsorize_limits: list[float] = None
) -> float:
    """
    Computes normalized spread instability score from bid-ask data.
    Applies optional winsorization to reduce outlier impact.
    """
    if df is None or df.empty or 'bid' not in df.columns or 'ask' not in df.columns:
        return 0.5  # default neutral instability

    df = df.copy()
    df["spread"] = df["ask"] - df["bid"]

    recent_spreads = df["spread"].tail(window_size)
    if len(recent_spreads) < 5:
        return 0.5

    # Optional winsorization
    if winsorize_limits:
        recent_spreads = winsorize(recent_spreads, limits=winsorize_limits)

    spread_std = recent_spreads.std()
    normalized = min(spread_std / high_vol_baseline, 1.0)
    return round(normalized, 4)