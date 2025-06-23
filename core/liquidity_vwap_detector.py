# liquidity_vwap_detector.py
# Production-grade VWAP-based liquidity sweep detection

import pandas as pd

class LiquidityVWAPDetector:
    ABOVE_SWEEP = "Above VWAP Sweep"
    BELOW_SWEEP = "Below VWAP Sweep"
    NO_SWEEP = "None"
    
    def __init__(self, std_window=30, threshold_factor=1.5):
        self.std_window = std_window
        self.threshold_factor = threshold_factor

    def compute_vwap(self, df):
        # Check for empty or malformed DataFrame input
        if df.empty or not all(col in df.columns for col in ["High", "Low", "Close", "Volume"]):
            raise ValueError("Input DataFrame is missing required columns or is empty.")
        df = df.copy()
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        cumulative_vp = (typical_price * df["Volume"]).cumsum()
        cumulative_volume = df["Volume"].cumsum()
        df["VWAP"] = cumulative_vp / cumulative_volume
        return df

    def detect_deviation_sweeps(self, df):
        df = self.compute_vwap(df)
        df["Deviation"] = df["Close"] - df["VWAP"]
        df["STD"] = df["Deviation"].rolling(self.std_window).std()
        df["Threshold"] = self.threshold_factor * df["STD"]

        df["LiquiditySweep"] = self.NO_SWEEP
        df.loc[df["Deviation"] > df["Threshold"], "LiquiditySweep"] = self.ABOVE_SWEEP
        df.loc[df["Deviation"] < -df["Threshold"], "LiquiditySweep"] = self.BELOW_SWEEP

        # Log or annotate the index when a liquidity sweep is triggered for auditability
        sweep_indices = df[df["LiquiditySweep"] != self.NO_SWEEP].index.tolist()
        df["SweepFlag"] = df.index.isin(sweep_indices)

        return df[["Close", "VWAP", "Deviation", "Threshold", "LiquiditySweep", "SweepFlag"]]


# No placeholder logic – to be integrated directly into execution pipelines
# Example usage is handled by upstream orchestrators or testing suites