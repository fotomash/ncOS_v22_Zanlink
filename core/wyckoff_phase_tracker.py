# wyckoff_phase_tracker.py
import pandas as pd

class WyckoffPhaseTracker:
    def __init__(self, window=50):
        self.window = window
        self.history = pd.DataFrame()

    def update(self, new_data: pd.DataFrame):
        df = new_data.copy()
        df["MID"] = (df["<BID>"] + df["<ASK>"]) / 2
        df = df.dropna(subset=["MID"])
        self.history = pd.concat([self.history, df]).sort_index()
        self.history = self.history[~self.history.index.duplicated(keep='last')]
        return self._analyze()

    def _analyze(self):
        df = self.history.copy()
        df["ROLL_MIN"] = df["MID"].rolling(self.window).min()
        df["ROLL_MAX"] = df["MID"].rolling(self.window).max()
        df["SPRING"] = (df["MID"] < df["ROLL_MIN"].shift(1)) & (df["MID"].pct_change() > 0.0004)
        df["AR"] = (df["MID"] > df["ROLL_MAX"].shift(1))
        df["ST"] = (df["MID"] < df["ROLL_MIN"].shift(1)) & (df["MID"].pct_change() < 0)
        df["LPS"] = (df["MID"] > df["ROLL_MIN"].shift(1)) & (df["MID"].pct_change() > 0)
        return df[["MID", "ROLL_MIN", "ROLL_MAX", "SPRING", "AR", "ST", "LPS"]].tail(50)

# Example usage:
# tracker = WyckoffPhaseTracker()
# latest_phases = tracker.update(latest_tick_df)
