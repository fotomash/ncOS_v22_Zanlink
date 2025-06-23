# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: wick_liquidity_monitor.py
# Description: Monitors wick strength and liquidity events across macro assets.

import pandas as pd
import os
import glob

# Input and Output Directories
MACRO_DATA_DIR = "intel_data/macro/"
WICK_REPORT_DIR = "intel_data/wick_reports/"
os.makedirs(WICK_REPORT_DIR, exist_ok=True)

# Thresholds for detecting wick strength (percentages)
STRONG_WICK_THRESHOLD = 0.6
INDICISION_WICK_THRESHOLD = 0.3

# Analyze a single asset
def analyze_wick_structure(filepath: str):
    asset_name = os.path.basename(filepath).replace(".csv", "")
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])

        if df.empty:
            print(f"[WARN] No data in {asset_name}")
            return

        latest_candle = df.iloc[-1]
        high = latest_candle["high"]
        low = latest_candle["low"]
        open_price = latest_candle["open"]
        close_price = latest_candle["close"]

        if (high - low) == 0:
            print(f"[WARN] Zero range candle in {asset_name}")
            return

        top_wick = (high - max(open_price, close_price)) / (high - low)
        bottom_wick = (min(open_price, close_price) - low) / (high - low)

        signals = []

        if top_wick > STRONG_WICK_THRESHOLD:
            signals.append("Strong Top Wick (Potential Selling Pressure)")
        if bottom_wick > STRONG_WICK_THRESHOLD:
            signals.append("Strong Bottom Wick (Potential Buying Pressure)")
        if top_wick > INDICISION_WICK_THRESHOLD and bottom_wick > INDICISION_WICK_THRESHOLD:
            signals.append("Indecision Candle (Both Wicks Strong)")

        report_path = os.path.join(WICK_REPORT_DIR, f"{asset_name}_wick_report.txt")

        with open(report_path, "w") as f:
            f.write(f"Asset: {asset_name}\n")
            f.write(f"Timestamp: {latest_candle['timestamp']}\n")
            f.write(f"Top Wick %: {top_wick:.2f}\n")
            f.write(f"Bottom Wick %: {bottom_wick:.2f}\n")
            f.write("Signals:\n")
            for sig in signals:
                f.write(f"- {sig}\n")

        print(f"[OK] Wick analysis complete for {asset_name}")

    except Exception as e:
        print(f"[ERR] Failed to analyze {asset_name}: {e}")

# Batch analyze all assets
def batch_analyze_wicks():
    csv_files = glob.glob(os.path.join(MACRO_DATA_DIR, "*.csv"))
    for filepath in csv_files:
        analyze_wick_structure(filepath)

if __name__ == "__main__":
    batch_analyze_wicks()