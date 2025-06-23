# Zanzibar v5.1 Core Module
# Version: 5.1.0
# Module: massive_macro_fetcher.py
# Description: Fetches M1 or M5 OHLCV data for critical macro assets (VIX, SPX, DXY, Gold, BTC, Bonds).

import pandas as pd
import yfinance as yf
import os

# Macro Assets to Track
MACRO_ASSETS = [
    {"symbol": "^VIX", "name": "VIX"},
    {"symbol": "^GSPC", "name": "SPX"},
    {"symbol": "DX-Y.NYB", "name": "DXY"},
    {"symbol": "^TNX", "name": "US10Y"},
    {"symbol": "GC=F", "name": "GOLD"},
    {"symbol": "BTC-USD", "name": "BTCUSD"},
    {"symbol": "CL=F", "name": "WTI_OIL"},
    {"symbol": "EURUSD=X", "name": "EURUSD"}
]

# Output Directory
MACRO_OUTPUT_DIR = "intel_data/macro/"
os.makedirs(MACRO_OUTPUT_DIR, exist_ok=True)

# Fetch Function
def fetch_macro_asset(symbol: str, name: str, interval: str = "5m"):
    try:
        df = yf.download(
            tickers=symbol,
            interval=interval,
            period="1d",
            auto_adjust=True,
            prepost=False
        )

        if not df.empty:
            df.reset_index(inplace=True)
            df.rename(columns={
                "Datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            save_path = os.path.join(MACRO_OUTPUT_DIR, f"{name}_{interval}.csv")
            df.to_csv(save_path, index=False)
            print(f"[OK] Fetched {name} ({symbol})")
        else:
            print(f"[WARN] No data for {name} ({symbol})")
    except Exception as e:
        print(f"[ERR] Failed to fetch {name} ({symbol}): {e}")

# Batch Fetcher
def batch_fetch_macro(interval="5m"):
    for asset in MACRO_ASSETS:
        fetch_macro_asset(asset["symbol"], asset["name"], interval)

if __name__ == "__main__":
    batch_fetch_macro(interval="5m")  # Default fetch interval is 5-minute candles