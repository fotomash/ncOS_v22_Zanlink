# -*- coding: utf-8 -*-
"""
M1 fetcher via Finnhub
Usage: batch_fetch_m1(['BTCUSDT'], days_back=3)
"""
import os, time, pandas as pd, requests
from datetime import datetime, timedelta, timezone, date

API_KEY = os.getenv("FINNHUB_API_KEY")

def _query(symbol, start, end):
    url = "https://finnhub.io/api/v1/crypto/candle"
    params = {
        "symbol": symbol,
        "resolution": 1,            # 1-minute
        "from": int(start.timestamp()),
        "to":   int(end.timestamp()),
        "token": API_KEY
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    if j.get("s") != "ok": raise ValueError("Finnhub err: %s" % j.get("s"))
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(j["t"], unit="s", utc=True),
        "open":  j["o"], "high": j["h"],
        "low":   j["l"], "close": j["c"],
        "volume": j["v"]
    })
    return df

def batch_fetch_m1(symbols, days_back=3, out_dir="tick_data/m1/"):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start =  now - timedelta(days=days_back)
    os.makedirs(out_dir, exist_ok=True)
    for sym in symbols:
        try:
            df = _query(sym, start, now)
            out = f"{out_dir}/{sym.split(':')[-1]}_M1_{start:%Y%m%d}_{now:%Y%m%d%H%M}.csv"
            df.to_csv(out, index=False)
            print(f"[OK] {sym} -> {out} ({len(df)} rows)")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

# Quick test
if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("Set FINNHUB_API_KEY env-var first")
    batch_fetch_m1(["BINANCE:BTCUSDT"], days_back=1)