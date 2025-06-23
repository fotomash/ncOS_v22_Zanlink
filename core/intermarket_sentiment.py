# core/intermarket_sentiment.py
# Author: TL
# Purpose: Fetch and analyze intermarket data to generate macro sentiment JSON

# Scalping-aware macro context loader
import logging

import yfinance as yf
import datetime
import json
import os


def fetch_symbol_data(symbol, period="5d", interval="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def derive_trend(df):
    if df is None or df.empty:
        return "Unknown"
    close = df["Close"]
    if close.iloc[-1] > close.mean():
        return "Bullish"
    elif close.iloc[-1] < close.mean():
        return "Bearish"
    else:
        return "Neutral"


def generate_sentiment():
    symbols = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "US10Y": "^TNX",
        "US2Y": "^IRX",
        "SPY": "SPY",
        "QQQ": "QQQ",
        "CRUDE": "CL=F"
    }

    result = {}
    vix_data = fetch_symbol_data(symbols["VIX"])
    dxy_data = fetch_symbol_data(symbols["DXY"])
    us10y_data = fetch_symbol_data(symbols["US10Y"])
    us2y_data = fetch_symbol_data(symbols["US2Y"])

    result["vix_sentiment"] = derive_trend(vix_data)
    result["dxy_trend"] = derive_trend(dxy_data)
    result["bonds_yield"] = "Rising" if (us10y_data["Close"].iloc[-1] > us2y_data["Close"].iloc[-1]) else "Falling"

    if result["vix_sentiment"] == "Bearish" and result["dxy_trend"] == "Bullish":
        result["macro_bias"] = "Risk-Off"
    else:
        result["macro_bias"] = "Risk-On"

    result["scalping_module"] = {
        "enabled": True,
        "rr_threshold": 1.8,
        "tick_window": 5,
        "drift_pips_min": 1.0,
        "spread_max": 1.8,
        "consistency_ratio": 0.6
    }

    return result


def save_sentiment_to_journal(sentiment):
    os.makedirs("journal", exist_ok=True)
    with open("journal/sentiment_snapshot.json", "w") as f:
        json.dump(sentiment, f, indent=2)
    logging.info("Sentiment and scalping config saved to journal/sentiment_snapshot.json")


# Hook for orchestrator integration

def intermarket_context_hook():
    sentiment = generate_sentiment()
    save_sentiment_to_journal(sentiment)
    return sentiment


if __name__ == "__main__":
    s = intermarket_context_hook()
    print(json.dumps(s, indent=2))
