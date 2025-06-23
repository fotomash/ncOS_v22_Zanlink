# finnhub_data_fetcher.py
# Upgraded module for fetching and aggregating OHLCV data from Finnhub
# Returns a status dictionary: {'status': 'ok', 'data': {...}} or {'status': 'error', 'message': ...}
# Version: 1.1.0 (Using Provided API Key)

import os
import time
import traceback
import pandas as pd
import finnhub # Ensure finnhub library is installed: pip install finnhub-python
from datetime import datetime, timedelta, timezone
from pathlib import Path # For potential CSV saving

print("[DEBUG] ✅ finnhub_data_fetcher.py loaded and active")

# --- Initialize Finnhub client ---
finnhub_client = None
# Use the provided API key, fallback to environment variable or placeholder
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d07lgo1r01qrslhp3q3gd07lgo1r01qrslhp3q40") # Using provided key

# Define base path for saving raw M1 data (consistent with data_pipeline)
M1_SAVE_DIR = Path("tick_data/m1")
M1_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

try:
    if not FINNHUB_API_KEY or len(FINNHUB_API_KEY) < 20: # Basic check for placeholder/invalid key
         print(f"[WARN] Invalid or missing Finnhub API Key ('{FINNHUB_API_KEY}'). Finnhub client not initialized.")
    else:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        # Optional: Test connection (e.g., fetch profile for a common stock)
        # try:
        #     profile = finnhub_client.company_profile2(symbol='AAPL')
        #     if profile: print(f"[INFO] Finnhub client initialized and connection tested successfully.")
        #     else: print(f"[WARN] Finnhub client initialized, but test query failed. Check API key validity/permissions.")
        # except Exception as test_err:
        #      print(f"[ERROR] Finnhub client initialization test failed: {test_err}. Check API key/network.")
        #      finnhub_client = None # Disable client if test fails
        print(f"[INFO] Finnhub client initialized.") # Keep simple init message for now

except Exception as e:
    print(f"[ERROR] Failed to initialize Finnhub client: {e}")
    finnhub_client = None

# --- Symbol Resolution Logic ---
def _resolve_symbol_type(symbol: str) -> str:
    """ Determines asset type based on symbol format. """
    symbol = symbol.upper()
    # Simple checks, refine as needed
    if "BINANCE:" in symbol or "COINBASE:" in symbol or symbol in ["BTCUSD", "ETHUSD"]:
        return "crypto"
    elif ":" in symbol and any(s in symbol for s in ["OANDA:", "FXCM:", "PEPPERSTONE:", "ICMARKETS:"]):
        return "forex"
    elif any(s in symbol for s in ["XAU", "XAG", "GOLD", "SILVER"]):
         return "forex" # Finnhub often uses forex endpoint for metals CFDs
    elif any(s in symbol for s in ["SPX", "US500", "NAS100", "USTEC", "US30", "DE40", "UK100", "^"]):
        return "index_cfd" # Or potentially 'stock' if using stock endpoint for index CFDs
    # Default fallback
    return "stock"

# --- Core Raw Fetcher ---
def _fetch_raw_candles(symbol: str, resolution: str, start_unix: int, end_unix: int) -> dict:
    """ Fetches raw candle data from appropriate Finnhub endpoint. """
    if not finnhub_client:
        return {'status': 'error', 'source': 'finnhub', 'message': 'Finnhub client not initialized'}

    sym_type = _resolve_symbol_type(symbol)
    print(f"[INFO][FinnhubFetch] Fetching {symbol} ({sym_type}), Res: {resolution}, From: {datetime.fromtimestamp(start_unix, tz=timezone.utc):%Y-%m-%d %H:%M}, To: {datetime.fromtimestamp(end_unix, tz=timezone.utc):%Y-%m-%d %H:%M}")

    try:
        # Choose the correct Finnhub API endpoint based on symbol type
        if sym_type == "crypto":
            result = finnhub_client.crypto_candles(symbol, resolution, start_unix, end_unix)
        elif sym_type == "forex":
            # Ensure symbol format is correct for forex (e.g., OANDA:EUR_USD)
            if ":" not in symbol:
                 print(f"[WARN][FinnhubFetch] Forex symbol '{symbol}' might need a prefix (e.g., OANDA:). Attempting fetch anyway.")
            result = finnhub_client.forex_candles(symbol, resolution, start_unix, end_unix)
        elif sym_type == "index_cfd":
             # Index CFDs might use stock or forex endpoints depending on broker feed in Finnhub
             # Try stock first, then forex as fallback? Or require specific prefix?
             # Let's assume stock endpoint works for common index symbols like SPX, NDX
             print(f"[INFO][FinnhubFetch] Attempting index fetch for {symbol} using stock_candles endpoint.")
             result = finnhub_client.stock_candles(symbol, resolution, start_unix, end_unix)
             if result.get("s") != "ok" and ":" not in symbol: # Fallback for indices without prefix
                 print(f"[WARN][FinnhubFetch] Stock endpoint failed for index {symbol}. Trying with OANDA: prefix via forex_candles.")
                 try: result = finnhub_client.forex_candles(f"OANDA:{symbol}", resolution, start_unix, end_unix)
                 except Exception: pass # Ignore error if fallback also fails
        else: # Default to stock endpoint
            result = finnhub_client.stock_candles(symbol, resolution, start_unix, end_unix)

        # --- Process Result ---
        if result.get("s") == "no_data":
            print(f"[INFO][FinnhubFetch] No data returned from Finnhub for {symbol} in the specified range.")
            # Return empty DataFrame matching expected structure
            empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            empty_df.index.name = 'Timestamp'
            return {'status': 'ok', 'data': empty_df}

        if result.get("s") != "ok":
            error_msg = f"API returned status: {result.get('s')}"
            print(f"[ERROR][FinnhubFetch] Finnhub API error for {symbol}: {error_msg}")
            return {'status': 'error', 'source': 'finnhub', 'message': error_msg}

        # Convert to DataFrame
        df = pd.DataFrame({
            'Timestamp': pd.to_datetime(result.get("t", []), unit='s', utc=True),
            'Open': result.get("o", []),
            'High': result.get("h", []),
            'Low': result.get("l", []),
            'Close': result.get("c", []),
            'Volume': result.get("v", [])
        }).set_index("Timestamp")

        # Ensure correct dtypes
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Drop rows with NaN OHLC
        df['Volume'].fillna(0, inplace=True) # Fill NaN Volume

        print(f"[INFO][FinnhubFetch] Successfully fetched {len(df)} raw candles for {symbol}.")
        return {'status': 'ok', 'data': df}

    except finnhub.FinnhubAPIException as api_e:
         print(f"[ERROR][FinnhubFetch] Finnhub API Exception for {symbol}: {api_e}")
         return {'status': 'error', 'source': 'finnhub', 'message': f"API Exception: {api_e}"}
    except Exception as e:
        print(f"[ERROR][FinnhubFetch] Unexpected Exception fetching {symbol}: {e}")
        traceback.print_exc()
        return {'status': 'error', 'source': 'finnhub', 'message': f"Exception: {e}"}

# --- Aggregation Interface ---
def load_and_aggregate_m1(symbol: str, start_dt: datetime, end_dt: datetime, save_raw_m1: bool = True) -> dict:
    """
    Fetches raw M1 data, saves it (optional), and aggregates to standard timeframes.

    Args:
        symbol (str): The trading symbol.
        start_dt (datetime): Start datetime (timezone aware).
        end_dt (datetime): End datetime (timezone aware).
        save_raw_m1 (bool): If True, saves the fetched M1 data to tick_data/m1/.

    Returns:
        dict: {'status': 'ok', 'data': {'m1': df, 'm5': df, ...}} or error dict.
    """
    print(f"[INFO][FinnhubFetch] Requesting M1 data load & aggregate for {symbol}...")

    # Ensure datetimes are UTC for Finnhub
    start_dt_utc = start_dt.astimezone(timezone.utc)
    end_dt_utc = end_dt.astimezone(timezone.utc)

    # Fetch raw M1 data
    result = _fetch_raw_candles(symbol, "1", int(start_dt_utc.timestamp()), int(end_dt_utc.timestamp()))

    if result["status"] != "ok":
        return result # Propagate fetch error

    df_m1 = result["data"]
    if df_m1.empty:
        print(f"[INFO][FinnhubFetch] No M1 data found for {symbol} in range. Returning empty dataset.")
        return {'status': 'ok', 'data': {}} # Return ok with empty data dict

    # --- Save Raw M1 Data (Required for Data Pipeline Resampler) ---
    if save_raw_m1:
        try:
            # Create a filename compatible with resampler (adjust if needed)
            safe_symbol_name = symbol.replace(":", "_").replace("/", "_")
            filename = f"{safe_symbol_name}_M1_{start_dt_utc:%Y%m%d%H%M}_{end_dt_utc:%Y%m%d%H%M}.csv"
            filepath = M1_SAVE_DIR / filename
            # Save with appropriate format (e.g., comma-separated, include index)
            df_m1.to_csv(filepath, sep=',', index=True, header=True) # Save with comma for standard processing
            print(f"[INFO][FinnhubFetch] Saved raw M1 data ({len(df_m1)} rows) to {filepath}")
        except Exception as save_err:
            print(f"[ERROR][FinnhubFetch] Failed to save raw M1 data for {symbol}: {save_err}")
            # Continue with aggregation even if saving fails? Or return error? Let's continue.

    # --- Aggregate to Higher Timeframes ---
    aggregated = {"m1": df_m1}
    rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    # Standard timeframes for aggregation
    timeframes = {"m5": "5min", "m15": "15min", "m30": "30min", "h1": "1H", "h4": "4H", "d1": "D", "w1": "W"}

    print(f"[INFO][FinnhubFetch] Aggregating M1 data for {symbol}...")
    try:
        for tf_key, freq in timeframes.items():
            # Use label='left', closed='left' for standard OHLC aggregation alignment
            df_tf = df_m1.resample(freq, label='left', closed='left').agg(rules)
            # Drop rows where resampling resulted in all NaNs (common for periods with no M1 data)
            df_tf.dropna(how='all', subset=['Open', 'High', 'Low', 'Close'], inplace=True)
            # Fill volume NaNs that might occur during resampling gaps
            df_tf['Volume'].fillna(0, inplace=True)
            if not df_tf.empty:
                aggregated[tf_key] = df_tf
                print(f"  -> Aggregated {tf_key.upper()}: {len(df_tf)} rows")
            else:
                 print(f"  -> Aggregated {tf_key.upper()}: 0 rows (after dropna)")

        print(f"[INFO][FinnhubFetch] Aggregation complete for {symbol}.")
        return {'status': 'ok', 'data': aggregated}

    except Exception as agg_err:
        print(f"[ERROR][FinnhubFetch] Aggregation failed for {symbol}: {agg_err}")
        traceback.print_exc()
        return {'status': 'error', 'source': 'resample', 'message': str(agg_err)}

# --- Deprecated Function (Direct Fetch) ---
def fetch_finnhub_ohlcv(symbol: str, start_time: datetime, end_time: datetime, resolution: str = '1') -> pd.DataFrame:
    """ Deprecated: Use load_and_aggregate_m1 instead for consistent workflow. """
    print("[WARN][FinnhubFetch] DEPRECATED: fetch_finnhub_ohlcv called directly. Use load_and_aggregate_m1.")
    # Basic implementation for backward compatibility if absolutely needed
    start_unix = int(start_time.timestamp())
    end_unix = int(end_time.timestamp())
    result = _fetch_raw_candles(symbol, resolution, start_unix, end_unix)
    if result['status'] == 'ok':
        return result['data']
    else:
        # Return empty dataframe on error to match expected type hint (though error is lost)
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


# --- CLI Testing Support ---
# Removed for operational code
# if __name__ == "__main__":
#    ... (testing code removed) ...
