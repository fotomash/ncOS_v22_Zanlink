# Zanzibar v5.2 Core Module
# Version: 5.2.1 (Updated Resampling Logic)
# Module: data_pipeline.py
# Author: Captain Zanzibar Crew
# --------------------------------------------------
# Unified data-flow engine: fetch M1, fetch macro, resample to HTFs.
# Now uses core/resample_m1_to_htf.py for standardization.
# Assumes finnhub_data_fetcher is modified to SAVE raw M1 CSVs.
# --------------------------------------------------

# --- Zanzibar dynamic version tag and startup timestamp ---
from datetime import datetime
VERSION = "5.2.1"
STARTED_AT = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
print(f"[DataPipeline] Zanzibar {VERSION} initialized at {STARTED_AT} UTC")

from concurrent.futures import ThreadPoolExecutor # Still useful for fetching
import os
import glob
import pandas as pd
from datetime import datetime, timezone, timedelta # Added timezone, timedelta
import traceback # For detailed error logging
from pathlib import Path # Added Path for robust directory handling

# Local imports (assumes modules already exist in core/ or indicators/)
# Import the specific fetchers
try:
    # Assuming the provided finnhub_data_fetcher handles M1 pairs
    # IMPORTANT: This fetcher needs modification to SAVE raw M1 CSVs to 'tick_data/m1/'
    # for the resample_htf step to work correctly.
    from core.finnhub_data_fetcher import load_and_aggregate_m1 # Use the new fetcher
    M1_FETCHER_AVAILABLE = True
except ImportError:
    print("[ERROR][DataPipeline] Cannot import finnhub_data_fetcher. M1 fetching disabled.")
    load_and_aggregate_m1 = None
    M1_FETCHER_AVAILABLE = False

try:
    from core import massive_macro_fetcher
    MACRO_FETCHER_AVAILABLE = True
except ImportError:
     print("[WARN][DataPipeline] massive_macro_fetcher.py not found. Macro fetching disabled.")
     massive_macro_fetcher = None
     MACRO_FETCHER_AVAILABLE = False

# Import the chosen standardized resampler
try:
    # This function expects M1 CSV files in 'tick_data/m1/'
    from core.resample_m1_to_htf import resample_all_symbols_parallel # Use the function from the standardized script
    RESAMPLER_AVAILABLE = True
except ImportError:
    print("[ERROR][DataPipeline] Cannot import resample_m1_to_htf. Resampling disabled.")
    resample_all_symbols_parallel = None
    RESAMPLER_AVAILABLE = False


# ----------------------------------------------------------------------------
# Helper: simple timestamped logger
# ----------------------------------------------------------------------------

def _log(msg: str):
    """Logs messages with a timestamp."""
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DataPipeline] {stamp} | {msg}")

# ----------------------------------------------------------------------------
# DataPipeline class
# ----------------------------------------------------------------------------

class DataPipeline:
    """Unified fetch + resample orchestrator."""

    def __init__(self, symbols_m1=None, macro_interval="5m", workers: int = 6, m1_fetch_days=5):
        """
        Initializes the DataPipeline.

        Args:
            symbols_m1 (list, optional): List of symbols for M1 fetching. Defaults to common FX/Indices.
            macro_interval (str, optional): Interval for macro asset fetching. Defaults to "5m".
            workers (int, optional): Number of parallel workers for fetching (if applicable). Defaults to 6.
            m1_fetch_days (int, optional): How many days back to fetch M1 data. Defaults to 5.
        """
        self.symbols_m1 = symbols_m1 or ["OANDA:EUR_USD", "OANDA:GBP_USD", "OANDA:XAU_USD"] # Updated defaults
        self.macro_interval = macro_interval
        self.workers = workers
        self.m1_fetch_days = m1_fetch_days
        # Define base paths using pathlib
        self.base_dir = Path(".") # Assuming script runs from project root
        self.log_dir = self.base_dir / "logs"
        self.m1_data_dir = self.base_dir / "tick_data" / "m1"
        self.ohlcv_data_dir = self.base_dir / "ohlcv_data"
        self.macro_data_dir = self.base_dir / "intel_data" / "macro" # Define macro dir

        # Ensure necessary directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.m1_data_dir.mkdir(parents=True, exist_ok=True)
        self.ohlcv_data_dir.mkdir(parents=True, exist_ok=True)
        self.macro_data_dir.mkdir(parents=True, exist_ok=True) # Ensure macro dir exists

    # --------------------------------------
    # 1. Fetch intraday pairs (M1) - Using new fetcher
    # --------------------------------------
    def fetch_pairs(self):
        """
        Fetches M1 data for specified symbols using finnhub_data_fetcher.
        **ASSUMES** load_and_aggregate_m1 is modified to SAVE raw M1 CSVs
        to 'tick_data/m1/' in addition to returning aggregated data.
        """
        if not M1_FETCHER_AVAILABLE:
            _log("Skipping M1 pair fetching: finnhub_data_fetcher not available.")
            return

        _log(f"Fetching M1 pairs via finnhub_data_fetcher for symbols: {self.symbols_m1}")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self.m1_fetch_days)

        successful_fetches = 0
        # Consider using ThreadPoolExecutor here if fetching is slow
        # Sequential fetching (simpler for now, modify if parallel needed)
        for symbol in self.symbols_m1:
            _log(f"Fetching M1 for {symbol} (and potentially saving raw M1)...")
            try:
                fetch_result = load_and_aggregate_m1(symbol, start_dt, end_dt)
                if fetch_result['status'] == 'ok':
                    _log(f"Fetch successful for {symbol}.")
                    successful_fetches += 1
                    # Save raw M1 data to CSV
                    raw_m1_df = fetch_result.get('raw_m1_data')
                    if raw_m1_df is not None and not raw_m1_df.empty:
                        safe_symbol = symbol.replace(":", "_").replace("/", "_")
                        filename = f"{safe_symbol}_M1_{start_dt:%Y%m%d}_{end_dt:%Y%m%d%H%M}.csv"
                        filepath = self.m1_data_dir / filename
                        raw_m1_df.to_csv(filepath)
                        _log(f"Saved raw M1 data to {filepath}")
                    else:
                        _log(f"[WARN][DataPipeline] No raw M1 data returned for {symbol}; CSV not saved.")
                else:
                    _log(f"Fetch failed for {symbol}: {fetch_result.get('message')}")
            except Exception as e:
                _log(f"Exception during fetch for {symbol}: {e}")
                traceback.print_exc()

        _log(f"M1 Pair fetching complete. Successful fetches: {successful_fetches}/{len(self.symbols_m1)}")


    # --------------------------------------
    # 2. Fetch macro set (VIX, SPX…)
    # --------------------------------------
    def fetch_macro(self):
        """Fetches macro asset data using massive_macro_fetcher."""
        if not MACRO_FETCHER_AVAILABLE:
            _log("Skipping macro fetching: massive_macro_fetcher not available.")
            return

        _log(f"Fetching macro assets (Interval: {self.macro_interval})...")
        try:
            # Ensure massive_macro_fetcher saves to the correct directory
            # Modify massive_macro_fetcher.py if needed to use self.macro_data_dir
            # Check if the module has the attribute before setting it
            if hasattr(massive_macro_fetcher, 'MACRO_OUTPUT_DIR'):
                 massive_macro_fetcher.MACRO_OUTPUT_DIR = str(self.macro_data_dir) # Override if possible
            else:
                 _log(f"WARN: Cannot set MACRO_OUTPUT_DIR in massive_macro_fetcher. Using its default.")

            massive_macro_fetcher.batch_fetch_macro(interval=self.macro_interval)
            _log("Macro asset fetching complete.")
        except Exception as e:
            _log(f"Error during macro fetching: {e}")
            traceback.print_exc()

    # --------------------------------------
    # 3. Resample M1 to HTF (using standardized resampler)
    # --------------------------------------
    def resample_htf(self):
        """Resamples M1 data found in tick_data/m1/ to HTFs using resample_m1_to_htf."""
        if not RESAMPLER_AVAILABLE:
            _log("Skipping HTF resampling: resample_m1_to_htf module not available.")
            return
        _log(f"Resampling all M1 datasets found in {self.m1_data_dir} -> HTF...")
        try:
            # Resample using specified directories
            if callable(resample_all_symbols_parallel):
                resample_all_symbols_parallel(
                    m1_dir=str(self.m1_data_dir),
                    output_dir=str(self.ohlcv_data_dir)
                )
                _log("HTF resampling complete.")
            else:
                _log("ERROR: resample_all_symbols_parallel function not callable.")
        except Exception as e:
            _log(f"Error during HTF resampling: {e}")
            traceback.print_exc()

    # --------------------------------------
    # 4. Run full chain
    # --------------------------------------
    def run_full(self):
        """Runs the full data pipeline: Fetch Pairs -> Fetch Macro -> Resample HTF."""
        _log("--- Starting Full Data Pipeline Run ---")
        self.fetch_pairs() # Assumes this step results in M1 CSVs being saved
        self.fetch_macro()
        self.resample_htf() # Resamples the saved M1 CSVs
        _log("--- DataPipeline Full Run Complete ✅ ---")

# ----------------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    _log("Initializing DataPipeline from CLI...")
    # Example: Customize symbols or fetch days if needed
    # pipeline = DataPipeline(symbols_m1=["OANDA:EUR_USD", "BINANCE:BTCUSDT"], m1_fetch_days=7)
    pipeline = DataPipeline()
    pipeline.run_full()
