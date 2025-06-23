import pandas as pd
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
import sys
import traceback

# --- Import functions from the data manager ---
try:
    from ingest_data_manager import (
        load_tick_csv,
        convert_ticks_to_ohlcv # Assuming this exists and works
        # CANONICAL_HEADERS # If needed, ensure it's defined
    )
    INGEST_MANAGER_LOADED = True
except ImportError:
    print("[ERROR][IngestionEngine] Failed to import from ingest_data_manager.py! Ensure it exists and is accessible.")
    INGEST_MANAGER_LOADED = False
    # Define dummy functions if import fails, allowing script structure to remain
    def load_tick_csv(*args, **kwargs): print("Error: load_tick_csv not loaded."); return None
    def convert_ticks_to_ohlcv(*args, **kwargs): print("Error: convert_ticks_to_ohlcv not loaded."); return None

# --- Import Marker Enrichment Engine --- ### MODIFICATION START ###
try:
    # Using the user-provided scaffold name
    from marker_enrichment_engine import add_all_indicators
    ENRICHMENT_ENGINE_LOADED = True
    print("[INFO][IngestionEngine] Successfully imported 'add_all_indicators' from marker_enrichment_engine.")
except ImportError:
    print("[WARN][IngestionEngine] marker_enrichment_engine.py not found or failed to import 'add_all_indicators'. Enrichment step will be skipped.")
    def add_all_indicators(df, **kwargs): print("Warning: add_all_indicators not loaded."); return df # Dummy pass-through
    ENRICHMENT_ENGINE_LOADED = False
# ### MODIFICATION END ###

DEFAULT_OUTPUT_DIR = Path("./data/processed_ohlcv")
DEFAULT_PROFILE_PATH = Path("tick_header_profiles.json")
# Default timeframes if none specified via CLI or profile
DEFAULT_RESAMPLE_DEPTHS = ["1T", "5T", "15T", "1H", "4H", "1D"]

# Updated function signature to include max_candles ### MODIFICATION START ###
def run_ingestion_pipeline(
    input_file: str,
    header_profile: Optional[str] = None,
    header_map_override: Optional[Dict] = None,
    profile_path: str = str(DEFAULT_PROFILE_PATH),
    output_timeframes: List[str] = DEFAULT_RESAMPLE_DEPTHS, # Use default list
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    save_csv: bool = True,
    return_data: bool = False,
    max_candles: Optional[int] = None, # Max candles per timeframe (None for no limit/backtest)
    read_csv_kwargs: Optional[Dict] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Orchestrates the tick data ingestion pipeline for a single file.
    Includes OHLCV conversion, optional marker enrichment, and optional candle trimming.
    """
    # ### MODIFICATION END ###

    if not INGEST_MANAGER_LOADED:
        print("[ERROR][IngestionEngine] Core data manager functions not loaded. Aborting.")
        return None

    print(f"\n--- Starting Ingestion Pipeline for: {input_file} ---")
    print(f"Header Profile: {header_profile or 'None (using override or auto-detect)'}")
    print(f"Output Timeframes: {output_timeframes}")
    # Log max_candles info - None implies backtest mode or no limit applied
    print(f"Max Candles Limit: {max_candles if max_candles is not None else 'None (Backtest Mode)'}")
    print(f"Save Output CSVs: {save_csv}")
    print(f"Return DataFrames: {return_data}")

    # --- 1. Load & Normalize Tick Data ---
    print("\nStep 1: Loading and Normalizing Tick Data...")
    tick_df = load_tick_csv(
        file_path=input_file,
        header_profile=header_profile,
        header_map_override=header_map_override,
        profile_path=profile_path,
        **(read_csv_kwargs or {})
    )
    if tick_df is None:
        print("[ERROR][IngestionEngine] Failed to load or validate tick data. Aborting pipeline.")
        return None
    print(f"Step 1 SUCCESS: Loaded and validated {len(tick_df)} ticks.")

    # --- 2. Convert Ticks to OHLCV Timeframes ---
    print("\nStep 2: Converting Ticks to OHLCV Timeframes...")
    ohlcv_data_dict = {}
    # Iterate through the dynamically provided list of timeframes
    for tf in output_timeframes: # Uses the list passed as argument
        print(f"  - Generating {tf}...")
        try:
            # Assuming convert_ticks_to_ohlcv returns a DataFrame with OHLCV columns
            ohlcv_df = convert_ticks_to_ohlcv(tick_df, freq=tf)
            # Check for valid DataFrame before adding
            if ohlcv_df is not None and not ohlcv_df.empty and isinstance(ohlcv_df, pd.DataFrame):
                ohlcv_data_dict[tf] = ohlcv_df
                print(f"    -> Generated {len(ohlcv_df)} candles.")
            else:
                print(f"[WARN][IngestionEngine] Failed to generate or empty result for timeframe: {tf}")
                # Decide if pipeline should continue if one TF fails
        except Exception as convert_err:
             print(f"[ERROR][IngestionEngine] Error converting to timeframe {tf}: {convert_err}")
             traceback.print_exc() # Print full traceback for conversion errors

    if not ohlcv_data_dict:
        print("[ERROR][IngestionEngine] No OHLCV data could be generated. Aborting.")
        return None
    print("Step 2 COMPLETE: OHLCV Conversion finished.")

    # --- 3. Trim Candles (if max_candles specified, before enrichment) --- ### MODIFICATION START ###
    if max_candles is not None:
        print(f"\nStep 3: Applying max_candles limit: {max_candles}...")
        trimmed_data_dict = {}
        for tf, df_original in ohlcv_data_dict.items():
            if len(df_original) > max_candles:
                # Use .copy() to avoid potential SettingWithCopyWarning later
                trimmed_data_dict[tf] = df_original.tail(max_candles).copy()
                print(f"  - Trimmed {tf} from {len(df_original)} to {max_candles} candles.")
            else:
                trimmed_data_dict[tf] = df_original # No trimming needed, keep original
        ohlcv_data_dict = trimmed_data_dict # Update dict with trimmed (or original) data
        print("Step 3 COMPLETE: Candle trimming finished.")
    else:
        print("\nStep 3: Skipping candle trimming (Backtest mode or no limit set).")
    # ### MODIFICATION END ###


    # --- 4. Add Enrichment Markers --- ### MODIFICATION START ###
    if ENRICHMENT_ENGINE_LOADED:
        print("\nStep 4: Applying indicator and marker enrichment...")
        enriched_data_dict = {}
        for tf, df_to_enrich in ohlcv_data_dict.items():
            # Ensure we have data to enrich for this timeframe
            if df_to_enrich is None or df_to_enrich.empty:
                 print(f"  - Skipping enrichment for {tf}: No data.")
                 enriched_data_dict[tf] = df_to_enrich # Keep None or empty df
                 continue

            print(f"  - Enriching {tf} data...")
            try:
                # Pass timeframe for context-aware indicators (like session VWAP)
                # The add_all_indicators function should handle missing columns gracefully
                enriched_df = add_all_indicators(df_to_enrich, timeframe=tf)
                enriched_data_dict[tf] = enriched_df
            except Exception as enrich_err:
                print(f"[ERROR][IngestionEngine] Failed to enrich {tf} data: {enrich_err}")
                print(traceback.format_exc())
                enriched_data_dict[tf] = df_to_enrich # Keep original (trimmed) df if enrichment fails
        ohlcv_data_dict = enriched_data_dict # Update dict with enriched data
        print("Step 4 COMPLETE: Enrichment finished.")
    else:
        print("\nStep 4: Skipping marker enrichment (Engine not loaded).")
    # ### MODIFICATION END ###


    # --- 5. Save OHLCV Data (Optional) --- ### Step renumbered ###
    if save_csv:
        print("\nStep 5: Saving final OHLCV Data to CSV...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_filename = Path(input_file).stem
        for tf, df in ohlcv_data_dict.items():
            # Check if df is valid before saving
            if df is None or df.empty:
                print(f"  - Skipping save for {tf}: No data.")
                continue
            # Sanitize TF string for filename (e.g., '1T' -> 'M1', '15s' -> 'S15')
            tf_filename_part = tf.replace('T', 'M').replace('s', 'S')
            # Indicate if the saved file is enriched
            suffix = "_enriched" if ENRICHMENT_ENGINE_LOADED else ""
            output_filepath = output_path / f"{base_filename}_{tf_filename_part}{suffix}.csv"
            try:
                df.to_csv(output_filepath)
                print(f"  - Saved {tf} data ({'enriched' if ENRICHMENT_ENGINE_LOADED else 'raw'}) to: {output_filepath}")
            except Exception as e:
                print(f"[ERROR][IngestionEngine] Failed to save {tf} data to {output_filepath}: {e}")
        print("Step 5 COMPLETE: CSV Saving finished.")
    else:
        print("\nStep 5: Skipping CSV Save.")

    print("\n--- Ingestion Pipeline Finished ---")
    if return_data:
        print("Returning final OHLCV data dictionary.")
        return ohlcv_data_dict
    else:
        return None


# --- CLI Argument Parsing & Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="ZANZIBAR Tick Data Ingestion Pipeline: Load, Normalize, Resample, Enrich.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    # Arguments
    parser.add_argument("input_file", nargs='?', default=None, help="Path to the input tick data CSV file (required unless --list-profiles is used).")
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save output CSV files.")
    parser.add_argument("--no-save", action="store_true", help="Do not save generated OHLCV data to CSV files.")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument("-p", "--profile", help="Name of the header profile to use (from profiles JSON).")
    profile_group.add_argument("-m", "--header-map", type=json.loads, help='JSON string for header map override.')
    parser.add_argument("-pf", "--profile-file", default=str(DEFAULT_PROFILE_PATH), help="Path to the header profiles JSON file.")
    parser.add_argument("--list-profiles", action="store_true", help="List available header profiles defined in the profile file and exit.")
    parser.add_argument(
        "-tf", "--timeframes", nargs='+', default=DEFAULT_RESAMPLE_DEPTHS,
        help='List of output OHLCV timeframes (Pandas freq strings like "1s", "15s", "1T", "5T", "1H").'
        )
    # Add max_candles argument for CLI use ### MODIFICATION START ###
    parser.add_argument(
        "--max-candles", type=int, default=None,
        help="Maximum number of candles per timeframe (default: None, no limit/backtest)."
        )
    # ### MODIFICATION END ###
    # CSV reading args (passed via read_csv_kwargs)
    parser.add_argument("-d", "--delimiter", default=",", help="Delimiter for input CSV file.")
    parser.add_argument("--skiprows", type=int, default=0, help="Number of rows to skip at the beginning of the CSV.")


    args = parser.parse_args()

    # --- Handle --list-profiles action ---
    if args.list_profiles:
        profile_p = Path(args.profile_file)
        print(f"--- Available Header Profiles in '{profile_p.name}' ---")
        if not profile_p.is_file(): print(f"Error: Profile file not found at '{profile_p.resolve()}'"); sys.exit(1)
        try:
            with open(profile_p, 'r') as f: profiles = json.load(f)
            if not profiles or not isinstance(profiles, dict): print("Error: Profile file is empty or not valid JSON."); sys.exit(1)
            # Iterate through top-level keys as profile names
            for name, content in profiles.items():
                if name.startswith("_"): continue # Skip keys starting with underscore
                # Look for description field, fall back if missing
                comment = content.get("description", "No description")
                print(f"- {name}: {comment}")
        except Exception as e: print(f"Error reading profile file: {e}"); sys.exit(1)
        sys.exit(0)

    if not args.input_file:
        parser.error("the following arguments are required: input_file (unless --list-profiles is used)")

    # --- Prepare read_csv_kwargs ---
    # Pass only if they differ from potential profile defaults? Or always pass?
    # For now, pass them; load_tick_csv should prioritize profile if applicable.
    read_csv_args = {'delimiter': args.delimiter, 'skiprows': args.skiprows}

    # --- Run the main pipeline ---
    run_ingestion_pipeline(
        input_file=args.input_file,
        header_profile=args.profile,
        header_map_override=args.header_map,
        profile_path=args.profile_file,
        output_timeframes=args.timeframes, # Passes the list from CLI or default
        output_dir=args.output_dir,
        save_csv=not args.no_save,
        return_data=False, # Don't return data when run from CLI
        max_candles=args.max_candles, # Pass max_candles from CLI ### MODIFICATION START ###
        read_csv_kwargs=read_csv_args
    )
    # ### MODIFICATION END ###

if __name__ == "__main__":
    # Basic check for module availability before running CLI
    if not INGEST_MANAGER_LOADED:
         print("\n[FATAL] Cannot run script because ingest_data_manager module is missing. Check imports/installation.")
         sys.exit(1)
    main()

