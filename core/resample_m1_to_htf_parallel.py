import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Parallel resample M1 CSVs to HTFs")
parser.add_argument("--m1-dir", type=str, required=True, help="Directory containing M1 CSV files")
parser.add_argument("--htf-dir", type=str, required=True, help="Output directory for HTF CSV files")
parser.add_argument("--keep-empty", action="store_true", help="Keep empty bars, filling prices and zero volumes")
args = parser.parse_args()

M1_FOLDER = args.m1_dir
HTF_FOLDER = args.htf_dir
KEEP_EMPTY = args.keep_empty

def resample_symbol(symbol_csv_path):
    df = pd.read_csv(symbol_csv_path, sep=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample targets
    timeframes = {
        'M5': '5T', 'M15': '15T', 'M30': '30T',
        'H1': '1H', 'H4': '4H', 'H12': '12H',
        'D': '1D', 'W': '1W'
    }

    htf_folder = Path(HTF_FOLDER)
    symbol_name = Path(symbol_csv_path).stem.replace('_m1', '')

    for tf, rule in timeframes.items():
        df_tf = df.resample(rule, label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        if KEEP_EMPTY:
            df_tf = df_tf.asfreq(rule)
            for col in ['open','high','low','close']:
                df_tf[col].ffill(inplace=True)
            df_tf['volume'].fillna(0, inplace=True)
        else:
            df_tf.dropna(subset=['open','high','low','close'], inplace=True)

        save_path = htf_folder / f"{symbol_name}_{tf}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_tf.to_csv(save_path)

def run_resampling():
    m1_folder = Path(M1_FOLDER)
    htf_folder = Path(HTF_FOLDER)
    csv_files = list(m1_folder.glob("*.csv"))
    with ThreadPoolExecutor() as executor:
        executor.map(resample_symbol, csv_files)

if __name__ == "__main__":
    run_resampling()