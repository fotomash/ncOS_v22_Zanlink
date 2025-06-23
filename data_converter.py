#!/usr/bin/env python3
"""
ncOS v23 Data Converter
Converts various data formats for ncOS processing
"""

import pandas as pd
import json
from datetime import datetime
import argparse

def convert_mt4_to_ncos(input_file: str, output_file: str):
    """Convert MT4 tick data to ncOS format"""
    # Read MT4 format
    df = pd.read_csv(input_file)

    # Standardize columns
    df.columns = df.columns.str.lower()

    # Ensure required columns
    required = ['timestamp', 'bid', 'ask']
    if not all(col in df.columns for col in required):
        print(f"❌ Missing required columns: {required}")
        return False

    # Calculate spread if not present
    if 'spread_points' not in df.columns:
        df['spread_points'] = (df['ask'] - df['bid']) * 10000  # For forex

    if 'volume' not in df.columns:
        df['volume'] = 0.0

    # Save in ncOS format
    df.to_csv(output_file, index=False)
    print(f"✅ Converted {len(df)} ticks to {output_file}")
    return True

def aggregate_ticks_to_ohlcv(tick_file: str, output_file: str, timeframe: str = '5Min'):
    """Aggregate tick data to OHLCV bars"""
    # Read tick data
    df = pd.read_csv(tick_file)

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Use bid prices for OHLC
    ohlcv = df['bid'].resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    # Add volume
    ohlcv['volume'] = df['volume'].resample(timeframe).sum()

    # Remove NaN rows
    ohlcv.dropna(inplace=True)

    # Save
    ohlcv.to_csv(output_file)
    print(f"✅ Created {len(ohlcv)} {timeframe} bars")
    return True

def create_sample_data():
    """Create sample data for testing"""
    # Sample tick data
    ticks = {
        'timestamp': pd.date_range('2025-06-23 05:00:00', periods=100, freq='10S'),
        'bid': [3357.50 + (i % 10) * 0.1 for i in range(100)],
        'ask': [3357.80 + (i % 10) * 0.1 for i in range(100)],
        'spread_points': [30] * 100,
        'volume': [100 + (i % 5) * 50 for i in range(100)]
    }

    tick_df = pd.DataFrame(ticks)
    tick_df.to_csv('data/sample_ticks.csv', index=False)

    # Sample OHLCV data
    ohlcv = {
        'timestamp': pd.date_range('2025-06-23', periods=24, freq='H'),
        'open': [3357.50 + (i % 5) for i in range(24)],
        'high': [3358.00 + (i % 5) for i in range(24)],
        'low': [3357.00 + (i % 5) for i in range(24)],
        'close': [3357.75 + (i % 5) for i in range(24)],
        'volume': [1000 + (i % 10) * 100 for i in range(24)]
    }

    ohlcv_df = pd.DataFrame(ohlcv)
    ohlcv_df.to_csv('data/sample_ohlcv.csv', index=False)

    print("✅ Created sample data files in data/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ncOS Data Converter')
    parser.add_argument('--convert', help='Convert MT4 to ncOS format')
    parser.add_argument('--aggregate', help='Aggregate ticks to OHLCV')
    parser.add_argument('--timeframe', default='5Min', help='Aggregation timeframe')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--sample', action='store_true', help='Create sample data')

    args = parser.parse_args()

    if args.sample:
        create_sample_data()
    elif args.convert and args.output:
        convert_mt4_to_ncos(args.convert, args.output)
    elif args.aggregate and args.output:
        aggregate_ticks_to_ohlcv(args.aggregate, args.output, args.timeframe)
    else:
        parser.print_help()
