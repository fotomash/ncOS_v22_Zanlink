#!/usr/bin/env python3
"""
ncOS - Ultimate Trading Data Processor
Default behavior: Scan entire folder for CSV files and process everything
"""

import pandas as pd
import numpy as np
import talib
import os
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    directory: str = "."
    file_pattern: str = "*.csv"
    timeframes: List[str] = None
    output_dir: str = "processed_data"
    process_all_indicators: bool = True
    process_all_timeframes: bool = True
    delimiter: str = "auto"  # auto-detect between comma and tab

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1min', '5min', '15min', '30min', '1H', '4H', '1D']

class TimeframeDetector:
    """Detects timeframe from filename patterns"""

    TIMEFRAME_PATTERNS = {
        r'[_\-\s]?m1[_\-\s]?': '1min',
        r'[_\-\s]?m5[_\-\s]?': '5min',
        r'[_\-\s]?m15[_\-\s]?': '15min',
        r'[_\-\s]?m30[_\-\s]?': '30min',
        r'[_\-\s]?h1[_\-\s]?': '1H',
        r'[_\-\s]?h4[_\-\s]?': '4H',
        r'[_\-\s]?d1[_\-\s]?': '1D',
        r'[_\-\s]?1min[_\-\s]?': '1min',
        r'[_\-\s]?5min[_\-\s]?': '5min',
        r'[_\-\s]?15min[_\-\s]?': '15min',
        r'[_\-\s]?30min[_\-\s]?': '30min',
        r'[_\-\s]?1h[_\-\s]?': '1H',
        r'[_\-\s]?4h[_\-\s]?': '4H',
        r'[_\-\s]?1d[_\-\s]?': '1D',
    }

    @classmethod
    def detect_timeframe(cls, filename: str) -> Optional[str]:
        """Detect timeframe from filename"""
        filename_lower = filename.lower()

        for pattern, timeframe in cls.TIMEFRAME_PATTERNS.items():
            if re.search(pattern, filename_lower):
                return timeframe

        return '1min'  # Default fallback

class PairDetector:
    """Detects currency pair from filename"""

    PAIR_PATTERNS = [
        r'([A-Z]{6})',  # EURUSD, GBPUSD, etc.
        r'([A-Z]{3}[A-Z]{3})',  # EUR USD as EURUSD
        r'(XAU[A-Z]{3})',  # XAUUSD, XAUEUR, etc.
        r'(XAG[A-Z]{3})',  # XAGUSD, etc.
        r'([A-Z]{3}JPY)',  # USDJPY, EURJPY, etc.
    ]

    @classmethod
    def detect_pair(cls, filename: str) -> str:
        """Detect currency pair from filename"""
        filename_upper = filename.upper()

        for pattern in cls.PAIR_PATTERNS:
            match = re.search(pattern, filename_upper)
            if match:
                return match.group(1)

        # Fallback to filename without extension
        return Path(filename).stem.upper()

class TechnicalIndicators:
    """Comprehensive technical indicators calculator"""

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators"""
        df = data.copy()

        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required OHLC columns. Available: {list(df.columns)}")
            return df

        try:
            # Basic price data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
                df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
                df[f'WMA_{period}'] = talib.WMA(close, timeperiod=period)

            # MACD
            macd, macdsignal, macdhist = talib.MACD(close)
            df['MACD'] = macd
            df['MACD_Signal'] = macdsignal
            df['MACD_Hist'] = macdhist

            # RSI
            for period in [14, 21]:
                df[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            df['STOCH_K'] = slowk
            df['STOCH_D'] = slowd

            # Bollinger Bands
            for period in [20, 50]:
                upper, middle, lower = talib.BBANDS(close, timeperiod=period)
                df[f'BB_Upper_{period}'] = upper
                df[f'BB_Middle_{period}'] = middle
                df[f'BB_Lower_{period}'] = lower
                df[f'BB_Width_{period}'] = (upper - lower) / middle

            # ATR
            for period in [14, 21]:
                df[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)

            # ADX
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            df['DI_Plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['DI_Minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # CCI
            df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

            # Williams %R
            df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

            # Momentum
            df['MOM'] = talib.MOM(close, timeperiod=10)

            # Rate of Change
            df['ROC'] = talib.ROC(close, timeperiod=10)

            # Parabolic SAR
            df['SAR'] = talib.SAR(high, low)

            # Volume indicators (if volume is available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['OBV'] = talib.OBV(close, volume)
                df['AD'] = talib.AD(high, low, close, volume)
                df['ADOSC'] = talib.ADOSC(high, low, close, volume)

            # Price patterns
            df['DOJI'] = talib.CDLDOJI(open_price, high, low, close)
            df['HAMMER'] = talib.CDLHAMMER(open_price, high, low, close)
            df['ENGULFING'] = talib.CDLENGULFING(open_price, high, low, close)
            df['HARAMI'] = talib.CDLHARAMI(open_price, high, low, close)

            # Support and Resistance levels
            df['Pivot'] = (high + low + close) / 3
            df['R1'] = 2 * df['Pivot'] - low
            df['S1'] = 2 * df['Pivot'] - high
            df['R2'] = df['Pivot'] + (high - low)
            df['S2'] = df['Pivot'] - (high - low)

            # Custom indicators
            df['HL_Ratio'] = (high - low) / close
            df['Price_Change'] = close - open_price
            df['Price_Change_Pct'] = (close - open_price) / open_price * 100
            df['High_Low_Pct'] = (high - low) / low * 100

            # Trend strength
            df['Trend_Strength'] = np.where(
                df['EMA_20'] > df['EMA_50'], 1,
                np.where(df['EMA_20'] < df['EMA_50'], -1, 0)
            )

            print(f"✓ Calculated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")

        except Exception as e:
            print(f"Warning: Error calculating some indicators: {e}")

        return df

class DataProcessor:
    """Main data processing engine"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_files = []
        self.session_stats = {
            'start_time': datetime.now(),
            'files_processed': 0,
            'total_indicators': 0,
            'errors': []
        }

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.output_dir, 'processing.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _detect_delimiter(self, file_path: str) -> str:
        """Auto-detect file delimiter"""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line or first_line.count('\t') > first_line.count(','):
                    return '\t'
                else:
                    return ','
        except:
            return ','

    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and validate data from file"""
        try:
            # Auto-detect delimiter if needed
            delimiter = self.config.delimiter
            if delimiter == "auto":
                delimiter = self._detect_delimiter(file_path)

            # Try different ways to load the data
            try:
                df = pd.read_csv(file_path, delimiter=delimiter)
            except:
                df = pd.read_csv(file_path, sep='\t')

            # Clean column names
            df.columns = df.columns.str.strip().str.lower()

            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                # Try alternative column names
                col_mapping = {
                    'time': 'timestamp',
                    'date': 'timestamp',
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'vol': 'volume',
                    'v': 'volume'
                }

                for old_name, new_name in col_mapping.items():
                    if old_name in df.columns and new_name in missing_cols:
                        df.rename(columns={old_name: new_name}, inplace=True)
                        missing_cols.remove(new_name)

            if missing_cols:
                self.logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                return None

            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Ensure numeric types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN in OHLC
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

            if len(df) == 0:
                self.logger.warning(f"No valid data in {file_path}")
                return None

            self.logger.info(f"✓ Loaded {len(df)} rows from {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            self.session_stats['errors'].append(f"Load error in {file_path}: {e}")
            return None

    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        timeframe_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D'
        }

        if timeframe not in timeframe_map:
            return df

        freq = timeframe_map[timeframe]

        try:
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df.columns else 'mean'
            }).dropna()

            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling to {timeframe}: {e}")
            return df

    def _save_results(self, df: pd.DataFrame, original_file: str, timeframe: str, pair: str):
        """Save processed results"""
        try:
            # Create pair directory
            pair_dir = os.path.join(self.config.output_dir, pair)
            os.makedirs(pair_dir, exist_ok=True)

            # Generate filename
            base_name = Path(original_file).stem
            output_file = f"{base_name}_{timeframe}_processed.csv"
            output_path = os.path.join(pair_dir, output_file)

            # Save the data
            df.to_csv(output_path)

            # Track processed file
            self.processed_files.append({
                'original': original_file,
                'processed': output_path,
                'timeframe': timeframe,
                'pair': pair,
                'rows': len(df),
                'indicators': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
            })

            self.logger.info(f"✓ Saved {timeframe} data to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.session_stats['errors'].append(f"Save error: {e}")

    def _generate_journal(self):
        """Generate processing journal"""
        try:
            journal_path = os.path.join(self.config.output_dir, 'processing_journal.json')

            journal = {
                'session_info': {
                    'start_time': self.session_stats['start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': (datetime.now() - self.session_stats['start_time']).total_seconds() / 60,
                    'files_processed': len(self.processed_files),
                    'total_indicators_calculated': sum(f['indicators'] for f in self.processed_files),
                    'errors': self.session_stats['errors']
                },
                'configuration': {
                    'directory_scanned': self.config.directory,
                    'file_pattern': self.config.file_pattern,
                    'timeframes_processed': self.config.timeframes,
                    'output_directory': self.config.output_dir
                },
                'processed_files': self.processed_files,
                'summary': {
                    'unique_pairs': len(set(f['pair'] for f in self.processed_files)),
                    'unique_timeframes': len(set(f['timeframe'] for f in self.processed_files)),
                    'total_rows_processed': sum(f['rows'] for f in self.processed_files)
                }
            }

            with open(journal_path, 'w') as f:
                json.dump(journal, f, indent=2)

            self.logger.info(f"✓ Generated processing journal: {journal_path}")

        except Exception as e:
            self.logger.error(f"Error generating journal: {e}")

    def process_all_files(self):
        """Process all files in the directory"""
        try:
            # Find all CSV files
            search_pattern = os.path.join(self.config.directory, self.config.file_pattern)
            csv_files = glob.glob(search_pattern)

            if not csv_files:
                self.logger.warning(f"No CSV files found in {self.config.directory} with pattern {self.config.file_pattern}")
                return

            self.logger.info(f"Found {len(csv_files)} CSV files to process")

            for file_path in csv_files:
                self.logger.info(f"\n🔄 Processing: {file_path}")

                # Load original data
                original_data = self._load_data(file_path)
                if original_data is None:
                    continue

                # Detect pair and original timeframe
                pair = PairDetector.detect_pair(os.path.basename(file_path))
                detected_timeframe = TimeframeDetector.detect_timeframe(os.path.basename(file_path))

                self.logger.info(f"   📊 Detected: {pair} | {detected_timeframe}")

                # Process all requested timeframes
                for target_timeframe in self.config.timeframes:
                    try:
                        # Resample if needed
                        if target_timeframe == detected_timeframe:
                            processed_data = original_data.copy()
                        else:
                            processed_data = self._resample_data(original_data, target_timeframe)

                        if len(processed_data) == 0:
                            continue

                        # Calculate all indicators
                        if self.config.process_all_indicators:
                            processed_data = TechnicalIndicators.calculate_all_indicators(processed_data)

                        # Save results
                        self._save_results(processed_data, file_path, target_timeframe, pair)

                        self.logger.info(f"   ✅ {target_timeframe}: {len(processed_data)} rows with indicators")

                    except Exception as e:
                        self.logger.error(f"   ❌ Error processing {target_timeframe}: {e}")
                        self.session_stats['errors'].append(f"{file_path} - {target_timeframe}: {e}")

            # Generate final journal
            self._generate_journal()

            # Print summary
            self._print_summary()

        except Exception as e:
            self.logger.error(f"Critical error in processing: {e}")

    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*80)
        print("🎉 PROCESSING COMPLETE!")
        print("="*80)
        print(f"📁 Files processed: {len(set(f['original'] for f in self.processed_files))}")
        print(f"📊 Total outputs: {len(self.processed_files)}")
        print(f"💱 Currency pairs: {len(set(f['pair'] for f in self.processed_files))}")
        print(f"⏰ Timeframes: {', '.join(set(f['timeframe'] for f in self.processed_files))}")
        print(f"📈 Total indicators: {sum(f['indicators'] for f in self.processed_files)}")
        print(f"📂 Output directory: {self.config.output_dir}")
        if self.session_stats['errors']:
            print(f"⚠️  Errors encountered: {len(self.session_stats['errors'])}")
        print("="*80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ncOS - Ultimate Trading Data Processor (Default: Process everything)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                           # Process all CSV files with all timeframes and indicators
  python script.py --dir /path/to/data       # Process files in specific directory
  python script.py --pattern "*EURUSD*"      # Process only EURUSD files
  python script.py --timeframes 1min 5min    # Process only specific timeframes
  python script.py --output results          # Custom output directory
        """
    )

    # All arguments are optional with sensible defaults
    parser.add_argument('--dir', '--directory', 
                       default='.',
                       help='Directory to scan for CSV files (default: current directory)')

    parser.add_argument('--pattern', 
                       default='*.csv',
                       help='File pattern to match (default: *.csv)')

    parser.add_argument('--timeframes', 
                       nargs='+',
                       default=['1min', '5min', '15min', '30min', '1H', '4H', '1D'],
                       choices=['1min', '5min', '15min', '30min', '1H', '4H', '1D'],
                       help='Timeframes to process (default: all)')

    parser.add_argument('--output', 
                       default='processed_data',
                       help='Output directory (default: processed_data)')

    parser.add_argument('--delimiter',
                       default='auto',
                       choices=['auto', ',', '\t'],
                       help='CSV delimiter (default: auto-detect)')

    args = parser.parse_args()

    # Create configuration
    config = ProcessingConfig(
        directory=args.dir,
        file_pattern=args.pattern,
        timeframes=args.timeframes,
        output_dir=args.output,
        delimiter=args.delimiter
    )

    # Print startup info
    print("🚀 ncOS - Ultimate Trading Data Processor")
    print("="*50)
    print(f"📁 Scanning directory: {config.directory}")
    print(f"🔍 File pattern: {config.file_pattern}")
    print(f"⏰ Timeframes: {', '.join(config.timeframes)}")
    print(f"📂 Output directory: {config.output_dir}")
    print("="*50)

    # Process files
    processor = DataProcessor(config)
    processor.process_all_files()

if __name__ == "__main__":
    main()
