#!/usr/bin/env python3
"""
ncOS - Ultimate Trading Data Processor with JSON Support
Default behavior: Scan for CSV AND JSON files, process everything including tick data
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
from typing import Dict, List, Optional, Tuple, Any, Union
import re
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    directory: str = "."
    file_pattern: str = "*"  # Changed to include both CSV and JSON by default
    file_types: List[str] = None  # ['csv', 'json'] or ['json'] for json-only
    timeframes: List[str] = None
    output_dir: str = "processed_data"
    process_all_indicators: bool = True
    process_all_timeframes: bool = True
    process_tick_data: bool = True
    delimiter: str = "auto"
    json_only: bool = False

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1min', '5min', '15min', '30min', '1H', '4H', '1D']
        if self.file_types is None:
            self.file_types = ['json'] if self.json_only else ['csv', 'json']

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
        # Tick data patterns
        r'tick': 'tick',
        r'ticks': 'tick',
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

class JSONAnalysisProcessor:
    """Processes JSON analysis files"""

    @staticmethod
    def load_json_analysis(file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON analysis file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    @staticmethod
    def extract_timeframe_data(json_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Extract timeframe data from JSON analysis"""
        timeframe_data = {}

        if 'results' not in json_data:
            return timeframe_data

        timeframe_mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        for tf_key, tf_data in json_data['results'].items():
            if tf_key in timeframe_mapping:
                timeframe = timeframe_mapping[tf_key]

                # Extract basic data
                df_data = {
                    'timestamp': pd.date_range(
                        start=tf_data['date_range']['start'],
                        end=tf_data['date_range']['end'],
                        periods=tf_data['data_points']
                    ),
                    'open': [tf_data['price_summary']['open']] * tf_data['data_points'],
                    'high': [tf_data['price_summary']['high']] * tf_data['data_points'],
                    'low': [tf_data['price_summary']['low']] * tf_data['data_points'],
                    'close': [tf_data['price_summary']['close']] * tf_data['data_points'],
                    'volume': [tf_data['price_summary']['volume']] * tf_data['data_points']
                }

                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)

                # Add pre-calculated indicators
                if 'indicators' in tf_data:
                    indicators = tf_data['indicators']
                    for indicator, value in indicators.items():
                        if value is not None:
                            df[f'JSON_{indicator.upper()}'] = value

                # Add pattern analysis
                if 'patterns' in tf_data and 'counts' in tf_data['patterns']:
                    patterns = tf_data['patterns']['counts']
                    for pattern, counts in patterns.items():
                        df[f'JSON_{pattern.upper()}_bullish'] = counts.get('bullish', 0)
                        df[f'JSON_{pattern.upper()}_bearish'] = counts.get('bearish', 0)
                        df[f'JSON_{pattern.upper()}_total'] = counts.get('total', 0)

                # Add Wyckoff analysis
                if 'wyckoff' in tf_data:
                    wyckoff = tf_data['wyckoff']
                    df['JSON_WYCKOFF_phase'] = wyckoff.get('current_phase', 'unknown')
                    if 'phases' in wyckoff:
                        for phase, count in wyckoff['phases'].items():
                            df[f'JSON_WYCKOFF_{phase}'] = count

                # Add SMC analysis
                if 'smc' in tf_data:
                    smc = tf_data['smc']
                    df['JSON_SMC_structure'] = smc.get('structure', 'unknown')
                    if 'key_levels' in smc:
                        df['JSON_SMC_support'] = smc['key_levels'].get('support', 0)
                        df['JSON_SMC_resistance'] = smc['key_levels'].get('resistance', 0)

                # Add trend analysis
                if 'trend_analysis' in tf_data:
                    trend = tf_data['trend_analysis']
                    df['JSON_TREND_direction'] = trend.get('direction', 'unknown')
                    df['JSON_TREND_price_vs_sma20'] = trend.get('price_vs_sma20', 'unknown')

                timeframe_data[timeframe] = df

        return timeframe_data

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
            df['SHOOTING_STAR'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)

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

            # Tick data specific indicators (if applicable)
            if 'tick' in df.columns or len(df) > 10000:  # Assume tick data if many rows
                df['Tick_Direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
                df['Tick_Volume_Weighted'] = df['close'] * df['volume'] if 'volume' in df.columns else df['close']
                df['Tick_Spread'] = df['high'] - df['low']
                df['Tick_Volatility'] = df['close'].rolling(window=100).std()

            print(f"✓ Calculated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")

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
            'json_files_processed': 0,
            'csv_files_processed': 0,
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

    def _find_files(self) -> List[str]:
        """Find all files to process based on configuration"""
        all_files = []

        for file_type in self.config.file_types:
            if file_type == 'csv':
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}.csv")
                csv_files = glob.glob(pattern)
                all_files.extend(csv_files)
            elif file_type == 'json':
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}.json")
                json_files = glob.glob(pattern)
                all_files.extend(json_files)

        # Also check for files without specific extensions if pattern doesn't include extension
        if '.' not in self.config.file_pattern:
            for ext in self.config.file_types:
                pattern = os.path.join(self.config.directory, f"{self.config.file_pattern}*.{ext}")
                files = glob.glob(pattern)
                all_files.extend(files)

        return list(set(all_files))  # Remove duplicates

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

    def _load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and validate CSV data from file"""
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
                    'datetime': 'timestamp',
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

            self.logger.info(f"✓ Loaded {len(df)} rows from CSV: {file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {e}")
            self.session_stats['errors'].append(f"CSV load error in {file_path}: {e}")
            return None

    def _load_json_data(self, file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load and process JSON analysis data"""
        try:
            json_data = JSONAnalysisProcessor.load_json_analysis(file_path)
            if json_data is None:
                return None

            timeframe_data = JSONAnalysisProcessor.extract_timeframe_data(json_data)

            if timeframe_data:
                self.logger.info(f"✓ Loaded {len(timeframe_data)} timeframes from JSON: {file_path}")
                return timeframe_data
            else:
                self.logger.warning(f"No timeframe data found in JSON: {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading JSON {file_path}: {e}")
            self.session_stats['errors'].append(f"JSON load error in {file_path}: {e}")
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

    def _save_results(self, df: pd.DataFrame, original_file: str, timeframe: str, pair: str, source_type: str = 'csv'):
        """Save processed results"""
        try:
            # Create pair directory
            pair_dir = os.path.join(self.config.output_dir, pair)
            os.makedirs(pair_dir, exist_ok=True)

            # Generate filename
            base_name = Path(original_file).stem
            output_file = f"{base_name}_{timeframe}_{source_type}_processed.csv"
            output_path = os.path.join(pair_dir, output_file)

            # Save the data
            df.to_csv(output_path)

            # Track processed file
            self.processed_files.append({
                'original': original_file,
                'processed': output_path,
                'timeframe': timeframe,
                'pair': pair,
                'source_type': source_type,
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
                    'total_files_processed': len(self.processed_files),
                    'csv_files_processed': self.session_stats['csv_files_processed'],
                    'json_files_processed': self.session_stats['json_files_processed'],
                    'total_indicators_calculated': sum(f['indicators'] for f in self.processed_files),
                    'errors': self.session_stats['errors']
                },
                'configuration': {
                    'directory_scanned': self.config.directory,
                    'file_pattern': self.config.file_pattern,
                    'file_types': self.config.file_types,
                    'json_only_mode': self.config.json_only,
                    'timeframes_processed': self.config.timeframes,
                    'output_directory': self.config.output_dir,
                    'tick_data_processing': self.config.process_tick_data
                },
                'processed_files': self.processed_files,
                'summary': {
                    'unique_pairs': len(set(f['pair'] for f in self.processed_files)),
                    'unique_timeframes': len(set(f['timeframe'] for f in self.processed_files)),
                    'total_rows_processed': sum(f['rows'] for f in self.processed_files),
                    'source_types': list(set(f['source_type'] for f in self.processed_files))
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
            # Find all files
            all_files = self._find_files()

            if not all_files:
                file_types_str = ', '.join(self.config.file_types)
                self.logger.warning(f"No {file_types_str} files found in {self.config.directory}")
                return

            csv_files = [f for f in all_files if f.endswith('.csv')]
            json_files = [f for f in all_files if f.endswith('.json')]

            self.logger.info(f"Found {len(csv_files)} CSV files and {len(json_files)} JSON files to process")

            # Process CSV files
            for file_path in csv_files:
                self.logger.info(f"\n🔄 Processing CSV: {file_path}")

                # Load original data
                original_data = self._load_csv_data(file_path)
                if original_data is None:
                    continue

                self.session_stats['csv_files_processed'] += 1

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
                        self._save_results(processed_data, file_path, target_timeframe, pair, 'csv')

                        self.logger.info(f"   ✅ {target_timeframe}: {len(processed_data)} rows with indicators")

                    except Exception as e:
                        self.logger.error(f"   ❌ Error processing {target_timeframe}: {e}")
                        self.session_stats['errors'].append(f"{file_path} - {target_timeframe}: {e}")

            # Process JSON files
            for file_path in json_files:
                self.logger.info(f"\n🔄 Processing JSON: {file_path}")

                # Load JSON data (already contains multiple timeframes)
                timeframe_data = self._load_json_data(file_path)
                if timeframe_data is None:
                    continue

                self.session_stats['json_files_processed'] += 1

                # Detect pair
                pair = PairDetector.detect_pair(os.path.basename(file_path))

                self.logger.info(f"   📊 Detected: {pair} | {len(timeframe_data)} timeframes")

                # Process each timeframe from JSON
                for timeframe, json_df in timeframe_data.items():
                    try:
                        # Calculate additional indicators if requested
                        if self.config.process_all_indicators:
                            processed_data = TechnicalIndicators.calculate_all_indicators(json_df)
                        else:
                            processed_data = json_df

                        # Save results
                        self._save_results(processed_data, file_path, timeframe, pair, 'json')

                        self.logger.info(f"   ✅ {timeframe}: {len(processed_data)} rows with JSON analysis + indicators")

                    except Exception as e:
                        self.logger.error(f"   ❌ Error processing JSON {timeframe}: {e}")
                        self.session_stats['errors'].append(f"{file_path} - {timeframe}: {e}")

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
        print(f"📁 CSV files processed: {self.session_stats['csv_files_processed']}")
        print(f"📄 JSON files processed: {self.session_stats['json_files_processed']}")
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
        description='ncOS - Ultimate Trading Data Processor with JSON Support (Default: Process CSV & JSON)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py                           # Process all CSV & JSON files with everything
  python script.py --json-only               # Process ONLY JSON files
  python script.py --dir /path/to/data       # Process files in specific directory
  python script.py --pattern "*EURUSD*"      # Process only EURUSD files
  python script.py --timeframes 1min 5min    # Process only specific timeframes
  python script.py --output results          # Custom output directory
  python script.py --no-tick-data            # Skip tick data processing
        """
    )

    # All arguments are optional with sensible defaults
    parser.add_argument('--dir', '--directory', 
                       default='.',
                       help='Directory to scan for files (default: current directory)')

    parser.add_argument('--pattern', 
                       default='*',
                       help='File pattern to match (default: *)')

    parser.add_argument('--json-only', 
                       action='store_true',
                       help='Process ONLY JSON files (ignore CSV files)')

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

    parser.add_argument('--no-tick-data',
                       action='store_true',
                       help='Skip tick data processing')

    args = parser.parse_args()

    # Create configuration
    config = ProcessingConfig(
        directory=args.dir,
        file_pattern=args.pattern,
        timeframes=args.timeframes,
        output_dir=args.output,
        delimiter=args.delimiter,
        json_only=args.json_only,
        process_tick_data=not args.no_tick_data
    )

    # Print startup info
    print("🚀 ncOS - Ultimate Trading Data Processor with JSON Support")
    print("="*60)
    print(f"📁 Scanning directory: {config.directory}")
    print(f"🔍 File pattern: {config.file_pattern}")
    print(f"📄 File types: {', '.join(config.file_types)}")
    print(f"⏰ Timeframes: {', '.join(config.timeframes)}")
    print(f"🎯 JSON only mode: {'ON' if config.json_only else 'OFF'}")
    print(f"📊 Tick data processing: {'ON' if config.process_tick_data else 'OFF'}")
    print(f"📂 Output directory: {config.output_dir}")
    print("="*60)

    # Process files
    processor = DataProcessor(config)
    processor.process_all_files()

if __name__ == "__main__":
    main()
