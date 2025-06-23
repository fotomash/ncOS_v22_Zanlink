import os
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import yaml

# Technical Analysis Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not installed. Using pandas_ta as fallback.")

import pandas_ta as ta

# Smart Money Concepts Market Structure Analysis
try:
    from market_structure_analyzer_smc import analyze_market_structure
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    warnings.warn("market_structure_analyzer_smc not available. SMC analysis will be skipped.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats"""
    ANNOTATED = "annotated"  # Your tick CSVs with <COLUMN> format
    MT5 = "mt5"  # MetaTrader 5 exports
    GENERIC = "generic"  # Generic OHLCV format


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    timeframes: List[str] = None
    indicators: Dict[str, Dict] = None
    output_dir: str = "processed_output"
    parallel_processing: bool = True
    max_workers: int = None
    use_tick_volume_for_volume_indicators: bool = True
    process_single_timeframe: bool = True  # NEW: Process only detected timeframe

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D', '1W', '1M']
        if self.max_workers is None:
            self.max_workers = max(1, os.cpu_count() - 1)


class TimeframeDetector:
    """Detect timeframe from filename"""
    
    TIMEFRAME_PATTERNS = {
        '_m1': '1T',
        '_1m': '1T',
        '_1min': '1T',
        '_m5': '5T',
        '_5m': '5T',
        '_5min': '5T',
        '_m15': '15T',
        '_15m': '15T',
        '_15min': '15T',
        '_m30': '30T',
        '_30m': '30T',
        '_30min': '30T',
        '_h1': '1H',
        '_1h': '1H',
        '_60min': '1H',
        '_h4': '4H',
        '_4h': '4H',
        '_240min': '4H',
        '_d1': '1D',
        '_1d': '1D',
        '_daily': '1D',
        '_w1': '1W',
        '_1w': '1W',
        '_weekly': '1W',
        '_mn1': '1M',
        '_1mo': '1M',
        '_monthly': '1M'
    }
    
    @classmethod
    def detect_timeframe(cls, file_path: Path) -> Optional[str]:
        """Detect timeframe from filename"""
        filename_lower = file_path.stem.lower()
        
        for pattern, timeframe in cls.TIMEFRAME_PATTERNS.items():
            if pattern in filename_lower:
                logger.info(f"Detected timeframe {timeframe} from pattern '{pattern}' in {file_path.name}")
                return timeframe
        
        return None


class CSVFormatDetector:
    """Auto-detect CSV format and delimiter"""

    @staticmethod
    def detect_format(file_path: Path) -> Tuple[DataFormat, str]:
        """Detect CSV format and delimiter"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]

        # Check delimiter
        delimiter = '\t' if '\t' in first_lines[0] else ','

        # Check format by column names
        header = first_lines[0].strip().split(delimiter)
        header_lower = [col.lower().strip() for col in header]

        if any('<' in col and '>' in col for col in header):
            return DataFormat.ANNOTATED, delimiter
        elif 'time' in header_lower and 'tick volume' in header_lower:
            return DataFormat.MT5, delimiter
        else:
            return DataFormat.GENERIC, delimiter

    @staticmethod
    def normalize_columns(df: pd.DataFrame, format_type: DataFormat) -> pd.DataFrame:
        """Normalize column names based on format"""
        if format_type == DataFormat.ANNOTATED:
            # Handle <COLUMN> format
            column_mapping = {
                '<DATE>': 'date',
                '<TIME>': 'time',
                '<OPEN>': 'open',
                '<HIGH>': 'high',
                '<LOW>': 'low',
                '<CLOSE>': 'close',
                '<TICKVOL>': 'tickvol',
                '<VOL>': 'volume',
                '<SPREAD>': 'spread'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Log volume data status
            if 'volume' in df.columns and 'tickvol' in df.columns:
                vol_sum = df['volume'].sum()
                tick_sum = df['tickvol'].sum()
                logger.info(f"Volume data: real volume sum={vol_sum}, tick volume sum={tick_sum}")

        elif format_type == DataFormat.MT5:
            # Handle MT5 format
            column_mapping = {
                'Time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Tick Volume': 'tickvol',
                'Volume': 'volume',
                'Spread': 'spread'
            }
            df.rename(columns=column_mapping, inplace=True)

        else:
            # Generic format - lowercase all
            df.columns = df.columns.str.lower().str.strip()

        return df


class TechnicalIndicatorEngine:
    """Comprehensive technical indicator calculator with robust error handling"""

    def __init__(self, use_talib: bool = TALIB_AVAILABLE, use_tick_volume: bool = True):
        self.use_talib = use_talib
        self.use_tick_volume = use_tick_volume
        logger.info(f"Indicator engine initialized. Using TA-Lib: {use_talib}, Use tick volume: {use_tick_volume}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all professional trading indicators"""
        df = df.copy()

        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return df

        # Handle volume data
        if 'volume' not in df.columns and 'tickvol' in df.columns:
            df['volume'] = df['tickvol']
            logger.info("Using tick volume as volume")
        elif 'volume' not in df.columns:
            df['volume'] = 0
            logger.warning("No volume data available")

        # Check if we have real volume or just tick volume
        has_real_volume = df['volume'].sum() > 0 if 'volume' in df.columns else False
        has_tick_volume = df['tickvol'].sum() > 0 if 'tickvol' in df.columns else False

        logger.info(f"Data has real volume: {has_real_volume}, has tick volume: {has_tick_volume}")

        # Calculate indicators by category with error handling
        logger.info("Calculating trend indicators...")
        df = self._calculate_trend_indicators_safe(df)

        logger.info("Calculating momentum indicators...")
        df = self._calculate_momentum_indicators_safe(df)

        logger.info("Calculating volatility indicators...")
        df = self._calculate_volatility_indicators_safe(df)

        if has_real_volume or (has_tick_volume and self.use_tick_volume):
            logger.info("Calculating volume indicators...")
            df = self._calculate_volume_indicators_safe(df)
        else:
            logger.info("Skipping volume indicators - no volume data")

        logger.info("Calculating support/resistance levels...")
        df = self._calculate_structure_indicators_safe(df)

        logger.info("Calculating advanced indicators...")
        df = self._calculate_advanced_indicators_safe(df)

        logger.info("Detecting patterns...")
        df = self._detect_patterns_safe(df)

        logger.info("Calculating risk metrics...")
        df = self._calculate_risk_metrics_safe(df)

        return df

    # [Keep all the _calculate_*_safe methods unchanged from original]
    def _calculate_trend_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators with error handling"""
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:  # Only calculate if we have enough data
                try:
                    df[f'sma_{period}'] = ta.sma(df['close'], length=period)
                    df[f'ema_{period}'] = ta.ema(df['close'], length=period)
                    df[f'wma_{period}'] = ta.wma(df['close'], length=period)
                except Exception as e:
                    logger.warning(f"MA calculation failed for period {period}: {e}")

        # Advanced MAs
        try:
            df['hma'] = ta.hma(df['close'], length=20)
        except: pass

        try:
            df['kama'] = ta.kama(df['close'], length=10)
        except: pass

        try:
            df['t3'] = ta.t3(df['close'], length=10)
        except: pass

        try:
            df['zlema'] = ta.zlma(df['close'], length=20)
        except: pass

        # Parabolic SAR
        try:
            psar = ta.psar(df['high'], df['low'], df['close'])
            if isinstance(psar, pd.DataFrame) and len(psar.columns) > 0:
                for col in psar.columns:
                    if 'PSARl' in col or 'PSAR_long' in col or col.endswith('_0.02_0.2'):
                        if 'long' in col.lower() or 'PSARl' in col:
                            df['psar_long'] = psar[col]
                        elif 'short' in col.lower() or 'PSARs' in col:
                            df['psar_short'] = psar[col]
        except Exception as e:
            logger.warning(f"PSAR calculation failed: {e}")

        # ADX
        try:
            adx = ta.adx(df['high'], df['low'], df['close'])
            if isinstance(adx, pd.DataFrame) and len(adx.columns) > 0:
                for col in adx.columns:
                    if 'ADX' in col and not any(x in col for x in ['DMP', 'DMN', '+DI', '-DI']):
                        df['adx'] = adx[col]
                    elif 'DMP' in col or '+DI' in col:
                        df['di_plus'] = adx[col]
                    elif 'DMN' in col or '-DI' in col:
                        df['di_minus'] = adx[col]
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")

        # Aroon
        try:
            aroon = ta.aroon(df['high'], df['low'])
            if isinstance(aroon, pd.DataFrame) and len(aroon.columns) > 0:
                logger.debug(f"Aroon columns: {aroon.columns.tolist()}")
                for col in aroon.columns:
                    col_str = str(col)
                    if any(pattern in col_str for pattern in ['AROONU', 'AroonUp', 'Aroon_Up']):
                        df['aroon_up'] = aroon[col]
                    elif any(pattern in col_str for pattern in ['AROOND', 'AroonDown', 'Aroon_Down']):
                        df['aroon_down'] = aroon[col]
                    elif any(pattern in col_str for pattern in ['AROONOSC', 'AroonOsc', 'Aroon_Osc']):
                        df['aroon_osc'] = aroon[col]
        except Exception as e:
            logger.warning(f"Aroon calculation failed: {e}")

        # CCI
        try:
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        except: pass

        # Linear Regression
        try:
            df['linreg'] = ta.linreg(df['close'], length=14)
        except: pass

        return df

    def _calculate_momentum_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators with error handling"""
        # RSI
        try:
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
        except: pass

        # Stochastic
        try:
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if isinstance(stoch, pd.DataFrame) and len(stoch.columns) > 0:
                for col in stoch.columns:
                    col_str = str(col)
                    if any(pattern in col_str for pattern in ['STOCHk', 'K%', 'stoch_k']):
                        df['stoch_k'] = stoch[col]
                    elif any(pattern in col_str for pattern in ['STOCHd', 'D%', 'stoch_d']):
                        df['stoch_d'] = stoch[col]
        except: pass

        # Williams %R
        try:
            df['willr'] = ta.willr(df['high'], df['low'], df['close'])
        except: pass

        # ROC and Momentum
        try:
            df['roc'] = ta.roc(df['close'], length=10)
            df['mom'] = ta.mom(df['close'], length=10)
        except: pass

        # MACD
        try:
            macd = ta.macd(df['close'])
            if isinstance(macd, pd.DataFrame) and len(macd.columns) > 0:
                for col in macd.columns:
                    col_str = str(col)
                    if 'MACD_' in col_str and not any(x in col_str for x in ['h', 's']):
                        df['macd'] = macd[col]
                    elif 'MACDs' in col_str or 'signal' in col_str.lower():
                        df['macd_signal'] = macd[col]
                    elif 'MACDh' in col_str or 'hist' in col_str.lower():
                        df['macd_hist'] = macd[col]
        except: pass

        # Other momentum indicators
        try:
            df['ultimate'] = ta.uo(df['high'], df['low'], df['close'])
        except: pass

        try:
            df['cmo'] = ta.cmo(df['close'])
        except: pass

        try:
            df['trix'] = ta.trix(df['close'])
        except: pass

        return df

    def _calculate_volatility_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators with error handling"""
        # Bollinger Bands
        try:
            bbands = ta.bbands(df['close'])
            if isinstance(bbands, pd.DataFrame) and len(bbands.columns) > 0:
                for col in bbands.columns:
                    col_str = str(col)
                    if 'BBU' in col_str or 'upper' in col_str.lower():
                        df['bb_upper'] = bbands[col]
                    elif 'BBM' in col_str or 'mid' in col_str.lower():
                        df['bb_middle'] = bbands[col]
                    elif 'BBL' in col_str or 'lower' in col_str.lower():
                        df['bb_lower'] = bbands[col]
                    elif 'BBB' in col_str or 'width' in col_str.lower():
                        df['bb_width'] = bbands[col]
                    elif 'BBP' in col_str or 'percent' in col_str.lower():
                        df['bb_percent'] = bbands[col]
        except: pass

        # ATR
        try:
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            df['natr'] = ta.natr(df['high'], df['low'], df['close'])
            df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])
        except: pass

        return df

    def _calculate_volume_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators with proper handling of zero volume"""
        # Check if we have any non-zero volume data
        has_real_volume = False
        volume_col = 'volume'

        if 'volume' in df.columns:
            has_real_volume = df['volume'].sum() > 0

        # If no real volume, use tick volume if available and enabled
        if not has_real_volume and 'tickvol' in df.columns and self.use_tick_volume:
            if df['tickvol'].sum() > 0:
                logger.info("Using tick volume for volume-based indicators")
                volume_col = 'tickvol'
                has_real_volume = True

        if not has_real_volume:
            logger.warning("No volume data available. Skipping all volume indicators.")
            return df

        # Now calculate volume indicators using the appropriate column
        try:
            df['obv'] = ta.obv(df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"OBV calculation failed: {e}")

        try:
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"A/D calculation failed: {e}")

        try:
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"CMF calculation failed: {e}")

        try:
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"MFI calculation failed: {e}")

        try:
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"VWAP calculation failed: {e}")

        try:
            df['pvt'] = ta.pvt(df['close'], df[volume_col])
        except Exception as e:
            logger.warning(f"PVT calculation failed: {e}")

        return df

    def _calculate_structure_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate structure indicators with error handling"""
        try:
            # Pivot Points
            high = df['high'].rolling(1).max()
            low = df['low'].rolling(1).min()
            close = df['close']

            df['pivot'] = (high + low + close) / 3
            df['r1'] = 2 * df['pivot'] - low
            df['s1'] = 2 * df['pivot'] - high
            df['r2'] = df['pivot'] + (high - low)
            df['s2'] = df['pivot'] - (high - low)

            # Add Fractal Analysis
            window = 2  # bars on each side

            # Initialize fractal columns
            df['fractal_high'] = np.nan
            df['fractal_low'] = np.nan

            # Calculate fractals
            for i in range(window, len(df) - window):
                # Check for fractal high
                is_fractal_high = True
                for j in range(1, window + 1):
                    if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                        is_fractal_high = False
                        break

                if is_fractal_high:
                    df.loc[df.index[i], 'fractal_high'] = df['high'].iloc[i]

                # Check for fractal low
                is_fractal_low = True
                for j in range(1, window + 1):
                    if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                        is_fractal_low = False
                        break

                if is_fractal_low:
                    df.loc[df.index[i], 'fractal_low'] = df['low'].iloc[i]

            # Basic Market Structure Analysis
            df['structure'] = 'neutral'

            # Forward fill fractals for structure analysis
            last_fractal_high = df['fractal_high'].ffill()
            last_fractal_low = df['fractal_low'].ffill()

            # Determine structure based on fractal breaks
            for i in range(1, len(df)):
                if pd.notna(last_fractal_high.iloc[i]) and pd.notna(last_fractal_low.iloc[i]):
                    if (df['high'].iloc[i] > last_fractal_high.iloc[i-1] if pd.notna(last_fractal_high.iloc[i-1]) else False):
                        df.loc[df.index[i], 'structure'] = 'bullish'
                    elif (df['low'].iloc[i] < last_fractal_low.iloc[i-1] if pd.notna(last_fractal_low.iloc[i-1]) else False):
                        df.loc[df.index[i], 'structure'] = 'bearish'
                    else:
                        df.loc[df.index[i], 'structure'] = 'neutral'

            # Advanced SMC Market Structure Analysis (if available)
            if SMC_AVAILABLE and len(df) > 20:
                try:
                    # Prepare data for SMC analysis
                    smc_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
                    smc_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                    # Run SMC analysis
                    smc_result = analyze_market_structure(smc_df, swing_n=5, bos_look_forward=20)

                    # Extract structure points
                    structure_points = smc_result.get('structure_points', [])
                    if structure_points:
                        df['smc_structure'] = 'neutral'
                        df['structure_type'] = np.nan
                        df['structure_strength'] = np.nan

                        for point in structure_points:
                            timestamp = point.get('timestamp')
                            point_type = point.get('type', '')

                            if timestamp in df.index:
                                if 'Strong High' in point_type:
                                    df.loc[timestamp, 'smc_structure'] = 'strong_high'
                                    df.loc[timestamp, 'structure_strength'] = 1.0
                                elif 'Weak High' in point_type:
                                    df.loc[timestamp, 'smc_structure'] = 'weak_high'
                                    df.loc[timestamp, 'structure_strength'] = 0.5
                                elif 'Strong Low' in point_type:
                                    df.loc[timestamp, 'smc_structure'] = 'strong_low'
                                    df.loc[timestamp, 'structure_strength'] = -1.0
                                elif 'Weak Low' in point_type:
                                    df.loc[timestamp, 'smc_structure'] = 'weak_low'
                                    df.loc[timestamp, 'structure_strength'] = -0.5

                        df['smc_structure'] = df['smc_structure'].ffill()

                        # Add HTF bias
                        htf_bias = smc_result.get('htf_bias', 'Uncertain')
                        df['htf_bias'] = htf_bias.lower()

                        # Add trading range info
                        valid_range = smc_result.get('valid_trading_range')
                        if valid_range:
                            range_type = valid_range.get('type', 'Unknown')
                            df['range_type'] = range_type.lower()

                            # Mark range boundaries
                            start_ts = valid_range.get('start', {}).get('timestamp')
                            end_ts = valid_range.get('end', {}).get('timestamp')

                            if start_ts in df.index:
                                df.loc[start_ts, 'range_boundary'] = 'start'
                            if end_ts in df.index:
                                df.loc[end_ts, 'range_boundary'] = 'end'

                        # Add discount/premium zones
                        dp_info = smc_result.get('discount_premium')
                        if dp_info and 'midpoint' in dp_info:
                            midpoint = dp_info['midpoint']
                            df['price_zone'] = df['close'].apply(
                                lambda x: 'premium' if x > midpoint else 'discount'
                            )

                    logger.info("SMC market structure analysis completed successfully")

                except Exception as e:
                    logger.warning(f"SMC analysis failed: {e}. Using basic structure analysis.")

        except Exception as e:
            logger.warning(f"Error calculating structure indicators: {e}")
            pass

        return df

    def _calculate_advanced_indicators_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced indicators with error handling"""
        try:
            # Ichimoku
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            if isinstance(ichimoku, pd.DataFrame) and len(ichimoku) > 0:
                for col in ichimoku.columns:
                    df[f'ichimoku_{col}'] = ichimoku[col]
        except: pass

        return df

    def _detect_patterns_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns with error handling"""
        # Simple divergence detection
        try:
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14']
                df['rsi_bull_div'] = (
                    (df['close'] < df['close'].shift(1)) & 
                    (rsi > rsi.shift(1))
                )
                df['rsi_bear_div'] = (
                    (df['close'] > df['close'].shift(1)) & 
                    (rsi < rsi.shift(1))
                )
        except: pass

        return df

    def _calculate_risk_metrics_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk metrics with error handling"""
        try:
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Rolling metrics
            window = min(252, len(df) // 4)
            if window > 20:  # Only calculate if we have enough data
                # Sharpe Ratio
                df['sharpe_ratio'] = (
                    df['returns'].rolling(window).mean() * np.sqrt(252) /
                    df['returns'].rolling(window).std()
                )

                # Volatility
                df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)
        except: pass

        return df


class DataProcessor:
    """Main data processing orchestrator"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.detector = CSVFormatDetector()
        self.timeframe_detector = TimeframeDetector()
        self.indicator_engine = TechnicalIndicatorEngine(
            use_tick_volume=config.use_tick_volume_for_volume_indicators
        )

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize results storage
        self.processing_results = {
            'successful': [],
            'failed': [],
            'statistics': {}
        }

    def process_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """Process a single CSV file"""
        logger.info(f"Processing {file_path.name}...")

        try:
            # Detect format and delimiter
            data_format, delimiter = self.detector.detect_format(file_path)
            logger.info(f"Detected format: {data_format.value}, delimiter: {repr(delimiter)}")

            # Read CSV
            df = pd.read_csv(file_path, delimiter=delimiter)
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")

            # Normalize columns
            df = self.detector.normalize_columns(df, data_format)

            # Create timestamp index
            df = self._create_timestamp_index(df, data_format)

            # Preserve original columns
            original_columns = [col for col in df.columns if col not in 
                              ['open', 'high', 'low', 'close', 'volume', 'tickvol']]

            # Detect timeframe if single timeframe processing is enabled
            timeframes_to_process = self.config.timeframes
            if self.config.process_single_timeframe:
                detected_timeframe = self.timeframe_detector.detect_timeframe(file_path)
                if detected_timeframe:
                    timeframes_to_process = [detected_timeframe]
                    logger.info(f"Single timeframe mode: Processing only {detected_timeframe}")
                else:
                    logger.info("No timeframe detected in filename, processing all configured timeframes")

            # Process each timeframe
            timeframe_results = {}

            for timeframe in timeframes_to_process:
                logger.info(f"  Processing timeframe {timeframe}...")

                # For single timeframe mode with matching timeframe, skip resampling
                if self.config.process_single_timeframe and len(timeframes_to_process) == 1:
                    # Use data as-is without resampling
                    logger.info(f"  Using original data for {timeframe} (no resampling)")
                    processed = df.copy()
                else:
                    # Resample to target timeframe
                    logger.info(f"  Resampling to {timeframe}...")
                    processed = self._resample_ohlcv(df, timeframe)

                    # Skip if not enough data
                    if len(processed) < 2:
                        logger.warning(f"  Skipping {timeframe} - not enough data points ({len(processed)} rows)")
                        continue

                # Preserve original columns
                for col in original_columns:
                    if col in df.columns and timeframe != timeframes_to_process[0]:
                        processed[col] = df[col].resample(timeframe).last()

                # Calculate indicators
                logger.info(f"  Calculating indicators for {timeframe}...")
                enriched = self.indicator_engine.calculate_all_indicators(processed)

                timeframe_results[timeframe] = enriched

            # Save results
            self._save_results(file_path, timeframe_results)

            # Generate journal
            self._generate_journal(file_path, timeframe_results)

            self.processing_results['successful'].append(file_path.name)
            logger.info(f"Successfully processed {file_path.name}")

            return timeframe_results

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}", exc_info=True)
            self.processing_results['failed'].append({
                'file': file_path.name,
                'error': str(e)
            })
            return {}

    def _create_timestamp_index(self, df: pd.DataFrame, data_format: DataFormat) -> pd.DataFrame:
        """Create proper timestamp index"""
        if data_format == DataFormat.ANNOTATED:
            # Combine date and time columns
            if 'date' in df.columns and 'time' in df.columns:
                # Handle the date format YYYY.MM.DD
                df['date'] = df['date'].astype(str).str.replace('.', '-')
                df['timestamp'] = pd.to_datetime(
                    df['date'] + ' ' + df['time'].astype(str)
                )
            else:
                raise ValueError("Missing date/time columns")

        elif data_format == DataFormat.MT5:
            # Parse MT5 timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], 
                    format='%Y.%m.%d %H:%M:%S',
                    errors='coerce'
                )
            else:
                raise ValueError("Missing timestamp column")

        else:
            # Generic format - try to find timestamp column
            timestamp_cols = ['timestamp', 'datetime', 'date', 'time']
            for col in timestamp_cols:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                    break
            else:
                raise ValueError("No timestamp column found")

        # Set index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data"""
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        if 'tickvol' in df.columns:
            agg_dict['tickvol'] = 'sum'
        if 'spread' in df.columns:
            agg_dict['spread'] = 'mean'

        resampled = df.resample(timeframe).agg(agg_dict)
        resampled.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        return resampled

    def _save_results(self, file_path: Path, results: Dict[str, pd.DataFrame]):
        """Save processed results"""
        base_name = file_path.stem
        
        # Group by timeframe if only one timeframe
        if len(results) == 1:
            timeframe = list(results.keys())[0]
            # Create output directory grouped by timeframe
            output_dir = Path(self.config.output_dir) / timeframe.replace('T', 'min').replace('H', 'h').replace('D', 'd').replace('W', 'w').replace('M', 'mo')
        else:
            output_dir = Path(self.config.output_dir) / base_name
            
        output_dir.mkdir(parents=True, exist_ok=True)

        for timeframe, df in results.items():
            output_file = output_dir / f"{base_name}_{timeframe}_enriched.csv"
            df.to_csv(output_file)
            logger.info(f"  Saved: {output_file}")

    def _generate_journal(self, file_path: Path, results: Dict[str, pd.DataFrame]):
        """Generate trading journal with signals"""
        base_name = file_path.stem
        
        # Group by timeframe if only one timeframe
        if len(results) == 1:
            timeframe = list(results.keys())[0]
            output_dir = Path(self.config.output_dir) / timeframe.replace('T', 'min').replace('H', 'h').replace('D', 'd').replace('W', 'w').replace('M', 'mo')
        else:
            output_dir = Path(self.config.output_dir) / base_name

        journal_data = {
            'file': file_path.name,
            'processed_at': datetime.now().isoformat(),
            'timeframes': list(results.keys()),
            'indicators_calculated': [],
            'signal_summary': {},
            'session_state': {
                'detected_timeframe': self.timeframe_detector.detect_timeframe(file_path),
                'single_timeframe_mode': self.config.process_single_timeframe
            }
        }

        # Collect indicator names
        if results:
            sample_df = next(iter(results.values()))
            journal_data['indicators_calculated'] = [
                col for col in sample_df.columns 
                if col not in ['open', 'high', 'low', 'close', 'volume', 'tickvol', 'spread']
            ]

        # Generate signal summary for each timeframe
        for timeframe, df in results.items():
            signals = self._generate_signals(df)
            journal_data['signal_summary'][timeframe] = signals

        # Save journal
        journal_file = output_dir / f"{base_name}_journal.json"
        with open(journal_file, 'w') as f:
            json.dump(journal_data, f, indent=2)

        logger.info(f"  Journal saved: {journal_file}")

    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals from indicators"""
        signals = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'details': []
        }

        if len(df) < 2:
            return signals

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # RSI signals
        if 'rsi_14' in df.columns and not pd.isna(latest['rsi_14']):
            if latest['rsi_14'] < 30:
                signals['bullish'] += 1
                signals['details'].append('RSI oversold')
            elif latest['rsi_14'] > 70:
                signals['bearish'] += 1
                signals['details'].append('RSI overbought')

        # MACD signals
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
                if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                    signals['bullish'] += 1
                    signals['details'].append('MACD bullish crossover')
                elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                    signals['bearish'] += 1
                    signals['details'].append('MACD bearish crossover')

        # Moving average signals
        if all(col in df.columns for col in ['close', 'sma_50', 'sma_200']):
            if not pd.isna(latest['sma_50']) and not pd.isna(latest['sma_200']):
                if latest['sma_50'] > latest['sma_200'] and prev['sma_50'] <= prev['sma_200']:
                    signals['bullish'] += 1
                    signals['details'].append('Golden cross')
                elif latest['sma_50'] < latest['sma_200'] and prev['sma_50'] >= prev['sma_200']:
                    signals['bearish'] += 1
                    signals['details'].append('Death cross')

        # Calculate overall signal
        total_signals = signals['bullish'] + signals['bearish']
        if total_signals == 0:
            signals['neutral'] = 1

        return signals

    def process_all_files(self, directory: Path):
        """Process all CSV files in directory"""
        csv_files = list(directory.glob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files to process")

        if self.config.parallel_processing and len(csv_files) > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self.process_file, file_path): file_path
                    for file_path in csv_files
                }

                for future in tqdm(as_completed(futures), total=len(csv_files)):
                    file_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
        else:
            # Sequential processing
            for file_path in tqdm(csv_files):
                self.process_file(file_path)

        # Generate summary report
        self._generate_summary_report()

    def _generate_summary_report(self):
        """Generate processing summary report"""
        total_files = len(self.processing_results['successful']) + len(self.processing_results['failed'])

        report = {
            'processing_summary': {
                'total_files': total_files,
                'successful': len(self.processing_results['successful']),
                'failed': len(self.processing_results['failed']),
                'success_rate': len(self.processing_results['successful']) / total_files if total_files > 0 else 0
            },
            'successful_files': self.processing_results['successful'],
            'failed_files': self.processing_results['failed'],
            'processed_at': datetime.now().isoformat(),
            'configuration': {
                'timeframes': self.config.timeframes,
                'output_directory': self.config.output_dir,
                'use_tick_volume_for_volume_indicators': self.config.use_tick_volume_for_volume_indicators,
                'process_single_timeframe': self.config.process_single_timeframe
            }
        }

        report_file = Path(self.config.output_dir) / 'processing_summary.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Total files: {report['processing_summary']['total_files']}")
        logger.info(f"  Successful: {report['processing_summary']['successful']}")
        logger.info(f"  Failed: {report['processing_summary']['failed']}")
        logger.info(f"  Success rate: {report['processing_summary']['success_rate']:.1%}")
        logger.info(f"  Report saved: {report_file}")


def main():
    """Main execution function"""
    # Configuration
    config = ProcessingConfig(
        timeframes=['1T', '5T', '15T', '30T', '1H', '4H', '1D', '1W', '1M'],
        output_dir='processed_output',
        parallel_processing=True,
        use_tick_volume_for_volume_indicators=True,
        process_single_timeframe=True  # NEW: Enable single timeframe processing
    )

    # Initialize processor
    processor = DataProcessor(config)

    # Process all files in current directory
    current_dir = Path('.')
    processor.process_all_files(current_dir)


if __name__ == "__main__":
    main()