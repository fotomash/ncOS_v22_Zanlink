
import os
import sys
import re
import warnings
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Technical Analysis Libraries
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not installed. Using pandas_ta as fallback.")

import pandas_ta as ta

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
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
    ANNOTATED = "annotated"
    MT5 = "mt5"
    GENERIC = "generic"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class ProcessingConfig:
    timeframes: List[str] = None
    output_dir: str = "processed_output"
    parallel_processing: bool = True
    max_workers: int = None
    use_tick_volume_for_volume_indicators: bool = True
    process_single_timeframe: bool = False
    calculate_signals: bool = True
    include_market_structure: bool = True
    include_pattern_detection: bool = True
    include_risk_metrics: bool = True

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1T', '5T', '15T', '30T', '1H', '4H', '1D', '1W', '1M']
        if self.max_workers is None:
            self.max_workers = max(1, os.cpu_count() - 1)


class TimeframeDetector:
    TIMEFRAME_PATTERNS = {
        '_m1': '1T', '_1m': '1T', '_1min': '1T',
        '_m5': '5T', '_5m': '5T', '_5min': '5T',
        '_m15': '15T', '_15m': '15T', '_15min': '15T',
        '_m30': '30T', '_30m': '30T', '_30min': '30T',
        '_h1': '1H', '_1h': '1H', '_60min': '1H',
        '_h4': '4H', '_4h': '4H', '_240min': '4H',
        '_d1': '1D', '_1d': '1D', '_daily': '1D',
        '_w1': '1W', '_1w': '1W', '_weekly': '1W',
        '_mn1': '1M', '_1mo': '1M', '_monthly': '1M'
    }

    @classmethod
    def detect_timeframe(cls, file_path: Path) -> Optional[str]:
        filename_lower = file_path.stem.lower()
        for pattern, timeframe in cls.TIMEFRAME_PATTERNS.items():
            if pattern in filename_lower:
                return timeframe
        return None


class PairDetector:
    PAIR_PATTERNS = [r'^([A-Z]{6})', r'^([A-Z]{3}[-_.]?[A-Z]{3})']

    @classmethod
    def detect_pair(cls, file_path: Path) -> Optional[str]:
        filename = file_path.stem
        for pattern in cls.PAIR_PATTERNS:
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                pair = match.group(1).upper().replace('-', '').replace('.', '').replace('_', '')
                if len(pair) == 6:
                    return pair
        return None


class CSVFormatDetector:
    @staticmethod
    def detect_format(file_path: Path) -> Tuple[DataFormat, str]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline() for _ in range(5)]
        delimiter = '\t' if '\t' in first_lines[0] else ','
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
        if format_type == DataFormat.ANNOTATED:
            column_mapping = {
                '<DATE>': 'date', '<TIME>': 'time', '<OPEN>': 'open', 
                '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close', 
                '<TICKVOL>': 'tickvol', '<VOL>': 'volume', '<SPREAD>': 'spread'
            }
            df.rename(columns=column_mapping, inplace=True)
        elif format_type == DataFormat.MT5:
            column_mapping = {
                'Time': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Tick Volume': 'tickvol', 'Volume': 'volume', 'Spread': 'spread'
            }
            df.rename(columns=column_mapping, inplace=True)
        else:
            df.columns = df.columns.str.lower().str.strip()
        return df


class TechnicalIndicatorEngine:
    """Ultra-comprehensive technical indicator calculator for professional trading"""

    def __init__(self, use_tick_volume: bool = True):
        self.use_tick_volume = use_tick_volume
        logger.info(f"Advanced indicator engine initialized. Using tick volume: {use_tick_volume}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive suite of professional trading indicators"""
        df = df.copy()

        # Validate required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.error(f"Missing required OHLC columns. Found: {df.columns.tolist()}")
            return df

        # Prepare volume column
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            if self.use_tick_volume and 'tickvol' in df.columns and df['tickvol'].sum() > 0:
                df['volume'] = df['tickvol']
            else:
                df['volume'] = np.ones(len(df))  # Fallback volume

        # Calculate all indicator categories
        df = self._calculate_trend_indicators(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_volatility_indicators(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_cycle_indicators(df)
        df = self._calculate_statistical_indicators(df)
        df = self._calculate_market_structure_indicators(df)
        df = self._calculate_risk_metrics(df)
        df = self._detect_patterns(df)
        df = self._calculate_signals(df)
        df = self._identify_market_regime(df)

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive trend analysis indicators"""
        try:
            # Moving Averages - Multiple periods
            ma_periods = [5, 8, 13, 21, 34, 55, 89, 144, 200]
            for period in ma_periods:
                if len(df) >= period:
                    df[f'sma_{period}'] = ta.sma(df['close'], length=period)
                    df[f'ema_{period}'] = ta.ema(df['close'], length=period)
                    df[f'wma_{period}'] = ta.wma(df['close'], length=period)

                    # MA slopes and trends
                    df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5)
                    df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(5)

            # Hull Moving Average
            df['hma_21'] = ta.hma(df['close'], length=21)

            # TEMA (Triple Exponential Moving Average)
            df['tema_21'] = ta.tema(df['close'], length=21)

            # ADX System
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_data is not None and not adx_data.empty:
                df = df.join(adx_data, rsuffix='_adx')

            # Parabolic SAR
            psar = ta.psar(df['high'], df['low'], df['close'])
            if psar is not None and not psar.empty:
                df = df.join(psar, rsuffix='_psar')

            # Ichimoku Cloud
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            if ichimoku is not None and not ichimoku.empty:
                df = df.join(ichimoku, rsuffix='_ichimoku')

            # Supertrend
            supertrend = ta.supertrend(df['high'], df['low'], df['close'])
            if supertrend is not None and not supertrend.empty:
                df = df.join(supertrend, rsuffix='_st')

            # Moving Average Convergence
            df['ma_convergence_fast'] = (df['ema_8'] - df['ema_21']) / df['ema_21'] * 100
            df['ma_convergence_slow'] = (df['ema_21'] - df['ema_55']) / df['ema_55'] * 100

        except Exception as e:
            logger.warning(f"Trend indicator calculation failed: {e}")
        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced momentum oscillators and indicators"""
        try:
            # RSI family
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

            # Stochastic family
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                df = df.join(stoch, rsuffix='_stoch')

            # MACD system
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df = df.join(macd, rsuffix='_macd')

            # Additional MACD periods
            macd_fast = ta.macd(df['close'], fast=5, slow=13)
            if macd_fast is not None and not macd_fast.empty:
                df = df.join(macd_fast.add_suffix('_fast'), rsuffix='_macd_fast')

            # Williams %R
            df['willr'] = ta.willr(df['high'], df['low'], df['close'])

            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])

            # Rate of Change
            df['roc_1'] = ta.roc(df['close'], length=1)
            df['roc_5'] = ta.roc(df['close'], length=5)
            df['roc_10'] = ta.roc(df['close'], length=10)

            # Momentum
            df['momentum_10'] = ta.mom(df['close'], length=10)

            # Ultimate Oscillator
            df['uo'] = ta.uo(df['high'], df['low'], df['close'])

            # Fisher Transform
            df['fisher'] = ta.fisher(df['high'], df['low'])

        except Exception as e:
            logger.warning(f"Momentum indicator calculation failed: {e}")
        return df

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive volatility analysis"""
        try:
            # Bollinger Bands
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None and not bbands.empty:
                df = df.join(bbands, rsuffix='_bb')

                # Bollinger Band derived metrics
                if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
                    df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
                    df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

            # ATR system
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_21'] = ta.atr(df['high'], df['low'], df['close'], length=21)
            df['atr_ratio'] = df['atr_14'] / df['atr_21']

            # Normalized ATR
            df['natr'] = ta.natr(df['high'], df['low'], df['close'])

            # True Range
            df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])

            # Keltner Channels
            keltner = ta.kc(df['high'], df['low'], df['close'])
            if keltner is not None and not keltner.empty:
                df = df.join(keltner, rsuffix='_kc')

            # Donchian Channels
            donchian = ta.donchian(df['high'], df['low'])
            if donchian is not None and not donchian.empty:
                df = df.join(donchian, rsuffix='_dc')

            # Volatility metrics
            df['price_volatility'] = df['close'].rolling(20).std()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['realized_volatility'] = df['log_returns'].rolling(20).std() * np.sqrt(252)

        except Exception as e:
            logger.warning(f"Volatility indicator calculation failed: {e}")
        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volume analysis"""
        try:
            if df['volume'].sum() == 0:
                logger.info("No volume data available, skipping volume indicators")
                return df

            # Core volume indicators
            df['obv'] = ta.obv(df['close'], df['volume'])
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])

            # Accumulation/Distribution Line
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

            # Chaikin Oscillator
            df['adosc'] = ta.adosc(df['high'], df['low'], df['close'], df['volume'])

            # Volume-based moving averages
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

            # Price Volume Trend
            df['pvt'] = ta.pvt(df['close'], df['volume'])

            # Volume Rate of Change
            df['vroc'] = ta.roc(df['volume'], length=10)

            # VWAP bands
            vwap_std = df['close'].rolling(20).std()
            df['vwap_upper'] = df['vwap'] + (2 * vwap_std)
            df['vwap_lower'] = df['vwap'] - (2 * vwap_std)

        except Exception as e:
            logger.warning(f"Volume indicator calculation failed: {e}")
        return df

    def _calculate_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cycle and wave analysis indicators"""
        try:
            # Hilbert Transform indicators
            if TALIB_AVAILABLE:
                df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'].values)
                df['ht_dcphase'] = talib.HT_DCPHASE(df['close'].values)
                df['ht_phasor_inphase'], df['ht_phasor_quadrature'] = talib.HT_PHASOR(df['close'].values)
                df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(df['close'].values)
                df['ht_trendmode'] = talib.HT_TRENDMODE(df['close'].values)

            # Detrended Price Oscillator
            df['dpo'] = ta.dpo(df['close'], length=20)

        except Exception as e:
            logger.warning(f"Cycle indicator calculation failed: {e}")
        return df

    def _calculate_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical and mathematical indicators"""
        try:
            # Z-Score
            df['zscore'] = ta.zscore(df['close'], length=20)

            # Entropy
            df['entropy'] = ta.entropy(df['close'], length=10)

            # Skewness and Kurtosis of returns
            returns = df['close'].pct_change()
            df['returns_skew'] = returns.rolling(20).skew()
            df['returns_kurtosis'] = returns.rolling(20).kurtosis()

            # Linear regression indicators
            df['linreg'] = ta.linreg(df['close'], length=14)
            df['linreg_slope'] = ta.linreg(df['close'], length=14).diff()

            # Standard deviation
            df['stdev'] = ta.stdev(df['close'], length=20)

            # Variance
            df['variance'] = ta.variance(df['close'], length=20)

        except Exception as e:
            logger.warning(f"Statistical indicator calculation failed: {e}")
        return df

    def _calculate_market_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market structure and Smart Money Concepts indicators"""
        try:
            # Pivot points
            df['pivot_high'] = ta.pivot_high(df['high'], df['low'], length=5)
            df['pivot_low'] = ta.pivot_low(df['high'], df['low'], length=5)

            # Support and resistance levels
            df = self._calculate_support_resistance(df)

            # Market structure breaks
            df = self._calculate_structure_breaks(df)

            # Order blocks
            df = self._identify_order_blocks(df)

            # Fair value gaps
            df = self._identify_fair_value_gaps(df)

        except Exception as e:
            logger.warning(f"Market structure calculation failed: {e}")
        return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        try:
            # Rolling highs and lows
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            df['resistance_50'] = df['high'].rolling(50).max()
            df['support_50'] = df['low'].rolling(50).min()

            # Distance to support/resistance
            df['dist_to_resistance'] = (df['resistance_20'] - df['close']) / df['close'] * 100
            df['dist_to_support'] = (df['close'] - df['support_20']) / df['close'] * 100

        except Exception as e:
            logger.warning(f"Support/Resistance calculation failed: {e}")
        return df

    def _calculate_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure breaks"""
        try:
            # Basic structure break logic
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            df['structure_break'] = df['higher_high'] | df['lower_low']

        except Exception as e:
            logger.warning(f"Structure break calculation failed: {e}")
        return df

    def _identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify institutional order blocks"""
        try:
            # Simplified order block identification
            body_size = abs(df['close'] - df['open'])
            avg_body = body_size.rolling(20).mean()

            df['large_body'] = body_size > (avg_body * 2)
            df['bullish_order_block'] = df['large_body'] & (df['close'] > df['open'])
            df['bearish_order_block'] = df['large_body'] & (df['close'] < df['open'])

        except Exception as e:
            logger.warning(f"Order block identification failed: {e}")
        return df

    def _identify_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify fair value gaps (imbalances)"""
        try:
            # Bullish FVG: Previous high < Current low
            df['bullish_fvg'] = df['high'].shift(1) < df['low']
            # Bearish FVG: Previous low > Current high  
            df['bearish_fvg'] = df['low'].shift(1) > df['high']

            df['fvg_size'] = np.where(df['bullish_fvg'], 
                                    df['low'] - df['high'].shift(1),
                                    np.where(df['bearish_fvg'],
                                           df['low'].shift(1) - df['high'], 0))

        except Exception as e:
            logger.warning(f"Fair value gap identification failed: {e}")
        return df

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk management metrics"""
        try:
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Rolling volatility
            df['volatility_10'] = df['returns'].rolling(10).std()
            df['volatility_20'] = df['returns'].rolling(20).std()

            # Value at Risk (simple percentile method)
            df['var_95'] = df['returns'].rolling(20).quantile(0.05)
            df['var_99'] = df['returns'].rolling(20).quantile(0.01)

            # Maximum Drawdown
            df['cumulative_returns'] = (1 + df['returns']).cumprod()
            df['running_max'] = df['cumulative_returns'].expanding().max()
            df['drawdown'] = (df['cumulative_returns'] - df['running_max']) / df['running_max']
            df['max_drawdown'] = df['drawdown'].expanding().min()

            # Sharpe ratio (rolling)
            excess_returns = df['returns'] - 0.02/252  # Assuming 2% risk-free rate
            df['sharpe_ratio'] = excess_returns.rolling(20).mean() / df['returns'].rolling(20).std()

        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
        return df

    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pattern detection and recognition"""
        try:
            # Candlestick patterns using TA-Lib if available
            if TALIB_AVAILABLE:
                df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
                df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
                df['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
                df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
                df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

            # Simple pattern detection
            body_size = abs(df['close'] - df['open'])
            candle_range = df['high'] - df['low']
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

            df['small_body'] = body_size < (candle_range * 0.3)
            df['long_upper_shadow'] = upper_shadow > (body_size * 2)
            df['long_lower_shadow'] = lower_shadow > (body_size * 2)

        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        return df

    def _calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        try:
            # Trend following signals
            df['ma_cross_signal'] = np.where((df['ema_8'] > df['ema_21']) & 
                                           (df['ema_8'].shift(1) <= df['ema_21'].shift(1)), 1,
                                           np.where((df['ema_8'] < df['ema_21']) & 
                                                  (df['ema_8'].shift(1) >= df['ema_21'].shift(1)), -1, 0))

            # RSI signals
            df['rsi_signal'] = np.where((df['rsi_14'] < 30) & (df['rsi_14'].shift(1) >= 30), 1,
                                      np.where((df['rsi_14'] > 70) & (df['rsi_14'].shift(1) <= 70), -1, 0))

            # MACD signals
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                df['macd_signal'] = np.where((df['MACD_12_26_9'] > df['MACDs_12_26_9']) & 
                                           (df['MACD_12_26_9'].shift(1) <= df['MACDs_12_26_9'].shift(1)), 1,
                                           np.where((df['MACD_12_26_9'] < df['MACDs_12_26_9']) & 
                                                  (df['MACD_12_26_9'].shift(1) >= df['MACDs_12_26_9'].shift(1)), -1, 0))

            # Bollinger Band signals
            if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
                df['bb_signal'] = np.where((df['close'] < df['BBL_20_2.0']) & (df['rsi_14'] < 30), 1,
                                         np.where((df['close'] > df['BBU_20_2.0']) & (df['rsi_14'] > 70), -1, 0))

            # Composite signal
            signal_columns = [col for col in df.columns if col.endswith('_signal')]
            if signal_columns:
                df['composite_signal'] = df[signal_columns].sum(axis=1)
                df['strong_buy'] = df['composite_signal'] >= 2
                df['strong_sell'] = df['composite_signal'] <= -2

        except Exception as e:
            logger.warning(f"Signal calculation failed: {e}")
        return df

    def _identify_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify current market regime"""
        try:
            # Trend strength
            if 'ADX_14' in df.columns:
                trend_strength = df['ADX_14']
            else:
                # Fallback trend strength calculation
                trend_strength = abs(df['ema_21'].diff(10)) / df['atr_14'] * 100

            # Volatility regime
            vol_percentile = df['atr_14'].rolling(50).rank(pct=True)

            # Market regime classification
            df['market_regime'] = 'ranging'
            df.loc[(trend_strength > 25) & (df['ema_8'] > df['ema_21']), 'market_regime'] = 'trending_up'
            df.loc[(trend_strength > 25) & (df['ema_8'] < df['ema_21']), 'market_regime'] = 'trending_down'
            df.loc[vol_percentile > 0.8, 'market_regime'] = 'volatile'

            # Regime confidence
            df['regime_confidence'] = np.where(trend_strength > 30, 'high',
                                             np.where(trend_strength > 20, 'medium', 'low'))

        except Exception as e:
            logger.warning(f"Market regime identification failed: {e}")
        return df


class DataProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.csv_detector = CSVFormatDetector()
        self.timeframe_detector = TimeframeDetector()
        self.pair_detector = PairDetector()
        self.indicator_engine = TechnicalIndicatorEngine(
            use_tick_volume=config.use_tick_volume_for_volume_indicators
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.processing_results = {'successful': [], 'failed': [], 'skipped': []}

    def process_file(self, file_path: Path) -> None:
        logger.info(f"Starting comprehensive processing for {file_path.name}...")
        try:
            timeframes_to_process = []
            should_resample = True

            if self.config.process_single_timeframe:
                detected_timeframe = self.timeframe_detector.detect_timeframe(file_path)
                if detected_timeframe:
                    timeframes_to_process = [detected_timeframe]
                    should_resample = False
                    logger.info(f"Single-timeframe mode: Processing native timeframe '{detected_timeframe}' without resampling.")
                else:
                    logger.warning(f"SKIPPING: {file_path.name} - No timeframe pattern found in filename while in single-timeframe mode.")
                    self.processing_results['skipped'].append({'file': file_path.name, 'reason': 'No timeframe pattern found.'})
                    return
            else:
                timeframes_to_process = self.config.timeframes
                should_resample = True
                logger.info(f"Multi-timeframe mode: Resampling to all configured timeframes.")

            data_format, delimiter = self.csv_detector.detect_format(file_path)
            df = pd.read_csv(file_path, delimiter=delimiter)
            df = self.csv_detector.normalize_columns(df, data_format)
            df = self._create_timestamp_index(df, data_format)

            timeframe_results = {}
            for timeframe in timeframes_to_process:
                processed_df = self._resample_ohlcv(df, timeframe) if should_resample else df.copy()
                if len(processed_df) < 200:  # Need substantial data for comprehensive analysis
                    logger.warning(f"Insufficient data for timeframe {timeframe} (need 200, have {len(processed_df)}). Skipping.")
                    continue

                logger.info(f"  Calculating comprehensive indicators for {timeframe}...")
                enriched_df = self.indicator_engine.calculate_all_indicators(processed_df)
                timeframe_results[timeframe] = enriched_df

            self._save_results(file_path, timeframe_results)
            self._generate_analysis_report(file_path, timeframe_results)
            self.processing_results['successful'].append(file_path.name)
            logger.info(f"Successfully processed and saved comprehensive results for {file_path.name}")

        except Exception as e:
            logger.error(f"FAILED to process {file_path.name}: {str(e)}", exc_info=True)
            self.processing_results['failed'].append({'file': file_path.name, 'error': str(e)})

    def _create_timestamp_index(self, df: pd.DataFrame, data_format: DataFormat) -> pd.DataFrame:
        if data_format == DataFormat.ANNOTATED and 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'].astype(str).str.replace('.', '-') + ' ' + df['time'].astype(str))
        elif data_format == DataFormat.MT5 and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
        else:
            for col in ['timestamp', 'datetime', 'date', 'time']:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col], errors='coerce')
                    break
            else: 
                raise ValueError("No timestamp column found")
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df[~df.index.duplicated(keep='first')]

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        if 'volume' in df.columns: agg_dict['volume'] = 'sum'
        if 'tickvol' in df.columns: agg_dict['tickvol'] = 'sum'
        if 'spread' in df.columns: agg_dict['spread'] = 'mean'
        resampled = df.resample(timeframe).agg(agg_dict)
        return resampled.dropna(subset=['open'])

    def _save_results(self, file_path: Path, results: Dict[str, pd.DataFrame]):
        base_name = file_path.stem
        detected_pair = self.pair_detector.detect_pair(file_path)
        output_dir = Path(self.config.output_dir) / detected_pair if detected_pair else Path(self.config.output_dir) / base_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for timeframe, df in results.items():
            output_file = output_dir / f"{base_name}_COMPREHENSIVE_{timeframe}.csv"
            df.to_csv(output_file)
            logger.info(f"  Saved comprehensive dataset: {output_file}")

            # Save summary statistics
            summary_file = output_dir / f"{base_name}_SUMMARY_{timeframe}.json"
            summary = self._generate_summary_stats(df)
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the processed data"""
        try:
            stats = {
                'data_info': {
                    'total_rows': len(df),
                    'date_range': {
                        'start': df.index.min(),
                        'end': df.index.max()
                    },
                    'total_columns': len(df.columns)
                },
                'price_stats': {
                    'price_range': {
                        'min': float(df['low'].min()),
                        'max': float(df['high'].max())
                    },
                    'volatility': float(df['atr_14'].mean()) if 'atr_14' in df.columns else None,
                    'trend': 'bullish' if df['close'].iloc[-1] > df['close'].iloc[0] else 'bearish'
                },
                'signal_summary': {
                    'strong_buy_signals': int(df['strong_buy'].sum()) if 'strong_buy' in df.columns else 0,
                    'strong_sell_signals': int(df['strong_sell'].sum()) if 'strong_sell' in df.columns else 0
                },
                'market_regime': {
                    'current': df['market_regime'].iloc[-1] if 'market_regime' in df.columns else None,
                    'distribution': df['market_regime'].value_counts().to_dict() if 'market_regime' in df.columns else None
                }
            }
            return stats
        except Exception as e:
            logger.warning(f"Summary stats generation failed: {e}")
            return {}

    def _generate_analysis_report(self, file_path: Path, results: Dict[str, pd.DataFrame]):
        """Generate comprehensive analysis report"""
        base_name = file_path.stem
        detected_pair = self.pair_detector.detect_pair(file_path)
        output_dir = Path(self.config.output_dir) / detected_pair if detected_pair else Path(self.config.output_dir) / base_name

        report_data = {
            'file_info': {
                'original_file': file_path.name,
                'detected_pair': detected_pair,
                'detected_timeframe': self.timeframe_detector.detect_timeframe(file_path),
                'processed_at': datetime.now().isoformat()
            },
            'processing_config': {
                'single_timeframe_mode': self.config.process_single_timeframe,
                'timeframes_processed': list(results.keys()),
                'total_indicators': len([col for df in results.values() for col in df.columns])
            },
            'timeframe_analysis': {}
        }

        for timeframe, df in results.items():
            report_data['timeframe_analysis'][timeframe] = self._generate_summary_stats(df)

        report_file = output_dir / f"{base_name}_ANALYSIS_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        logger.info(f"  Analysis report saved: {report_file}")

    def process_files_with_pattern(self, directory: Path, pattern: Optional[str]):
        csv_files = [f for f in directory.glob('*.csv') if not pattern or pattern.upper() in f.name.upper()]
        logger.info(f"Found {len(csv_files)} CSV files to process with comprehensive analysis.")

        if not csv_files:
            logger.warning(f"No CSV files found in {directory} with pattern '{pattern}'")
            return

        if self.config.parallel_processing and len(csv_files) > 1:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                list(tqdm(executor.map(self.process_file, csv_files), total=len(csv_files), desc="Processing files"))
        else:
            for file_path in tqdm(csv_files, desc="Processing files"):
                self.process_file(file_path)
        self._generate_summary_report()

    def _generate_summary_report(self):
        total = len(self.processing_results['successful']) + len(self.processing_results['failed']) + len(self.processing_results['skipped'])
        report = {
            'processing_summary': {
                'total_files_found': total,
                'successful': len(self.processing_results['successful']),
                'failed': len(self.processing_results['failed']),
                'skipped': len(self.processing_results['skipped']),
                'success_rate': f"{len(self.processing_results['successful'])/total*100:.1f}%" if total > 0 else "0%"
            },
            'successful_files': self.processing_results['successful'],
            'failed_files': self.processing_results['failed'],
            'skipped_files': self.processing_results['skipped'],
            'processed_at': datetime.now().isoformat(),
            'configuration': {k: str(v) for k, v in self.config.__dict__.items()}
        }
        report_file = Path(self.config.output_dir) / 'COMPREHENSIVE_PROCESSING_SUMMARY.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Comprehensive summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Professional Trading Data Processor')
    parser.add_argument('--dir', '-d', type=str, default='.', help='Directory containing CSV files (default: current directory)')
    parser.add_argument('--pattern', '-p', type=str, default=None, help='Filter filenames (e.g., "TICK")')
    parser.add_argument('--output', '-o', type=str, default='comprehensive_output', help='Output directory (default: comprehensive_output)')
    parser.add_argument('--single-timeframe', '-s', action='store_true', help='Process only the timeframe detected in filename')
    parser.add_argument('--no-signals', action='store_true', help='Skip signal generation (faster processing)')
    parser.add_argument('--basic-only', action='store_true', help='Calculate only basic indicators (faster processing)')
    args = parser.parse_args()

    config = ProcessingConfig(
        output_dir=args.output,
        parallel_processing=True,
        use_tick_volume_for_volume_indicators=True,
        process_single_timeframe=args.single_timeframe,
        calculate_signals=not args.no_signals,
        include_market_structure=not args.basic_only,
        include_pattern_detection=not args.basic_only,
        include_risk_metrics=not args.basic_only
    )

    logger.info("="*60)
    logger.info("COMPREHENSIVE PROFESSIONAL TRADING DATA PROCESSOR")
    logger.info("="*60)
    logger.info(f"Configuration: {config}")

    processor = DataProcessor(config)
    input_dir = Path(args.dir)
    if not input_dir.is_dir():
        logger.error(f"Directory not found: {input_dir}")
        return

    processor.process_files_with_pattern(input_dir, args.pattern)
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()
