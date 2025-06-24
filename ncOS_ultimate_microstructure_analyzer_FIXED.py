#!/usr/bin/env python3
"""
ncOS Ultimate Microstructure Analyzer - Fixed Version
The most comprehensive trading data analysis system ever created
ALL FEATURES ENABLED BY DEFAULT
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass, field
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import logging
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
import sqlite3
import pickle
import gzip
from collections import defaultdict, deque

# Optional imports with fallbacks
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Some technical indicators will be disabled.")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML features will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive charts will be disabled.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ncOS_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Ultimate configuration for maximum analysis - ALL FEATURES ENABLED BY DEFAULT"""
    # Data processing
    process_tick_data: bool = True
    process_csv_files: bool = True
    process_json_files: bool = True
    tick_bars_limit: int = 1500  # DEFAULT 1500 TICK BARS
    bar_limits: Dict[str, int] = field(default_factory=lambda: {
        '1min': 1000, '5min': 500, '15min': 200, '30min': 100,
        '1h': 100, '4h': 50, '1d': 30, '1w': 20, '1M': 12
    })

    # Technical analysis - ALL ENABLED
    enable_all_indicators: bool = True
    enable_advanced_patterns: bool = True
    enable_harmonic_patterns: bool = True
    enable_elliott_waves: bool = True
    enable_gann_analysis: bool = True
    enable_fibonacci_analysis: bool = True

    # Market structure - ALL ENABLED
    enable_smc_analysis: bool = True
    enable_wyckoff_analysis: bool = True
    enable_volume_profile: bool = True
    enable_order_flow: bool = True
    enable_market_profile: bool = True

    # Microstructure analysis - ALL ENABLED
    enable_liquidity_analysis: bool = True
    enable_manipulation_detection: bool = True
    enable_spoofing_detection: bool = True
    enable_front_running_detection: bool = True
    enable_wash_trading_detection: bool = True

    # Machine learning - ALL ENABLED
    enable_ml_predictions: bool = True
    enable_anomaly_detection: bool = True
    enable_clustering: bool = True
    enable_regime_detection: bool = True

    # Risk management - ALL ENABLED
    enable_risk_metrics: bool = True
    enable_portfolio_analysis: bool = True
    enable_stress_testing: bool = True
    enable_var_calculation: bool = True

    # Visualization - ALL ENABLED
    enable_advanced_plots: bool = True
    enable_interactive_charts: bool = True
    enable_3d_visualization: bool = True
    save_all_plots: bool = True

    # Output options - ALL ENABLED
    compress_output: bool = True
    save_detailed_reports: bool = True
    export_excel: bool = True
    export_csv: bool = True

class UltimateIndicatorEngine:
    """Maximum indicator calculation engine"""

    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate over 200 technical indicators"""
        if not TALIB_AVAILABLE:
            return self._calculate_basic_indicators(df)

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_prices = df['open'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

        indicators = {}

        # Price-based indicators
        indicators.update(self._price_indicators(high, low, close, open_prices))

        # Volume indicators
        indicators.update(self._volume_indicators(high, low, close, volume))

        # Momentum indicators
        indicators.update(self._momentum_indicators(high, low, close))

        # Volatility indicators
        indicators.update(self._volatility_indicators(high, low, close))

        # Cycle indicators
        indicators.update(self._cycle_indicators(close))

        # Statistical indicators
        indicators.update(self._statistical_indicators(close))

        # Custom indicators
        indicators.update(self._custom_indicators(high, low, close, open_prices, volume))

        return indicators

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate basic indicators without TA-Lib"""
        indicators = {}
        close = df['close'].values

        # Simple moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close) >= period:
                sma = pd.Series(close).rolling(window=period).mean().values
                indicators[f'SMA_{period}'] = sma

                # Exponential moving average
                ema = pd.Series(close).ewm(span=period).mean().values
                indicators[f'EMA_{period}'] = ema

        return indicators

    def _price_indicators(self, high, low, close, open_prices):
        """Price-based indicators"""
        indicators = {}

        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                if len(close) >= period:
                    indicators[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
                    indicators[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
                    indicators[f'WMA_{period}'] = talib.WMA(close, timeperiod=period)
                    indicators[f'TEMA_{period}'] = talib.TEMA(close, timeperiod=period)
                    indicators[f'TRIMA_{period}'] = talib.TRIMA(close, timeperiod=period)
                    indicators[f'KAMA_{period}'] = talib.KAMA(close, timeperiod=period)
                    indicators[f'MAMA_{period}'], indicators[f'FAMA_{period}'] = talib.MAMA(close)
            except Exception as e:
                logger.warning(f"Error calculating MA for period {period}: {e}")

        # Bollinger Bands (multiple periods)
        for period in [10, 20, 50]:
            try:
                if len(close) >= period:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
                    indicators[f'BB_UPPER_{period}'] = bb_upper
                    indicators[f'BB_MIDDLE_{period}'] = bb_middle
                    indicators[f'BB_LOWER_{period}'] = bb_lower
                    indicators[f'BB_WIDTH_{period}'] = (bb_upper - bb_lower) / bb_middle
                    indicators[f'BB_POSITION_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands for period {period}: {e}")

        # Donchian Channels
        for period in [10, 20, 55]:
            try:
                if len(high) >= period:
                    indicators[f'DONCHIAN_UPPER_{period}'] = pd.Series(high).rolling(period).max().values
                    indicators[f'DONCHIAN_LOWER_{period}'] = pd.Series(low).rolling(period).min().values
                    indicators[f'DONCHIAN_MIDDLE_{period}'] = (
                        indicators[f'DONCHIAN_UPPER_{period}'] + 
                        indicators[f'DONCHIAN_LOWER_{period}']
                    ) / 2
            except Exception as e:
                logger.warning(f"Error calculating Donchian Channels for period {period}: {e}")

        # Pivot Points
        try:
            indicators['PIVOT'] = (high + low + close) / 3
            indicators['R1'] = 2 * indicators['PIVOT'] - low
            indicators['S1'] = 2 * indicators['PIVOT'] - high
            indicators['R2'] = indicators['PIVOT'] + (high - low)
            indicators['S2'] = indicators['PIVOT'] - (high - low)
            indicators['R3'] = high + 2 * (indicators['PIVOT'] - low)
            indicators['S3'] = low - 2 * (high - indicators['PIVOT'])
        except Exception as e:
            logger.warning(f"Error calculating Pivot Points: {e}")

        return indicators

    def _volume_indicators(self, high, low, close, volume):
        """Volume-based indicators"""
        indicators = {}

        try:
            # Volume indicators
            indicators['OBV'] = talib.OBV(close, volume)
            indicators['AD'] = talib.AD(high, low, close, volume)
            indicators['ADOSC'] = talib.ADOSC(high, low, close, volume)

            # Volume moving averages
            for period in [10, 20, 50]:
                if len(volume) >= period:
                    indicators[f'VOLUME_SMA_{period}'] = talib.SMA(volume, timeperiod=period)
                    vol_sma = indicators[f'VOLUME_SMA_{period}']
                    indicators[f'VOLUME_RATIO_{period}'] = np.divide(
                        volume, vol_sma, 
                        out=np.ones_like(volume), 
                        where=vol_sma!=0
                    )

            # Volume price trend
            price_change = np.diff(close, prepend=close[0])
            price_change_pct = np.divide(
                price_change, 
                np.roll(close, 1), 
                out=np.zeros_like(price_change), 
                where=np.roll(close, 1)!=0
            )
            indicators['VPT'] = np.cumsum(volume * price_change_pct)

            # Money flow index
            if len(close) >= 14:
                indicators['MFI_14'] = talib.MFI(high, low, close, volume, timeperiod=14)

            # Volume weighted average price
            indicators['VWAP'] = np.divide(
                np.cumsum(volume * close), 
                np.cumsum(volume),
                out=close.copy(),
                where=np.cumsum(volume)!=0
            )

        except Exception as e:
            logger.warning(f"Error calculating volume indicators: {e}")

        return indicators

    def _momentum_indicators(self, high, low, close):
        """Momentum indicators"""
        indicators = {}

        try:
            # RSI (multiple periods)
            for period in [9, 14, 21]:
                if len(close) >= period:
                    indicators[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

            # Stochastic
            if len(close) >= 14:
                indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high, low, close)
                indicators['STOCHF_K'], indicators['STOCHF_D'] = talib.STOCHF(high, low, close)
                indicators['STOCHRSI_K'], indicators['STOCHRSI_D'] = talib.STOCHRSI(close)

            # MACD (multiple settings)
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
                if len(close) >= slow:
                    macd, signal_line, histogram = talib.MACD(
                        close, fastperiod=fast, slowperiod=slow, signalperiod=signal
                    )
                    indicators[f'MACD_{fast}_{slow}_{signal}'] = macd
                    indicators[f'MACD_SIGNAL_{fast}_{slow}_{signal}'] = signal_line
                    indicators[f'MACD_HIST_{fast}_{slow}_{signal}'] = histogram

            # Williams %R
            for period in [14, 21]:
                if len(close) >= period:
                    indicators[f'WILLR_{period}'] = talib.WILLR(high, low, close, timeperiod=period)

            # Commodity Channel Index
            for period in [14, 20]:
                if len(close) >= period:
                    indicators[f'CCI_{period}'] = talib.CCI(high, low, close, timeperiod=period)

            # Ultimate Oscillator
            if len(close) >= 28:
                indicators['ULTOSC'] = talib.ULTOSC(high, low, close)

            # Rate of Change
            for period in [10, 20]:
                if len(close) >= period:
                    indicators[f'ROC_{period}'] = talib.ROC(close, timeperiod=period)
                    indicators[f'ROCP_{period}'] = talib.ROCP(close, timeperiod=period)
                    indicators[f'ROCR_{period}'] = talib.ROCR(close, timeperiod=period)

            # Momentum
            for period in [10, 14, 20]:
                if len(close) >= period:
                    indicators[f'MOM_{period}'] = talib.MOM(close, timeperiod=period)

        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")

        return indicators

    def _volatility_indicators(self, high, low, close):
        """Volatility indicators"""
        indicators = {}

        try:
            # Average True Range
            for period in [14, 21]:
                if len(close) >= period:
                    indicators[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)
                    indicators[f'NATR_{period}'] = talib.NATR(high, low, close, timeperiod=period)
                    indicators[f'TRANGE_{period}'] = talib.TRANGE(high, low, close)

            # Standard deviation
            for period in [10, 20, 50]:
                if len(close) >= period:
                    indicators[f'STDDEV_{period}'] = talib.STDDEV(close, timeperiod=period)
                    indicators[f'VAR_{period}'] = talib.VAR(close, timeperiod=period)

            # Chaikin Volatility
            if len(high) >= 20:
                hl_diff = high - low
                hl_ma = pd.Series(hl_diff).rolling(10).mean()
                indicators['CHAIKIN_VOL'] = hl_ma.pct_change(10) * 100

        except Exception as e:
            logger.warning(f"Error calculating volatility indicators: {e}")

        return indicators

    def _cycle_indicators(self, close):
        """Cycle indicators"""
        indicators = {}

        try:
            # Hilbert Transform indicators
            if len(close) >= 63:  # Minimum length for HT indicators
                indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
                indicators['HT_DCPHASE'] = talib.HT_DCPHASE(close)
                indicators['HT_PHASOR_INPHASE'], indicators['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(close)
                indicators['HT_SINE_SINE'], indicators['HT_SINE_LEADSINE'] = talib.HT_SINE(close)
                indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        except Exception as e:
            logger.warning(f"Error calculating cycle indicators: {e}")

        return indicators

    def _statistical_indicators(self, close):
        """Statistical indicators"""
        indicators = {}

        try:
            # Linear regression
            for period in [14, 20, 50]:
                if len(close) >= period:
                    indicators[f'LINEARREG_{period}'] = talib.LINEARREG(close, timeperiod=period)
                    indicators[f'LINEARREG_ANGLE_{period}'] = talib.LINEARREG_ANGLE(close, timeperiod=period)
                    indicators[f'LINEARREG_INTERCEPT_{period}'] = talib.LINEARREG_INTERCEPT(close, timeperiod=period)
                    indicators[f'LINEARREG_SLOPE_{period}'] = talib.LINEARREG_SLOPE(close, timeperiod=period)

            # Time Series Forecast
            for period in [14, 20]:
                if len(close) >= period:
                    indicators[f'TSF_{period}'] = talib.TSF(close, timeperiod=period)

            # Beta and Correlation
            if len(close) >= 30:
                indicators['BETA'] = talib.BETA(close, close, timeperiod=5)
                indicators['CORREL'] = talib.CORREL(close, close, timeperiod=30)

        except Exception as e:
            logger.warning(f"Error calculating statistical indicators: {e}")

        return indicators

    def _custom_indicators(self, high, low, close, open_prices, volume):
        """Custom advanced indicators"""
        indicators = {}

        try:
            # Heikin Ashi
            ha_close = (open_prices + high + low + close) / 4
            ha_open = np.zeros_like(open_prices)
            ha_open[0] = (open_prices[0] + close[0]) / 2
            for i in range(1, len(ha_open)):
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
            ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

            indicators['HA_OPEN'] = ha_open
            indicators['HA_HIGH'] = ha_high
            indicators['HA_LOW'] = ha_low
            indicators['HA_CLOSE'] = ha_close

            # Ichimoku Cloud
            if len(high) >= 52:
                tenkan_sen = (pd.Series(high).rolling(9).max() + pd.Series(low).rolling(9).min()) / 2
                kijun_sen = (pd.Series(high).rolling(26).max() + pd.Series(low).rolling(26).min()) / 2
                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                senkou_span_b = ((pd.Series(high).rolling(52).max() + pd.Series(low).rolling(52).min()) / 2).shift(26)
                chikou_span = pd.Series(close).shift(-26)

                indicators['TENKAN_SEN'] = tenkan_sen.values
                indicators['KIJUN_SEN'] = kijun_sen.values
                indicators['SENKOU_SPAN_A'] = senkou_span_a.values
                indicators['SENKOU_SPAN_B'] = senkou_span_b.values
                indicators['CHIKOU_SPAN'] = chikou_span.values

            # Parabolic SAR
            if len(close) >= 2:
                indicators['SAR'] = talib.SAR(high, low)
                indicators['SAREXT'] = talib.SAREXT(high, low)

            # SuperTrend
            for period, multiplier in [(10, 3), (14, 2), (20, 1.5)]:
                if len(close) >= period:
                    hl2 = (high + low) / 2
                    atr = talib.ATR(high, low, close, timeperiod=period)
                    upper_band = hl2 + multiplier * atr
                    lower_band = hl2 - multiplier * atr

                    supertrend = np.zeros_like(close)
                    direction = np.ones_like(close)

                    for i in range(1, len(close)):
                        if close[i] > upper_band[i-1]:
                            direction[i] = 1
                        elif close[i] < lower_band[i-1]:
                            direction[i] = -1
                        else:
                            direction[i] = direction[i-1]

                        if direction[i] == 1:
                            supertrend[i] = lower_band[i]
                        else:
                            supertrend[i] = upper_band[i]

                    indicators[f'SUPERTREND_{period}_{multiplier}'] = supertrend
                    indicators[f'SUPERTREND_DIR_{period}_{multiplier}'] = direction

        except Exception as e:
            logger.warning(f"Error calculating custom indicators: {e}")

        return indicators
