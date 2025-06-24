#!/usr/bin/env python3
"""
ncOS Ultimate Microstructure Analyzer - Maximum Edition
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
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

    def _price_indicators(self, high, low, close, open_prices):
        """Price-based indicators"""
        indicators = {}

        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                indicators[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
                indicators[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
                indicators[f'WMA_{period}'] = talib.WMA(close, timeperiod=period)
                indicators[f'TEMA_{period}'] = talib.TEMA(close, timeperiod=period)
                indicators[f'TRIMA_{period}'] = talib.TRIMA(close, timeperiod=period)
                indicators[f'KAMA_{period}'] = talib.KAMA(close, timeperiod=period)
                indicators[f'MAMA_{period}'], indicators[f'FAMA_{period}'] = talib.MAMA(close)
            except:
                pass

        # Bollinger Bands (multiple periods)
        for period in [10, 20, 50]:
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
                indicators[f'BB_UPPER_{period}'] = bb_upper
                indicators[f'BB_MIDDLE_{period}'] = bb_middle
                indicators[f'BB_LOWER_{period}'] = bb_lower
                indicators[f'BB_WIDTH_{period}'] = (bb_upper - bb_lower) / bb_middle
                indicators[f'BB_POSITION_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
            except:
                pass

        # Donchian Channels
        for period in [10, 20, 55]:
            try:
                indicators[f'DONCHIAN_UPPER_{period}'] = pd.Series(high).rolling(period).max().values
                indicators[f'DONCHIAN_LOWER_{period}'] = pd.Series(low).rolling(period).min().values
                indicators[f'DONCHIAN_MIDDLE_{period}'] = (indicators[f'DONCHIAN_UPPER_{period}'] + indicators[f'DONCHIAN_LOWER_{period}']) / 2
            except:
                pass

        # Pivot Points
        try:
            indicators['PIVOT'] = (high + low + close) / 3
            indicators['R1'] = 2 * indicators['PIVOT'] - low
            indicators['S1'] = 2 * indicators['PIVOT'] - high
            indicators['R2'] = indicators['PIVOT'] + (high - low)
            indicators['S2'] = indicators['PIVOT'] - (high - low)
            indicators['R3'] = high + 2 * (indicators['PIVOT'] - low)
            indicators['S3'] = low - 2 * (high - indicators['PIVOT'])
        except:
            pass

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
                indicators[f'VOLUME_SMA_{period}'] = talib.SMA(volume, timeperiod=period)
                indicators[f'VOLUME_RATIO_{period}'] = volume / indicators[f'VOLUME_SMA_{period}']

            # Volume price trend
            indicators['VPT'] = np.cumsum(volume * (close - np.roll(close, 1)) / np.roll(close, 1))

            # Money flow index
            indicators['MFI_14'] = talib.MFI(high, low, close, volume, timeperiod=14)

            # Volume weighted average price
            indicators['VWAP'] = np.cumsum(volume * close) / np.cumsum(volume)

        except:
            pass

        return indicators

    def _momentum_indicators(self, high, low, close):
        """Momentum indicators"""
        indicators = {}

        try:
            # RSI (multiple periods)
            for period in [9, 14, 21]:
                indicators[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

            # Stochastic
            indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high, low, close)
            indicators['STOCHF_K'], indicators['STOCHF_D'] = talib.STOCHF(high, low, close)
            indicators['STOCHRSI_K'], indicators['STOCHRSI_D'] = talib.STOCHRSI(close)

            # MACD (multiple settings)
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
                macd, signal_line, histogram = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
                indicators[f'MACD_{fast}_{slow}_{signal}'] = macd
                indicators[f'MACD_SIGNAL_{fast}_{slow}_{signal}'] = signal_line
                indicators[f'MACD_HIST_{fast}_{slow}_{signal}'] = histogram

            # Williams %R
            for period in [14, 21]:
                indicators[f'WILLR_{period}'] = talib.WILLR(high, low, close, timeperiod=period)

            # Commodity Channel Index
            for period in [14, 20]:
                indicators[f'CCI_{period}'] = talib.CCI(high, low, close, timeperiod=period)

            # Ultimate Oscillator
            indicators['ULTOSC'] = talib.ULTOSC(high, low, close)

            # Rate of Change
            for period in [10, 20]:
                indicators[f'ROC_{period}'] = talib.ROC(close, timeperiod=period)
                indicators[f'ROCP_{period}'] = talib.ROCP(close, timeperiod=period)
                indicators[f'ROCR_{period}'] = talib.ROCR(close, timeperiod=period)

            # Momentum
            for period in [10, 14, 20]:
                indicators[f'MOM_{period}'] = talib.MOM(close, timeperiod=period)

        except:
            pass

        return indicators

    def _volatility_indicators(self, high, low, close):
        """Volatility indicators"""
        indicators = {}

        try:
            # Average True Range
            for period in [14, 21]:
                indicators[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)
                indicators[f'NATR_{period}'] = talib.NATR(high, low, close, timeperiod=period)
                indicators[f'TRANGE_{period}'] = talib.TRANGE(high, low, close)

            # Standard deviation
            for period in [10, 20, 50]:
                indicators[f'STDDEV_{period}'] = talib.STDDEV(close, timeperiod=period)
                indicators[f'VAR_{period}'] = talib.VAR(close, timeperiod=period)

            # Chaikin Volatility
            indicators['CHAIKIN_VOL'] = ((high - low).rolling(10).mean().pct_change(10) * 100)

        except:
            pass

        return indicators

    def _cycle_indicators(self, close):
        """Cycle indicators"""
        indicators = {}

        try:
            # Hilbert Transform indicators
            indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
            indicators['HT_DCPHASE'] = talib.HT_DCPHASE(close)
            indicators['HT_PHASOR_INPHASE'], indicators['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(close)
            indicators['HT_SINE_SINE'], indicators['HT_SINE_LEADSINE'] = talib.HT_SINE(close)
            indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        except:
            pass

        return indicators

    def _statistical_indicators(self, close):
        """Statistical indicators"""
        indicators = {}

        try:
            # Linear regression
            for period in [14, 20, 50]:
                indicators[f'LINEARREG_{period}'] = talib.LINEARREG(close, timeperiod=period)
                indicators[f'LINEARREG_ANGLE_{period}'] = talib.LINEARREG_ANGLE(close, timeperiod=period)
                indicators[f'LINEARREG_INTERCEPT_{period}'] = talib.LINEARREG_INTERCEPT(close, timeperiod=period)
                indicators[f'LINEARREG_SLOPE_{period}'] = talib.LINEARREG_SLOPE(close, timeperiod=period)

            # Time Series Forecast
            for period in [14, 20]:
                indicators[f'TSF_{period}'] = talib.TSF(close, timeperiod=period)

            # Beta
            indicators['BETA'] = talib.BETA(close, close, timeperiod=5)
            indicators['CORREL'] = talib.CORREL(close, close, timeperiod=30)

        except:
            pass

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
            indicators['SAR'] = talib.SAR(high, low)
            indicators['SAREXT'] = talib.SAREXT(high, low)

            # SuperTrend
            for period, multiplier in [(10, 3), (14, 2), (20, 1.5)]:
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

class HarmonicPatternDetector:
    """Advanced harmonic pattern detection"""

    def __init__(self):
        self.patterns = {
            'GARTLEY': {'XA': 0.618, 'AB': 0.382, 'BC': 0.886, 'CD': 0.786},
            'BUTTERFLY': {'XA': 0.786, 'AB': 0.382, 'BC': 0.886, 'CD': 1.27},
            'BAT': {'XA': 0.382, 'AB': 0.382, 'BC': 0.886, 'CD': 0.886},
            'CRAB': {'XA': 0.382, 'AB': 0.382, 'BC': 0.886, 'CD': 1.618},
            'SHARK': {'XA': 0.5, 'AB': 0.5, 'BC': 1.618, 'CD': 0.886},
            'CYPHER': {'XA': 0.382, 'AB': 0.382, 'BC': 1.272, 'CD': 0.786},
            'THREE_DRIVES': {'XA': 0.618, 'AB': 0.618, 'BC': 0.618, 'CD': 0.786},
            'ABCD': {'AB': 0.618, 'BC': 0.618, 'CD': 1.272}
        }

    def detect_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all harmonic patterns"""
        patterns_found = []

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Find pivot points
            pivots = self._find_pivots(high, low, close)

            # Check for patterns
            for pattern_name, ratios in self.patterns.items():
                pattern_results = self._check_pattern(pivots, ratios, pattern_name)
                patterns_found.extend(pattern_results)

        except Exception as e:
            logger.warning(f"Error detecting harmonic patterns: {e}")

        return patterns_found

    def _find_pivots(self, high, low, close, window=5):
        """Find pivot points"""
        pivots = []

        for i in range(window, len(high) - window):
            # Pivot high
            if all(high[i] >= high[i-j] for j in range(1, window+1)) and                all(high[i] >= high[i+j] for j in range(1, window+1)):
                pivots.append({'index': i, 'price': high[i], 'type': 'high'})

            # Pivot low
            if all(low[i] <= low[i-j] for j in range(1, window+1)) and                all(low[i] <= low[i+j] for j in range(1, window+1)):
                pivots.append({'index': i, 'price': low[i], 'type': 'low'})

        return sorted(pivots, key=lambda x: x['index'])

    def _check_pattern(self, pivots, ratios, pattern_name):
        """Check for specific harmonic pattern"""
        patterns = []

        if len(pivots) < 5:
            return patterns

        # Check all possible 5-point combinations
        for i in range(len(pivots) - 4):
            points = pivots[i:i+5]

            # Validate pattern structure
            if self._validate_pattern_structure(points, ratios):
                patterns.append({
                    'pattern': pattern_name,
                    'points': points,
                    'ratios': ratios,
                    'completion_time': points[-1]['index'],
                    'completion_price': points[-1]['price']
                })

        return patterns

    def _validate_pattern_structure(self, points, ratios):
        """Validate if points form a valid harmonic pattern"""
        try:
            # Calculate ratios between points
            xa = abs(points[1]['price'] - points[0]['price'])
            ab = abs(points[2]['price'] - points[1]['price'])
            bc = abs(points[3]['price'] - points[2]['price'])
            cd = abs(points[4]['price'] - points[3]['price'])

            tolerance = 0.05  # 5% tolerance

            # Check each ratio
            for ratio_name, expected_ratio in ratios.items():
                if ratio_name == 'XA':
                    actual_ratio = ab / xa if xa != 0 else 0
                elif ratio_name == 'AB':
                    actual_ratio = bc / ab if ab != 0 else 0
                elif ratio_name == 'BC':
                    actual_ratio = cd / bc if bc != 0 else 0
                elif ratio_name == 'CD':
                    actual_ratio = cd / xa if xa != 0 else 0
                else:
                    continue

                if abs(actual_ratio - expected_ratio) > tolerance:
                    return False

            return True

        except:
            return False

class SMCAnalyzer:
    """Smart Money Concepts Analysis"""

    def __init__(self):
        self.structure_levels = []
        self.liquidity_zones = []
        self.order_blocks = []
        self.fair_value_gaps = []

    def analyze_smc(self, df: pd.DataFrame) -> Dict:
        """Comprehensive SMC analysis"""
        results = {}

        try:
            # Market structure
            results['market_structure'] = self._analyze_market_structure(df)

            # Liquidity zones
            results['liquidity_zones'] = self._identify_liquidity_zones(df)

            # Order blocks
            results['order_blocks'] = self._identify_order_blocks(df)

            # Fair value gaps
            results['fair_value_gaps'] = self._identify_fair_value_gaps(df)

            # Breaker blocks
            results['breaker_blocks'] = self._identify_breaker_blocks(df)

            # Mitigation blocks
            results['mitigation_blocks'] = self._identify_mitigation_blocks(df)

            # Displacement analysis
            results['displacement'] = self._analyze_displacement(df)

            # Inducement analysis
            results['inducement'] = self._analyze_inducement(df)

        except Exception as e:
            logger.warning(f"Error in SMC analysis: {e}")

        return results

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Find swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(2, len(high) - 2):
            if high[i] > high[i-1] and high[i] > high[i+1] and                high[i] > high[i-2] and high[i] > high[i+2]:
                swing_highs.append({'index': i, 'price': high[i]})

            if low[i] < low[i-1] and low[i] < low[i+1] and                low[i] < low[i-2] and low[i] < low[i+2]:
                swing_lows.append({'index': i, 'price': low[i]})

        # Determine trend
        trend = self._determine_trend(swing_highs, swing_lows)

        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend': trend,
            'structure_breaks': self._find_structure_breaks(swing_highs, swing_lows)
        }

    def _identify_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Identify liquidity zones"""
        zones = []

        try:
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Equal highs/lows with high volume
            for i in range(1, len(high) - 1):
                # Check for equal highs
                if abs(high[i] - high[i-1]) < (high[i] * 0.001) and volume[i] > np.mean(volume):
                    zones.append({
                        'type': 'liquidity_high',
                        'price': high[i],
                        'index': i,
                        'strength': volume[i] / np.mean(volume)
                    })

                # Check for equal lows
                if abs(low[i] - low[i-1]) < (low[i] * 0.001) and volume[i] > np.mean(volume):
                    zones.append({
                        'type': 'liquidity_low',
                        'price': low[i],
                        'index': i,
                        'strength': volume[i] / np.mean(volume)
                    })

        except Exception as e:
            logger.warning(f"Error identifying liquidity zones: {e}")

        return zones

    def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify order blocks"""
        order_blocks = []

        try:
            open_prices = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            for i in range(1, len(df) - 1):
                # Bullish order block
                if close[i] > open_prices[i] and close[i+1] > high[i]:
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'top': high[i],
                        'bottom': open_prices[i],
                        'index': i,
                        'strength': (close[i] - open_prices[i]) / open_prices[i]
                    })

                # Bearish order block
                if close[i] < open_prices[i] and close[i+1] < low[i]:
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'top': open_prices[i],
                        'bottom': low[i],
                        'index': i,
                        'strength': (open_prices[i] - close[i]) / open_prices[i]
                    })

        except Exception as e:
            logger.warning(f"Error identifying order blocks: {e}")

        return order_blocks

    def _identify_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Identify fair value gaps"""
        gaps = []

        try:
            high = df['high'].values
            low = df['low'].values

            for i in range(2, len(df)):
                # Bullish FVG
                if low[i] > high[i-2]:
                    gaps.append({
                        'type': 'bullish_fvg',
                        'top': low[i],
                        'bottom': high[i-2],
                        'index': i,
                        'size': low[i] - high[i-2]
                    })

                # Bearish FVG
                if high[i] < low[i-2]:
                    gaps.append({
                        'type': 'bearish_fvg',
                        'top': low[i-2],
                        'bottom': high[i],
                        'index': i,
                        'size': low[i-2] - high[i]
                    })

        except Exception as e:
            logger.warning(f"Error identifying fair value gaps: {e}")

        return gaps

    def _identify_breaker_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify breaker blocks"""
        breakers = []

        try:
            # Implementation of breaker block identification
            order_blocks = self._identify_order_blocks(df)
            close = df['close'].values

            for ob in order_blocks:
                # Check if order block was broken
                for i in range(ob['index'] + 1, len(close)):
                    if ob['type'] == 'bullish_ob' and close[i] < ob['bottom']:
                        breakers.append({
                            'type': 'bearish_breaker',
                            'original_ob': ob,
                            'break_index': i,
                            'break_price': close[i]
                        })
                        break
                    elif ob['type'] == 'bearish_ob' and close[i] > ob['top']:
                        breakers.append({
                            'type': 'bullish_breaker',
                            'original_ob': ob,
                            'break_index': i,
                            'break_price': close[i]
                        })
                        break

        except Exception as e:
            logger.warning(f"Error identifying breaker blocks: {e}")

        return breakers

    def _identify_mitigation_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify mitigation blocks"""
        mitigations = []

        try:
            order_blocks = self._identify_order_blocks(df)
            close = df['close'].values

            for ob in order_blocks:
                # Check for mitigation (price returning to order block)
                for i in range(ob['index'] + 1, len(close)):
                    if ob['type'] == 'bullish_ob':
                        if ob['bottom'] <= close[i] <= ob['top']:
                            mitigations.append({
                                'type': 'bullish_mitigation',
                                'original_ob': ob,
                                'mitigation_index': i,
                                'mitigation_price': close[i]
                            })
                            break
                    elif ob['type'] == 'bearish_ob':
                        if ob['bottom'] <= close[i] <= ob['top']:
                            mitigations.append({
                                'type': 'bearish_mitigation',
                                'original_ob': ob,
                                'mitigation_index': i,
                                'mitigation_price': close[i]
                            })
                            break

        except Exception as e:
            logger.warning(f"Error identifying mitigation blocks: {e}")

        return mitigations

    def _analyze_displacement(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze displacement moves"""
        displacements = []

        try:
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Calculate price changes and volume
            price_changes = np.diff(close) / close[:-1]
            avg_change = np.mean(np.abs(price_changes))
            avg_volume = np.mean(volume)

            for i in range(1, len(price_changes)):
                # Significant price movement with volume
                if abs(price_changes[i]) > 2 * avg_change and volume[i] > 1.5 * avg_volume:
                    displacements.append({
                        'index': i,
                        'direction': 'bullish' if price_changes[i] > 0 else 'bearish',
                        'magnitude': abs(price_changes[i]),
                        'volume_ratio': volume[i] / avg_volume,
                        'price_change': price_changes[i]
                    })

        except Exception as e:
            logger.warning(f"Error analyzing displacement: {e}")

        return displacements

    def _analyze_inducement(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze inducement patterns"""
        inducements = []

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            for i in range(2, len(df) - 2):
                # Bullish inducement (fake breakout to downside)
                if low[i] < min(low[i-2:i]) and close[i] > low[i] and close[i+1] > close[i]:
                    inducements.append({
                        'type': 'bullish_inducement',
                        'index': i,
                        'low': low[i],
                        'recovery': close[i+1] - low[i]
                    })

                # Bearish inducement (fake breakout to upside)
                if high[i] > max(high[i-2:i]) and close[i] < high[i] and close[i+1] < close[i]:
                    inducements.append({
                        'type': 'bearish_inducement',
                        'index': i,
                        'high': high[i],
                        'decline': high[i] - close[i+1]
                    })

        except Exception as e:
            logger.warning(f"Error analyzing inducement: {e}")

        return inducements

    def _determine_trend(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> str:
        """Determine market trend"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'sideways'

        # Higher highs and higher lows = uptrend
        recent_highs = sorted(swing_highs, key=lambda x: x['index'])[-2:]
        recent_lows = sorted(swing_lows, key=lambda x: x['index'])[-2:]

        if (recent_highs[1]['price'] > recent_highs[0]['price'] and 
            recent_lows[1]['price'] > recent_lows[0]['price']):
            return 'uptrend'
        elif (recent_highs[1]['price'] < recent_highs[0]['price'] and 
              recent_lows[1]['price'] < recent_lows[0]['price']):
            return 'downtrend'
        else:
            return 'sideways'

    def _find_structure_breaks(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """Find market structure breaks"""
        breaks = []

        # Implementation of structure break detection
        # This would identify when price breaks through significant swing levels

        return breaks

class WyckoffAnalyzer:
    """Micro Wyckoff Analysis"""

    def __init__(self):
        self.phases = ['accumulation', 'markup', 'distribution', 'markdown']
        self.events = ['PS', 'SC', 'AR', 'ST', 'BC', 'LPS', 'SOS', 'LPSY', 'UTAD', 'LPSY2']

    def analyze_wyckoff(self, df: pd.DataFrame) -> Dict:
        """Comprehensive Wyckoff analysis"""
        results = {}

        try:
            # Identify Wyckoff phases
            results['phases'] = self._identify_phases(df)

            # Identify key events
            results['events'] = self._identify_events(df)

            # Volume analysis
            results['volume_analysis'] = self._analyze_volume_patterns(df)

            # Effort vs Result
            results['effort_vs_result'] = self._analyze_effort_vs_result(df)

            # Supply and Demand
            results['supply_demand'] = self._analyze_supply_demand(df)

        except Exception as e:
            logger.warning(f"Error in Wyckoff analysis: {e}")

        return results

    def _identify_phases(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Wyckoff phases"""
        phases = []

        try:
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Use rolling statistics to identify phases
            price_volatility = pd.Series(close).rolling(20).std()
            volume_avg = pd.Series(volume).rolling(20).mean()

            for i in range(20, len(df) - 20):
                # Low volatility + high volume = potential accumulation
                if (price_volatility.iloc[i] < price_volatility.rolling(50).mean().iloc[i] and
                    volume[i] > volume_avg.iloc[i]):
                    phases.append({
                        'phase': 'accumulation',
                        'start': i - 10,
                        'end': i + 10,
                        'strength': volume[i] / volume_avg.iloc[i]
                    })

                # High volatility + increasing volume = potential markup
                elif (price_volatility.iloc[i] > price_volatility.rolling(50).mean().iloc[i] and
                      close[i] > close[i-5] and volume[i] > volume_avg.iloc[i]):
                    phases.append({
                        'phase': 'markup',
                        'start': i - 5,
                        'end': i + 5,
                        'strength': (close[i] - close[i-5]) / close[i-5]
                    })

        except Exception as e:
            logger.warning(f"Error identifying Wyckoff phases: {e}")

        return phases

    def _identify_events(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Wyckoff events"""
        events = []

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Preliminary Support (PS) - first sign of buying interest
            for i in range(10, len(df) - 10):
                if (close[i] < close[i-5] and volume[i] > np.mean(volume[i-10:i]) and
                    close[i+1] > close[i]):
                    events.append({
                        'event': 'PS',
                        'index': i,
                        'price': close[i],
                        'volume': volume[i],
                        'description': 'Preliminary Support'
                    })

            # Selling Climax (SC) - panic selling
            for i in range(10, len(df) - 10):
                if (volume[i] > 2 * np.mean(volume[i-10:i]) and
                    close[i] < close[i-1] and
                    close[i+1] > close[i]):
                    events.append({
                        'event': 'SC',
                        'index': i,
                        'price': close[i],
                        'volume': volume[i],
                        'description': 'Selling Climax'
                    })

        except Exception as e:
            logger.warning(f"Error identifying Wyckoff events: {e}")

        return events

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        analysis = {}

        try:
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
            close = df['close'].values

            # Volume trend analysis
            volume_trend = np.polyfit(range(len(volume)), volume, 1)[0]

            # Volume divergence
            price_trend = np.polyfit(range(len(close)), close, 1)[0]

            analysis = {
                'volume_trend': volume_trend,
                'price_trend': price_trend,
                'divergence': (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0),
                'avg_volume': np.mean(volume),
                'volume_spikes': np.where(volume > 2 * np.mean(volume))[0].tolist()
            }

        except Exception as e:
            logger.warning(f"Error analyzing volume patterns: {e}")

        return analysis

    def _analyze_effort_vs_result(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze effort vs result"""
        analysis = []

        try:
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            for i in range(1, len(df)):
                price_change = abs(close[i] - close[i-1]) / close[i-1]
                volume_effort = volume[i] / np.mean(volume)

                if volume_effort > 1.5:  # High volume effort
                    if price_change < 0.001:  # Low price result
                        analysis.append({
                            'index': i,
                            'type': 'high_effort_low_result',
                            'effort': volume_effort,
                            'result': price_change,
                            'interpretation': 'Potential accumulation/distribution'
                        })
                    elif price_change > 0.01:  # High price result
                        analysis.append({
                            'index': i,
                            'type': 'high_effort_high_result',
                            'effort': volume_effort,
                            'result': price_change,
                            'interpretation': 'Strong move with support'
                        })

        except Exception as e:
            logger.warning(f"Error analyzing effort vs result: {e}")

        return analysis

    def _analyze_supply_demand(self, df: pd.DataFrame) -> Dict:
        """Analyze supply and demand"""
        analysis = {}

        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

            # Supply zones (resistance areas with high volume)
            supply_zones = []
            demand_zones = []

            for i in range(10, len(df) - 10):
                # Supply zone
                if (high[i] == max(high[i-5:i+5]) and
                    volume[i] > np.mean(volume[i-10:i+10])):
                    supply_zones.append({
                        'price': high[i],
                        'index': i,
                        'strength': volume[i] / np.mean(volume[i-10:i+10])
                    })

                # Demand zone
                if (low[i] == min(low[i-5:i+5]) and
                    volume[i] > np.mean(volume[i-10:i+10])):
                    demand_zones.append({
                        'price': low[i],
                        'index': i,
                        'strength': volume[i] / np.mean(volume[i-10:i+10])
                    })

            analysis = {
                'supply_zones': supply_zones,
                'demand_zones': demand_zones,
                'current_bias': self._determine_supply_demand_bias(close, volume)
            }

        except Exception as e:
            logger.warning(f"Error analyzing supply/demand: {e}")

        return analysis

    def _determine_supply_demand_bias(self, close, volume):
        """Determine current supply/demand bias"""
        try:
            recent_close = close[-10:]
            recent_volume = volume[-10:]

            # Up moves vs down moves with volume
            up_volume = sum(recent_volume[i] for i in range(1, len(recent_close)) if recent_close[i] > recent_close[i-1])
            down_volume = sum(recent_volume[i] for i in range(1, len(recent_close)) if recent_close[i] < recent_close[i-1])

            if up_volume > down_volume * 1.2:
                return 'demand_dominant'
            elif down_volume > up_volume * 1.2:
                return 'supply_dominant'
            else:
                return 'balanced'
        except:
            return 'unknown'

class TickDataAnalyzer:
    """Advanced tick data analysis"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.tick_limit = config.tick_bars_limit

    def analyze_tick_data(self, df: pd.DataFrame) -> Dict:
        """Comprehensive tick data analysis"""
        results = {}

        try:
            # Limit tick data if specified
            if len(df) > self.tick_limit:
                df = df.tail(self.tick_limit)
                logger.info(f"Limited tick data to last {self.tick_limit} ticks")

            # Microstructure analysis
            results['microstructure'] = self._analyze_microstructure(df)

            # Spread analysis
            results['spread_analysis'] = self._analyze_spreads(df)

            # Volume analysis
            results['volume_analysis'] = self._analyze_tick_volume(df)

            # Market manipulation detection
            results['manipulation'] = self._detect_manipulation(df)

            # Liquidity analysis
            results['liquidity'] = self._analyze_liquidity(df)

            # Order flow
            results['order_flow'] = self._analyze_order_flow(df)

        except Exception as e:
            logger.warning(f"Error in tick data analysis: {e}")

        return results

    def _analyze_microstructure(self, df: pd.DataFrame) -> Dict:
        """Analyze market microstructure"""
        analysis = {}

        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                bid = df['bid'].values
                ask = df['ask'].values

                # Mid price
                mid_price = (bid + ask) / 2

                # Price impact
                price_changes = np.diff(mid_price)

                # Tick rule (Lee-Ready algorithm)
                tick_rule = np.zeros(len(price_changes))
                for i in range(len(price_changes)):
                    if price_changes[i] > 0:
                        tick_rule[i] = 1  # Buy
                    elif price_changes[i] < 0:
                        tick_rule[i] = -1  # Sell
                    else:
                        tick_rule[i] = tick_rule[i-1] if i > 0 else 0

                analysis = {
                    'mid_price': mid_price,
                    'price_changes': price_changes,
                    'tick_rule': tick_rule,
                    'buy_ratio': np.sum(tick_rule == 1) / len(tick_rule),
                    'sell_ratio': np.sum(tick_rule == -1) / len(tick_rule),
                    'volatility': np.std(price_changes),
                    'autocorrelation': np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1] if len(price_changes) > 1 else 0
                }

        except Exception as e:
            logger.warning(f"Error analyzing microstructure: {e}")

        return analysis

    def _analyze_spreads(self, df: pd.DataFrame) -> Dict:
        """Analyze bid-ask spreads"""
        analysis = {}

        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                bid = df['bid'].values
                ask = df['ask'].values
                spread = ask - bid

                analysis = {
                    'mean_spread': np.mean(spread),
                    'median_spread': np.median(spread),
                    'std_spread': np.std(spread),
                    'max_spread': np.max(spread),
                    'min_spread': np.min(spread),
                    'spread_percentiles': {
                        '25th': np.percentile(spread, 25),
                        '75th': np.percentile(spread, 75),
                        '95th': np.percentile(spread, 95),
                        '99th': np.percentile(spread, 99)
                    },
                    'wide_spread_count': np.sum(spread > np.mean(spread) + 2 * np.std(spread)),
                    'narrow_spread_count': np.sum(spread < np.mean(spread) - np.std(spread))
                }

        except Exception as e:
            logger.warning(f"Error analyzing spreads: {e}")

        return analysis

    def _analyze_tick_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze tick volume patterns"""
        analysis = {}

        try:
            if 'volume' in df.columns:
                volume = df['volume'].values

                # Volume distribution
                analysis = {
                    'total_volume': np.sum(volume),
                    'mean_volume': np.mean(volume),
                    'median_volume': np.median(volume),
                    'volume_std': np.std(volume),
                    'volume_skewness': stats.skew(volume),
                    'volume_kurtosis': stats.kurtosis(volume),
                    'zero_volume_ticks': np.sum(volume == 0),
                    'high_volume_ticks': np.sum(volume > np.mean(volume) + 2 * np.std(volume)),
                    'volume_percentiles': {
                        '50th': np.percentile(volume, 50),
                        '75th': np.percentile(volume, 75),
                        '90th': np.percentile(volume, 90),
                        '95th': np.percentile(volume, 95),
                        '99th': np.percentile(volume, 99)
                    }
                }

        except Exception as e:
            logger.warning(f"Error analyzing tick volume: {e}")

        return analysis

    def _detect_manipulation(self, df: pd.DataFrame) -> Dict:
        """Detect market manipulation patterns"""
        manipulation = {}

        try:
            # Spoofing detection
            manipulation['spoofing'] = self._detect_spoofing(df)

            # Layering detection
            manipulation['layering'] = self._detect_layering(df)

            # Wash trading
            manipulation['wash_trading'] = self._detect_wash_trading(df)

            # Quote stuffing
            manipulation['quote_stuffing'] = self._detect_quote_stuffing(df)

            # Momentum ignition
            manipulation['momentum_ignition'] = self._detect_momentum_ignition(df)

        except Exception as e:
            logger.warning(f"Error detecting manipulation: {e}")

        return manipulation

    def _detect_spoofing(self, df: pd.DataFrame) -> List[Dict]:
        """Detect spoofing patterns"""
        spoofing_events = []

        try:
            if 'bid' in df.columns and 'ask' in df.columns and 'volume' in df.columns:
                bid = df['bid'].values
                ask = df['ask'].values
                volume = df['volume'].values

                # Look for large bid/ask changes with no volume
                for i in range(1, len(df)):
                    bid_change = abs(bid[i] - bid[i-1])
                    ask_change = abs(ask[i] - ask[i-1])

                    # Large price change with zero volume (potential spoofing)
                    if (bid_change > np.std(np.diff(bid)) * 3 or 
                        ask_change > np.std(np.diff(ask)) * 3) and volume[i] == 0:
                        spoofing_events.append({
                            'index': i,
                            'timestamp': df.index[i] if hasattr(df.index[i], 'timestamp') else i,
                            'bid_change': bid_change,
                            'ask_change': ask_change,
                            'volume': volume[i],
                            'severity': max(bid_change, ask_change)
                        })

        except Exception as e:
            logger.warning(f"Error detecting spoofing: {e}")

        return spoofing_events

    def _detect_layering(self, df: pd.DataFrame) -> List[Dict]:
        """Detect layering patterns"""
        layering_events = []

        try:
            # Layering detection would require order book data
            # This is a simplified implementation
            if 'bid' in df.columns and 'ask' in df.columns:
                bid = df['bid'].values
                ask = df['ask'].values

                # Look for rapid bid/ask adjustments
                for i in range(5, len(df)):
                    recent_bid_changes = np.diff(bid[i-5:i])
                    recent_ask_changes = np.diff(ask[i-5:i])

                    # Multiple rapid changes in same direction
                    if (np.sum(recent_bid_changes > 0) >= 3 or 
                        np.sum(recent_ask_changes > 0) >= 3):
                        layering_events.append({
                            'index': i,
                            'pattern': 'rapid_adjustments',
                            'bid_changes': recent_bid_changes.tolist(),
                            'ask_changes': recent_ask_changes.tolist()
                        })

        except Exception as e:
            logger.warning(f"Error detecting layering: {e}")

        return layering_events

    def _detect_wash_trading(self, df: pd.DataFrame) -> List[Dict]:
        """Detect wash trading patterns"""
        wash_events = []

        try:
            if 'volume' in df.columns and 'last' in df.columns:
                volume = df['volume'].values
                last_price = df['last'].values

                # Look for high volume with minimal price movement
                for i in range(10, len(df)):
                    recent_volume = np.sum(volume[i-10:i])
                    price_range = np.max(last_price[i-10:i]) - np.min(last_price[i-10:i])
                    avg_price = np.mean(last_price[i-10:i])

                    # High volume, low price movement
                    if (recent_volume > np.mean(volume) * 5 and 
                        price_range < avg_price * 0.001):
                        wash_events.append({
                            'index': i,
                            'volume': recent_volume,
                            'price_range': price_range,
                            'suspicion_level': recent_volume / (price_range + 1e-10)
                        })

        except Exception as e:
            logger.warning(f"Error detecting wash trading: {e}")

        return wash_events

    def _detect_quote_stuffing(self, df: pd.DataFrame) -> Dict:
        """Detect quote stuffing"""
        analysis = {}

        try:
            # Count rapid quote updates
            timestamps = df.index if hasattr(df, 'index') else range(len(df))

            # Calculate time between quotes
            if len(timestamps) > 1:
                time_diffs = np.diff([t.timestamp() if hasattr(t, 'timestamp') else t for t in timestamps])

                # Very rapid quotes (< 1ms apart) could indicate stuffing
                rapid_quotes = np.sum(time_diffs < 0.001)

                analysis = {
                    'total_quotes': len(df),
                    'rapid_quotes': rapid_quotes,
                    'rapid_quote_ratio': rapid_quotes / len(df),
                    'mean_time_between_quotes': np.mean(time_diffs),
                    'min_time_between_quotes': np.min(time_diffs),
                    'stuffing_suspected': rapid_quotes > len(df) * 0.1
                }

        except Exception as e:
            logger.warning(f"Error detecting quote stuffing: {e}")

        return analysis

    def _detect_momentum_ignition(self, df: pd.DataFrame) -> List[Dict]:
        """Detect momentum ignition patterns"""
        ignition_events = []

        try:
            if 'last' in df.columns and 'volume' in df.columns:
                last_price = df['last'].values
                volume = df['volume'].values

                # Look for sudden price spikes with high volume
                price_changes = np.diff(last_price) / last_price[:-1]

                for i in range(1, len(price_changes)):
                    # Sudden large price movement with high volume
                    if (abs(price_changes[i]) > np.std(price_changes) * 3 and
                        volume[i] > np.mean(volume) * 2):
                        ignition_events.append({
                            'index': i,
                            'price_change': price_changes[i],
                            'volume': volume[i],
                            'intensity': abs(price_changes[i]) * volume[i]
                        })

        except Exception as e:
            logger.warning(f"Error detecting momentum ignition: {e}")

        return ignition_events

    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict:
        """Analyze liquidity patterns"""
        analysis = {}

        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                spread = df['ask'] - df['bid']

                # Liquidity proxies
                analysis = {
                    'mean_spread': np.mean(spread),
                    'spread_volatility': np.std(spread),
                    'liquidity_score': 1 / (np.mean(spread) + 1e-10),
                    'illiquidity_periods': self._identify_illiquid_periods(df),
                    'liquidity_trend': self._analyze_liquidity_trend(spread)
                }

        except Exception as e:
            logger.warning(f"Error analyzing liquidity: {e}")

        return analysis

    def _identify_illiquid_periods(self, df: pd.DataFrame) -> List[Dict]:
        """Identify periods of low liquidity"""
        periods = []

        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                spread = df['ask'] - df['bid']
                high_spread_threshold = np.mean(spread) + 2 * np.std(spread)

                in_period = False
                start_idx = 0

                for i, s in enumerate(spread):
                    if s > high_spread_threshold and not in_period:
                        start_idx = i
                        in_period = True
                    elif s <= high_spread_threshold and in_period:
                        periods.append({
                            'start': start_idx,
                            'end': i,
                            'duration': i - start_idx,
                            'avg_spread': np.mean(spread[start_idx:i])
                        })
                        in_period = False

        except Exception as e:
            logger.warning(f"Error identifying illiquid periods: {e}")

        return periods

    def _analyze_liquidity_trend(self, spread: np.ndarray) -> Dict:
        """Analyze liquidity trend"""
        try:
            # Rolling average of spread
            window = min(100, len(spread) // 4)
            rolling_spread = pd.Series(spread).rolling(window).mean()

            # Trend analysis
            x = np.arange(len(rolling_spread))
            valid_idx = ~np.isnan(rolling_spread)

            if np.sum(valid_idx) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[valid_idx], rolling_spread[valid_idx]
                )

                return {
                    'trend_slope': slope,
                    'trend_r_squared': r_value**2,
                    'trend_p_value': p_value,
                    'improving_liquidity': slope < 0,
                    'trend_strength': abs(r_value)
                }
        except:
            pass

        return {}

    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze order flow"""
        analysis = {}

        try:
            if 'bid' in df.columns and 'ask' in df.columns and 'last' in df.columns:
                bid = df['bid'].values
                ask = df['ask'].values
                last = df['last'].values
                volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

                # Classify trades as buy/sell
                trade_classification = []
                for i in range(len(df)):
                    mid_price = (bid[i] + ask[i]) / 2
                    if last[i] > mid_price:
                        trade_classification.append('buy')
                    elif last[i] < mid_price:
                        trade_classification.append('sell')
                    else:
                        trade_classification.append('neutral')

                # Calculate order flow metrics
                buy_volume = sum(volume[i] for i, t in enumerate(trade_classification) if t == 'buy')
                sell_volume = sum(volume[i] for i, t in enumerate(trade_classification) if t == 'sell')

                analysis = {
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'net_order_flow': buy_volume - sell_volume,
                    'order_flow_ratio': buy_volume / (sell_volume + 1e-10),
                    'trade_classification': trade_classification,
                    'aggressive_buy_ratio': np.sum([1 for t in trade_classification if t == 'buy']) / len(trade_classification),
                    'aggressive_sell_ratio': np.sum([1 for t in trade_classification if t == 'sell']) / len(trade_classification)
                }

        except Exception as e:
            logger.warning(f"Error analyzing order flow: {e}")

        return analysis

class UltimateDataProcessor:
    """Ultimate data processing engine with all features enabled by default"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.indicator_engine = UltimateIndicatorEngine()
        self.harmonic_detector = HarmonicPatternDetector()
        self.smc_analyzer = SMCAnalyzer()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        self.tick_analyzer = TickDataAnalyzer(config)

    async def process_file(self, file_path: str) -> Dict:
        """Process a single file with maximum analysis"""
        logger.info(f"Processing file: {file_path}")

        try:
            # Load data
            df = await self._load_data(file_path)
            if df is None or df.empty:
                return {'error': 'Failed to load data'}

            # Apply limits
            df = self._apply_limits(df, file_path)

            # Initialize results
            results = {
                'file_path': file_path,
                'total_bars': len(df),
                'timeframe': self._detect_timeframe(file_path),
                'currency_pair': self._detect_currency_pair(file_path),
                'analysis_timestamp': datetime.now().isoformat()
            }

            # Basic statistics
            results['basic_stats'] = self._calculate_basic_stats(df)

            # Technical indicators (ALWAYS ENABLED)
            results['indicators'] = self.indicator_engine.calculate_all_indicators(df)

            # Pattern recognition (ALWAYS ENABLED)
            results['harmonic_patterns'] = self.harmonic_detector.detect_patterns(df)

            # SMC Analysis (ALWAYS ENABLED)
            results['smc_analysis'] = self.smc_analyzer.analyze_smc(df)

            # Wyckoff Analysis (ALWAYS ENABLED)
            results['wyckoff_analysis'] = self.wyckoff_analyzer.analyze_wyckoff(df)

            # Tick data analysis (ALWAYS ENABLED if tick data)
            if self._is_tick_data(file_path):
                results['tick_analysis'] = self.tick_analyzer.analyze_tick_data(df)

            # Advanced analytics (ALWAYS ENABLED)
            results['advanced_analytics'] = await self._perform_advanced_analytics(df)

            # Risk metrics (ALWAYS ENABLED)
            results['risk_metrics'] = self._calculate_risk_metrics(df)

            # Machine learning features (ALWAYS ENABLED)
            results['ml_features'] = await self._extract_ml_features(df)

            logger.info(f"Completed processing: {file_path}")
            return results

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {'error': str(e), 'file_path': file_path}

    async def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from CSV and JSON files"""
        try:
            if file_path.endswith('.csv'):
                # Try different separators
                for sep in ['	', ',', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep)
                        if len(df.columns) > 1:  # Successfully parsed
                            break
                    except:
                        continue

                # Convert timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)

                return df

            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Convert to DataFrame based on structure
                if isinstance(data, dict) and 'data' in data:
                    return pd.DataFrame(data['data'])
                elif isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame([data])

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _apply_limits(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Apply bar limits based on timeframe"""
        try:
            if self._is_tick_data(file_path):
                if len(df) > self.config.tick_bars_limit:
                    df = df.tail(self.config.tick_bars_limit)
                    logger.info(f"Limited to {self.config.tick_bars_limit} tick bars")
            else:
                timeframe = self._detect_timeframe(file_path)
                if timeframe in self.config.bar_limits:
                    limit = self.config.bar_limits[timeframe]
                    if len(df) > limit:
                        df = df.tail(limit)
                        logger.info(f"Limited to {limit} bars for {timeframe}")

            return df
        except:
            return df

    def _detect_timeframe(self, file_path: str) -> str:
        """Detect timeframe from filename"""
        timeframe_patterns = {
            r'(?i)tick': 'tick',
            r'(?i)m1|1min|1minute': '1min',
            r'(?i)m5|5min|5minute': '5min',
            r'(?i)m15|15min|15minute': '15min',
            r'(?i)m30|30min|30minute': '30min',
            r'(?i)h1|1h|1hour': '1h',
            r'(?i)h4|4h|4hour': '4h',
            r'(?i)d1|1d|1day|daily': '1d',
            r'(?i)w1|1w|1week|weekly': '1w',
            r'(?i)mn1|1mn|1month|monthly': '1M'
        }

        filename = Path(file_path).name.lower()
        for pattern, tf in timeframe_patterns.items():
            if re.search(pattern, filename):
                return tf

        return 'unknown'

    def _detect_currency_pair(self, file_path: str) -> str:
        """Detect currency pair from filename"""
        pair_patterns = [
            r'(?i)(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD|XAU|XAG|BTC|ETH|OIL|GAS|SPX|NAS|DJI|DAX|FTSE|NIKKEI)[A-Z]{3}',
            r'(?i)(GOLD|SILVER|CRUDE|BRENT)',
            r'(?i)(BITCOIN|ETHEREUM|LITECOIN|RIPPLE)'
        ]

        filename = Path(file_path).name
        for pattern in pair_patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(0).upper()

        return 'UNKNOWN'

    def _is_tick_data(self, file_path: str) -> bool:
        """Check if file contains tick data"""
        return 'tick' in Path(file_path).name.lower()

    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics"""
        stats = {}

        try:
            if 'close' in df.columns:
                close = df['close']
                stats.update({
                    'first_price': close.iloc[0],
                    'last_price': close.iloc[-1],
                    'high_price': close.max(),
                    'low_price': close.min(),
                    'price_change': close.iloc[-1] - close.iloc[0],
                    'price_change_pct': ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100,
                    'volatility': close.pct_change().std() * np.sqrt(252),
                    'skewness': close.pct_change().skew(),
                    'kurtosis': close.pct_change().kurtosis()
                })

            if 'volume' in df.columns:
                volume = df['volume']
                stats.update({
                    'total_volume': volume.sum(),
                    'avg_volume': volume.mean(),
                    'volume_std': volume.std(),
                    'max_volume': volume.max(),
                    'min_volume': volume.min()
                })

            # Time-based stats
            if hasattr(df.index, 'to_pydatetime'):
                stats.update({
                    'start_time': df.index[0].isoformat(),
                    'end_time': df.index[-1].isoformat(),
                    'duration_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600
                })

        except Exception as e:
            logger.warning(f"Error calculating basic stats: {e}")

        return stats

    async def _perform_advanced_analytics(self, df: pd.DataFrame) -> Dict:
        """Perform advanced analytics"""
        analytics = {}

        try:
            if 'close' in df.columns:
                close = df['close'].values

                # Fractal dimension
                analytics['fractal_dimension'] = self._calculate_fractal_dimension(close)

                # Hurst exponent
                analytics['hurst_exponent'] = self._calculate_hurst_exponent(close)

                # Approximate entropy
                analytics['approximate_entropy'] = self._calculate_approximate_entropy(close)

                # Detrended fluctuation analysis
                analytics['dfa_alpha'] = self._calculate_dfa(close)

                # Market efficiency ratio
                analytics['efficiency_ratio'] = self._calculate_efficiency_ratio(close)

                # Trend strength
                analytics['trend_strength'] = self._calculate_trend_strength(close)

                # Support and resistance levels
                analytics['support_resistance'] = self._find_support_resistance(df)

                # Fibonacci levels
                analytics['fibonacci_levels'] = self._calculate_fibonacci_levels(close)

        except Exception as e:
            logger.warning(f"Error in advanced analytics: {e}")

        return analytics

    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box counting method"""
        try:
            def box_count(data, r):
                n = len(data)
                boxes = np.ceil(n / r).astype(int)
                count = 0
                for i in range(boxes):
                    start = i * r
                    end = min((i + 1) * r, n)
                    if np.max(data[start:end]) - np.min(data[start:end]) > 0:
                        count += 1
                return count

            scales = np.logspace(0.5, 2, 10).astype(int)
            scales = np.unique(scales)
            counts = [box_count(data, r) for r in scales]

            # Fit line to log-log plot
            coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
            return -coeffs[0]

        except:
            return np.nan

    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent"""
        try:
            lags = range(2, min(100, len(data) // 4))
            tau = [np.var(np.diff(data, lag)) for lag in lags]

            # Fit line to log-log plot
            coeffs = np.polyfit(np.log(lags), np.log(tau), 1)
            return coeffs[0] / 2.0

        except:
            return np.nan

    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        try:
            if r is None:
                r = 0.2 * np.std(data)

            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])

            def _phi(m):
                N = len(data)
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)

                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], N, m) <= r:
                            C[i] += 1.0

                C = C / float(N - m + 1.0)
                phi = np.mean(np.log(C))
                return phi

            return _phi(m) - _phi(m + 1)

        except:
            return np.nan

    def _calculate_dfa(self, data: np.ndarray) -> float:
        """Calculate Detrended Fluctuation Analysis alpha"""
        try:
            # Integrate the data
            y = np.cumsum(data - np.mean(data))

            # Define scales
            scales = np.unique(np.logspace(0.5, 2.5, 20).astype(int))
            scales = scales[scales < len(data) // 4]

            F = []
            for scale in scales:
                # Divide into segments
                segments = len(y) // scale
                if segments == 0:
                    continue

                # Calculate fluctuation for each segment
                fluctuations = []
                for i in range(segments):
                    start = i * scale
                    end = (i + 1) * scale
                    segment = y[start:end]

                    # Fit polynomial trend
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)

                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((segment - trend)**2))
                    fluctuations.append(fluctuation)

                F.append(np.mean(fluctuations))

            # Fit line to log-log plot
            if len(F) > 1:
                valid_scales = scales[:len(F)]
                coeffs = np.polyfit(np.log(valid_scales), np.log(F), 1)
                return coeffs[0]

        except:
            pass

        return np.nan

    def _calculate_efficiency_ratio(self, data: np.ndarray, period: int = 20) -> float:
        """Calculate Market Efficiency Ratio"""
        try:
            if len(data) < period:
                return np.nan

            # Price change over period
            change = abs(data[-1] - data[-period])

            # Sum of absolute daily changes
            volatility = np.sum(np.abs(np.diff(data[-period:])))

            if volatility == 0:
                return 0

            return change / volatility

        except:
            return np.nan

    def _calculate_trend_strength(self, data: np.ndarray) -> Dict:
        """Calculate trend strength metrics"""
        try:
            # Linear regression slope
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

            # ADX-like calculation
            high = data  # Simplified for close price data
            low = data

            tr = np.maximum(high[1:] - low[1:], 
                           np.maximum(abs(high[1:] - data[:-1]), 
                                    abs(low[1:] - data[:-1])))

            atr = np.mean(tr)

            return {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_strength': abs(slope) / (atr + 1e-10),
                'trend_direction': 'up' if slope > 0 else 'down'
            }

        except:
            return {}

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Find local extrema
            highs = []
            lows = []

            for i in range(2, len(high) - 2):
                if high[i] > high[i-1] and high[i] > high[i+1] and                    high[i] > high[i-2] and high[i] > high[i+2]:
                    highs.append(high[i])

                if low[i] < low[i-1] and low[i] < low[i+1] and                    low[i] < low[i-2] and low[i] < low[i+2]:
                    lows.append(low[i])

            # Cluster similar levels
            resistance_levels = self._cluster_levels(highs)
            support_levels = self._cluster_levels(lows)

            return {
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'current_price': close[-1],
                'nearest_resistance': min(resistance_levels, key=lambda x: abs(x - close[-1])) if resistance_levels else None,
                'nearest_support': min(support_levels, key=lambda x: abs(x - close[-1])) if support_levels else None
            }

        except:
            return {}

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.001) -> List[float]:
        """Cluster price levels that are close together"""
        if not levels:
            return []

        try:
            levels = np.array(levels)
            clustered = []

            while len(levels) > 0:
                level = levels[0]
                cluster = levels[abs(levels - level) / level < tolerance]
                clustered.append(np.mean(cluster))
                levels = levels[abs(levels - level) / level >= tolerance]

            return sorted(clustered)

        except:
            return levels

    def _calculate_fibonacci_levels(self, data: np.ndarray) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            high = np.max(data)
            low = np.min(data)
            diff = high - low

            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

            # Retracement levels (from high)
            retracement_levels = {
                f"fib_{int(ratio*1000)}": high - (diff * ratio) 
                for ratio in fib_ratios
            }

            # Extension levels (beyond high)
            extension_levels = {
                f"ext_{int(ratio*1000)}": high + (diff * ratio) 
                for ratio in fib_ratios
            }

            return {
                'swing_high': high,
                'swing_low': low,
                'retracement_levels': retracement_levels,
                'extension_levels': extension_levels,
                'current_price': data[-1]
            }

        except:
            return {}

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        metrics = {}

        try:
            if 'close' in df.columns:
                close = df['close']
                returns = close.pct_change().dropna()

                # Basic risk metrics
                metrics.update({
                    'volatility_daily': returns.std(),
                    'volatility_annualized': returns.std() * np.sqrt(252),
                    'var_95': np.percentile(returns, 5),
                    'var_99': np.percentile(returns, 1),
                    'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
                    'cvar_99': returns[returns <= np.percentile(returns, 1)].mean(),
                    'max_drawdown': self._calculate_max_drawdown(close),
                    'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
                    'calmar_ratio': returns.mean() * 252 / abs(self._calculate_max_drawdown(close)) if self._calculate_max_drawdown(close) != 0 else 0
                })

                # Advanced risk metrics
                metrics.update({
                    'downside_deviation': returns[returns < 0].std(),
                    'up_capture_ratio': self._calculate_up_capture_ratio(returns),
                    'down_capture_ratio': self._calculate_down_capture_ratio(returns),
                    'beta': self._calculate_beta(returns),
                    'alpha': self._calculate_alpha(returns),
                    'information_ratio': self._calculate_information_ratio(returns),
                    'treynor_ratio': self._calculate_treynor_ratio(returns),
                })

        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")

        return metrics

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            return drawdown.min()
        except:
            return 0

    def _calculate_up_capture_ratio(self, returns: pd.Series) -> float:
        """Calculate up capture ratio (simplified)"""
        try:
            positive_returns = returns[returns > 0]
            if len(positive_returns) > 0:
                return positive_returns.mean() / returns.mean() if returns.mean() > 0 else 0
        except:
            pass
        return 0

    def _calculate_down_capture_ratio(self, returns: pd.Series) -> float:
        """Calculate down capture ratio (simplified)"""
        try:
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                return negative_returns.mean() / returns.mean() if returns.mean() < 0 else 0
        except:
            pass
        return 0

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta (simplified using market returns approximation)"""
        try:
            # Simplified beta calculation
            market_returns = returns  # Using same series as proxy
            covariance = np.cov(returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 0 else 1
        except:
            return 1

    def _calculate_alpha(self, returns: pd.Series) -> float:
        """Calculate alpha (simplified)"""
        try:
            beta = self._calculate_beta(returns)
            market_return = returns.mean()  # Simplified
            risk_free_rate = 0.02 / 252  # Simplified daily risk-free rate
            return returns.mean() - (risk_free_rate + beta * (market_return - risk_free_rate))
        except:
            return 0

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio"""
        try:
            benchmark_returns = returns.mean()  # Simplified benchmark
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std()
            return active_returns.mean() / tracking_error if tracking_error > 0 else 0
        except:
            return 0

    def _calculate_treynor_ratio(self, returns: pd.Series) -> float:
        """Calculate Treynor ratio"""
        try:
            beta = self._calculate_beta(returns)
            risk_free_rate = 0.02 / 252  # Simplified daily risk-free rate
            return (returns.mean() - risk_free_rate) / beta if beta != 0 else 0
        except:
            return 0

    async def _extract_ml_features(self, df: pd.DataFrame) -> Dict:
        """Extract machine learning features"""
        features = {}

        try:
            if 'close' in df.columns:
                close = df['close'].values

                # Technical features
                features['technical'] = self._extract_technical_features(df)

                # Statistical features
                features['statistical'] = self._extract_statistical_features(close)

                # Time-based features
                features['temporal'] = self._extract_temporal_features(df)

                # Microstructure features (if available)
                if 'bid' in df.columns and 'ask' in df.columns:
                    features['microstructure'] = self._extract_microstructure_features(df)

                # Regime detection
                features['regime'] = self._detect_market_regime(close)

                # Anomaly detection
                features['anomalies'] = self._detect_anomalies(close)

        except Exception as e:
            logger.warning(f"Error extracting ML features: {e}")

        return features

    def _extract_technical_features(self, df: pd.DataFrame) -> Dict:
        """Extract technical analysis features"""
        features = {}

        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # Price-based features
            features['price_position'] = (close[-1] - np.min(close)) / (np.max(close) - np.min(close))
            features['price_momentum'] = (close[-1] - close[-10]) / close[-10] if len(close) > 10 else 0

            # Volatility features
            returns = np.diff(close) / close[:-1]
            features['volatility_regime'] = 'high' if np.std(returns) > np.percentile(returns, 75) else 'low'

            # Trend features
            features['trend_strength'] = self._calculate_trend_strength(close)

        except Exception as e:
            logger.warning(f"Error extracting technical features: {e}")

        return features

    def _extract_statistical_features(self, data: np.ndarray) -> Dict:
        """Extract statistical features"""
        features = {}

        try:
            returns = np.diff(data) / data[:-1]

            features.update({
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'skew_return': stats.skew(returns),
                'kurt_return': stats.kurtosis(returns),
                'jb_statistic': stats.jarque_bera(returns)[0],
                'jb_pvalue': stats.jarque_bera(returns)[1],
                'ljung_box_stat': stats.diagnostic.acorr_ljungbox(returns, lags=10, return_df=True)['lb_stat'].iloc[-1] if len(returns) > 10 else 0,
                'adf_statistic': stats.adfuller(returns)[0] if len(returns) > 10 else 0,
                'adf_pvalue': stats.adfuller(returns)[1] if len(returns) > 10 else 0
            })

        except Exception as e:
            logger.warning(f"Error extracting statistical features: {e}")

        return features

    def _extract_temporal_features(self, df: pd.DataFrame) -> Dict:
        """Extract time-based features"""
        features = {}

        try:
            if hasattr(df.index, 'hour'):
                features['hour_of_day'] = df.index[-1].hour
                features['day_of_week'] = df.index[-1].dayofweek
                features['month'] = df.index[-1].month

                # Session features (assuming UTC time)
                hour = df.index[-1].hour
                if 0 <= hour < 8:
                    features['session'] = 'asian'
                elif 8 <= hour < 16:
                    features['session'] = 'european'
                else:
                    features['session'] = 'american'

        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")

        return features

    def _extract_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Extract microstructure features"""
        features = {}

        try:
            if 'bid' in df.columns and 'ask' in df.columns:
                spread = df['ask'] - df['bid']
                features.update({
                    'avg_spread': spread.mean(),
                    'spread_volatility': spread.std(),
                    'relative_spread': spread.mean() / df['close'].mean(),
                    'spread_regime': 'wide' if spread.mean() > spread.quantile(0.75) else 'narrow'
                })

        except Exception as e:
            logger.warning(f"Error extracting microstructure features: {e}")

        return features

    def _detect_market_regime(self, data: np.ndarray) -> Dict:
        """Detect market regime using clustering"""
        regime = {}

        try:
            # Calculate features for regime detection
            returns = np.diff(data) / data[:-1]
            volatility = pd.Series(returns).rolling(20).std().values

            # Create feature matrix
            features = []
            for i in range(20, len(returns)):
                features.append([
                    np.mean(returns[i-20:i]),  # Average return
                    volatility[i],  # Volatility
                    np.std(returns[i-20:i]),  # Return volatility
                ])

            if len(features) > 10:
                features = np.array(features)

                # Normalize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # K-means clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)

                # Interpret clusters
                current_regime = clusters[-1]
                regime_names = ['low_vol', 'high_vol', 'trending']

                regime = {
                    'current_regime': regime_names[current_regime],
                    'regime_probability': np.sum(clusters == current_regime) / len(clusters),
                    'regime_stability': 1 - len(np.unique(clusters[-10:])) / 10 if len(clusters) >= 10 else 0
                }

        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")

        return regime

    def _detect_anomalies(self, data: np.ndarray) -> Dict:
        """Detect anomalies in the data"""
        anomalies = {}

        try:
            returns = np.diff(data) / data[:-1]

            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(returns.reshape(-1, 1))

            # Statistical outliers
            z_scores = np.abs(stats.zscore(returns))
            statistical_outliers = z_scores > 3

            anomalies = {
                'isolation_forest_anomalies': np.sum(anomaly_scores == -1),
                'statistical_outliers': np.sum(statistical_outliers),
                'anomaly_ratio': np.sum(anomaly_scores == -1) / len(anomaly_scores),
                'recent_anomalies': np.sum(anomaly_scores[-20:] == -1) if len(anomaly_scores) >= 20 else 0
            }

        except Exception as e:
            logger.warning(f"Error detecting anomalies: {e}")

        return anomalies

async def main():
    """Main function with ALL features enabled by default"""
    parser = argparse.ArgumentParser(description='ncOS Ultimate Microstructure Analyzer - ALL FEATURES ENABLED')

    # Input options
    parser.add_argument('--directory', '-d', type=str, help='Directory to scan for files')
    parser.add_argument('--file', '-f', type=str, help='Single file to process')
    parser.add_argument('--pattern', '-p', type=str, default='*', help='File pattern to match (supports CSV and JSON)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursive directory scan')

    # Processing options
    parser.add_argument('--tick_bars', type=int, default=1500, help='Maximum number of tick bars to process (DEFAULT: 1500)')
    parser.add_argument('--timeframes', nargs='+', default=['1min', '5min', '15min', '30min', '1h', '4h', '1d'], 
                        help='Timeframes to process')
    parser.add_argument('--bar_limits', type=str, help='JSON string of bar limits per timeframe')

    # Output options
    parser.add_argument('--output_dir', '-o', type=str, default='./ultimate_analysis_results', help='Output directory')
    parser.add_argument('--no_compression', action='store_true', help='Disable output compression')

    args = parser.parse_args()

    # Create configuration with ALL FEATURES ENABLED BY DEFAULT
    config = AnalysisConfig(
        tick_bars_limit=args.tick_bars,
        bar_limits=json.loads(args.bar_limits) if args.bar_limits else {},
        # ALL ANALYSIS FEATURES ENABLED BY DEFAULT
        enable_all_indicators=True,
        enable_harmonic_patterns=True,
        enable_smc_analysis=True,
        enable_wyckoff_analysis=True,
        process_tick_data=True,
        enable_ml_predictions=True,
        enable_risk_metrics=True,
        save_all_plots=True,
        compress_output=not args.no_compression,
        process_csv_files=True,
        process_json_files=True
    )

    # Create processor
    processor = UltimateDataProcessor(config)

    # Get files to process (CSV and JSON by default)
    files_to_process = []

    if args.file:
        files_to_process.append(args.file)
    elif args.directory:
        directory = Path(args.directory)
        if directory.exists():
            # Process both CSV and JSON files by default
            csv_pattern = args.pattern.replace('*', '*.csv') if not args.pattern.endswith('.csv') else args.pattern
            json_pattern = args.pattern.replace('*', '*.json') if not args.pattern.endswith('.json') else args.pattern

            if args.recursive:
                files_to_process.extend(directory.rglob('*.csv'))
                files_to_process.extend(directory.rglob('*.json'))
            else:
                files_to_process.extend(directory.glob('*.csv'))
                files_to_process.extend(directory.glob('*.json'))

    if not files_to_process:
        logger.error("No CSV or JSON files found to process")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 ULTIMATE ANALYSIS STARTING - Processing {len(files_to_process)} files with ALL FEATURES ENABLED")
    logger.info(f"📊 Tick bars limit: {args.tick_bars}")
    logger.info(f"📁 Processing: CSV and JSON files")
    logger.info(f"💎 All indicators, SMC, Wyckoff, ML, Risk metrics ENABLED")

    # Process files
    results = {}

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(multiprocessing.cpu_count(), len(files_to_process))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(asyncio.run, processor.process_file(str(file_path))): file_path 
            for file_path in files_to_process
        }

        # Collect results
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results[str(file_path)] = result
                logger.info(f"✅ Completed: {file_path}")
            except Exception as e:
                logger.error(f"❌ Failed: {file_path} - {e}")
                results[str(file_path)] = {'error': str(e)}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ultimate_analysis_{timestamp}.json"

    # Save JSON results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Compress if requested
    if config.compress_output:
        with open(output_file, 'rb') as f_in:
            with gzip.open(f"{output_file}.gz", 'wb') as f_out:
                f_out.writelines(f_in)
        output_file.unlink()  # Remove uncompressed file
        logger.info(f"📦 Results compressed and saved to: {output_file}.gz")
    else:
        logger.info(f"💾 Results saved to: {output_file}")

    # Generate summary report
    summary_file = output_dir / f"summary_report_{timestamp}.json"
    summary = {
        'total_files_processed': len(files_to_process),
        'successful_analyses': len([r for r in results.values() if 'error' not in r]),
        'failed_analyses': len([r for r in results.values() if 'error' in r]),
        'analysis_timestamp': datetime.now().isoformat(),
        'configuration': {
            'tick_bars_limit': config.tick_bars_limit,
            'all_features_enabled': True,
            'csv_and_json_processing': True
        },
        'files_processed': list(results.keys())
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"📋 Summary report saved to: {summary_file}")
    logger.info(f"🎉 ULTIMATE ANALYSIS COMPLETE! Processed {len(files_to_process)} files")
    logger.info(f"✅ Success: {summary['successful_analyses']} | ❌ Failed: {summary['failed_analyses']}")

if __name__ == "__main__":
    asyncio.run(main())
