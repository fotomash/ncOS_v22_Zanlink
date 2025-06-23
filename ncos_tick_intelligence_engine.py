# ncos_tick_intelligence_engine.py
"""
ncOS Tick Intelligence Engine
Extracts institutional footprints from raw tick data for LLM consumption
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import asyncio
import json
from enum import Enum
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstitutionalEvent(Enum):
    """Types of institutional events we detect"""
    LIQUIDITY_SWEEP = "liquidity_sweep"
    MICRO_TRAP = "micro_trap"
    VOLUME_SURGE = "volume_surge"
    DELTA_DIVERGENCE = "delta_divergence"
    SPREAD_ANOMALY = "spread_anomaly"
    ABSORPTION = "absorption"
    ICEBERG_ORDER = "iceberg_order"
    STOP_RUN = "stop_run"


@dataclass
class MicroTrap:
    """Detected micro-trap pattern"""
    timestamp: datetime
    trap_type: str  # 'bull_trap' or 'bear_trap'
    entry_price: float
    trigger_price: float
    reversal_price: float
    volume_profile: Dict[str, float]
    confidence: float
    timeframe_seconds: int

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'trap_type': self.trap_type,
            'entry_price': self.entry_price,
            'trigger_price': self.trigger_price,
            'reversal_price': self.reversal_price,
            'volume_profile': self.volume_profile,
            'confidence': self.confidence,
            'timeframe_seconds': self.timeframe_seconds
        }


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep event"""
    timestamp: datetime
    sweep_type: str  # 'stop_hunt' or 'liquidity_grab'
    swept_level: float
    sweep_depth: float
    volume_burst: float
    recovery_speed: float  # seconds to recover
    participants_trapped: int  # estimated

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'sweep_type': self.sweep_type,
            'swept_level': self.swept_level,
            'sweep_depth': self.sweep_depth,
            'volume_burst': self.volume_burst,
            'recovery_speed': self.recovery_speed,
            'participants_trapped': self.participants_trapped
        }


@dataclass 
class TickBar:
    """Enhanced 1-second bar with institutional metrics"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_delta: float
    vwap: float
    spread_avg: float
    spread_max: float
    large_trades: int
    absorption_score: float

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'tick_count': self.tick_count,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'delta': self.delta,
            'cumulative_delta': self.cumulative_delta,
            'vwap': self.vwap,
            'spread_avg': self.spread_avg,
            'spread_max': self.spread_max,
            'large_trades': self.large_trades,
            'absorption_score': self.absorption_score
        }


class TickIntelligenceEngine:
    """
    Core engine for extracting institutional footprints from tick data
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

        # Buffers for streaming analysis
        self.tick_buffer = deque(maxlen=10000)  # Last 10k ticks
        self.bar_buffer = deque(maxlen=3600)    # Last hour of 1s bars
        self.event_buffer = deque(maxlen=100)   # Recent events

        # State tracking
        self.cumulative_delta = 0
        self.session_volume_profile = {}
        self.detected_levels = []
        self.last_process_time = datetime.now()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'micro_trap_window': 30,        # seconds
            'sweep_threshold_atr': 1.5,     # ATR multiplier
            'large_trade_threshold': 100,    # lots/contracts
            'delta_divergence_threshold': 2.0,
            'spread_anomaly_zscore': 3.0,
            'absorption_threshold': 0.7,
            'min_volume_surge': 2.5,        # multiplier vs average
        }

    def preprocess_ticks(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw tick data and add initial features

        Expected columns: timestamp, price, volume, bid, ask, [side]
        """
        df = tick_df.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("tick_df must have 'timestamp' column")

        # Calculate spread
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
        else:
            # Estimate spread from price movements
            df['spread'] = df['price'].diff().abs().rolling(10).mean()

        # Classify trades (buy/sell) if not provided
        if 'side' not in df.columns:
            df['side'] = self._classify_trades(df)

        # Calculate tick-level metrics
        df['price_change'] = df['price'].diff()
        df['volume_delta'] = df['volume'] * df['side'].map({1: 1, -1: -1})
        df['cumulative_delta'] = df['volume_delta'].cumsum()

        # Identify large trades
        volume_mean = df['volume'].rolling(100).mean()
        volume_std = df['volume'].rolling(100).std()
        df['large_trade'] = df['volume'] > (volume_mean + 2 * volume_std)

        # Time between ticks (for intensity analysis)
        df['tick_duration'] = df['timestamp'].diff().dt.total_seconds()
        df['tick_intensity'] = 1 / df['tick_duration'].clip(lower=0.001)

        return df

    def _classify_trades(self, df: pd.DataFrame) -> pd.Series:
        """Classify trades as buy (1) or sell (-1) using tick rule"""
        price_change = df['price'].diff()

        # Tick rule: if price up -> buy, if price down -> sell
        side = pd.Series(0, index=df.index)
        side[price_change > 0] = 1
        side[price_change < 0] = -1

        # Forward fill for unchanged prices
        side = side.replace(0, np.nan).fillna(method='ffill').fillna(1)

        return side.astype(int)

    def build_tick_bars(self, tick_df: pd.DataFrame, bar_size: str = '1S') -> List[TickBar]:
        """
        Build enhanced bars from tick data
        """
        bars = []

        # Resample to specified bar size
        grouped = tick_df.set_index('timestamp').resample(bar_size)

        for timestamp, group in grouped:
            if len(group) == 0:
                continue

            # Calculate buy/sell volumes
            buy_volume = group[group['side'] == 1]['volume'].sum()
            sell_volume = group[group['side'] == -1]['volume'].sum()

            # VWAP calculation
            if group['volume'].sum() > 0:
                vwap = (group['price'] * group['volume']).sum() / group['volume'].sum()
            else:
                vwap = group['price'].mean()

            # Absorption detection (price stays flat despite volume)
            price_range = group['price'].max() - group['price'].min()
            avg_range = tick_df['price'].rolling(100).apply(lambda x: x.max() - x.min()).mean()
            absorption_score = 1 - (price_range / avg_range) if avg_range > 0 else 0

            bar = TickBar(
                timestamp=timestamp,
                open=group['price'].iloc[0],
                high=group['price'].max(),
                low=group['price'].min(),
                close=group['price'].iloc[-1],
                volume=group['volume'].sum(),
                tick_count=len(group),
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                delta=buy_volume - sell_volume,
                cumulative_delta=group['cumulative_delta'].iloc[-1],
                vwap=vwap,
                spread_avg=group['spread'].mean() if 'spread' in group else 0,
                spread_max=group['spread'].max() if 'spread' in group else 0,
                large_trades=group['large_trade'].sum(),
                absorption_score=absorption_score
            )

            bars.append(bar)

        return bars

    def detect_micro_traps(self, tick_df: pd.DataFrame, window_seconds: int = 30) -> List[MicroTrap]:
        """
        Detect micro-trap patterns in tick data

        Micro-traps are quick fake breakouts designed to trap retail traders
        """
        traps = []

        # Need at least some data
        if len(tick_df) < 100:
            return traps

        # Calculate rolling metrics
        df = tick_df.copy()
        df['price_ma'] = df['price'].rolling(20).mean()
        df['volume_ma'] = df['volume'].rolling(50).mean()

        # Find potential trap zones (price spikes with volume)
        df['price_spike'] = (df['price'] - df['price_ma']).abs() / df['price_ma']
        df['volume_spike'] = df['volume'] / df['volume_ma']

        # Identify reversal points
        df['price_reversal'] = (
            (df['price'].diff() * df['price'].diff().shift(-1)) < 0
        )

        # Look for trap patterns
        spike_threshold = 0.001  # 0.1% price spike
        volume_threshold = 2.0   # 2x average volume

        potential_traps = df[
            (df['price_spike'] > spike_threshold) & 
            (df['volume_spike'] > volume_threshold) &
            (df['price_reversal'])
        ]

        for idx in potential_traps.index:
            # Analyze the trap pattern
            start_idx = max(0, idx - window_seconds)
            end_idx = min(len(df) - 1, idx + window_seconds)

            window = df.iloc[start_idx:end_idx]

            # Determine trap type
            pre_move = df.iloc[idx]['price'] - df.iloc[start_idx]['price']
            post_move = df.iloc[end_idx]['price'] - df.iloc[idx]['price']

            if pre_move > 0 and post_move < 0:
                trap_type = 'bull_trap'
            elif pre_move < 0 and post_move > 0:
                trap_type = 'bear_trap'
            else:
                continue

            # Calculate confidence based on reversal strength
            reversal_strength = abs(post_move / pre_move)
            confidence = min(1.0, reversal_strength)

            # Volume profile during trap
            volume_profile = {
                'entry_volume': window.iloc[:len(window)//3]['volume'].sum(),
                'trap_volume': window.iloc[len(window)//3:2*len(window)//3]['volume'].sum(),
                'exit_volume': window.iloc[2*len(window)//3:]['volume'].sum()
            }

            trap = MicroTrap(
                timestamp=df.iloc[idx]['timestamp'],
                trap_type=trap_type,
                entry_price=df.iloc[start_idx]['price'],
                trigger_price=df.iloc[idx]['price'],
                reversal_price=df.iloc[end_idx]['price'],
                volume_profile=volume_profile,
                confidence=confidence,
                timeframe_seconds=window_seconds
            )

            traps.append(trap)

        return traps

    def detect_liquidity_sweeps(self, tick_df: pd.DataFrame, bars: List[TickBar]) -> List[LiquiditySweep]:
        """
        Detect liquidity sweep events (stop hunts)
        """
        sweeps = []

        if len(bars) < 20:
            return sweeps

        # Convert bars to DataFrame for easier analysis
        bar_df = pd.DataFrame([bar.to_dict() for bar in bars])
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])

        # Calculate ATR for threshold
        bar_df['tr'] = bar_df[['high', 'low', 'close']].apply(
            lambda x: max(x['high'] - x['low'], 
                         abs(x['high'] - x['close']), 
                         abs(x['low'] - x['close'])), axis=1
        )
        bar_df['atr'] = bar_df['tr'].rolling(14).mean()

        # Find recent highs/lows (potential stop zones)
        window = 20
        bar_df['recent_high'] = bar_df['high'].rolling(window).max()
        bar_df['recent_low'] = bar_df['low'].rolling(window).min()

        # Detect sweeps
        for i in range(window, len(bar_df) - 5):
            current = bar_df.iloc[i]

            # Check for high sweep
            if current['high'] > bar_df.iloc[i-1]['recent_high']:
                # Check if price quickly reverses
                next_bars = bar_df.iloc[i+1:i+6]
                reversal = next_bars['close'].min()

                if reversal < current['close'] - (0.5 * current['atr']):
                    # Calculate sweep metrics
                    sweep_depth = current['high'] - bar_df.iloc[i-1]['recent_high']
                    volume_burst = current['volume'] / bar_df.iloc[i-20:i]['volume'].mean()

                    # Find recovery time
                    recovery_idx = next_bars[next_bars['close'] < current['close']].index
                    recovery_speed = (recovery_idx[0] - i) if len(recovery_idx) > 0 else 5

                    # Estimate trapped participants (based on volume profile)
                    trapped_volume = current['buy_volume']
                    avg_trade_size = bar_df.iloc[i-20:i]['volume'].sum() / bar_df.iloc[i-20:i]['tick_count'].sum()
                    participants_trapped = int(trapped_volume / avg_trade_size)

                    sweep = LiquiditySweep(
                        timestamp=current['timestamp'],
                        sweep_type='stop_hunt',
                        swept_level=bar_df.iloc[i-1]['recent_high'],
                        sweep_depth=sweep_depth,
                        volume_burst=volume_burst,
                        recovery_speed=recovery_speed,
                        participants_trapped=participants_trapped
                    )

                    sweeps.append(sweep)

            # Check for low sweep (similar logic)
            elif current['low'] < bar_df.iloc[i-1]['recent_low']:
                next_bars = bar_df.iloc[i+1:i+6]
                reversal = next_bars['close'].max()

                if reversal > current['close'] + (0.5 * current['atr']):
                    sweep_depth = bar_df.iloc[i-1]['recent_low'] - current['low']
                    volume_burst = current['volume'] / bar_df.iloc[i-20:i]['volume'].mean()

                    recovery_idx = next_bars[next_bars['close'] > current['close']].index
                    recovery_speed = (recovery_idx[0] - i) if len(recovery_idx) > 0 else 5

                    trapped_volume = current['sell_volume']
                    avg_trade_size = bar_df.iloc[i-20:i]['volume'].sum() / bar_df.iloc[i-20:i]['tick_count'].sum()
                    participants_trapped = int(trapped_volume / avg_trade_size)

                    sweep = LiquiditySweep(
                        timestamp=current['timestamp'],
                        sweep_type='stop_hunt',
                        swept_level=bar_df.iloc[i-1]['recent_low'],
                        sweep_depth=sweep_depth,
                        volume_burst=volume_burst,
                        recovery_speed=recovery_speed,
                        participants_trapped=participants_trapped
                    )

                    sweeps.append(sweep)

        return sweeps

    def calculate_volume_dynamics(self, bars: List[TickBar]) -> Dict[str, Any]:
        """
        Calculate advanced volume dynamics and imbalances
        """
        if not bars:
            return {}

        # Convert to arrays for faster computation
        volumes = np.array([bar.volume for bar in bars])
        buy_volumes = np.array([bar.buy_volume for bar in bars])
        sell_volumes = np.array([bar.sell_volume for bar in bars])
        deltas = np.array([bar.delta for bar in bars])

        # Calculate rolling metrics
        window = min(20, len(bars))

        # Volume momentum
        volume_ma = np.convolve(volumes, np.ones(window)/window, mode='valid')
        current_volume_ratio = volumes[-1] / volume_ma[-1] if len(volume_ma) > 0 else 1

        # Delta divergence (price up but delta down = bearish divergence)
        price_changes = np.array([bars[i].close - bars[i-1].close for i in range(1, len(bars))])
        if len(price_changes) > 0 and len(deltas) > 1:
            # Calculate correlation between price changes and delta
            if len(price_changes) >= 10:
                recent_correlation = np.corrcoef(price_changes[-10:], deltas[-10:])[0, 1]
            else:
                recent_correlation = 0
        else:
            recent_correlation = 0

        # Cumulative Delta divergence
        cum_delta = np.cumsum(deltas)

        # Volume imbalance
        buy_pressure = np.sum(buy_volumes[-window:])
        sell_pressure = np.sum(sell_volumes[-window:])
        total_pressure = buy_pressure + sell_pressure

        if total_pressure > 0:
            buy_ratio = buy_pressure / total_pressure
            imbalance = buy_ratio - 0.5  # -0.5 to 0.5 scale
        else:
            buy_ratio = 0.5
            imbalance = 0

        # Large player activity
        large_trades = sum(bar.large_trades for bar in bars[-window:])

        # Absorption detection
        absorption_scores = [bar.absorption_score for bar in bars[-window:]]
        avg_absorption = np.mean(absorption_scores) if absorption_scores else 0

        return {
            'current_volume_ratio': float(current_volume_ratio),
            'delta_price_correlation': float(recent_correlation),
            'cumulative_delta': float(cum_delta[-1]) if len(cum_delta) > 0 else 0,
            'buy_sell_ratio': float(buy_ratio),
            'volume_imbalance': float(imbalance),
            'large_trades_count': int(large_trades),
            'absorption_score': float(avg_absorption),
            'divergence_detected': recent_correlation < -0.3,
            'volume_surge': current_volume_ratio > self.config['min_volume_surge']
        }

    def detect_spread_anomalies(self, tick_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect abnormal spread behavior indicating institutional activity
        """
        anomalies = []

        if 'spread' not in tick_df.columns or len(tick_df) < 100:
            return anomalies

        df = tick_df.copy()

        # Calculate spread statistics
        df['spread_ma'] = df['spread'].rolling(50).mean()
        df['spread_std'] = df['spread'].rolling(50).std()
        df['spread_zscore'] = (df['spread'] - df['spread_ma']) / df['spread_std']

        # Detect anomalies
        threshold = self.config['spread_anomaly_zscore']
        anomaly_mask = df['spread_zscore'].abs() > threshold

        for idx in df[anomaly_mask].index:
            anomaly = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat(),
                'spread': float(df.loc[idx, 'spread']),
                'normal_spread': float(df.loc[idx, 'spread_ma']),
                'zscore': float(df.loc[idx, 'spread_zscore']),
                'anomaly_type': 'wide' if df.loc[idx, 'spread_zscore'] > 0 else 'tight',
                'volume': float(df.loc[idx, 'volume'])
            }

            anomalies.append(anomaly)

        return anomalies

    def build_tick_summary(self, 
                          tick_df: pd.DataFrame,
                          window_minutes: int = 30) -> Dict[str, Any]:
        """
        Build comprehensive tick analysis summary for LLM consumption
        """
        # Preprocess ticks
        processed_ticks = self.preprocess_ticks(tick_df)

        # Build bars
        bars_1s = self.build_tick_bars(processed_ticks, '1S')
        bars_1m = self.build_tick_bars(processed_ticks, '1T')

        # Detect patterns
        micro_traps = self.detect_micro_traps(processed_ticks)
        liquidity_sweeps = self.detect_liquidity_sweeps(processed_ticks, bars_1s)

        # Calculate dynamics
        volume_dynamics = self.calculate_volume_dynamics(bars_1s)
        spread_anomalies = self.detect_spread_anomalies(processed_ticks)

        # Get recent bars for context
        recent_bars = bars_1s[-60:] if len(bars_1s) > 60 else bars_1s

        # Build institutional footprint score
        footprint_score = self._calculate_institutional_footprint(
            micro_traps, liquidity_sweeps, volume_dynamics, spread_anomalies
        )

        # Create summary
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_range': {
                'start': processed_ticks['timestamp'].min().isoformat(),
                'end': processed_ticks['timestamp'].max().isoformat(),
                'tick_count': len(processed_ticks),
                'duration_seconds': (processed_ticks['timestamp'].max() - 
                                   processed_ticks['timestamp'].min()).total_seconds()
            },
            'institutional_footprint': {
                'score': footprint_score,
                'interpretation': self._interpret_footprint_score(footprint_score)
            },
            'micro_traps': [trap.to_dict() for trap in micro_traps[-5:]],  # Last 5
            'liquidity_sweeps': [sweep.to_dict() for sweep in liquidity_sweeps[-5:]],
            'volume_dynamics': volume_dynamics,
            'spread_anomalies': spread_anomalies[-10:],  # Last 10
            'recent_bars': {
                '1s': [bar.to_dict() for bar in recent_bars],
                '1m': [bar.to_dict() for bar in bars_1m[-30:]]  # Last 30 minutes
            },
            'key_levels': self._identify_key_levels(bars_1m),
            'market_regime': self._classify_market_regime(bars_1s, volume_dynamics)
        }

        return summary

    def _calculate_institutional_footprint(self,
                                         traps: List[MicroTrap],
                                         sweeps: List[LiquiditySweep], 
                                         dynamics: Dict,
                                         anomalies: List) -> float:
        """Calculate overall institutional activity score (0-100)"""
        score = 0

        # Weight different signals
        score += min(20, len(traps) * 5)      # Max 20 points for traps
        score += min(30, len(sweeps) * 10)    # Max 30 points for sweeps

        # Volume dynamics contribution
        if dynamics.get('volume_surge', False):
            score += 15
        if dynamics.get('divergence_detected', False):
            score += 10
        if dynamics.get('absorption_score', 0) > 0.7:
            score += 10

        # Spread anomalies
        score += min(15, len(anomalies) * 3)

        return min(100, score)

    def _interpret_footprint_score(self, score: float) -> str:
        """Interpret the institutional footprint score"""
        if score >= 70:
            return "Heavy institutional activity detected - potential major move incoming"
        elif score >= 50:
            return "Moderate institutional presence - accumulation/distribution likely"
        elif score >= 30:
            return "Some institutional activity - monitor for development"
        else:
            return "Low institutional footprint - retail-driven market"

    def _identify_key_levels(self, bars: List[TickBar]) -> Dict[str, List[float]]:
        """Identify key price levels from bars"""
        if not bars:
            return {'support': [], 'resistance': []}

        prices = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]

        # Volume-weighted levels
        vwap_levels = []
        for i in range(0, len(bars), 60):  # Every hour
            window = bars[i:i+60]
            if window:
                total_volume = sum(b.volume for b in window)
                if total_volume > 0:
                    vwap = sum(b.close * b.volume for b in window) / total_volume
                    vwap_levels.append(vwap)

        # High volume nodes (potential support/resistance)
        volume_threshold = np.percentile(volumes, 80)
        high_volume_prices = [bars[i].close for i, v in enumerate(volumes) if v > volume_threshold]

        # Cluster nearby levels
        support_levels = []
        resistance_levels = []

        current_price = bars[-1].close if bars else 0

        for level in set(vwap_levels + high_volume_prices):
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)

        return {
            'support': sorted(support_levels, reverse=True)[:3],
            'resistance': sorted(resistance_levels)[:3]
        }

    def _classify_market_regime(self, 
                               bars: List[TickBar],
                               dynamics: Dict) -> Dict[str, str]:
        """Classify current market regime"""
        if not bars or len(bars) < 10:
            return {'regime': 'unknown', 'confidence': 'low'}

        # Price action analysis
        recent_bars = bars[-20:]
        price_changes = [recent_bars[i].close - recent_bars[i-1].close 
                        for i in range(1, len(recent_bars))]

        # Trend detection
        trend = 'neutral'
        if len(price_changes) > 0:
            positive_changes = sum(1 for p in price_changes if p > 0)
            if positive_changes > len(price_changes) * 0.65:
                trend = 'bullish'
            elif positive_changes < len(price_changes) * 0.35:
                trend = 'bearish'

        # Volatility
        if len(price_changes) > 1:
            volatility = np.std(price_changes)
            avg_bar_range = np.mean([bar.high - bar.low for bar in recent_bars])
        else:
            volatility = 0
            avg_bar_range = 0

        # Classify regime
        if dynamics.get('absorption_score', 0) > 0.7:
            regime = 'accumulation' if trend == 'bearish' else 'distribution'
        elif dynamics.get('volume_surge', False) and volatility > avg_bar_range:
            regime = 'breakout'
        elif volatility < avg_bar_range * 0.5:
            regime = 'range'
        else:
            regime = trend

        return {
            'regime': regime,
            'trend': trend,
            'volatility': 'high' if volatility > avg_bar_range else 'normal',
            'confidence': 'high' if len(bars) > 50 else 'medium'
        }

    def full_tick_analysis(self, tick_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for complete tick analysis
        Returns LLM-ready summary
        """
        logger.info(f"Analyzing {len(tick_df)} ticks...")

        try:
            summary = self.build_tick_summary(tick_df)
            logger.info(f"Analysis complete. Institutional footprint score: {summary['institutional_footprint']['score']}")
            return summary
        except Exception as e:
            logger.error(f"Tick analysis failed: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
