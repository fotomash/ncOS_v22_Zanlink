# xanflow_quarry_tools/swing_engine.py
"""
NCOS v11.6 Swing Engine
Critical utility for detecting swing points and labeling BOS/CHoCH
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum

class SwingType(Enum):
    HIGH = "swing_high"
    LOW = "swing_low"

class StructureType(Enum):
    BOS = "break_of_structure"
    CHOCH = "change_of_character"
    CONTINUATION = "continuation"
    NONE = "none"

class SwingEngine:
    """
    Detects swing highs/lows and labels market structure breaks
    """

    def __init__(self, lookback: int = 9, min_swing_strength: float = 0.0):
        self.lookback = lookback
        self.min_swing_strength = min_swing_strength
        self.swings_cache = {}

    def detect_swings(self, data: pd.DataFrame, lookback: Optional[int] = None) -> Dict:
        """
        Detect swing highs and lows using fractal/pivot logic

        Args:
            data: DataFrame with OHLC data
            lookback: Number of bars to look back (overrides instance default)

        Returns:
            Dict with 'highs' and 'lows' containing swing points
        """
        if lookback is None:
            lookback = self.lookback

        # Ensure we have required columns
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        swing_highs = []
        swing_lows = []

        # Need at least lookback*2 + 1 bars
        if len(data) < lookback * 2 + 1:
            return {'highs': swing_highs, 'lows': swing_lows}

        # Detect swing highs
        for i in range(lookback, len(data) - lookback):
            is_swing_high = True
            current_high = data.iloc[i]['high']

            # Check left side
            for j in range(i - lookback, i):
                if data.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break

            # Check right side
            if is_swing_high:
                for j in range(i + 1, i + lookback + 1):
                    if j < len(data) and data.iloc[j]['high'] > current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': current_high,
                    'timestamp': data.index[i] if hasattr(data.index, 'to_pydatetime') else i,
                    'type': SwingType.HIGH.value
                })

        # Detect swing lows
        for i in range(lookback, len(data) - lookback):
            is_swing_low = True
            current_low = data.iloc[i]['low']

            # Check left side
            for j in range(i - lookback, i):
                if data.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break

            # Check right side
            if is_swing_low:
                for j in range(i + 1, i + lookback + 1):
                    if j < len(data) and data.iloc[j]['low'] < current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': current_low,
                    'timestamp': data.index[i] if hasattr(data.index, 'to_pydatetime') else i,
                    'type': SwingType.LOW.value
                })

        return {
            'highs': swing_highs,
            'lows': swing_lows,
            'lookback_used': lookback,
            'total_bars': len(data)
        }

    def label_bos_choch(self, swings: Dict, price_data: pd.DataFrame, 
                       current_index: Optional[int] = None) -> Dict:
        """
        Label Break of Structure (BOS) and Change of Character (CHoCH)

        BOS: Price breaks previous swing in direction of trend
        CHoCH: Price breaks previous swing against trend direction

        Args:
            swings: Dict containing swing highs and lows
            price_data: DataFrame with price data
            current_index: Current bar index to check up to

        Returns:
            Dict with structure breaks labeled
        """
        if current_index is None:
            current_index = len(price_data) - 1

        highs = sorted(swings.get('highs', []), key=lambda x: x['index'])
        lows = sorted(swings.get('lows', []), key=lambda x: x['index'])

        structure_breaks = []

        # Track trend direction
        trend_direction = None  # 'bullish' or 'bearish'

        # Combine and sort all swings by index
        all_swings = []
        for high in highs:
            all_swings.append({**high, 'swing_type': 'high'})
        for low in lows:
            all_swings.append({**low, 'swing_type': 'low'})
        all_swings.sort(key=lambda x: x['index'])

        # Need at least 2 swings to determine structure
        if len(all_swings) < 2:
            return {'structure_breaks': structure_breaks, 'current_trend': trend_direction}

        # Analyze structure breaks
        for i in range(1, len(all_swings)):
            if all_swings[i]['index'] > current_index:
                break

            current_swing = all_swings[i]

            # Find the most recent opposite swing
            prev_opposite_swing = None
            for j in range(i - 1, -1, -1):
                if all_swings[j]['swing_type'] != current_swing['swing_type']:
                    prev_opposite_swing = all_swings[j]
                    break

            if not prev_opposite_swing:
                continue

            # Check for structure break
            structure_break = None

            if current_swing['swing_type'] == 'high':
                # Check if we broke above previous high
                prev_high = None
                for j in range(i - 1, -1, -1):
                    if all_swings[j]['swing_type'] == 'high':
                        prev_high = all_swings[j]
                        break

                if prev_high and current_swing['price'] > prev_high['price']:
                    # Broke above previous high
                    if trend_direction == 'bearish':
                        structure_break = StructureType.CHOCH
                        trend_direction = 'bullish'
                    else:
                        structure_break = StructureType.BOS
                        trend_direction = 'bullish'

            else:  # swing_type == 'low'
                # Check if we broke below previous low
                prev_low = None
                for j in range(i - 1, -1, -1):
                    if all_swings[j]['swing_type'] == 'low':
                        prev_low = all_swings[j]
                        break

                if prev_low and current_swing['price'] < prev_low['price']:
                    # Broke below previous low
                    if trend_direction == 'bullish':
                        structure_break = StructureType.CHOCH
                        trend_direction = 'bearish'
                    else:
                        structure_break = StructureType.BOS
                        trend_direction = 'bearish'

            if structure_break:
                # Find the exact candle that broke structure
                break_candle_index = self._find_break_candle(
                    price_data, 
                    prev_opposite_swing['index'] if prev_opposite_swing else current_swing['index'] - 1,
                    current_swing['index'],
                    current_swing['swing_type'],
                    prev_high['price'] if current_swing['swing_type'] == 'high' and prev_high else 
                    prev_low['price'] if current_swing['swing_type'] == 'low' and prev_low else None
                )

                structure_breaks.append({
                    'type': structure_break.value,
                    'swing_index': current_swing['index'],
                    'break_candle_index': break_candle_index,
                    'swing_price': current_swing['price'],
                    'swing_type': current_swing['swing_type'],
                    'timestamp': current_swing['timestamp'],
                    'trend_after': trend_direction
                })

        return {
            'structure_breaks': structure_breaks,
            'current_trend': trend_direction,
            'total_swings_analyzed': len(all_swings)
        }

    def _find_break_candle(self, price_data: pd.DataFrame, start_idx: int, 
                          end_idx: int, swing_type: str, break_level: float) -> int:
        """
        Find the exact candle that broke the structure level
        """
        if break_level is None:
            return end_idx

        for i in range(start_idx + 1, min(end_idx + 1, len(price_data))):
            if swing_type == 'high' and price_data.iloc[i]['close'] > break_level:
                return i
            elif swing_type == 'low' and price_data.iloc[i]['close'] < break_level:
                return i

        return end_idx

    def get_recent_structure(self, data: pd.DataFrame, lookback_bars: int = 100) -> Dict:
        """
        Get recent market structure for quick analysis

        Args:
            data: Price data
            lookback_bars: Number of bars to analyze

        Returns:
            Dict with recent swings and structure breaks
        """
        # Use only recent data for efficiency
        recent_data = data.tail(lookback_bars) if len(data) > lookback_bars else data

        # Detect swings
        swings = self.detect_swings(recent_data)

        # Label structure
        structure = self.label_bos_choch(swings, recent_data)

        # Get the most recent structure break
        recent_break = None
        if structure['structure_breaks']:
            recent_break = structure['structure_breaks'][-1]

        return {
            'swings': swings,
            'structure': structure,
            'recent_break': recent_break,
            'current_trend': structure.get('current_trend'),
            'analysis_bars': len(recent_data)
        }

    def validate_structure_break(self, break_info: Dict, price_data: pd.DataFrame,
                               min_impulse_bars: int = 3) -> bool:
        """
        Validate if a structure break is significant

        Args:
            break_info: Structure break information
            price_data: Price data
            min_impulse_bars: Minimum bars for impulse move

        Returns:
            bool: True if break is valid
        """
        break_idx = break_info.get('break_candle_index')
        if break_idx is None or break_idx >= len(price_data) - min_impulse_bars:
            return False

        # Check for impulse move after break
        if break_info['swing_type'] == 'high':
            # For bullish break, check upward impulse
            impulse_high = price_data.iloc[break_idx:break_idx + min_impulse_bars]['high'].max()
            break_close = price_data.iloc[break_idx]['close']
            impulse_strength = (impulse_high - break_close) / break_close

        else:
            # For bearish break, check downward impulse
            impulse_low = price_data.iloc[break_idx:break_idx + min_impulse_bars]['low'].min()
            break_close = price_data.iloc[break_idx]['close']
            impulse_strength = (break_close - impulse_low) / break_close

        # Require at least 0.1% move (adjustable based on instrument)
        return impulse_strength > 0.001

# Utility functions for external use

def quick_structure_check(data: pd.DataFrame, lookback: int = 9) -> Dict:
    """
    Quick structure check for integration with other modules
    """
    engine = SwingEngine(lookback=lookback)
    return engine.get_recent_structure(data, lookback_bars=100)

def find_nearest_swing(data: pd.DataFrame, index: int, swing_type: str = 'any',
                      lookback: int = 9, search_bars: int = 50) -> Optional[Dict]:
    """
    Find the nearest swing point to a given index

    Args:
        data: Price data
        index: Current index
        swing_type: 'high', 'low', or 'any'
        lookback: Swing detection lookback
        search_bars: How many bars to search

    Returns:
        Dict with swing information or None
    """
    engine = SwingEngine(lookback=lookback)

    # Get data window
    start_idx = max(0, index - search_bars)
    end_idx = min(len(data), index + 1)
    data_window = data.iloc[start_idx:end_idx]

    # Detect swings
    swings = engine.detect_swings(data_window)

    # Find nearest based on type
    candidates = []
    if swing_type in ['high', 'any']:
        candidates.extend(swings.get('highs', []))
    if swing_type in ['low', 'any']:
        candidates.extend(swings.get('lows', []))

    if not candidates:
        return None

    # Adjust indices to original data
    for swing in candidates:
        swing['original_index'] = swing['index'] + start_idx

    # Find nearest to target index
    nearest = min(candidates, key=lambda x: abs(x['original_index'] - index))

    return nearest
