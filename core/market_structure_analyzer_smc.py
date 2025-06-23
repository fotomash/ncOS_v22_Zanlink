
# Smart Money Concepts Market Structure Analyzer
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import deque
import asyncio

class MarketStructureAnalyzer:
    """
    Analyzes market structure using Smart Money Concepts (SMC).
    Identifies key levels, order blocks, and institutional movements.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.structure_points: List[Dict[str, Any]] = []
        self.order_blocks: List[Dict[str, Any]] = []
        self.liquidity_zones: List[Dict[str, Any]] = []
        self.current_bias: Optional[str] = None
        self.price_history: deque = deque(maxlen=config.get('lookback_period', 100))
        self.volume_history: deque = deque(maxlen=config.get('lookback_period', 100))

    async def initialize(self):
        """Initialize the analyzer"""
        self.lookback_period = self.config.get('lookback_period', 100)
        self.min_structure_distance = self.config.get('min_structure_distance', 10)

    async def analyze(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market structure from tick data"""
        price = tick_data['price']
        volume = tick_data['volume']
        timestamp = tick_data['timestamp']

        # Update structure points
        await self._update_structure_points(price, volume, timestamp)

        # Identify order blocks
        order_block = await self._identify_order_blocks()

        # Detect market bias
        bias = await self._detect_market_bias()

        # Check for structure breaks
        structure_break = await self._check_structure_break(price)

        if structure_break or order_block:
            return {
                'timestamp': timestamp,
                'structure_break': structure_break,
                'order_block': order_block,
                'market_bias': bias,
                'key_levels': self._get_key_levels(),
                'confidence': self._calculate_confidence()
            }

        return None

    # ------------------------------------------------------------------
    async def _update_structure_points(self, price: float, volume: float, timestamp: Any) -> None:
        """Maintain a rolling list of swing highs and lows."""
        self.price_history.append((timestamp, price))
        self.volume_history.append(volume)

        if len(self.price_history) < 3:
            return

        prev_time, prev_price = self.price_history[-2]
        before_price = self.price_history[-3][1]
        after_price = self.price_history[-1][1]

        if prev_price > before_price and prev_price > after_price:
            self.structure_points.append({'type': 'swing_high', 'price': prev_price, 'timestamp': prev_time})
        elif prev_price < before_price and prev_price < after_price:
            self.structure_points.append({'type': 'swing_low', 'price': prev_price, 'timestamp': prev_time})

    # ------------------------------------------------------------------
    async def _identify_order_blocks(self) -> Optional[Dict[str, Any]]:
        """Detect order blocks by comparing current volume to average volume."""
        if len(self.volume_history) < 5:
            return None

        avg_vol = float(np.mean(self.volume_history))
        last_vol = self.volume_history[-1]

        if last_vol >= avg_vol * 1.5 and self.structure_points:
            last_point = self.structure_points[-1]
            block_type = 'bullish' if self.current_bias == 'bullish' else 'bearish'
            block = {
                'type': block_type,
                'price': last_point['price'],
                'timestamp': last_point['timestamp'],
                'strength': min(1.0, last_vol / (avg_vol + 1e-9))
            }
            self.order_blocks.append(block)
            return block
        return None

    # ------------------------------------------------------------------
    async def _detect_market_bias(self) -> str:
        """Determine trend bias using a simple moving average crossover."""
        prices = [p[1] for p in self.price_history]
        if len(prices) < 5:
            self.current_bias = 'neutral'
            return self.current_bias

        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices)

        if ma_short > ma_long:
            self.current_bias = 'bullish'
        elif ma_short < ma_long:
            self.current_bias = 'bearish'
        else:
            self.current_bias = 'neutral'
        return self.current_bias

    # ------------------------------------------------------------------
    async def _check_structure_break(self, price: float) -> Optional[str]:
        """Check if price breaks the last swing level."""
        if not self.structure_points:
            return None

        last = self.structure_points[-1]
        if last['type'] == 'swing_high' and price > last['price']:
            return 'BOS'
        if last['type'] == 'swing_low' and price < last['price']:
            return 'BOS'
        return None

    # ------------------------------------------------------------------
    def _get_key_levels(self) -> Dict[str, List[float]]:
        """Return recent swing highs and lows as key levels."""
        highs = [p['price'] for p in self.structure_points if p['type'] == 'swing_high'][-3:]
        lows = [p['price'] for p in self.structure_points if p['type'] == 'swing_low'][-3:]
        return {'highs': highs, 'lows': lows}

    # ------------------------------------------------------------------
    def _calculate_confidence(self) -> float:
        """Calculate a simple confidence score."""
        factors = []
        if self.order_blocks:
            factors.append(min(1.0, self.order_blocks[-1]['strength']))
        if self.current_bias in ('bullish', 'bearish'):
            factors.append(0.6)
        return float(sum(factors) / len(factors)) if factors else 0.5
