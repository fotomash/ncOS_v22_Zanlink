
# Liquidity Engine for Smart Money Concepts
import numpy as np
from typing import Any, Dict, List, Optional
from collections import deque
import asyncio

class LiquidityEngine:
    """
    Identifies and tracks liquidity zones where smart money operates.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.liquidity_pools: List[Dict[str, Any]] = []
        self.swept_liquidity = deque(maxlen=50)
        self.price_history: deque = deque(maxlen=100)
        self.volume_history: deque = deque(maxlen=100)

    async def initialize(self):
        """Initialize the liquidity engine"""
        self.sensitivity = self.config.get('liquidity_sensitivity', 0.7)
        self.min_pool_size = self.config.get('min_pool_size', 100000)

    # ------------------------------------------------------------------
    async def analyze(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a market tick and detect liquidity events."""
        price = tick_data['price']
        volume = tick_data['volume']
        timestamp = tick_data['timestamp']

        self.price_history.append(price)
        self.volume_history.append(volume)

        new_pool = self._detect_liquidity_pool(price, volume, timestamp)
        sweep = self._detect_liquidity_sweep(price, timestamp)

        result = {
            'timestamp': timestamp,
            'current_price': price,
            'liquidity_pools': self.liquidity_pools[-5:],
            'sweep_detected': sweep is not None,
            'swept_liquidity': sweep
        }

        if new_pool:
            result['new_pool'] = new_pool

        return result

    # ------------------------------------------------------------------
    def _detect_liquidity_pool(self, price: float, volume: float, timestamp: Any) -> Optional[Dict[str, Any]]:
        """Identify potential liquidity pools based on spikes in volume."""
        if len(self.volume_history) < 5:
            return None

        avg_vol = float(np.mean(self.volume_history))
        if volume >= avg_vol * (1 + self.sensitivity):
            pool = {
                'level': price,
                'size': volume,
                'timestamp': timestamp,
            }
            self.liquidity_pools.append(pool)
            return pool
        return None

    # ------------------------------------------------------------------
    def _detect_liquidity_sweep(self, price: float, timestamp: Any) -> Optional[Dict[str, Any]]:
        """Detect sweeps of existing liquidity pools."""
        if not self.liquidity_pools:
            return None

        last_pool = self.liquidity_pools[-1]
        level = last_pool['level']
        threshold = self.config.get('sweep_threshold', 0.001)

        if abs(price - level) / level >= threshold:
            sweep = {'level': level, 'timestamp': timestamp}
            self.swept_liquidity.append(sweep)
            return sweep
        return None
