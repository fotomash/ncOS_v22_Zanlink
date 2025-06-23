"""
ncOS Unified v5.0 - Market Maker Agent
NYSE/NASDAQ-style market making with vector-native analysis
"""

from datetime import datetime
from typing import Dict, List

import numpy as np


class MarketMaker:
    """Market maker following NYSE/NASDAQ terminology and practices"""

    def __init__(self, config: Dict):
        self.config = config
        self.position = 0
        self.inventory = {}
        self.spread_target = config.get("spread_target", 0.0002)  # 2 pips
        self.max_position = config.get("max_position", 1000000)  # $1M
        self.risk_limits = config.get("risk_limits", {})

    async def process(self, zbars: List, embeddings: np.ndarray) -> Dict:
        """Process market data and generate market making signals"""

        # Analyze current market microstructure
        microstructure = self._analyze_microstructure(zbars[-10:])  # Last 10 bars

        # Calculate optimal bid/ask spreads
        spreads = self._calculate_spreads(microstructure, embeddings)

        # Generate market making signals
        signals = self._generate_signals(spreads, microstructure)

        return {
            "agent": "market_maker",
            "timestamp": datetime.now().isoformat(),
            "microstructure": microstructure,
            "spreads": spreads,
            "signals": signals,
            "position": self.position,
            "inventory": self.inventory,
        }

    def _analyze_microstructure(self, recent_zbars: List) -> Dict:
        """Analyze market microstructure from recent bars"""
        if not recent_zbars:
            return {"order_flow": 0, "volatility": 0, "liquidity": "low"}

        # Extract order flow metrics
        order_flows = [getattr(zbar, "order_flow_imbalance", 0) for zbar in recent_zbars]
        avg_order_flow = np.mean(order_flows) if order_flows else 0

        # Calculate volatility
        prices = [getattr(zbar, "close", 0) for zbar in recent_zbars]
        volatility = np.std(prices) if len(prices) > 1 else 0

        # Assess liquidity
        volumes = [getattr(zbar, "volume", 0) for zbar in recent_zbars]
        avg_volume = np.mean(volumes) if volumes else 0

        liquidity = "high" if avg_volume > 1000 else "medium" if avg_volume > 500 else "low"

        return {
            "order_flow": avg_order_flow,
            "volatility": volatility,
            "liquidity": liquidity,
            "avg_volume": avg_volume,
        }

    def _calculate_spreads(self, microstructure: Dict, embeddings: np.ndarray) -> Dict:
        """Calculate optimal bid/ask spreads"""
        base_spread = self.spread_target

        # Adjust spread based on volatility
        volatility_multiplier = 1 + (microstructure["volatility"] * 0.1)

        # Adjust spread based on liquidity
        liquidity_multiplier = {
            "high": 0.8,
            "medium": 1.0,
            "low": 1.5,
        }.get(microstructure["liquidity"], 1.0)

        optimal_spread = base_spread * volatility_multiplier * liquidity_multiplier

        return {
            "base_spread": base_spread,
            "optimal_spread": optimal_spread,
            "volatility_adj": volatility_multiplier,
            "liquidity_adj": liquidity_multiplier,
        }

    def _generate_signals(self, spreads: Dict, microstructure: Dict) -> List[Dict]:
        """Generate market making signals"""
        signals = []

        if spreads["optimal_spread"] > 0.002:
            raise ValueError("ERROR: optimal spread exceeds 0.002")

        # Generate bid signal
        if microstructure["order_flow"] < -0.5:  # Selling pressure
            signals.append(
                {
                    "side": "bid",
                    "action": "increase_size",
                    "reason": "selling_pressure_detected",
                    "confidence": 0.8,
                }
            )

        # Generate ask signal
        if microstructure["order_flow"] > 0.5:  # Buying pressure
            signals.append(
                {
                    "side": "ask",
                    "action": "increase_size",
                    "reason": "buying_pressure_detected",
                    "confidence": 0.8,
                }
            )

        # Spread adjustment signal
        if spreads["optimal_spread"] > spreads["base_spread"] * 1.5:
            signals.append(
                {
                    "side": "both",
                    "action": "widen_spread",
                    "reason": "high_volatility",
                    "confidence": 0.9,
                }
            )

        return signals

    def get_health_status(self) -> str:
        """Return agent health status"""
        if abs(self.position) > self.max_position * 0.8:
            return "warning"
        return "healthy"
