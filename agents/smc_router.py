"""
SMCRouter Agent Implementation
Smart Market Classifier routing for multi-strategy execution
"""

import statistics
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Tuple


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STABLE = "stable"
    UNKNOWN = "unknown"


class RouteType(Enum):
    """Routing decision types"""
    MAZ2 = "maz2"
    TMC = "tmc"
    HYBRID = "hybrid"
    HOLD = "hold"


class SMCRouter:
    """
    Smart Market Classifier for routing decisions.
    Analyzes market conditions and routes to appropriate execution strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.trend_threshold = config.get('trend_threshold', 0.01)
        self.volume_threshold = config.get('volume_threshold', 1000000)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

        # Routing statistics
        self.routing_stats = {
            'total_routes': 0,
            'maz2_routes': 0,
            'tmc_routes': 0,
            'hybrid_routes': 0,
            'hold_decisions': 0
        }

        # Market state cache
        self.market_state = {
            'regime': MarketRegime.UNKNOWN,
            'volatility': 0.0,
            'trend_strength': 0.0,
            'volume_profile': 'normal',
            'last_update': None
        }

    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

        if not returns:
            return 0.0

        return statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])

    def calculate_trend_strength(self, prices: List[float]) -> Tuple[float, str]:
        """Calculate trend strength and direction"""
        if len(prices) < 2:
            return 0.0, 'neutral'

        # Simple linear regression slope
        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n

        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0, 'neutral'

        slope = numerator / denominator

        # Normalize slope
        price_range = max(prices) - min(prices)
        if price_range > 0:
            normalized_slope = slope / price_range
        else:
            normalized_slope = 0

        direction = 'up' if normalized_slope > self.trend_threshold else 'down' if normalized_slope < -self.trend_threshold else 'neutral'

        return abs(normalized_slope), direction

    def classify_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Classify current market regime"""
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])

        if not prices:
            return MarketRegime.UNKNOWN

        volatility = self.calculate_volatility(prices)
        trend_strength, trend_direction = self.calculate_trend_strength(prices)

        # Update market state
        self.market_state['volatility'] = volatility
        self.market_state['trend_strength'] = trend_strength

        # Classify regime
        if volatility > self.volatility_threshold * 2:
            return MarketRegime.VOLATILE
        elif volatility < self.volatility_threshold * 0.5:
            return MarketRegime.STABLE
        elif trend_strength > self.trend_threshold and trend_direction == 'up':
            return MarketRegime.TRENDING_UP
        elif trend_strength > self.trend_threshold and trend_direction == 'down':
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def calculate_routing_confidence(self, market_data: Dict[str, Any],
                                     regime: MarketRegime) -> float:
        """Calculate confidence score for routing decision"""
        confidence_factors = []

        # Data quality factor
        prices = market_data.get('prices', [])
        if len(prices) >= 100:
            confidence_factors.append(1.0)
        elif len(prices) >= 50:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)

        # Regime clarity factor
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence_factors.append(0.9)
        elif regime == MarketRegime.VOLATILE:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)

        # Volume consistency factor
        volumes = market_data.get('volumes', [])
        if volumes and statistics.mean(volumes) > self.volume_threshold:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)

        return statistics.mean(confidence_factors) if confidence_factors else 0.5

    def determine_route(self, market_data: Dict[str, Any],
                        execution_params: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal routing based on market conditions"""
        # Classify market regime
        regime = self.classify_market_regime(market_data)
        self.market_state['regime'] = regime
        self.market_state['last_update'] = datetime.now()

        # Calculate routing confidence
        confidence = self.calculate_routing_confidence(market_data, regime)

        # Routing logic based on regime
        route_decision = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime.value,
            'confidence': confidence,
            'route_type': RouteType.HOLD,
            'reasoning': '',
            'parameters': {}
        }

        if confidence < self.confidence_threshold:
            route_decision['route_type'] = RouteType.HOLD
            route_decision['reasoning'] = f"Low confidence ({confidence:.2f})"
            self.routing_stats['hold_decisions'] += 1

        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Strong trends favor MAZ2
            route_decision['route_type'] = RouteType.MAZ2
            route_decision['reasoning'] = f"Strong {regime.value} detected"
            route_decision['parameters'] = {
                'trend_strength': self.market_state['trend_strength'],
                'execution_mode': 'aggressive' if confidence > 0.85 else 'moderate'
            }
            self.routing_stats['maz2_routes'] += 1

        elif regime == MarketRegime.RANGING:
            # Range-bound markets favor TMC
            route_decision['route_type'] = RouteType.TMC
            route_decision['reasoning'] = "Range-bound market conditions"
            route_decision['parameters'] = {
                'range_type': 'tight' if self.market_state['volatility'] < self.volatility_threshold else 'wide',
                'execution_mode': 'patient'
            }
            self.routing_stats['tmc_routes'] += 1

        elif regime == MarketRegime.VOLATILE:
            # High volatility may require hybrid approach
            route_decision['route_type'] = RouteType.HYBRID
            route_decision['reasoning'] = "High volatility requires adaptive strategy"
            route_decision['parameters'] = {
                'primary_strategy': 'tmc',
                'secondary_strategy': 'maz2',
                'split_ratio': 0.7  # 70% TMC, 30% MAZ2
            }
            self.routing_stats['hybrid_routes'] += 1

        else:
            # Default conservative approach
            route_decision['route_type'] = RouteType.TMC
            route_decision['reasoning'] = "Default conservative routing"
            route_decision['parameters'] = {'execution_mode': 'conservative'}
            self.routing_stats['tmc_routes'] += 1

        self.routing_stats['total_routes'] += 1
        return route_decision

    def batch_route(self, market_snapshots: List[Dict[str, Any]],
                    execution_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process multiple market snapshots and return routing decisions"""
        routing_decisions = []

        for snapshot in market_snapshots:
            decision = self.determine_route(snapshot, execution_params)
            routing_decisions.append(decision)

        return routing_decisions

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        stats = self.routing_stats.copy()
        stats['market_state'] = self.market_state.copy()

        if stats['total_routes'] > 0:
            stats['route_distribution'] = {
                'maz2_percentage': (stats['maz2_routes'] / stats['total_routes']) * 100,
                'tmc_percentage': (stats['tmc_routes'] / stats['total_routes']) * 100,
                'hybrid_percentage': (stats['hybrid_routes'] / stats['total_routes']) * 100,
                'hold_percentage': (stats['hold_decisions'] / stats['total_routes']) * 100
            }

        return stats

    def reset_stats(self):
        """Reset routing statistics"""
        self.routing_stats = {
            'total_routes': 0,
            'maz2_routes': 0,
            'tmc_routes': 0,
            'hybrid_routes': 0,
            'hold_decisions': 0
        }
