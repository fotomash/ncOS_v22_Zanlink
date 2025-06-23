"""
MAZ2Executor Agent Implementation
Mean-reversion Adaptive Zone v2 execution strategy
"""

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any


@dataclass
class ExecutionOrder:
    """Represents an execution order"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit', 'stop'
    status: str  # 'pending', 'filled', 'cancelled'
    metadata: Dict[str, Any]


class MAZ2Executor:
    """
    Mean-reversion Adaptive Zone v2 executor.
    Implements advanced mean-reversion strategies with adaptive zones.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_period = config.get('lookback_period', 20)
        self.zone_multiplier = config.get('zone_multiplier', 2.0)
        self.reversion_threshold = config.get('reversion_threshold', 0.8)
        self.max_position_size = config.get('max_position_size', 10000)
        self.stop_loss_percentage = config.get('stop_loss_percentage', 0.02)

        # Execution state
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history: List[ExecutionOrder] = []
        self.position_tracker = {
            'current_position': 0,
            'average_price': 0,
            'unrealized_pnl': 0
        }

        # Performance metrics
        self.performance_metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'win_rate': 0.0,
            'average_return': 0.0
        }

    def calculate_adaptive_zones(self, prices: List[float]) -> Dict[str, float]:
        """Calculate adaptive mean-reversion zones"""
        if len(prices) < self.lookback_period:
            return {
                'mean': statistics.mean(prices) if prices else 0,
                'upper_zone': 0,
                'lower_zone': 0,
                'zone_width': 0
            }

        # Calculate rolling statistics
        recent_prices = prices[-self.lookback_period:]
        mean_price = statistics.mean(recent_prices)
        std_dev = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0

        # Calculate adaptive zones
        zone_width = std_dev * self.zone_multiplier

        # Adjust zones based on recent volatility
        volatility_factor = self._calculate_volatility_adjustment(prices)
        adjusted_width = zone_width * volatility_factor

        return {
            'mean': mean_price,
            'upper_zone': mean_price + adjusted_width,
            'lower_zone': mean_price - adjusted_width,
            'zone_width': adjusted_width,
            'volatility_factor': volatility_factor
        }

    def _calculate_volatility_adjustment(self, prices: List[float]) -> float:
        """Calculate volatility-based adjustment factor"""
        if len(prices) < 10:
            return 1.0

        # Compare recent vs historical volatility
        recent_vol = self._calculate_volatility(prices[-10:])
        historical_vol = self._calculate_volatility(prices[-50:]) if len(prices) >= 50 else recent_vol

        if historical_vol > 0:
            vol_ratio = recent_vol / historical_vol
            # Clamp between 0.5 and 2.0
            return max(0.5, min(2.0, vol_ratio))
        return 1.0

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0

        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
        return statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])

    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate execution signals based on mean-reversion strategy"""
        signals = []
        prices = market_data.get('prices', [])
        current_price = market_data.get('current_price', prices[-1] if prices else 0)

        if not prices or len(prices) < self.lookback_period:
            return signals

        # Calculate adaptive zones
        zones = self.calculate_adaptive_zones(prices)

        # Calculate reversion score
        distance_from_mean = current_price - zones['mean']
        reversion_score = abs(distance_from_mean) / zones['zone_width'] if zones['zone_width'] > 0 else 0

        # Generate signals based on zone position
        if current_price > zones['upper_zone'] and reversion_score > self.reversion_threshold:
            # Overbought - potential short/sell signal
            signals.append({
                'type': 'sell',
                'strength': min(reversion_score, 2.0),  # Cap signal strength
                'price_target': zones['mean'],
                'stop_loss': current_price * (1 + self.stop_loss_percentage),
                'zones': zones,
                'reasoning': f"Price {current_price:.2f} above upper zone {zones['upper_zone']:.2f}"
            })

        elif current_price < zones['lower_zone'] and reversion_score > self.reversion_threshold:
            # Oversold - potential long/buy signal
            signals.append({
                'type': 'buy',
                'strength': min(reversion_score, 2.0),
                'price_target': zones['mean'],
                'stop_loss': current_price * (1 - self.stop_loss_percentage),
                'zones': zones,
                'reasoning': f"Price {current_price:.2f} below lower zone {zones['lower_zone']:.2f}"
            })

        # Check for mean reversion completion
        if self.position_tracker['current_position'] != 0:
            position_side = 'long' if self.position_tracker['current_position'] > 0 else 'short'

            if position_side == 'long' and current_price >= zones['mean']:
                signals.append({
                    'type': 'close_long',
                    'strength': 1.0,
                    'reasoning': "Price reverted to mean - taking profit"
                })
            elif position_side == 'short' and current_price <= zones['mean']:
                signals.append({
                    'type': 'close_short',
                    'strength': 1.0,
                    'reasoning': "Price reverted to mean - taking profit"
                })

        return signals

    def calculate_position_size(self, signal: Dict[str, Any],
                                account_balance: float) -> float:
        """Calculate appropriate position size based on signal and risk"""
        base_size = min(self.max_position_size, account_balance * 0.1)  # Max 10% per trade

        # Adjust based on signal strength
        strength_multiplier = signal.get('strength', 1.0)
        adjusted_size = base_size * min(strength_multiplier, 1.5)

        # Apply volatility adjustment
        if 'zones' in signal:
            vol_factor = signal['zones'].get('volatility_factor', 1.0)
            # Reduce size in high volatility
            adjusted_size = adjusted_size / max(vol_factor, 1.0)

        return round(adjusted_size, 2)

    def execute_order(self, signal: Dict[str, Any],
                      market_data: Dict[str, Any]) -> ExecutionOrder:
        """Execute order based on signal"""
        current_price = market_data.get('current_price', 0)
        symbol = market_data.get('symbol', 'UNKNOWN')

        # Generate order ID
        order_id = f"MAZ2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.execution_history)}"

        # Determine order parameters
        if signal['type'] in ['buy', 'sell']:
            side = signal['type']
            quantity = self.calculate_position_size(signal, market_data.get('account_balance', 100000))
        else:  # Close positions
            side = 'sell' if signal['type'] == 'close_long' else 'buy'
            quantity = abs(self.position_tracker['current_position'])

        # Create order
        order = ExecutionOrder(
            order_id=order_id,
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
            order_type='limit',
            status='pending',
            metadata={
                'signal': signal,
                'zones': signal.get('zones', {}),
                'price_target': signal.get('price_target'),
                'stop_loss': signal.get('stop_loss')
            }
        )

        # Track order
        self.active_orders[order_id] = order
        self.performance_metrics['total_orders'] += 1

        # Simulate immediate fill for demo
        self._fill_order(order, current_price)

        return order

    def _fill_order(self, order: ExecutionOrder, fill_price: float):
        """Process order fill"""
        order.status = 'filled'
        order.price = fill_price

        # Update position
        if order.side == 'buy':
            new_position = self.position_tracker['current_position'] + order.quantity
            total_cost = (self.position_tracker['current_position'] * self.position_tracker['average_price'] +
                          order.quantity * fill_price)
            self.position_tracker['average_price'] = total_cost / new_position if new_position > 0 else 0
            self.position_tracker['current_position'] = new_position
        else:  # sell
            self.position_tracker['current_position'] -= order.quantity

        # Move to history
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        self.execution_history.append(order)
        self.performance_metrics['filled_orders'] += 1

    def update_positions(self, current_price: float):
        """Update position tracking with current market price"""
        if self.position_tracker['current_position'] != 0:
            self.position_tracker['unrealized_pnl'] = (
                    (current_price - self.position_tracker['average_price']) *
                    self.position_tracker['current_position']
            )

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.performance_metrics.copy()
        stats['active_orders'] = len(self.active_orders)
        stats['position'] = self.position_tracker.copy()

        # Calculate win rate
        if self.execution_history:
            profitable_trades = sum(1 for order in self.execution_history
                                    if order.metadata.get('pnl', 0) > 0)
            stats['win_rate'] = profitable_trades / len(self.execution_history)

        return stats
