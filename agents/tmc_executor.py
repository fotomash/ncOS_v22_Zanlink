"""
TMCExecutor Agent Implementation
Trend-Momentum Confluence execution strategy
"""

import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Tuple


@dataclass
class TMCSignal:
    """Represents a TMC trading signal"""
    timestamp: datetime
    signal_type: str  # 'entry_long', 'entry_short', 'exit_long', 'exit_short'
    strength: float  # 0.0 to 1.0
    trend_score: float
    momentum_score: float
    confluence_score: float
    metadata: Dict[str, Any]


class TMCExecutor:
    """
    Trend-Momentum Confluence executor.
    Combines trend following with momentum indicators for high-probability trades.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trend_periods = config.get('trend_periods', [10, 20, 50])
        self.momentum_period = config.get('momentum_period', 14)
        self.confluence_threshold = config.get('confluence_threshold', 0.7)
        self.position_scale_factor = config.get('position_scale_factor', 1.0)
        self.max_drawdown_percent = config.get('max_drawdown_percent', 0.05)

        # Execution state
        self.active_positions = {}
        self.signal_history: List[TMCSignal] = []
        self.execution_metrics = {
            'total_signals': 0,
            'executed_trades': 0,
            'trend_accuracy': 0.0,
            'momentum_accuracy': 0.0,
            'peak_drawdown': 0.0
        }

        # Technical indicators cache
        self.indicator_cache = {
            'ema': {},
            'momentum': {},
            'atr': {}
        }

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return statistics.mean(prices) if prices else 0

        # Check cache
        cache_key = f"{period}_{len(prices)}"
        if cache_key in self.indicator_cache['ema']:
            return self.indicator_cache['ema'][cache_key]

        # Calculate EMA
        multiplier = 2 / (period + 1)
        ema = statistics.mean(prices[:period])  # SMA for first period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        self.indicator_cache['ema'][cache_key] = ema
        return ema

    def calculate_momentum(self, prices: List[float], period: int) -> float:
        """Calculate price momentum (rate of change)"""
        if len(prices) < period + 1:
            return 0.0

        current_price = prices[-1]
        past_price = prices[-(period + 1)]

        if past_price == 0:
            return 0.0

        momentum = ((current_price - past_price) / past_price) * 100
        return momentum

    def calculate_atr(self, high_prices: List[float], low_prices: List[float],
                      close_prices: List[float], period: int = 14) -> float:
        """Calculate Average True Range for volatility"""
        if len(high_prices) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(high_prices)):
            high = high_prices[i]
            low = low_prices[i]
            prev_close = close_prices[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if len(true_ranges) >= period:
            return statistics.mean(true_ranges[-period:])
        return statistics.mean(true_ranges) if true_ranges else 0.0

    def calculate_trend_score(self, prices: List[float]) -> Tuple[float, str]:
        """Calculate multi-timeframe trend score"""
        if len(prices) < max(self.trend_periods):
            return 0.0, 'neutral'

        current_price = prices[-1]
        trend_scores = []

        for period in self.trend_periods:
            ema = self.calculate_ema(prices, period)
            if ema > 0:
                # Price position relative to EMA
                position_score = (current_price - ema) / ema
                trend_scores.append(position_score)

        if not trend_scores:
            return 0.0, 'neutral'

        # Aggregate trend score
        avg_score = statistics.mean(trend_scores)

        # Determine trend direction
        if avg_score > 0.01:
            direction = 'bullish'
        elif avg_score < -0.01:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Normalize score to 0-1 range
        normalized_score = min(abs(avg_score) * 10, 1.0)

        return normalized_score, direction

    def calculate_momentum_score(self, prices: List[float],
                                 volumes: List[float] = None) -> float:
        """Calculate momentum score with volume confirmation"""
        if len(prices) < self.momentum_period + 1:
            return 0.0

        # Price momentum
        price_momentum = self.calculate_momentum(prices, self.momentum_period)

        # Volume-weighted momentum if volume data available
        if volumes and len(volumes) == len(prices):
            recent_volume = statistics.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            avg_volume = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else statistics.mean(volumes)

            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_factor = min(volume_ratio, 2.0)  # Cap at 2x
        else:
            volume_factor = 1.0

        # Calculate momentum score
        raw_score = abs(price_momentum) / 10  # Normalize to roughly 0-1
        weighted_score = raw_score * (0.7 + 0.3 * volume_factor)

        return min(weighted_score, 1.0)

    def calculate_confluence(self, trend_score: float, momentum_score: float,
                             trend_direction: str, current_price: float,
                             prices: List[float]) -> float:
        """Calculate trend-momentum confluence score"""
        # Base confluence from trend and momentum alignment
        base_confluence = (trend_score + momentum_score) / 2

        # Additional confluence factors
        confluence_multiplier = 1.0

        # Check if momentum confirms trend
        momentum = self.calculate_momentum(prices, self.momentum_period)
        if (trend_direction == 'bullish' and momentum > 0) or (trend_direction == 'bearish' and momentum < 0):
            confluence_multiplier *= 1.2
        else:
            confluence_multiplier *= 0.8

        # Check for trend consistency across timeframes
        ema_short = self.calculate_ema(prices, self.trend_periods[0])
        ema_long = self.calculate_ema(prices, self.trend_periods[-1])

        if (trend_direction == 'bullish' and ema_short > ema_long) or (
                trend_direction == 'bearish' and ema_short < ema_long):
            confluence_multiplier *= 1.1

        final_confluence = base_confluence * confluence_multiplier
        return min(final_confluence, 1.0)

    def generate_signals(self, market_data: Dict[str, Any]) -> List[TMCSignal]:
        """Generate TMC trading signals"""
        signals = []
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        current_price = market_data.get('current_price', prices[-1] if prices else 0)

        if not prices or len(prices) < max(self.trend_periods):
            return signals

        # Calculate scores
        trend_score, trend_direction = self.calculate_trend_score(prices)
        momentum_score = self.calculate_momentum_score(prices, volumes)
        confluence_score = self.calculate_confluence(
            trend_score, momentum_score, trend_direction, current_price, prices
        )

        # Generate signals based on confluence
        if confluence_score >= self.confluence_threshold:
            signal_type = None

            if trend_direction == 'bullish':
                # Check if we have an open short position to close
                if self._has_position('short'):
                    signals.append(TMCSignal(
                        timestamp=datetime.now(),
                        signal_type='exit_short',
                        strength=confluence_score,
                        trend_score=trend_score,
                        momentum_score=momentum_score,
                        confluence_score=confluence_score,
                        metadata={'reason': 'Trend reversal to bullish'}
                    ))

                # Entry signal for long
                if not self._has_position('long'):
                    signals.append(TMCSignal(
                        timestamp=datetime.now(),
                        signal_type='entry_long',
                        strength=confluence_score,
                        trend_score=trend_score,
                        momentum_score=momentum_score,
                        confluence_score=confluence_score,
                        metadata={
                            'entry_price': current_price,
                            'trend_direction': trend_direction,
                            'stop_loss': self._calculate_stop_loss(prices, 'long')
                        }
                    ))

            elif trend_direction == 'bearish':
                # Check if we have an open long position to close
                if self._has_position('long'):
                    signals.append(TMCSignal(
                        timestamp=datetime.now(),
                        signal_type='exit_long',
                        strength=confluence_score,
                        trend_score=trend_score,
                        momentum_score=momentum_score,
                        confluence_score=confluence_score,
                        metadata={'reason': 'Trend reversal to bearish'}
                    ))

                # Entry signal for short
                if not self._has_position('short'):
                    signals.append(TMCSignal(
                        timestamp=datetime.now(),
                        signal_type='entry_short',
                        strength=confluence_score,
                        trend_score=trend_score,
                        momentum_score=momentum_score,
                        confluence_score=confluence_score,
                        metadata={
                            'entry_price': current_price,
                            'trend_direction': trend_direction,
                            'stop_loss': self._calculate_stop_loss(prices, 'short')
                        }
                    ))

        # Check for exit conditions on existing positions
        self._check_exit_conditions(prices, signals)

        # Update metrics
        self.execution_metrics['total_signals'] += len(signals)
        self.signal_history.extend(signals)

        return signals

    def _has_position(self, position_type: str) -> bool:
        """Check if we have an active position of given type"""
        return position_type in self.active_positions

    def _calculate_stop_loss(self, prices: List[float], position_type: str) -> float:
        """Calculate stop loss based on ATR"""
        # Simplified ATR calculation using price ranges
        if len(prices) < 20:
            atr = statistics.stdev(prices) * 2 if len(prices) > 1 else prices[-1] * 0.02
        else:
            ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            atr = statistics.mean(ranges[-14:]) * 2

        current_price = prices[-1]

        if position_type == 'long':
            return current_price - atr
        else:  # short
            return current_price + atr

    def _check_exit_conditions(self, prices: List[float], signals: List[TMCSignal]):
        """Check if any positions should be exited"""
        current_price = prices[-1]

        for position_type, position_data in self.active_positions.items():
            entry_price = position_data['entry_price']
            stop_loss = position_data.get('stop_loss', 0)

            # Check stop loss
            if position_type == 'long' and current_price <= stop_loss:
                signals.append(TMCSignal(
                    timestamp=datetime.now(),
                    signal_type='exit_long',
                    strength=1.0,
                    trend_score=0,
                    momentum_score=0,
                    confluence_score=0,
                    metadata={'reason': 'Stop loss triggered'}
                ))
            elif position_type == 'short' and current_price >= stop_loss:
                signals.append(TMCSignal(
                    timestamp=datetime.now(),
                    signal_type='exit_short',
                    strength=1.0,
                    trend_score=0,
                    momentum_score=0,
                    confluence_score=0,
                    metadata={'reason': 'Stop loss triggered'}
                ))

            # Check for momentum exhaustion
            momentum = self.calculate_momentum(prices, self.momentum_period)
            if (position_type == 'long' and momentum < -5) or (position_type == 'short' and momentum > 5):
                exit_type = f'exit_{position_type}'
                signals.append(TMCSignal(
                    timestamp=datetime.now(),
                    signal_type=exit_type,
                    strength=0.8,
                    trend_score=0,
                    momentum_score=abs(momentum) / 10,
                    confluence_score=0,
                    metadata={'reason': 'Momentum exhaustion'}
                ))

    def execute_signal(self, signal: TMCSignal, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading signal"""
        result = {
            'signal': asdict(signal),
            'execution_status': 'pending',
            'position_size': 0,
            'execution_price': 0
        }

        current_price = market_data.get('current_price', 0)
        account_balance = market_data.get('account_balance', 100000)

        # Calculate position size based on signal strength and risk
        base_position = account_balance * 0.02  # 2% risk per trade
        position_size = base_position * signal.strength * self.position_scale_factor

        if signal.signal_type.startswith('entry'):
            position_type = 'long' if 'long' in signal.signal_type else 'short'

            # Add position
            self.active_positions[position_type] = {
                'entry_price': current_price,
                'position_size': position_size,
                'entry_time': datetime.now(),
                'stop_loss': signal.metadata.get('stop_loss', 0)
            }

            result['execution_status'] = 'filled'
            result['position_size'] = position_size
            result['execution_price'] = current_price
            self.execution_metrics['executed_trades'] += 1

        elif signal.signal_type.startswith('exit'):
            position_type = 'long' if 'long' in signal.signal_type else 'short'

            if position_type in self.active_positions:
                position_data = self.active_positions[position_type]

                # Calculate P&L
                entry_price = position_data['entry_price']
                if position_type == 'long':
                    pnl = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) / entry_price

                result['execution_status'] = 'filled'
                result['position_size'] = -position_data['position_size']
                result['execution_price'] = current_price
                result['pnl'] = pnl

                # Remove position
                del self.active_positions[position_type]

        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        stats = self.execution_metrics.copy()
        stats['active_positions'] = len(self.active_positions)
        stats['position_details'] = self.active_positions.copy()

        # Calculate signal accuracy if we have history
        if len(self.signal_history) > 10:
            recent_signals = self.signal_history[-10:]
            avg_trend = statistics.mean([s.trend_score for s in recent_signals])
            avg_momentum = statistics.mean([s.momentum_score for s in recent_signals])
            stats['recent_trend_strength'] = avg_trend
            stats['recent_momentum_strength'] = avg_momentum

        return stats
