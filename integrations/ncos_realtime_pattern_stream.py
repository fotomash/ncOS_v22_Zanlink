
# ncos_realtime_pattern_stream.py
import asyncio
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Callable
import time
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PatternAlert:
    timestamp: datetime
    pattern_type: str
    confidence: float
    timeframe: str
    entry_levels: Dict[str, float]
    risk_levels: Dict[str, float]
    metadata: Dict

class RealtimePatternDetector:
    '''
    High-performance real-time pattern detection for ncOS
    Optimized for low-latency execution
    '''

    def __init__(self, config: Dict):
        self.config = config
        self.buffers = {}  # Multi-timeframe buffers
        self.pattern_cache = {}  # LRU cache for patterns
        self.alert_callbacks = []
        self.performance_stats = {
            'avg_latency_ms': 0,
            'patterns_detected': 0,
            'cache_hits': 0
        }

        # Initialize timeframe buffers
        for tf in ['1m', '5m', '15m', '1h', '4h']:
            self.buffers[tf] = {
                'prices': deque(maxlen=500),
                'volumes': deque(maxlen=500),
                'timestamps': deque(maxlen=500)
            }

    async def process_tick(self, tick_data: Dict) -> Optional[List[PatternAlert]]:
        '''Process incoming tick with minimal latency'''
        start_time = time.time()

        # Update buffers
        await self._update_buffers(tick_data)

        # Parallel pattern detection across timeframes
        detection_tasks = []
        for timeframe in self.buffers.keys():
            if self._has_enough_data(timeframe):
                detection_tasks.append(
                    self._detect_patterns_async(timeframe, tick_data['timestamp'])
                )

        # Wait for all detections to complete
        results = await asyncio.gather(*detection_tasks)

        # Filter and merge results
        alerts = self._merge_pattern_alerts(results)

        # Update performance stats
        latency = (time.time() - start_time) * 1000
        self._update_performance_stats(latency, len(alerts))

        # Trigger callbacks
        if alerts:
            await self._trigger_alert_callbacks(alerts)

        return alerts if alerts else None

    async def _detect_patterns_async(self, timeframe: str, timestamp: datetime) -> List[PatternAlert]:
        '''Asynchronous pattern detection for a specific timeframe'''
        buffer = self.buffers[timeframe]

        # Convert to numpy arrays for faster computation
        prices = np.array(buffer['prices'])
        volumes = np.array(buffer['volumes'])

        alerts = []

        # Run pattern detections in parallel
        pattern_tasks = [
            self._detect_wyckoff_spring_optimized(prices, volumes, timeframe, timestamp),
            self._detect_order_blocks_optimized(prices, volumes, timeframe, timestamp),
            self._detect_liquidity_sweeps_optimized(prices, volumes, timeframe, timestamp),
            self._detect_smc_patterns_optimized(prices, volumes, timeframe, timestamp)
        ]

        pattern_results = await asyncio.gather(*pattern_tasks)

        for result in pattern_results:
            if result:
                alerts.extend(result)

        return alerts

    async def _detect_wyckoff_spring_optimized(
        self, prices: np.ndarray, volumes: np.ndarray, 
        timeframe: str, timestamp: datetime
    ) -> List[PatternAlert]:
        '''Optimized Wyckoff spring detection'''

        # Quick rejection checks
        if len(prices) < 50:
            return []

        # Cache key for this pattern
        cache_key = f"wyckoff_spring_{timeframe}_{len(prices)}"

        # Check cache first
        if cache_key in self.pattern_cache:
            cached_result, cached_time = self.pattern_cache[cache_key]
            if time.time() - cached_time < 5:  # 5 second cache
                self.performance_stats['cache_hits'] += 1
                return cached_result

        alerts = []

        # Fast numpy operations
        recent_low = np.min(prices[-20:])
        range_low = np.percentile(prices[-50:], 10)

        # Spring conditions
        if (prices[-1] < range_low * 1.01 and  # Near range low
            prices[-1] > recent_low * 0.995 and  # But not new low
            volumes[-1] > np.mean(volumes[-20:]) * 1.5):  # High volume

            # Calculate entry and stop levels
            entry_level = prices[-1] * 1.005
            stop_level = recent_low * 0.995
            target_level = np.percentile(prices[-50:], 75)

            alert = PatternAlert(
                timestamp=timestamp,
                pattern_type='wyckoff_spring',
                confidence=self._calculate_spring_confidence(prices, volumes),
                timeframe=timeframe,
                entry_levels={'limit': entry_level, 'stop': entry_level * 1.01},
                risk_levels={'stop_loss': stop_level, 'target': target_level},
                metadata={
                    'range_duration': 50,
                    'volume_surge': volumes[-1] / np.mean(volumes[-20:]),
                    'risk_reward': (target_level - entry_level) / (entry_level - stop_level)
                }
            )
            alerts.append(alert)

        # Cache result
        self.pattern_cache[cache_key] = (alerts, time.time())

        return alerts

    async def _detect_order_blocks_optimized(
        self, prices: np.ndarray, volumes: np.ndarray,
        timeframe: str, timestamp: datetime
    ) -> List[PatternAlert]:
        '''Optimized order block detection'''

        if len(prices) < 20:
            return []

        alerts = []

        # Vectorized operations for speed
        price_changes = np.diff(prices)
        volume_ratio = volumes[1:] / np.mean(volumes[:-1])

        # Find potential order blocks
        ob_mask = (volume_ratio > 2.0) & (np.abs(price_changes) > np.std(price_changes) * 1.5)
        ob_indices = np.where(ob_mask)[0]

        for idx in ob_indices[-3:]:  # Check last 3 potential OBs
            if idx < len(prices) - 5:  # Need future data to validate
                # Validate order block
                if self._validate_order_block(prices, volumes, idx):
                    ob_type = 'bullish' if price_changes[idx] > 0 else 'bearish'

                    alert = PatternAlert(
                        timestamp=timestamp,
                        pattern_type=f'{ob_type}_order_block',
                        confidence=min(0.9, volume_ratio[idx] / 3.0),
                        timeframe=timeframe,
                        entry_levels={
                            'limit': prices[idx],
                            'aggressive': prices[idx] * (1.002 if ob_type == 'bullish' else 0.998)
                        },
                        risk_levels={
                            'stop_loss': prices[idx] * (0.995 if ob_type == 'bullish' else 1.005),
                            'target': prices[idx] * (1.02 if ob_type == 'bullish' else 0.98)
                        },
                        metadata={
                            'volume_ratio': float(volume_ratio[idx]),
                            'ob_strength': self._calculate_ob_strength(prices, volumes, idx),
                            'mitigation_status': 'pending'
                        }
                    )
                    alerts.append(alert)

        return alerts

    async def _detect_liquidity_sweeps_optimized(
        self, prices: np.ndarray, volumes: np.ndarray,
        timeframe: str, timestamp: datetime
    ) -> List[PatternAlert]:
        '''Detect liquidity sweep patterns'''

        if len(prices) < 30:
            return []

        alerts = []

        # Find recent highs/lows (liquidity levels)
        window = 20
        recent_high_idx = np.argmax(prices[-window:]) + len(prices) - window
        recent_low_idx = np.argmin(prices[-window:]) + len(prices) - window

        recent_high = prices[recent_high_idx]
        recent_low = prices[recent_low_idx]

        # Check for sweep and reversal
        if (prices[-2] > recent_high and prices[-1] < recent_high * 0.999):  # Bearish sweep
            alert = PatternAlert(
                timestamp=timestamp,
                pattern_type='liquidity_sweep_bearish',
                confidence=0.7,
                timeframe=timeframe,
                entry_levels={'limit': prices[-1], 'market': prices[-1] * 0.999},
                risk_levels={'stop_loss': recent_high * 1.002, 'target': recent_low},
                metadata={'sweep_level': float(recent_high), 'sweep_type': 'resistance'}
            )
            alerts.append(alert)

        elif (prices[-2] < recent_low and prices[-1] > recent_low * 1.001):  # Bullish sweep
            alert = PatternAlert(
                timestamp=timestamp,
                pattern_type='liquidity_sweep_bullish',
                confidence=0.7,
                timeframe=timeframe,
                entry_levels={'limit': prices[-1], 'market': prices[-1] * 1.001},
                risk_levels={'stop_loss': recent_low * 0.998, 'target': recent_high},
                metadata={'sweep_level': float(recent_low), 'sweep_type': 'support'}
            )
            alerts.append(alert)

        return alerts

    def register_alert_callback(self, callback: Callable):
        '''Register callback for pattern alerts'''
        self.alert_callbacks.append(callback)

    async def _trigger_alert_callbacks(self, alerts: List[PatternAlert]):
        '''Trigger all registered callbacks'''
        for callback in self.alert_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(alerts)
            else:
                callback(alerts)

    def _calculate_spring_confidence(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        '''Calculate confidence score for Wyckoff spring'''
        # Multiple factors contribute to confidence
        volume_surge = volumes[-1] / np.mean(volumes[-20:])
        price_rejection = (prices[-1] - np.min(prices[-5:])) / (np.max(prices[-5:]) - np.min(prices[-5:]))
        range_position = (prices[-1] - np.min(prices[-50:])) / (np.max(prices[-50:]) - np.min(prices[-50:]))

        confidence = (
            min(1.0, volume_surge / 3.0) * 0.4 +
            price_rejection * 0.3 +
            (1 - range_position) * 0.3
        )

        return min(0.95, confidence)

    def get_performance_stats(self) -> Dict:
        '''Get performance statistics'''
        return self.performance_stats.copy()

# WebSocket integration for real-time data
class NCOSRealtimeSystem:
    '''Complete real-time system integration'''

    def __init__(self, ncos_config: Dict):
        self.pattern_detector = RealtimePatternDetector(ncos_config)
        self.active_patterns = {}
        self.trade_manager = TradeManager()

    async def start_realtime_analysis(self, data_feed):
        '''Start real-time pattern detection'''

        # Register alert handler
        self.pattern_detector.register_alert_callback(self.handle_pattern_alert)

        # Process incoming data
        async for tick in data_feed:
            alerts = await self.pattern_detector.process_tick(tick)

            # Update active patterns
            if alerts:
                for alert in alerts:
                    self.active_patterns[alert.pattern_type] = alert

            # Check pattern invalidation
            await self.check_pattern_invalidation(tick)

    async def handle_pattern_alert(self, alerts: List[PatternAlert]):
        '''Handle new pattern alerts'''
        for alert in alerts:
            print(f"[{alert.timestamp}] Pattern Detected: {alert.pattern_type}")
            print(f"  Confidence: {alert.confidence:.2%}")
            print(f"  Entry: {alert.entry_levels}")
            print(f"  Risk: {alert.risk_levels}")

            # Auto-execute if confidence is high
            if alert.confidence > 0.8:
                await self.trade_manager.execute_pattern_trade(alert)

class TradeManager:
    '''Manage trades based on pattern alerts'''

    async def execute_pattern_trade(self, alert: PatternAlert):
        '''Execute trade based on pattern alert'''
        # Implementation depends on your broker integration
        pass

# Example usage
async def main():
    config = {
        'min_confidence': 0.7,
        'risk_per_trade': 0.02,
        'max_concurrent_patterns': 5
    }

    system = NCOSRealtimeSystem(config)
    # await system.start_realtime_analysis(your_data_feed)

if __name__ == '__main__':
    # asyncio.run(main())
    pass
