"""Metrics Aggregator Agent"""
import logging
from datetime import datetime
from typing import Dict, Any


class MetricsAggregator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {}
        self.time_series = []

    async def aggregate(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Aggregate a metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf')
            }

        metric = self.metrics[metric_name]
        metric['count'] += 1
        metric['sum'] += value
        metric['min'] = min(metric['min'], value)
        metric['max'] = max(metric['max'], value)
        metric['avg'] = metric['sum'] / metric['count']

        # Store time series
        self.time_series.append({
            'timestamp': datetime.now().isoformat(),
            'metric': metric_name,
            'value': value,
            'tags': tags or {}
        })

        # Keep only recent data
        if len(self.time_series) > 10000:
            self.time_series = self.time_series[-10000:]

        return {'metric': metric_name, 'current': metric}

    def get_metrics(self):
        """Get all aggregated metrics"""
        return self.metrics

    def get_status(self):
        return {
            'metrics': list(self.metrics.keys()),
            'time_series_count': len(self.time_series)
        }
