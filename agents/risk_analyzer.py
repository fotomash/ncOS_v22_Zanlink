"""Risk Analyzer Agent"""
import logging
from typing import Dict, Any


class RiskAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_limits = config.get('risk_limits', {})
        self.alerts = []

    async def analyze_risk(self, portfolio: Dict[str, Any]):
        """Analyze portfolio risk"""
        risk_metrics = {
            'total_exposure': 0,
            'concentration_risk': 0,
            'alerts': []
        }

        # Calculate total exposure
        positions = portfolio.get('positions', {})
        for position in positions.values():
            risk_metrics['total_exposure'] += abs(position.get('size', 0) * position.get('current_price', 0))

        # Check risk limits
        for metric, limit in self.risk_limits.items():
            if metric in risk_metrics and risk_metrics[metric] > limit:
                alert = {
                    'metric': metric,
                    'value': risk_metrics[metric],
                    'limit': limit,
                    'severity': 'high'
                }
                risk_metrics['alerts'].append(alert)
                self.alerts.append(alert)

        return risk_metrics

    def get_status(self):
        return {
            'risk_limits': self.risk_limits,
            'active_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:]
        }
