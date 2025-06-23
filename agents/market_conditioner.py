"""Market Conditioner Agent"""
import logging
from typing import Dict, Any


class MarketConditioner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conditions = config.get('conditions', {})

    async def evaluate_conditions(self, market_data: Dict[str, Any]):
        """Evaluate market conditions"""
        results = {'conditions': {}, 'signals': []}

        for condition_name, condition_config in self.conditions.items():
            threshold = condition_config.get('threshold', 0)
            field = condition_config.get('field', 'price')

            if field in market_data:
                value = market_data[field]
                triggered = value > threshold
                results['conditions'][condition_name] = triggered

                if triggered:
                    results['signals'].append({
                        'condition': condition_name,
                        'value': value,
                        'threshold': threshold
                    })

        return results

    def get_status(self):
        return {'conditions': list(self.conditions.keys()), 'active': True}
