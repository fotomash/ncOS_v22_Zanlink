"""Strategy Evaluator Agent"""
import logging
from typing import Dict, Any


class StrategyEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategies = config.get('strategies', {})

    async def evaluate(self, context: Dict[str, Any]):
        """Evaluate strategies against current context"""
        results = {}

        for strategy_name, strategy_config in self.strategies.items():
            score = 0
            conditions_met = 0

            for condition in strategy_config.get('conditions', []):
                if self._check_condition(condition, context):
                    conditions_met += 1

            if strategy_config.get('conditions'):
                score = conditions_met / len(strategy_config['conditions'])

            results[strategy_name] = {
                'score': score,
                'conditions_met': conditions_met,
                'action': strategy_config.get('action', 'hold')
            }

        return results

    def _check_condition(self, condition: Dict[str, Any], context: Dict[str, Any]):
        """Check if a condition is met"""
        field = condition.get('field')
        operator = condition.get('operator', '>')
        value = condition.get('value')

        if field in context:
            context_value = context[field]
            if operator == '>':
                return context_value > value
            elif operator == '<':
                return context_value < value
            elif operator == '==':
                return context_value == value

        return False

    def get_status(self):
        return {'strategies': list(self.strategies.keys()), 'active': True}
