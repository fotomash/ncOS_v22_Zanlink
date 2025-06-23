"""Position Manager Agent"""
import logging
from typing import Dict, Any


class PositionManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions = {}
        self.max_positions = config.get('max_positions', 10)

    async def open_position(self, symbol: str, size: float, entry_price: float):
        """Open a new position"""
        if len(self.positions) >= self.max_positions:
            return {'error': 'Max positions reached'}

        position_id = f"{symbol}_{len(self.positions)}"
        self.positions[position_id] = {
            'symbol': symbol,
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'pnl': 0
        }

        return {'position_id': position_id, 'status': 'opened'}

    async def update_position(self, position_id: str, current_price: float):
        """Update position with current price"""
        if position_id in self.positions:
            position = self.positions[position_id]
            position['current_price'] = current_price
            position['pnl'] = (current_price - position['entry_price']) * position['size']
            return {'position_id': position_id, 'pnl': position['pnl']}

        return {'error': 'Position not found'}

    def get_status(self):
        total_pnl = sum(p['pnl'] for p in self.positions.values())
        return {
            'positions': len(self.positions),
            'max_positions': self.max_positions,
            'total_pnl': total_pnl
        }
