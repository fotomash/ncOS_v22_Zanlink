"""Signal Processor Agent"""
import logging
from typing import Dict, Any


class SignalProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.signal_buffer = []
        self.max_buffer_size = config.get('max_buffer_size', 1000)

    async def process_signal(self, signal: Dict[str, Any]):
        """Process incoming signal"""
        self.signal_buffer.append(signal)
        if len(self.signal_buffer) > self.max_buffer_size:
            self.signal_buffer.pop(0)

        return {'processed': True, 'buffer_size': len(self.signal_buffer)}

    async def get_signals(self, count: int = 10):
        """Get recent signals"""
        return self.signal_buffer[-count:]

    def get_status(self):
        return {'buffer_size': len(self.signal_buffer), 'max_size': self.max_buffer_size}
