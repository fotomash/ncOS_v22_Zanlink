#!/usr/bin/env python3
"""
ncOS v24 REAL DATA Engine for GPT
Standalone version that can be run in GPT code interpreter
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealTradingEngine:
    def __init__(self):
        self.tick_data = None
        self.ohlc_data = {}
        self.load_data()

    def load_data(self):
        """Load tick data from CSV"""
        try:
            # For GPT environment
            self.tick_data = pd.read_csv('/mnt/data/XAUUSD_TICKS_1days_20250623.csv')
            self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'], format='%Y.%m.%d %H:%M:%S')
            self.tick_data.set_index('timestamp', inplace=True)
            self.create_ohlc()
            print(f"✅ Loaded {len(self.tick_data)} ticks")
        except Exception as e:
            print(f"❌ Could not load data: {e}")

    def create_ohlc(self):
        """Convert to OHLC"""
        for tf, freq in {'M1': '1T', 'M5': '5T', 'M15': '15T', 'H1': '1H', 'H4': '4H'}.items():
            self.ohlc_data[tf] = self.tick_data['bid'].resample(freq).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()

    def scan(self):
        """Quick market scan"""
        if self.tick_data is None:
            return "No data loaded"

        current = float(self.tick_data['bid'].iloc[-1])
        h4 = self.ohlc_data['H4'].tail(20)

        # Simple trend check
        h4_trend = 'UP' if h4['close'].iloc[-1] > h4['close'].iloc[-10] else 'DOWN'

        return {
            'current_price': current,
            'h4_trend': h4_trend,
            'last_update': self.tick_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }

# Initialize
engine = RealTradingEngine()

# Helper functions
def scan():
    return engine.scan()
