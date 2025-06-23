# ncos_v5_chart_marker.py - FIXED VERSION
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class ChartMarker:
    def __init__(self):
        self.max_output_size = 50000  # 50KB limit
        self.max_candles = 500
        
    def generate_full_markup(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate optimized chart markings."""
        # Limit data size
        if len(data) > self.max_candles:
            # Use last N candles
            data = data.tail(self.max_candles)
            
        markings = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'candles_analyzed': len(data),
                'timeframe': self._detect_timeframe(data)
            },
            'summary': self._generate_summary(data),
            'key_levels': self._get_key_levels(data),
            'active_patterns': self._get_priority_patterns(data)
        }
        
        return markings
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detect timeframe from data."""
        if len(data) < 2:
            return "unknown"
        time_diff = (data.index[1] - data.index[0]).total_seconds()
        
        if time_diff < 60:
            return "tick"
        elif time_diff == 60:
            return "M1"
        elif time_diff == 300:
            return "M5"
        elif time_diff == 900:
            return "M15"
        elif time_diff == 1800:
            return "M30"
        elif time_diff == 3600:
            return "H1"
        elif time_diff == 14400:
            return "H4"
        else:
            return "other"
    
    def _generate_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate concise summary."""
        current = data.iloc[-1]
        
        return {
            'current_price': float(current['close']),
            'trend': self._calculate_trend(data),
            'volatility': float(data['close'].pct_change().std() * 100),
            'volume_trend': 'increasing' if current['volume'] > data['volume'].mean() else 'decreasing'
        }
    
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """Simple trend calculation."""
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        if len(data) < 50:
            return "insufficient_data"
            
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            return "bullish"
        else:
            return "bearish"
    
    def _get_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get only the most important levels."""
        return {
            'resistance_1': float(data['high'].rolling(20).max().iloc[-1]),
            'support_1': float(data['low'].rolling(20).min().iloc[-1]),
            'pivot': float((data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3),
            'vwap': float((data['close'] * data['volume']).sum() / data['volume'].sum()) if 'volume' in data.columns else float(data['close'].mean())
        }
    
    def _get_priority_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return only top 3 patterns."""
        patterns = []
        
        # Check for breakout
        resistance = data['high'].rolling(20).max()
        if data['close'].iloc[-1] > resistance.iloc[-2]:
            patterns.append({
                'type': 'breakout',
                'direction': 'bullish',
                'level': float(resistance.iloc[-2]),
                'strength': 'high'
            })
            
        # Check for reversal
        if len(data) >= 3:
            if data['low'].iloc[-2] < data['low'].iloc[-3] and data['low'].iloc[-2] < data['low'].iloc[-1]:
                patterns.append({
                    'type': 'reversal',
                    'direction': 'bullish',
                    'level': float(data['low'].iloc[-2]),
                    'strength': 'medium'
                })
                
        return patterns[:3]  # Max 3 patterns
    
    # Remove all the verbose marking methods
    def mark_wyckoff_phases(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simplified - return empty for now."""
        return []
        
    def mark_smc_zones(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simplified - return empty for now."""
        return []
        
    def mark_maz_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Simplified - return empty for now."""
        return []