# ncos_v5_unified_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

class StrategyType(Enum):
    """Available strategy types from all sources"""
    WYCKOFF_ACCUMULATION = "wyckoff_accumulation"
    WYCKOFF_DISTRIBUTION = "wyckoff_distribution"
    SMC_CHOCH = "smc_choch"
    SMC_BOS = "smc_bos"
    FVG_FILL = "fvg_fill"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    TRAP_TRADE = "trap_trade"
    JUDAS_SWEEP = "judas_sweep"
    INDUCEMENT_SWEEP = "inducement_sweep"

class UnifiedStrategy:
    """Combines all trading strategies from XanFlow, ZAnalytics, and ncOS"""
    
    def __init__(self):
        self.strategies = self._load_all_strategies()
        self.session_memory = {}
        
    def _load_all_strategies(self) -> Dict:
        """Load all strategy configurations"""
        return {
            StrategyType.WYCKOFF_ACCUMULATION: {
                'required_conditions': ['phase_accumulation', 'spring_test', 'volume_increase'],
                'entry_triggers': ['test_of_spring', 'sos_breakout'],
                'risk_reward': 3.0,
                'confidence_threshold': 0.75
            },
            StrategyType.SMC_CHOCH: {
                'required_conditions': ['structure_break', 'momentum_shift', 'volume_confirmation'],
                'entry_triggers': ['retest_of_break', 'fvg_entry'],
                'risk_reward': 2.5,
                'confidence_threshold': 0.70
            },
            StrategyType.FVG_FILL: {
                'required_conditions': ['fvg_present', 'trend_alignment', 'no_major_resistance'],
                'entry_triggers': ['price_enters_fvg', 'momentum_confirmation'],
                'risk_reward': 2.0,
                'confidence_threshold': 0.65
            },
            StrategyType.LIQUIDITY_SWEEP: {
                'required_conditions': ['liquidity_pool_identified', 'sweep_pattern', 'reversal_structure'],
                'entry_triggers': ['sweep_completion', 'structure_confirmation'],
                'risk_reward': 3.5,
                'confidence_threshold': 0.80
            },
            StrategyType.JUDAS_SWEEP: {
                'required_conditions': ['session_opening', 'initial_push', 'liquidity_grab'],
                'entry_triggers': ['reversal_after_sweep', 'volume_divergence'],
                'risk_reward': 4.0,
                'confidence_threshold': 0.85
            }
        }
    
    def analyze_all_strategies(self, data: pd.DataFrame, context: Dict) -> List[Dict]:
        """Analyze all strategies and return viable setups"""
        viable_setups = []
        
        for strategy_type, config in self.strategies.items():
            setup = self._evaluate_strategy(strategy_type, data, context, config)
            if setup['confidence'] >= config['confidence_threshold']:
                viable_setups.append(setup)
                
        # Sort by confidence
        return sorted(viable_setups, key=lambda x: x['confidence'], reverse=True)
    
    def _evaluate_strategy(self, strategy_type: StrategyType, data: pd.DataFrame, 
                          context: Dict, config: Dict) -> Dict:
        """Evaluate a specific strategy"""
        conditions_met = 0
        total_conditions = len(config['required_conditions'])
        
        details = {
            'strategy': strategy_type.value,
            'timestamp': datetime.now().isoformat(),
            'conditions': {}
        }
        
        # Check each condition
        for condition in config['required_conditions']:
            met = self._check_condition(condition, data, context)
            details['conditions'][condition] = met
            if met:
                conditions_met += 1
                
        confidence = conditions_met / total_conditions
        
        # Check for entry triggers
        triggers_active = []
        for trigger in config['entry_triggers']:
            if self._check_trigger(trigger, data, context):
                triggers_active.append(trigger)
                
        return {
            'strategy_type': strategy_type.value,
            'confidence': confidence,
            'conditions_met': f"{conditions_met}/{total_conditions}",
            'active_triggers': triggers_active,
            'risk_reward': config['risk_reward'],
            'details': details,
            'tradeable': confidence >= config['confidence_threshold'] and len(triggers_active) > 0
        }
    
    def _check_condition(self, condition: str, data: pd.DataFrame, context: Dict) -> bool:
        """Check if a specific condition is met"""
        if len(data) < 50:
            return False
            
        # Implement condition checks
        if condition == 'phase_accumulation':
            volume_declining = data['volume'].rolling(20).mean().iloc[-1] < data['volume'].rolling(50).mean().iloc[-1]
            price_ranging = data['close'].rolling(20).std().iloc[-1] < data['close'].rolling(50).std().iloc[-1]
            return volume_declining and price_ranging
            
        elif condition == 'structure_break':
            recent_high = data['high'].rolling(20).max().iloc[-20]
            return data['close'].iloc[-1] > recent_high
            
        elif condition == 'fvg_present':
            # Check last 10 candles for FVG
            for i in range(-10, -2):
                if data['low'].iloc[i] > data['high'].iloc[i-2]:
                    return True
            return False
            
        elif condition == 'liquidity_pool_identified':
            # Check for equal highs/lows
            recent_highs = data['high'].tail(20)
            return len(recent_highs[recent_highs == recent_highs.max()]) >= 2
            
        elif condition == 'session_opening':
            current_hour = datetime.now().hour
            return 8 <= current_hour <= 9  # London open
            
        # Add more conditions as needed
        return False
    
    def _check_trigger(self, trigger: str, data: pd.DataFrame, context: Dict) -> bool:
        """Check if a trigger is active"""
        if trigger == 'test_of_spring':
            # Check if price is testing a previous low
            recent_low = data['low'].tail(20).min()
            return abs(data['close'].iloc[-1] - recent_low) / recent_low < 0.002
            
        elif trigger == 'retest_of_break':
            # Check if retesting broken level
            return context.get('recent_break_retest', False)
            
        elif trigger == 'price_enters_fvg':
            # Check if price entered an FVG zone
            return context.get('in_fvg_zone', False)
            
        # Add more triggers as needed
        return False
    
    def get_entry_exit_levels(self, strategy_type: str, data: pd.DataFrame) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = data['close'].iloc[-1]
        atr = self._calculate_atr(data)
        
        config = self.strategies.get(StrategyType(strategy_type))
        if not config:
            return {}
            
        # Base calculations
        if 'bullish' in strategy_type.lower() or 'accumulation' in strategy_type.lower():
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * config['risk_reward'] * 1.5)
        else:
            entry = current_price
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * config['risk_reward'] * 1.5)
            
        return {
            'entry': round(entry, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'risk_amount': abs(entry - stop_loss),
            'reward_amount': abs(take_profit - entry),
            'risk_reward_ratio': config['risk_reward']
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr