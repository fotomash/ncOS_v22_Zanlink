
# advanced_pattern_recognition.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib
from scipy.signal import find_peaks
from scipy.stats import linregress
import joblib
from typing import Dict, List, Tuple, Optional

class AdvancedPatternRecognizer:
    '''
    Enhanced pattern recognition for ncOS using ML and advanced algorithms
    '''

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.models = {}
        self.pattern_memory = []
        self.initialize_models()

    def initialize_models(self):
        '''Initialize ML models for different pattern types'''
        self.models = {
            'wyckoff_spring': RandomForestClassifier(n_estimators=100, random_state=42),
            'order_block': RandomForestClassifier(n_estimators=100, random_state=42),
            'harmonic': RandomForestClassifier(n_estimators=50, random_state=42),
            'chart_patterns': RandomForestClassifier(n_estimators=150, random_state=42)
        }

    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Extract sophisticated features for pattern recognition'''
        features = pd.DataFrame(index=df.index)

        # Price Action Features
        features['price_momentum'] = talib.MOM(df['close'], timeperiod=10)
        features['price_acceleration'] = features['price_momentum'].diff()

        # Volume Profile Features
        features['volume_ratio'] = df['volume'] / talib.SMA(df['volume'], timeperiod=20)
        features['volume_momentum'] = talib.MOM(df['volume'], timeperiod=5)

        # Volatility Features
        features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['bb_width'] = self.calculate_bb_width(df)
        features['keltner_width'] = self.calculate_keltner_width(df)

        # Market Structure Features
        features['swing_high_distance'] = self.calculate_swing_distance(df, 'high')
        features['swing_low_distance'] = self.calculate_swing_distance(df, 'low')
        features['structure_score'] = self.calculate_structure_score(df)

        # Microstructure Features
        features['bid_ask_imbalance'] = self.estimate_bid_ask_imbalance(df)
        features['order_flow_toxicity'] = self.calculate_order_flow_toxicity(df)

        # Multi-timeframe Features
        for tf in [5, 15, 60]:  # 5min, 15min, 1hour
            features[f'mtf_trend_{tf}'] = self.calculate_mtf_trend(df, tf)

        return features.fillna(0)

    def detect_wyckoff_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        '''Advanced Wyckoff pattern detection using ML'''
        features = self.extract_wyckoff_features(df)

        patterns = {
            'accumulation': self.detect_accumulation_schematic(df, features),
            'distribution': self.detect_distribution_schematic(df, features),
            'spring': self.detect_spring_pattern(df, features),
            'upthrust': self.detect_upthrust_pattern(df, features)
        }

        # ML prediction if model is trained
        if hasattr(self.models['wyckoff_spring'], 'n_features_in_'):
            X = features.iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            patterns['ml_spring_probability'] = self.models['wyckoff_spring'].predict_proba(X_scaled)[0, 1]

        return patterns

    def extract_wyckoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Specific features for Wyckoff methodology'''
        features = pd.DataFrame(index=df.index)

        # Volume analysis
        features['volume_climax'] = self.detect_volume_climax(df)
        features['no_demand'] = self.detect_no_demand(df)
        features['stopping_volume'] = self.detect_stopping_volume(df)

        # Price spread analysis
        features['narrow_spread'] = (df['high'] - df['low']) < talib.ATR(df['high'], df['low'], df['close'], 14) * 0.5
        features['wide_spread'] = (df['high'] - df['low']) > talib.ATR(df['high'], df['low'], df['close'], 14) * 1.5

        # Cause and effect
        features['cause_built'] = self.calculate_cause_built(df)

        return features

    def detect_accumulation_schematic(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict:
        '''Detect Wyckoff accumulation phases'''
        result = {
            'phase': None,
            'confidence': 0,
            'key_levels': {}
        }

        # Find potential accumulation range
        range_high, range_low = self.find_trading_range(df)
        if not range_high or not range_low:
            return result

        # Phase A: Stopping action
        ps_level = self.find_preliminary_support(df, range_low)
        sc_level = self.find_selling_climax(df, features)

        # Phase B: Building cause
        if ps_level and sc_level:
            cause_index = self.calculate_cause_index(df, ps_level, sc_level)
            result['cause_built'] = cause_index

        # Phase C: Spring test
        spring = self.detect_spring_in_range(df, range_low, features)
        if spring:
            result['phase'] = 'C'
            result['key_levels']['spring'] = spring
            result['confidence'] = 0.8

        # Phase D: Markup
        if self.detect_sos(df, range_high):
            result['phase'] = 'D'
            result['confidence'] = 0.9

        return result

    def detect_harmonic_patterns(self, df: pd.DataFrame) -> List[Dict]:
        '''Detect harmonic patterns (Gartley, Butterfly, Bat, Crab)'''
        patterns = []

        # Find swing points
        highs, high_indices = self.find_swing_highs(df)
        lows, low_indices = self.find_swing_lows(df)

        # Check for 5-point patterns (XABCD)
        if len(highs) >= 3 and len(lows) >= 2:
            for i in range(len(highs) - 2):
                pattern = self.check_harmonic_ratios(
                    df, highs, lows, high_indices, low_indices, i
                )
                if pattern:
                    patterns.append(pattern)

        return patterns

    def check_harmonic_ratios(self, df, highs, lows, high_idx, low_idx, start_idx):
        '''Check Fibonacci ratios for harmonic patterns'''
        # Simplified example - implement full harmonic logic
        ratios = {
            'gartley': {'XA_BC': 0.618, 'AB_CD': 1.27, 'XA_AD': 0.786},
            'butterfly': {'XA_BC': 0.786, 'AB_CD': 1.618, 'XA_AD': 1.27},
            'bat': {'XA_BC': 0.50, 'AB_CD': 1.618, 'XA_AD': 0.886},
            'crab': {'XA_BC': 0.618, 'AB_CD': 2.618, 'XA_AD': 1.618}
        }

        # Calculate actual ratios from price swings
        # ... (implement ratio calculations)

        return None  # Return pattern if found

    def detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        '''Detect classical chart patterns'''
        patterns = []

        # Head and Shoulders
        h_and_s = self.detect_head_and_shoulders(df)
        if h_and_s:
            patterns.append(h_and_s)

        # Double Top/Bottom
        double_patterns = self.detect_double_patterns(df)
        patterns.extend(double_patterns)

        # Triangle patterns
        triangles = self.detect_triangle_patterns(df)
        patterns.extend(triangles)

        # Flag and Pennant
        continuation = self.detect_continuation_patterns(df)
        patterns.extend(continuation)

        return patterns

    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        '''Enhanced order block detection with ML'''
        order_blocks = []

        # Traditional detection
        for i in range(20, len(df)):
            if self.is_order_block_candidate(df, i):
                ob = self.analyze_order_block(df, i)
                order_blocks.append(ob)

        # ML enhancement
        if order_blocks and hasattr(self.models['order_block'], 'n_features_in_'):
            features = self.extract_order_block_features(df, order_blocks)
            X_scaled = self.scaler.transform(features)
            probabilities = self.models['order_block'].predict_proba(X_scaled)

            for i, ob in enumerate(order_blocks):
                ob['ml_validity_score'] = probabilities[i, 1]

        return order_blocks

    def calculate_structure_score(self, df: pd.DataFrame) -> pd.Series:
        '''Calculate market structure strength score'''
        scores = pd.Series(index=df.index, dtype=float)

        # Use rolling window to calculate structure
        for i in range(50, len(df)):
            window = df.iloc[i-50:i]

            # Count higher highs/lows for uptrend
            hh_count = sum(1 for j in range(1, len(window)) 
                          if window['high'].iloc[j] > window['high'].iloc[j-1])
            hl_count = sum(1 for j in range(1, len(window)) 
                          if window['low'].iloc[j] > window['low'].iloc[j-1])

            # Normalize to -1 to 1 scale
            uptrend_score = (hh_count + hl_count) / (2 * len(window))
            downtrend_score = 1 - uptrend_score

            scores.iloc[i] = uptrend_score - downtrend_score

        return scores

    def train_models(self, training_data: pd.DataFrame, labels: Dict[str, pd.Series]):
        '''Train ML models on historical patterns'''
        features = self.extract_advanced_features(training_data)
        X = features.fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        for model_name, model in self.models.items():
            if model_name in labels:
                y = labels[model_name]
                model.fit(X_scaled, y)

        # Save models
        joblib.dump(self.scaler, 'ncos_pattern_scaler.pkl')
        for name, model in self.models.items():
            joblib.dump(model, f'ncos_pattern_model_{name}.pkl')

    # Helper methods
    def calculate_bb_width(self, df: pd.DataFrame) -> pd.Series:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        return upper - lower

    def calculate_keltner_width(self, df: pd.DataFrame) -> pd.Series:
        ema = talib.EMA(df['close'], timeperiod=20)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
        return 2 * atr

    def find_swing_highs(self, df: pd.DataFrame, prominence=0.01) -> Tuple[List, List]:
        prices = df['high'].values
        peaks, properties = find_peaks(prices, prominence=prominence*prices.mean())
        return prices[peaks].tolist(), peaks.tolist()

    def find_swing_lows(self, df: pd.DataFrame, prominence=0.01) -> Tuple[List, List]:
        prices = -df['low'].values
        peaks, properties = find_peaks(prices, prominence=prominence*abs(prices.mean()))
        return df['low'].values[peaks].tolist(), peaks.tolist()

    def detect_volume_climax(self, df: pd.DataFrame) -> pd.Series:
        volume_sma = talib.SMA(df['volume'], timeperiod=20)
        return df['volume'] > volume_sma * 2

    def estimate_bid_ask_imbalance(self, df: pd.DataFrame) -> pd.Series:
        '''Estimate order flow imbalance from price and volume'''
        # Simplified implementation
        price_change = df['close'].pct_change()
        volume_weighted = price_change * df['volume']
        return talib.SMA(volume_weighted, timeperiod=10)

# Integration with ncOS
class NCOSPatternIntegration:
    '''Integrate advanced pattern recognition with existing ncOS system'''

    def __init__(self, ncos_config: Dict):
        self.pattern_recognizer = AdvancedPatternRecognizer(ncos_config)
        self.pattern_cache = {}

    def analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        '''Comprehensive pattern analysis'''

        # Extract all pattern types
        wyckoff = self.pattern_recognizer.detect_wyckoff_patterns(df)
        harmonics = self.pattern_recognizer.detect_harmonic_patterns(df)
        chart_patterns = self.pattern_recognizer.detect_chart_patterns(df)
        order_blocks = self.pattern_recognizer.detect_order_blocks(df)

        # Advanced features
        features = self.pattern_recognizer.extract_advanced_features(df)

        # Combine results
        analysis = {
            'timestamp': df.index[-1],
            'wyckoff_analysis': wyckoff,
            'harmonic_patterns': harmonics,
            'chart_patterns': chart_patterns,
            'order_blocks': order_blocks,
            'market_structure': {
                'score': features['structure_score'].iloc[-1],
                'atr': features['atr'].iloc[-1],
                'volatility_regime': self.classify_volatility(features)
            },
            'pattern_confluence': self.calculate_pattern_confluence(
                wyckoff, harmonics, chart_patterns, order_blocks
            )
        }

        return analysis

    def classify_volatility(self, features: pd.DataFrame) -> str:
        '''Classify current volatility regime'''
        current_vol = features['atr'].iloc[-1]
        vol_percentile = (features['atr'] < current_vol).sum() / len(features)

        if vol_percentile < 0.2:
            return 'low'
        elif vol_percentile < 0.8:
            return 'normal'
        else:
            return 'high'

    def calculate_pattern_confluence(self, *patterns) -> float:
        '''Calculate confluence score from multiple pattern types'''
        score = 0
        weights = [0.4, 0.2, 0.2, 0.2]  # Wyckoff, Harmonic, Chart, Order Blocks

        for pattern_set, weight in zip(patterns, weights):
            if pattern_set and any(pattern_set.values() if isinstance(pattern_set, dict) else pattern_set):
                score += weight

        return score

# Example usage
if __name__ == '__main__':
    # Initialize with your ncOS config
    config = {
        'lookback_period': 100,
        'pattern_sensitivity': 0.8
    }

    # Create pattern integration
    pattern_system = NCOSPatternIntegration(config)

    # Analyze patterns
    # df = pd.read_csv('your_data.csv')
    # analysis = pattern_system.analyze_patterns(df)
    # print(analysis)
