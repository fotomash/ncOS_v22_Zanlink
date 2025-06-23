# ncos_v5_tick_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from typing import Dict, Optional, List

class TickProcessor:
    def __init__(self):
        self.vector_dim = 1536
        self.session_cache = {}
        
    def find_best_data_source(self, pair: str, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load all available data sources for a pair."""
        sources = {}
        
        # Priority 1: Tick data
        tick_files = glob.glob(f"{data_path}/{pair}_TICKS*.csv")
        if tick_files:
            latest_tick = max(tick_files, key=os.path.getmtime)
            sources['ticks'] = self._load_tick_data(latest_tick)
            
        # Priority 2: M1 data
        m1_files = glob.glob(f"{data_path}/{pair}_M1*.csv")
        if m1_files:
            latest_m1 = max(m1_files, key=os.path.getmtime)
            sources['m1'] = pd.read_csv(latest_m1, parse_dates=['timestamp'])
            
        # Load all available timeframes for comprehensive analysis
        for tf in ['M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']:
            tf_files = glob.glob(f"{data_path}/{pair}_{tf}*.csv")
            if tf_files:
                latest = max(tf_files, key=os.path.getmtime)
                sources[tf.lower()] = pd.read_csv(latest, parse_dates=['timestamp'])
                
        return sources
    
    def _load_tick_data(self, filepath: str) -> pd.DataFrame:
        """Load and standardize tick data."""
        df = pd.read_csv(filepath)
        
        # Handle timestamp variations
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            
        df.set_index('timestamp', inplace=True)
        return df
    
    def resample_ticks_to_ohlc(self, ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Convert tick data to OHLC bars with proper aggregation."""
        # Define resampling rules
        rules = {
            'bid': 'ohlc',
            'ask': 'ohlc', 
            'volume': 'sum',
            'spread_points': 'mean'
        }
        
        # Resample
        ohlc = ticks.resample(timeframe).agg(rules)
        
        # Flatten multi-level columns
        ohlc.columns = ['_'.join(col).strip() for col in ohlc.columns.values]
        
        # Create standard OHLC from bid
        ohlc['open'] = ohlc['bid_open']
        ohlc['high'] = ohlc['bid_high']
        ohlc['low'] = ohlc['bid_low']
        ohlc['close'] = ohlc['bid_close']
        ohlc['volume'] = ohlc.get('volume_sum', 1)
        
        # Add microstructure features
        ohlc['spread_mean'] = ohlc.get('spread_points_mean', 0)
        ohlc['bid_ask_imbalance'] = (ohlc['ask_close'] - ohlc['bid_close']) / ohlc['bid_close']
        
        return ohlc[['open', 'high', 'low', 'close', 'volume', 'spread_mean', 'bid_ask_imbalance']]
    
    def enrich_with_microstructure(self, df: pd.DataFrame, tick_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add microstructure features from tick data."""
        if tick_data is not None:
            # Calculate tick-based features
            df['tick_count'] = tick_data.resample(df.index.freq).size()
            df['avg_spread'] = tick_data['spread_points'].resample(df.index.freq).mean()
            df['spread_volatility'] = tick_data['spread_points'].resample(df.index.freq).std()
            
            # Volume-weighted average price
            tick_data['mid_price'] = (tick_data['bid'] + tick_data['ask']) / 2
            df['vwap'] = (tick_data['mid_price'] * tick_data.get('volume', 1)).resample(df.index.freq).sum() / \
                         tick_data.get('volume', 1).resample(df.index.freq).sum()
        
        # Standard enrichments
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Generate vector embeddings for each row."""
        features = []
        
        # Price features
        features.extend([
            df['close'].pct_change(1),
            df['close'].pct_change(5),
            df['close'].pct_change(20),
            (df['high'] - df['low']) / df['close'],  # True range
            (df['close'] - df['open']) / df['open'],  # Body ratio
        ])
        
        # Volume features
        if 'volume' in df.columns:
            features.extend([
                df['volume'] / df['volume'].rolling(20).mean(),
                df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            ])
            
        # Microstructure features
        if 'spread_mean' in df.columns:
            features.append(df['spread_mean'] / df['close'])
            
        # Convert to embeddings (simplified - in production use proper embedding model)
        feature_matrix = pd.concat(features, axis=1).fillna(0)
        
        # Project to 1536 dimensions (placeholder for real embedding)
        embeddings = np.random.randn(len(df), self.vector_dim)
        embeddings[:, :feature_matrix.shape[1]] = feature_matrix.values
        
        return embeddings