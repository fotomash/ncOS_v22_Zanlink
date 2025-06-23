# ncos_v5_complete_tick_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import glob
from dataclasses import dataclass
from scipy.stats import zscore
import json

@dataclass
class TickData:
    timestamp: pd.DatetimeIndex
    bid: np.ndarray
    ask: np.ndarray
    volume: np.ndarray
    spread: np.ndarray
    embeddings: Optional[np.ndarray] = None

class VectorNativeTickProcessor:
    def __init__(self, config_path: str = 'ncos_v5_tick_config.yaml'):
        self.vector_dim = 1536
        self.session_state = {}
        self.embeddings_cache = {}
        
        # Load config
        with open(config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)['tick_processing']
    
    def process_pair_complete(self, pair: str, data_path: str) -> Dict[str, pd.DataFrame]:
        """Complete tick-to-signal processing pipeline."""
        # 1. Discovery
        available_data = self._discover_data_sources(pair, data_path)
        
        # 2. Load with priority
        primary_data = self._load_primary_source(available_data)
        
        # 3. Resample to all timeframes
        resampled_data = self._resample_all_timeframes(primary_data)
        
        # 4. Enrich each timeframe
        enriched_data = {}
        for tf, df in resampled_data.items():
            enriched_data[tf] = self._enrich_timeframe(df, primary_data)
            
        # 5. Generate embeddings
        for tf in enriched_data:
            enriched_data[tf] = self._add_embeddings(enriched_data[tf], tf)
            
        return enriched_data
    
    def _discover_data_sources(self, pair: str, data_path: str) -> Dict[str, List[str]]:
        """Find all available data files for a pair."""
        sources = {}
        
        # Check each timeframe
        timeframes = ['TICKS', 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        
        for tf in timeframes:
            pattern = os.path.join(data_path, f"{pair}_{tf}*.csv")
            files = glob.glob(pattern)
            if files:
                sources[tf] = sorted(files, key=os.path.getmtime, reverse=True)
                
        return sources
    
    def _load_primary_source(self, sources: Dict[str, List[str]]) -> pd.DataFrame:
        """Load data according to priority hierarchy."""
        priority = self.config['data_hierarchy']
        
        # Try each priority level
        if 'TICKS' in sources and sources['TICKS']:
            return self._load_tick_file(sources['TICKS'][0])
        elif 'M1' in sources and sources['M1']:
            return self._load_ohlc_file(sources['M1'][0])
        elif 'M5' in sources and sources['M5']:
            return self._load_ohlc_file(sources['M5'][0])
        else:
            # Load any available
            for tf, files in sources.items():
                if files:
                    return self._load_ohlc_file(files[0])
                    
        raise ValueError("No data sources found")
    
    def _load_tick_file(self, filepath: str) -> pd.DataFrame:
        """Load and standardize tick data."""
        df = pd.read_csv(filepath)
        
        # Standardize timestamp
        if 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df.set_index('timestamp', inplace=True)
        
        # Ensure required columns
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        
        return df
    
    def _load_ohlc_file(self, filepath: str) -> pd.DataFrame:
        """Load OHLC data file."""
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    
    def _resample_all_timeframes(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample to all target timeframes."""
        resampled = {}
        
        # Check if this is tick data
        is_tick_data = 'bid' in data.columns and 'ask' in data.columns
        
        for tf in self.config['resampling_targets']:
            if is_tick_data:
                resampled[tf] = self._resample_ticks(data, tf)
            else:
                # If already OHLC, only resample to larger timeframes
                resampled[tf] = self._resample_ohlc(data, tf)
                
        return resampled
    
    def _resample_ticks(self, ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample tick data to OHLC."""
        ohlc = pd.DataFrame()
        
        # Price from bid
        ohlc['open'] = ticks['bid'].resample(timeframe).first()
        ohlc['high'] = ticks['bid'].resample(timeframe).max()
        ohlc['low'] = ticks['bid'].resample(timeframe).min()
        ohlc['close'] = ticks['bid'].resample(timeframe).last()
        
        # Volume
        if 'volume' in ticks.columns:
            ohlc['volume'] = ticks['volume'].resample(timeframe).sum()
        else:
            ohlc['volume'] = ticks['bid'].resample(timeframe).count()
            
        # Microstructure
        ohlc['tick_count'] = ticks['bid'].resample(timeframe).count()
        ohlc['avg_spread'] = ticks['spread'].resample(timeframe).mean()
        ohlc['spread_std'] = ticks['spread'].resample(timeframe).std()
        ohlc['vwap'] = (ticks['mid'] * ticks.get('volume', 1)).resample(timeframe).sum() / \
                       ticks.get('volume', 1).resample(timeframe).sum()
        
        # Order flow
        ohlc['buy_volume'] = ticks[ticks['mid'] > ticks['mid'].shift(1)].get('volume', 1).resample(timeframe).sum()
        ohlc['sell_volume'] = ticks[ticks['mid'] < ticks['mid'].shift(1)].get('volume', 1).resample(timeframe).sum()
        ohlc['delta'] = ohlc['buy_volume'] - ohlc['sell_volume']
        ohlc['cumulative_delta'] = ohlc['delta'].cumsum()
        
        return ohlc.dropna()
    
    def _resample_ohlc(self, ohlc: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLC to larger timeframe."""
        resampled = ohlc.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return resampled.dropna()
    
    def _enrich_timeframe(self, df: pd.DataFrame, source_data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and microstructure features."""
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Volume analysis
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        
        # Market profile
        df['poc'] = df['close'].rolling(20).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean())
        df['value_area_high'] = df['high'].rolling(20).quantile(0.7)
        df['value_area_low'] = df['low'].rolling(20).quantile(0.3)
        
        # Structure
        df['swing_high'] = df['high'] == df['high'].rolling(5, center=True).max()
        df['swing_low'] = df['low'] == df['low'].rolling(5, center=True).min()
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _add_embeddings(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Generate vector embeddings for each row."""
        features = []
        
        # Select features for embedding
        feature_cols = [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio',
            'volatility_20', 'volume_ratio', 'obv'
        ]
        
        # Add SMA ratios
        for period in [5, 10, 20, 50]:
            if f'sma_{period}' in df.columns:
                df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                feature_cols.append(f'sma_{period}_ratio')
                
        # Normalize features
        feature_matrix = df[feature_cols].fillna(0)
        normalized = pd.DataFrame(
            zscore(feature_matrix, axis=0, nan_policy='omit'),
            index=feature_matrix.index,
            columns=feature_matrix.columns
        ).fillna(0)
        
        # Create embedding matrix
        n_features = len(feature_cols)
        n_rows = len(df)
        
        # Initialize with random projection
        embeddings = np.random.randn(n_rows, self.vector_dim) * 0.01
        
        # Inject features
        embeddings[:, :n_features] = normalized.values
        
        # Add temporal encoding
        time_encoding = np.sin(np.arange(n_rows)[:, None] * np.pi / n_rows)
        embeddings[:, n_features:n_features+1] = time_encoding
        
        # Store as list of arrays
        df['embeddings'] = [embeddings[i] for i in range(n_rows)]
        
        # Cache for session
        cache_key = f"{df.index[0]}_{df.index[-1]}_{timeframe}"
        self.embeddings_cache[cache_key] = embeddings
        
        return df
    
    def get_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix."""
        norm = np.linalg.norm(embeddings, axis=1)[:, None]
        normalized = embeddings / norm
        similarity = np.dot(normalized, normalized.T)
        return similarity
    
    def find_similar_patterns(self, df: pd.DataFrame, current_idx: int, top_k: int = 10) -> List[int]:
        """Find similar historical patterns using embeddings."""
        embeddings = np.array(df['embeddings'].tolist())
        current_embedding = embeddings[current_idx]
        
        # Calculate similarities
        similarities = np.dot(embeddings, current_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(current_embedding)
        )
        
        # Get top k similar (excluding current)
        similarities[current_idx] = -1
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return top_indices.tolist()