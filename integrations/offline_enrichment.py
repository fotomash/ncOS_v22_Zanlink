#!/usr/bin/env python3
"""
ncOS Offline Data Enrichment Engine - Zanlink Edition
Processes and enriches trading data for optimal LLM consumption
"""

import asyncio
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineEnrichmentEngine:
    """
    Offline engine for enriching market data with technical indicators,
    patterns, and pre-computed analytics for fast LLM access
    """

    def __init__(self, data_dir: str = "/app/data", cache_dir: str = "/app/data/cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = int(os.getenv("BATCH_SIZE", "1000"))
        self.enrichment_interval = int(os.getenv("ENRICHMENT_INTERVAL", "300"))

        # Technical indicator settings
        self.indicator_params = {
            "rsi_period": 14,
            "ema_fast": 9,
            "ema_slow": 21,
            "bb_period": 20,
            "atr_period": 14,
            "volume_ema": 20
        }

    async def enrich_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich OHLCV data with technical indicators and patterns
        """
        try:
            # Basic validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required OHLCV columns")
                return df

            # Technical indicators
            df = self._add_technical_indicators(df)

            # Pattern detection
            df = self._detect_patterns(df)

            # Market structure
            df = self._analyze_market_structure(df)

            # Microstructure features
            df = self._add_microstructure_features(df)

            # Pre-compute LLM-friendly summaries
            df = self._add_llm_summaries(df)

            return df

        except Exception as e:
            logger.error(f"Error enriching data: {e}")
            return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            df['close'], 
            window=self.indicator_params['rsi_period']
        ).rsi()

        # Moving averages
        df['ema_9'] = ta.trend.EMAIndicator(
            df['close'], 
            window=self.indicator_params['ema_fast']
        ).ema_indicator()

        df['ema_21'] = ta.trend.EMAIndicator(
            df['close'], 
            window=self.indicator_params['ema_slow']
        ).ema_indicator()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=self.indicator_params['bb_period']
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], 
            df['low'], 
            df['close'],
            window=self.indicator_params['atr_period']
        ).average_true_range()

        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Volume analysis
        df['volume_ema'] = ta.trend.EMAIndicator(
            df['volume'], 
            window=self.indicator_params['volume_ema']
        ).ema_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_ema']

        return df

    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading patterns"""
        # Candlestick patterns
        df['is_doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1
        df['is_bullish_engulfing'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )

        # Swing highs/lows
        df['swing_high'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(-1))
        )
        df['swing_low'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(-1))
        )

        # Order blocks
        df['potential_ob_bull'] = (
            df['is_bullish_engulfing'] & 
            (df['volume'] > df['volume_ema'] * 1.5)
        )
        df['potential_ob_bear'] = (
            (~df['is_bullish_engulfing']) & 
            (df['close'] < df['open']) &
            (df['volume'] > df['volume_ema'] * 1.5)
        )

        return df

    def _analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market structure"""
        # Trend identification
        df['ema_trend'] = np.where(
            df['ema_9'] > df['ema_21'], 
            'bullish', 
            'bearish'
        )

        # Higher highs/lows
        df['hh'] = (df['high'] > df['high'].shift(1)) & df['swing_high']
        df['ll'] = (df['low'] < df['low'].shift(1)) & df['swing_low']
        df['hl'] = (df['low'] > df['low'].shift(1)) & df['swing_low']
        df['lh'] = (df['high'] < df['high'].shift(1)) & df['swing_high']

        # Market structure
        conditions = [
            (df['hh'] & df['hl']),
            (df['ll'] & df['lh']),
            (df['ema_trend'] == 'bullish'),
            (df['ema_trend'] == 'bearish')
        ]
        choices = ['strong_bullish', 'strong_bearish', 'bullish', 'bearish']
        df['market_structure'] = np.select(conditions, choices, default='neutral')

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure features"""
        # Price action metrics
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_range'] = df['body_size'] / (df['high'] - df['low'])

        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # Order flow proxy
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['order_flow_imbalance'] = df['buy_pressure'] - df['sell_pressure']

        return df

    def _add_llm_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pre-computed summaries for LLM consumption"""
        # Current state summary
        df['llm_summary'] = df.apply(self._generate_row_summary, axis=1)

        # Pattern summary
        df['pattern_summary'] = df.apply(self._generate_pattern_summary, axis=1)

        # Action suggestion
        df['action_suggestion'] = df.apply(self._generate_action_suggestion, axis=1)

        return df

    def _generate_row_summary(self, row) -> str:
        """Generate LLM-friendly summary for a single row"""
        trend = "bullish" if row['ema_trend'] == 'bullish' else "bearish"
        rsi_state = "oversold" if row['rsi'] < 30 else "overbought" if row['rsi'] > 70 else "neutral"

        return f"Price at {row['close']:.2f}, {trend} trend, RSI {rsi_state} ({row['rsi']:.1f})"

    def _generate_pattern_summary(self, row) -> str:
        """Generate pattern summary"""
        patterns = []
        if row.get('is_doji', False):
            patterns.append("Doji")
        if row.get('is_bullish_engulfing', False):
            patterns.append("Bullish Engulfing")
        if row.get('swing_high', False):
            patterns.append("Swing High")
        if row.get('swing_low', False):
            patterns.append("Swing Low")
        if row.get('potential_ob_bull', False):
            patterns.append("Bullish Order Block")

        return ", ".join(patterns) if patterns else "No significant patterns"

    def _generate_action_suggestion(self, row) -> str:
        """Generate action suggestion"""
        if row['market_structure'] == 'strong_bullish' and row['rsi'] < 70:
            return "Consider long positions"
        elif row['market_structure'] == 'strong_bearish' and row['rsi'] > 30:
            return "Consider short positions"
        elif row['rsi'] < 30:
            return "Oversold - potential bounce"
        elif row['rsi'] > 70:
            return "Overbought - potential pullback"
        else:
            return "Wait for clearer signal"

    async def create_llm_cache(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Create cached data package for LLM"""
        try:
            # Get latest data
            latest = df.iloc[-1].to_dict()
            recent_data = df.tail(20)

            # Create cache package
            cache_package = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": latest['close'],
                "summary": latest.get('llm_summary', ''),
                "patterns": latest.get('pattern_summary', ''),
                "action": latest.get('action_suggestion', ''),
                "technical_data": {
                    "rsi": latest.get('rsi', 50),
                    "ema_trend": latest.get('ema_trend', 'neutral'),
                    "atr": latest.get('atr', 0),
                    "volume_ratio": latest.get('volume_ratio', 1)
                },
                "market_structure": {
                    "trend": latest.get('market_structure', 'neutral'),
                    "support": float(recent_data['low'].min()),
                    "resistance": float(recent_data['high'].max())
                },
                "statistics": {
                    "volatility": float(recent_data['volatility_20'].mean()),
                    "avg_volume": float(recent_data['volume'].mean()),
                    "price_change_24h": float((latest['close'] - recent_data.iloc[0]['close']) / recent_data.iloc[0]['close'] * 100)
                }
            }

            # Save to cache
            cache_file = self.cache_dir / f"{symbol}_{timeframe}_llm_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_package, f, indent=2)

            logger.info(f"Created LLM cache for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error creating LLM cache: {e}")

    async def run_enrichment_cycle(self):
        """Run a complete enrichment cycle"""
        logger.info("Starting enrichment cycle")

        # Process each data file
        data_files = list(self.data_dir.glob("*.csv"))

        for data_file in data_files:
            try:
                # Extract symbol and timeframe from filename
                # Expected format: SYMBOL_TIMEFRAME.csv
                parts = data_file.stem.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                else:
                    symbol = data_file.stem
                    timeframe = "H1"

                # Load data
                df = pd.read_csv(data_file)
                logger.info(f"Processing {symbol} {timeframe} - {len(df)} rows")

                # Enrich data
                enriched_df = await self.enrich_ohlcv_data(df)

                # Save enriched data
                enriched_file = self.data_dir / f"{symbol}_{timeframe}_enriched.parquet"
                enriched_df.to_parquet(enriched_file)

                # Create LLM cache
                await self.create_llm_cache(symbol, timeframe, enriched_df)

            except Exception as e:
                logger.error(f"Error processing {data_file}: {e}")

        logger.info("Enrichment cycle complete")

    async def run(self):
        """Main run loop"""
        while True:
            try:
                await self.run_enrichment_cycle()
                await asyncio.sleep(self.enrichment_interval)
            except Exception as e:
                logger.error(f"Error in enrichment loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

def main():
    """Main entry point"""
    engine = OfflineEnrichmentEngine()
    asyncio.run(engine.run())

if __name__ == "__main__":
    main()
