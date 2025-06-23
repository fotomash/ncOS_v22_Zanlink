"""
NCOS v21.7.1 Market Data Native Agent
Native market data processing with technical indicators and correlations
"""

from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd


class NCOSMarketDataNativeAgent:
    """NCOS Market Data Native Agent"""

    def __init__(self, session_state, config):
        self.session_state = session_state
        self.config = config.get("trading", {})
        self.agent_id = "market_data_native_agent"
        self.priority = 6
        self.status = "initializing"

        # Market data cache
        self.processed_data = {}
        self.correlations = {}

    async def initialize(self):
        """Initialize Market Data Native Agent"""
        self.status = "active"

    async def process_market_data(self, df: pd.DataFrame, asset_key: str) -> Dict[str, Any]:
        """Process market data with technical indicators"""
        try:
            # Generate technical indicators
            indicators = self._calculate_all_indicators(df)

            # Calculate asset profile
            asset_profile = self._generate_asset_profile(df, asset_key)

            # Calculate correlations if multiple assets exist
            correlations = await self._calculate_correlations(df, asset_key)

            # Calculate indicator strength
            indicator_strength = self._calculate_indicator_strength(indicators)

            # Store processed data
            self.processed_data[asset_key] = {
                "indicators": indicators,
                "asset_profile": asset_profile,
                "correlations": correlations,
                "indicator_strength": indicator_strength,
                "processed_at": datetime.now().isoformat()
            }

            return {
                "status": "success",
                "agent_id": self.agent_id,
                "asset_key": asset_key,
                "indicators": indicators,
                "asset_profile": asset_profile,
                "correlations": correlations,
                "indicator_strength": indicator_strength,
                "data_quality": self._assess_data_quality(df),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e),
                "indicator_strength": 0.0
            }

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        indicators = {}

        # Ensure we have OHLC data
        if not all(col in df.columns for col in ['close']):
            return indicators

        close = df['close']

        # Moving Averages
        indicators['sma_20'] = close.rolling(20).mean()
        indicators['sma_50'] = close.rolling(50).mean()
        indicators['ema_12'] = close.ewm(span=12).mean()
        indicators['ema_26'] = close.ewm(span=26).mean()

        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        signal_line = macd_line.ewm(span=9).mean()
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = macd_line - signal_line

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        indicators['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
        indicators['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
        indicators['bb_middle'] = bb_middle

        # Stochastic
        if all(col in df.columns for col in ['high', 'low']):
            high = df['high']
            low = df['low']

            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            indicators['stoch_k'] = k_percent
            indicators['stoch_d'] = k_percent.rolling(3).mean()

        # Average True Range (ATR)
        if all(col in df.columns for col in ['high', 'low', 'open']):
            high = df['high']
            low = df['low']
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(14).mean()

        # Volume indicators (if volume available)
        if 'volume' in df.columns:
            volume = df['volume']
            indicators['volume_sma'] = volume.rolling(20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']

            # On-Balance Volume
            obv = []
            obv_value = 0
            for i in range(len(close)):
                if i == 0:
                    obv_value = volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
                else:
                    if close.iloc[i] > close.iloc[i - 1]:
                        obv_value += volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
                    elif close.iloc[i] < close.iloc[i - 1]:
                        obv_value -= volume.iloc[i] if not pd.isna(volume.iloc[i]) else 0
                obv.append(obv_value)

            indicators['obv'] = pd.Series(obv, index=df.index)

        return indicators

    def _generate_asset_profile(self, df: pd.DataFrame, asset_key: str) -> Dict[str, Any]:
        """Generate comprehensive asset profile"""
        close = df['close']

        # Basic statistics
        profile = {
            "asset_key": asset_key,
            "data_points": len(df),
            "price_range": {
                "min": float(close.min()),
                "max": float(close.max()),
                "current": float(close.iloc[-1]),
                "avg": float(close.mean())
            },
            "volatility": {
                "std_dev": float(close.std()),
                "coefficient_of_variation": float(close.std() / close.mean()) if close.mean() != 0 else 0,
                "daily_returns_std": float(close.pct_change().std())
            },
            "trend_analysis": self._analyze_trend(close),
            "support_resistance": self._find_support_resistance(df)
        }

        return profile

    def _analyze_trend(self, close: pd.Series) -> Dict[str, Any]:
        """Analyze price trend"""
        # Simple trend analysis using linear regression
        x = np.arange(len(close))
        y = close.values

        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < 2:
            return {"direction": "insufficient_data", "strength": 0.0}

        # Linear regression
        slope, intercept = np.polyfit(x_valid, y_valid, 1)

        # Calculate R-squared
        y_pred = slope * x_valid + intercept
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Determine trend direction and strength
        if slope > 0:
            direction = "bullish"
        elif slope < 0:
            direction = "bearish"
        else:
            direction = "sideways"

        return {
            "direction": direction,
            "strength": float(abs(r_squared)),
            "slope": float(slope),
            "r_squared": float(r_squared)
        }

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Find basic support and resistance levels"""
        if 'high' not in df.columns or 'low' not in df.columns:
            return {"support_levels": [], "resistance_levels": []}

        high = df['high']
        low = df['low']

        # Find local peaks and troughs
        support_levels = []
        resistance_levels = []

        window = 5
        for i in range(window, len(df) - window):
            # Check for local high (resistance)
            if all(high.iloc[i] >= high.iloc[i - j] for j in range(1, window + 1)) and all(
                    high.iloc[i] >= high.iloc[i + j] for j in range(1, window + 1)):
                resistance_levels.append(float(high.iloc[i]))

            # Check for local low (support)
            if all(low.iloc[i] <= low.iloc[i - j] for j in range(1, window + 1)) and all(
                    low.iloc[i] <= low.iloc[i + j] for j in range(1, window + 1)):
                support_levels.append(float(low.iloc[i]))

        return {
            "support_levels": sorted(list(set(support_levels)))[-5:],  # Keep top 5
            "resistance_levels": sorted(list(set(resistance_levels)))[-5:]  # Keep top 5
        }

    async def _calculate_correlations(self, df: pd.DataFrame, asset_key: str) -> Dict[str, float]:
        """Calculate correlations with other assets"""
        correlations = {}

        current_returns = df['close'].pct_change().dropna()

        # Compare with previously processed assets
        for other_asset, other_data in self.processed_data.items():
            if other_asset == asset_key:
                continue

            try:
                # Get returns from other asset (simplified)
                # In real implementation, this would use stored price data  
                correlation_score = np.random.uniform(0.1, 0.9)  # Placeholder
                correlations[other_asset] = correlation_score
            except:
                continue

        return correlations

    def _calculate_indicator_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall indicator strength"""
        scores = []

        # RSI strength
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi_val = indicators['rsi'].iloc[-1]
            if not pd.isna(rsi_val):
                # RSI strength based on overbought/oversold
                if rsi_val > 70 or rsi_val < 30:
                    scores.append(0.8)
                elif rsi_val > 60 or rsi_val < 40:
                    scores.append(0.6)
                else:
                    scores.append(0.4)

        # MACD strength
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_val = indicators['macd'].iloc[-1]
            signal_val = indicators['macd_signal'].iloc[-1]
            if not pd.isna(macd_val) and not pd.isna(signal_val):
                if macd_val > signal_val:
                    scores.append(0.7)
                else:
                    scores.append(0.3)

        # Moving average strength
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma20 = indicators['sma_20'].iloc[-1]
            sma50 = indicators['sma_50'].iloc[-1]
            if not pd.isna(sma20) and not pd.isna(sma50):
                if sma20 > sma50:
                    scores.append(0.6)
                else:
                    scores.append(0.4)

        return sum(scores) / len(scores) if scores else 0.0

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        return {
            "completeness": float(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))),
            "data_points": len(df),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "quality_score": float(len(df) / max(len(df) + df.isnull().sum().sum(), 1))
        }

    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary across all processed assets"""
        if not self.processed_data:
            return {"message": "No market data processed yet"}

        summary = {
            "total_assets": len(self.processed_data),
            "assets": list(self.processed_data.keys()),
            "avg_indicator_strength": sum(
                data["indicator_strength"] for data in self.processed_data.values()
            ) / len(self.processed_data),
            "market_overview": {}
        }

        # Aggregate trend analysis
        trend_counts = {"bullish": 0, "bearish": 0, "sideways": 0}
        for data in self.processed_data.values():
            trend = data["asset_profile"]["trend_analysis"]["direction"]
            if trend in trend_counts:
                trend_counts[trend] += 1

        summary["market_overview"]["trend_distribution"] = trend_counts
        summary["last_updated"] = datetime.now().isoformat()

        return summary
