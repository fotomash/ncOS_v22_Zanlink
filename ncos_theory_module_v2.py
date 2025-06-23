
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta

class NCOSTheoryEngine:
    """
    The core intelligence engine for ncOS.
    Implements SMC, Wyckoff, MAZ strategies, and advanced pattern recognition.
    """

    def __init__(self):
        self.config = {
            "lookback_periods": {
                "short": 20,
                "medium": 50,
                "long": 200
            },
            "volatility_threshold": 0.02,
            "volume_spike_multiplier": 2.5,
            "structure_break_threshold": 0.003,
            "wyckoff_volume_confirmation": 1.5
        }

    def calculate_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies market structure: HH, HL, LL, LH patterns.
        Core SMC concept implementation.
        """
        highs = df['ask'].rolling(window=10).max()
        lows = df['bid'].rolling(window=10).min()

        # Identify swing points
        swing_highs = []
        swing_lows = []

        for i in range(20, len(df)-20):
            if highs.iloc[i] == df['ask'].iloc[i-10:i+10].max():
                swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] == df['bid'].iloc[i-10:i+10].min():
                swing_lows.append((i, lows.iloc[i]))

        # Determine structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_high = swing_highs[-1][1]
            prev_high = swing_highs[-2][1]
            last_low = swing_lows[-1][1]
            prev_low = swing_lows[-2][1]

            if last_high > prev_high and last_low > prev_low:
                structure = "BULLISH_STRUCTURE"
                strength = 0.8
            elif last_high < prev_high and last_low < prev_low:
                structure = "BEARISH_STRUCTURE"
                strength = 0.8
            else:
                structure = "RANGING"
                strength = 0.5
        else:
            structure = "UNDEFINED"
            strength = 0.3

        return {
            "structure": structure,
            "strength": strength,
            "swing_highs": len(swing_highs),
            "swing_lows": len(swing_lows),
            "last_high": swing_highs[-1][1] if swing_highs else None,
            "last_low": swing_lows[-1][1] if swing_lows else None
        }

    def detect_wyckoff_phase(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies current Wyckoff phase based on volume and price action.
        """
        price = df['bid'].values
        volume = df['volume'].values

        # Calculate volume moving average
        volume_ma = pd.Series(volume).rolling(window=20).mean()

        # Price range analysis
        price_range = price.max() - price.min()
        current_position = (price[-1] - price.min()) / price_range if price_range > 0 else 0.5

        # Volume analysis
        recent_volume = volume[-5:].mean() if len(volume) >= 5 else 0
        avg_volume = volume_ma.iloc[-1] if not volume_ma.empty else 0
        volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1

        # Determine phase
        if current_position < 0.3 and volume_spike > self.config["wyckoff_volume_confirmation"]:
            phase = "ACCUMULATION"
            confidence = 0.75
        elif current_position > 0.7 and volume_spike > self.config["wyckoff_volume_confirmation"]:
            phase = "DISTRIBUTION"
            confidence = 0.75
        elif 0.3 <= current_position <= 0.7 and volume_spike < 1.2:
            phase = "MARKUP" if df['bid'].iloc[-10:].mean() > df['bid'].iloc[-20:-10].mean() else "MARKDOWN"
            confidence = 0.65
        else:
            phase = "TRANSITION"
            confidence = 0.5

        return {
            "phase": phase,
            "confidence": confidence,
            "price_position": round(current_position, 3),
            "volume_spike": round(volume_spike, 2)
        }

    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifies bullish and bearish order blocks (SMC concept).
        """
        order_blocks = []

        # Look for strong moves preceded by consolidation
        for i in range(50, len(df)-10):
            # Calculate range before and after
            before_range = df['ask'].iloc[i-20:i].max() - df['bid'].iloc[i-20:i].min()
            after_range = df['ask'].iloc[i:i+10].max() - df['bid'].iloc[i:i+10].min()

            # Check for expansion
            if after_range > before_range * 2:
                # Determine direction
                move_direction = df['bid'].iloc[i+10] - df['bid'].iloc[i]

                if move_direction > 0:
                    block_type = "BULLISH_OB"
                    effectiveness = min(after_range / before_range / 2, 1.0)
                else:
                    block_type = "BEARISH_OB"
                    effectiveness = min(after_range / before_range / 2, 1.0)

                order_blocks.append({
                    "type": block_type,
                    "index": i,
                    "price_level": df['bid'].iloc[i],
                    "effectiveness": round(effectiveness, 2)
                })

        # Return only the most recent and relevant order blocks
        return sorted(order_blocks, key=lambda x: x['effectiveness'], reverse=True)[:3]

    def calculate_liquidity_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identifies key liquidity zones based on price clustering.
        """
        prices = df['bid'].values

        # Find price levels with high frequency
        price_bins = np.histogram(prices, bins=50)
        bin_centers = (price_bins[1][:-1] + price_bins[1][1:]) / 2

        # Identify high-frequency zones
        threshold = np.percentile(price_bins[0], 75)
        liquidity_zones = bin_centers[price_bins[0] > threshold]

        # Separate into buy and sell side liquidity
        current_price = prices[-1]
        buy_side_liquidity = [float(z) for z in liquidity_zones if z < current_price]
        sell_side_liquidity = [float(z) for z in liquidity_zones if z > current_price]

        return {
            "buy_side": sorted(buy_side_liquidity, reverse=True)[:3],
            "sell_side": sorted(sell_side_liquidity)[:3],
            "current_price": float(current_price)
        }

    def generate_comprehensive_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates a comprehensive trading signal using all analysis methods.
        """
        # Run all analysis
        structure = self.calculate_market_structure(df)
        wyckoff = self.detect_wyckoff_phase(df)
        order_blocks = self.identify_order_blocks(df)
        liquidity = self.calculate_liquidity_zones(df)

        # Calculate momentum
        short_ma = df['bid'].rolling(window=20).mean()
        long_ma = df['bid'].rolling(window=50).mean()
        momentum = "BULLISH" if short_ma.iloc[-1] > long_ma.iloc[-1] else "BEARISH"

        # Aggregate signals
        bull_score = 0
        bear_score = 0

        # Structure scoring
        if structure["structure"] == "BULLISH_STRUCTURE":
            bull_score += structure["strength"]
        elif structure["structure"] == "BEARISH_STRUCTURE":
            bear_score += structure["strength"]

        # Wyckoff scoring
        if wyckoff["phase"] == "ACCUMULATION":
            bull_score += wyckoff["confidence"]
        elif wyckoff["phase"] == "DISTRIBUTION":
            bear_score += wyckoff["confidence"]
        elif wyckoff["phase"] == "MARKUP":
            bull_score += wyckoff["confidence"] * 0.5
        elif wyckoff["phase"] == "MARKDOWN":
            bear_score += wyckoff["confidence"] * 0.5

        # Order block scoring
        for ob in order_blocks:
            if ob["type"] == "BULLISH_OB":
                bull_score += ob["effectiveness"] * 0.3
            else:
                bear_score += ob["effectiveness"] * 0.3

        # Momentum scoring
        if momentum == "BULLISH":
            bull_score += 0.4
        else:
            bear_score += 0.4

        # Final signal determination
        total_score = bull_score + bear_score
        if total_score == 0:
            signal = "HOLD"
            confidence = 0.0
        else:
            bull_ratio = bull_score / total_score

            if bull_ratio > 0.65:
                signal = "BUY"
                confidence = bull_ratio
            elif bull_ratio < 0.35:
                signal = "SELL"
                confidence = 1 - bull_ratio
            else:
                signal = "HOLD"
                confidence = 0.5

        # Generate detailed reason
        reasons = []
        reasons.append(f"Market Structure: {structure['structure']}")
        reasons.append(f"Wyckoff Phase: {wyckoff['phase']}")
        reasons.append(f"Momentum: {momentum}")
        if order_blocks:
            reasons.append(f"Active Order Blocks: {len(order_blocks)}")

        return {
            "signal": signal,
            "confidence": round(confidence, 3),
            "reason": " | ".join(reasons),
            "details": {
                "structure": structure,
                "wyckoff": wyckoff,
                "order_blocks": order_blocks,
                "liquidity": liquidity,
                "bull_score": round(bull_score, 3),
                "bear_score": round(bear_score, 3)
            }
        }

# Export the main function for backward compatibility
def get_trading_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Main entry point for getting a trading signal."""
    engine = NCOSTheoryEngine()
    return engine.generate_comprehensive_signal(df)

def get_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Get market structure analysis."""
    engine = NCOSTheoryEngine()
    return engine.calculate_market_structure(df)

def get_wyckoff_phase(df: pd.DataFrame) -> Dict[str, Any]:
    """Get Wyckoff phase analysis."""
    engine = NCOSTheoryEngine()
    return engine.detect_wyckoff_phase(df)

def get_order_blocks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Get order block analysis."""
    engine = NCOSTheoryEngine()
    return engine.identify_order_blocks(df)

def get_liquidity_zones(df: pd.DataFrame) -> Dict[str, List[float]]:
    """Get liquidity zone analysis."""
    engine = NCOSTheoryEngine()
    return engine.calculate_liquidity_zones(df)
