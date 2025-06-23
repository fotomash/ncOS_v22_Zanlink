"""
NCOS v21.7.1 Liquidity Analysis Agent
Advanced liquidity analysis with sweep detection and zone identification
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

import pandas as pd


class LiquidityZoneType(Enum):
    DEMAND = "demand_zone"
    SUPPLY = "supply_zone"
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"


class NCOSLiquidityAnalysisAgent:
    """NCOS Liquidity Analysis Agent"""

    def __init__(self, session_state, config):
        self.session_state = session_state
        self.config = config.get("trading", {})
        self.agent_id = "liquidity_analysis_agent"
        self.priority = 5
        self.status = "initializing"

        # Liquidity analysis parameters
        self.sweep_threshold = self.config.get("liquidity_sweep_threshold", 0.8)
        self.zone_strength_threshold = 0.6

    async def initialize(self):
        """Initialize Liquidity Analysis Agent"""
        self.status = "active"

    async def analyze_liquidity_complete(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete liquidity analysis"""
        try:
            candles = self._df_to_candles(df)

            # Liquidity zone detection
            zones = self._detect_liquidity_zones(candles)

            # Sweep probability analysis
            sweep_analysis = self._analyze_sweep_probabilities(candles, zones)

            # Equal highs/lows detection
            equal_levels = self._detect_equal_levels(candles)

            # Calculate overall liquidity probability
            overall_probability = self._calculate_overall_probability(zones, sweep_analysis, equal_levels)

            # Extract zones with a high probability of sweep based on strength
            high_prob_zones = []
            for zone in zones:
                zone_id = f"{zone['type']}_{zone['timestamp']}"
                prob = sweep_analysis.get(zone_id, {}).get("probability", zone["strength"])
                if prob > self.zone_strength_threshold:
                    enriched_zone = zone.copy()
                    enriched_zone["probability"] = prob
                    high_prob_zones.append(enriched_zone)

            return {
                "status": "success",
                "agent_id": self.agent_id,
                "zones": zones,
                "sweep_analysis": sweep_analysis,
                "equal_levels": equal_levels,
                "overall_probability": overall_probability,
                "zone_count": len(zones),
                "high_probability_zones": high_prob_zones,
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "agent_id": self.agent_id,
                "error": str(e),
                "overall_probability": 0.0,
                "zones": []
            }

    def _df_to_candles(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to candle format"""
        candles = []
        for _, row in df.iterrows():
            candle = {
                "open": float(row.get("open", row.get("Open", 0))),
                "high": float(row.get("high", row.get("High", 0))),
                "low": float(row.get("low", row.get("Low", 0))),
                "close": float(row.get("close", row.get("Close", 0))),
                "volume": float(row.get("volume", row.get("Volume", 0))),
                "timestamp": str(row.get("datetime", row.get("Date", "")))
            }
            candles.append(candle)
        return candles

    def _detect_liquidity_zones(self, candles: List[Dict]) -> List[Dict]:
        """Detect liquidity zones"""
        zones = []

        # Demand zones (areas where price bounced up)
        demand_zones = self._find_demand_zones(candles)
        zones.extend(demand_zones)

        # Supply zones (areas where price dropped)
        supply_zones = self._find_supply_zones(candles)
        zones.extend(supply_zones)

        return zones

    def _find_demand_zones(self, candles: List[Dict]) -> List[Dict]:
        """Find demand zones"""
        zones = []
        lookback = 20

        for i in range(lookback, len(candles) - lookback):
            current = candles[i]

            # Look for strong bullish reaction from a level
            reaction_strength = self._calculate_bullish_reaction_strength(candles, i)

            if reaction_strength > self.zone_strength_threshold:
                # Find the demand zone range
                zone_low = min(candles[j]["low"] for j in range(i - 5, i + 1))
                zone_high = max(candles[j]["high"] for j in range(i - 5, i + 1))

                zones.append({
                    "type": LiquidityZoneType.DEMAND.value,
                    "low": zone_low,
                    "high": zone_high,
                    "center": (zone_low + zone_high) / 2,
                    "strength": reaction_strength,
                    "timestamp": current["timestamp"],
                    "volume_profile": self._calculate_volume_profile(candles, i - 5, i + 1)
                })

        return zones

    def _find_supply_zones(self, candles: List[Dict]) -> List[Dict]:
        """Find supply zones"""
        zones = []
        lookback = 20

        for i in range(lookback, len(candles) - lookback):
            current = candles[i]

            # Look for strong bearish reaction from a level
            reaction_strength = self._calculate_bearish_reaction_strength(candles, i)

            if reaction_strength > self.zone_strength_threshold:
                # Find the supply zone range
                zone_low = min(candles[j]["low"] for j in range(i - 5, i + 1))
                zone_high = max(candles[j]["high"] for j in range(i - 5, i + 1))

                zones.append({
                    "type": LiquidityZoneType.SUPPLY.value,
                    "low": zone_low,
                    "high": zone_high,
                    "center": (zone_low + zone_high) / 2,
                    "strength": reaction_strength,
                    "timestamp": current["timestamp"],
                    "volume_profile": self._calculate_volume_profile(candles, i - 5, i + 1)
                })

        return zones

    def _calculate_bullish_reaction_strength(self, candles: List[Dict], index: int) -> float:
        """Calculate strength of bullish reaction"""
        if index < 5 or index >= len(candles) - 5:
            return 0.0

        # Calculate price movement strength
        low_point = min(candles[j]["low"] for j in range(index - 5, index + 1))
        high_after = max(candles[j]["high"] for j in range(index, index + 6))

        if low_point == 0:
            return 0.0

        price_change = (high_after - low_point) / low_point

        # Calculate volume confirmation
        volume_before = sum(candles[j]["volume"] for j in range(index - 5, index))
        volume_after = sum(candles[j]["volume"] for j in range(index, index + 5))

        volume_ratio = volume_after / volume_before if volume_before > 0 else 1.0

        # Combine price and volume strength
        strength = min(price_change * volume_ratio * 10, 1.0)
        return max(0.0, strength)

    def _calculate_bearish_reaction_strength(self, candles: List[Dict], index: int) -> float:
        """Calculate strength of bearish reaction"""
        if index < 5 or index >= len(candles) - 5:
            return 0.0

        # Calculate price movement strength
        high_point = max(candles[j]["high"] for j in range(index - 5, index + 1))
        low_after = min(candles[j]["low"] for j in range(index, index + 6))

        if high_point == 0:
            return 0.0

        price_change = (high_point - low_after) / high_point

        # Calculate volume confirmation
        volume_before = sum(candles[j]["volume"] for j in range(index - 5, index))
        volume_after = sum(candles[j]["volume"] for j in range(index, index + 5))

        volume_ratio = volume_after / volume_before if volume_before > 0 else 1.0

        # Combine price and volume strength
        strength = min(price_change * volume_ratio * 10, 1.0)
        return max(0.0, strength)

    def _calculate_volume_profile(self, candles: List[Dict], start: int, end: int) -> Dict:
        """Calculate volume profile for a range"""
        if start < 0 or end >= len(candles):
            return {"total_volume": 0, "avg_volume": 0}

        volumes = [candles[i]["volume"] for i in range(start, end + 1)]

        return {
            "total_volume": sum(volumes),
            "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
            "max_volume": max(volumes) if volumes else 0
        }

    def _analyze_sweep_probabilities(self, candles: List[Dict], zones: List[Dict]) -> Dict:
        """Analyze sweep probabilities"""
        sweep_probabilities = {}

        for zone in zones:
            zone_id = f"{zone['type']}_{zone['timestamp']}"

            # Calculate probability based on zone strength and recent price action
            base_probability = zone["strength"]

            # Check recent approaches to the zone
            recent_approaches = self._count_recent_approaches(candles, zone)
            approach_factor = min(recent_approaches * 0.2, 0.4)

            # Check volume near zone
            volume_factor = min(zone["volume_profile"]["avg_volume"] / 1000, 0.3)

            total_probability = min(base_probability + approach_factor + volume_factor, 1.0)

            sweep_probabilities[zone_id] = {
                "probability": total_probability,
                "components": {
                    "base_strength": base_probability,
                    "approach_factor": approach_factor,
                    "volume_factor": volume_factor
                },
                "recommendation": self._get_sweep_recommendation(total_probability)
            }

        return sweep_probabilities

    def _count_recent_approaches(self, candles: List[Dict], zone: Dict) -> int:
        """Count recent approaches to a zone"""
        approaches = 0
        lookback = min(50, len(candles))

        for i in range(len(candles) - lookback, len(candles)):
            candle = candles[i]

            # Check if price approached the zone
            if zone["type"] == LiquidityZoneType.DEMAND.value:
                if candle["low"] <= zone["high"] and candle["low"] >= zone["low"]:
                    approaches += 1
            elif zone["type"] == LiquidityZoneType.SUPPLY.value:
                if candle["high"] >= zone["low"] and candle["high"] <= zone["high"]:
                    approaches += 1

        return approaches

    def _detect_equal_levels(self, candles: List[Dict]) -> List[Dict]:
        """Detect equal highs and lows"""
        equal_levels = []
        tolerance = 0.001  # 0.1% tolerance for "equal" levels

        # Find equal highs
        highs = [c["high"] for c in candles]
        equal_highs = self._find_equal_prices(highs, tolerance)

        for level in equal_highs:
            equal_levels.append({
                "type": LiquidityZoneType.EQUAL_HIGHS.value,
                "price": level["price"],
                "occurrences": level["count"],
                "strength": min(level["count"] * 0.3, 1.0),
                "last_touched": level["last_index"]
            })

        # Find equal lows
        lows = [c["low"] for c in candles]
        equal_lows = self._find_equal_prices(lows, tolerance)

        for level in equal_lows:
            equal_levels.append({
                "type": LiquidityZoneType.EQUAL_LOWS.value,
                "price": level["price"],
                "occurrences": level["count"],
                "strength": min(level["count"] * 0.3, 1.0),
                "last_touched": level["last_index"]
            })

        return equal_levels

    def _find_equal_prices(self, prices: List[float], tolerance: float) -> List[Dict]:
        """Find equal price levels"""
        equal_levels = []
        used_indices = set()

        for i, price in enumerate(prices):
            if i in used_indices:
                continue

            matches = [i]
            for j, other_price in enumerate(prices[i + 1:], i + 1):
                if abs(price - other_price) / price <= tolerance:
                    matches.append(j)
                    used_indices.add(j)

            if len(matches) >= 2:  # At least 2 equal levels
                equal_levels.append({
                    "price": sum(prices[idx] for idx in matches) / len(matches),
                    "count": len(matches),
                    "indices": matches,
                    "last_index": max(matches)
                })
                used_indices.update(matches)

        return equal_levels

    def _calculate_overall_probability(self, zones: List[Dict], sweep_analysis: Dict,
                                       equal_levels: List[Dict]) -> float:
        """Calculate overall liquidity probability"""
        scores = []

        # Zone strength contribution (50% weight)
        if zones:
            avg_zone_strength = sum(z["strength"] for z in zones) / len(zones)
            scores.append(avg_zone_strength * 0.5)

        # Sweep probability contribution (30% weight)
        if sweep_analysis:
            avg_sweep_prob = sum(s["probability"] for s in sweep_analysis.values()) / len(sweep_analysis)
            scores.append(avg_sweep_prob * 0.3)

        # Equal levels contribution (20% weight)
        if equal_levels:
            avg_equal_strength = sum(l["strength"] for l in equal_levels) / len(equal_levels)
            scores.append(avg_equal_strength * 0.2)

        return sum(scores) if scores else 0.0

    def _get_sweep_recommendation(self, probability: float) -> str:
        """Get sweep recommendation"""
        if probability > 0.8:
            return "Very High - Strong Sweep Expected"
        elif probability > 0.6:
            return "High - Sweep Likely"
        elif probability > 0.4:
            return "Medium - Possible Sweep"
        else:
            return "Low - Sweep Unlikely"
