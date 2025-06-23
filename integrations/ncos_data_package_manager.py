
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import asyncio
from functools import lru_cache
import hashlib

@dataclass
class DataPackage:
    """Pre-processed data package for LLM consumption"""
    package_id: str
    timestamp: datetime
    data_type: str
    summary: str
    key_metrics: Dict[str, Any]
    insights: List[str]
    context: Dict[str, Any]
    ttl_minutes: int = 5

class DataPackageManager:
    """Manages pre-processed data packages for LLM consumption"""

    def __init__(self, cache_dir: str = "data/llm_cache"):
        self.cache_dir = cache_dir
        self.packages = {}
        self.package_templates = {
            "market_analysis": self._create_market_analysis_package,
            "trade_summary": self._create_trade_summary_package,
            "pattern_detection": self._create_pattern_detection_package,
            "risk_assessment": self._create_risk_assessment_package,
            "session_replay": self._create_session_replay_package
        }

    def create_package_id(self, data_type: str, params: Dict) -> str:
        """Create unique package ID based on type and parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{data_type}_{param_str}".encode()).hexdigest()[:12]

    @lru_cache(maxsize=100)
    def get_or_create_package(self, data_type: str, **params) -> DataPackage:
        """Get cached package or create new one"""
        package_id = self.create_package_id(data_type, params)

        # Check if package exists and is still valid
        if package_id in self.packages:
            package = self.packages[package_id]
            if datetime.now() - package.timestamp < timedelta(minutes=package.ttl_minutes):
                return package

        # Create new package
        if data_type in self.package_templates:
            package = self.package_templates[data_type](**params)
            self.packages[package_id] = package
            return package

        raise ValueError(f"Unknown package type: {data_type}")

    def _create_market_analysis_package(self, symbol: str, timeframe: str = "H1") -> DataPackage:
        """Create pre-processed market analysis package"""
        # Simulate data processing
        key_metrics = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": "bullish",
            "strength": 7.5,
            "volatility": "moderate",
            "volume_profile": "increasing",
            "key_levels": {
                "resistance": [1950.50, 1955.00],
                "support": [1945.30, 1940.00]
            },
            "indicators": {
                "rsi": 58.5,
                "macd": "bullish_cross",
                "ema_alignment": "bullish"
            }
        }

        insights = [
            f"{symbol} showing strong bullish momentum on {timeframe}",
            "Price consolidating below key resistance at 1950.50",
            "Volume profile suggests accumulation phase",
            "RSI at 58.5 - room for upward movement",
            "EMA alignment confirms bullish bias"
        ]

        summary = f"{symbol} {timeframe}: Bullish trend (7.5/10) with price at resistance. Volume accumulation detected. Next targets: 1955, 1960."

        return DataPackage(
            package_id=self.create_package_id("market_analysis", {"symbol": symbol, "timeframe": timeframe}),
            timestamp=datetime.now(),
            data_type="market_analysis",
            summary=summary,
            key_metrics=key_metrics,
            insights=insights,
            context={"symbol": symbol, "timeframe": timeframe},
            ttl_minutes=5
        )

    def _create_trade_summary_package(self, session_id: str, period: str = "today") -> DataPackage:
        """Create trade summary package"""
        key_metrics = {
            "total_trades": 12,
            "winning_trades": 9,
            "losing_trades": 3,
            "win_rate": 0.75,
            "profit_factor": 2.8,
            "total_pnl": 1250.50,
            "average_win": 175.25,
            "average_loss": -65.50,
            "best_trade": {"symbol": "XAUUSD", "pnl": 325.00, "pattern": "Wyckoff Spring"},
            "worst_trade": {"symbol": "EURUSD", "pnl": -95.00, "pattern": "Failed BOS"}
        }

        insights = [
            "Win rate of 75% exceeds target of 65%",
            "Profit factor of 2.8 indicates strong risk management",
            "Best performance on XAUUSD with Wyckoff patterns",
            "Losses contained well below risk limits",
            "Consider increasing position size on high-confidence setups"
        ]

        summary = f"Session {session_id}: Exceptional performance with 75% win rate and 2.8 profit factor. Total P&L: $1,250.50 across 12 trades."

        return DataPackage(
            package_id=self.create_package_id("trade_summary", {"session_id": session_id, "period": period}),
            timestamp=datetime.now(),
            data_type="trade_summary",
            summary=summary,
            key_metrics=key_metrics,
            insights=insights,
            context={"session_id": session_id, "period": period},
            ttl_minutes=10
        )

    def _create_pattern_detection_package(self, symbol: str, patterns: List[str] = None) -> DataPackage:
        """Create pattern detection package"""
        detected_patterns = patterns or ["Wyckoff Accumulation", "SMC Order Block", "Liquidity Sweep"]

        key_metrics = {
            "symbol": symbol,
            "patterns_detected": len(detected_patterns),
            "pattern_details": [
                {
                    "name": "Wyckoff Accumulation Phase C",
                    "confidence": 0.85,
                    "location": 1945.30,
                    "timeframe": "H4",
                    "action": "prepare_long"
                },
                {
                    "name": "SMC Bullish Order Block",
                    "confidence": 0.78,
                    "location": 1942.00,
                    "timeframe": "H1",
                    "action": "long_entry_zone"
                },
                {
                    "name": "Liquidity Sweep",
                    "confidence": 0.92,
                    "location": 1940.00,
                    "timeframe": "M15",
                    "action": "reversal_expected"
                }
            ],
            "confluence_score": 8.5,
            "recommended_bias": "bullish"
        }

        insights = [
            "Strong confluence of Wyckoff and SMC patterns",
            "Liquidity sweep at 1940 created optimal entry",
            "Multiple timeframe alignment confirms bullish bias",
            "High probability setup with 8.5/10 confluence score",
            "Risk entry at current levels, safer entry at 1942 retest"
        ]

        summary = f"{symbol}: {len(detected_patterns)} high-confidence patterns detected. Strong bullish confluence (8.5/10) with Wyckoff accumulation and SMC order blocks."

        return DataPackage(
            package_id=self.create_package_id("pattern_detection", {"symbol": symbol}),
            timestamp=datetime.now(),
            data_type="pattern_detection",
            summary=summary,
            key_metrics=key_metrics,
            insights=insights,
            context={"symbol": symbol, "patterns": detected_patterns},
            ttl_minutes=15
        )

    def export_for_llm(self, package: DataPackage) -> Dict[str, Any]:
        """Export package in LLM-friendly format"""
        return {
            "summary": package.summary,
            "key_points": package.insights[:3],  # Top 3 insights
            "metrics": package.key_metrics,
            "timestamp": package.timestamp.isoformat(),
            "context": package.context
        }

    def create_llm_prompt_context(self, packages: List[DataPackage]) -> str:
        """Create context string for LLM prompts"""
        context_parts = []
        for package in packages:
            context_parts.append(f"[{package.data_type.upper()}]")
            context_parts.append(package.summary)
            context_parts.append("Key insights:")
            for insight in package.insights[:2]:
                context_parts.append(f"- {insight}")
            context_parts.append("")

        return "\n".join(context_parts)

# Example usage function
def get_llm_ready_data(manager: DataPackageManager, request_type: str, **params) -> Dict[str, Any]:
    """Get pre-processed data ready for LLM consumption"""
    try:
        package = manager.get_or_create_package(request_type, **params)
        return manager.export_for_llm(package)
    except Exception as e:
        return {
            "error": str(e),
            "summary": "Unable to process request",
            "key_points": ["Error occurred during data processing"],
            "metrics": {},
            "context": params
        }
