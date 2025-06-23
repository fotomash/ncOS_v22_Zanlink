"""SMC Master Agent - Smart Money Concepts analysis with memory tracking."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory_manager import EnhancedMemoryManager


@dataclass
class SMCMasterAgentConfig:
    """Configuration options for :class:`SMCMasterAgent`."""

    agent_id: str = "smc_master"
    enabled: bool = True
    log_level: str = "INFO"
    structure_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    liquidity_sensitivity: float = 0.7
    order_block_threshold: float = 0.8
    custom_params: Dict[str, Any] = field(default_factory=dict)


class SMCMasterAgent:
    """Smart Money Concepts master analysis agent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = SMCMasterAgentConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.active = True
        self.analysis_history: List[Dict[str, Any]] = []
        self.memory_manager = EnhancedMemoryManager()

    # ------------------------------------------------------------------
    def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data request using SMC principles."""
        data = request.get("data", {})
        analysis_type = request.get("type", "market_analysis")
        try:
            if analysis_type == "trade_decision":
                return self._analyze_trade_decision(data)
            if analysis_type == "market_analysis":
                return self._analyze_market_structure(data)
            return self._default_analysis(data)
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "decision": "hold",
                "confidence": 0.0,
                "reasoning": f"SMC analysis error: {exc}",
                "smc_analysis": {},
            }

    # ------------------------------------------------------------------
    def _analyze_trade_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        symbol = data.get("symbol", "UNKNOWN")
        bid = data.get("bid", 0)
        ask = data.get("ask", 0)
        spread = ask - bid if ask and bid else 0

        smc_analysis = {
            "market_structure": self._analyze_market_structure_break(data),
            "liquidity_zones": self._identify_liquidity_zones(data),
            "order_blocks": self._identify_order_blocks(data),
            "fair_value_gaps": self._identify_fair_value_gaps(data),
            "inducement_levels": self._identify_inducement_levels(data),
        }

        decision = self._make_smc_decision(smc_analysis, data)
        confidence = self._calculate_smc_confidence(smc_analysis)
        reasoning = self._generate_smc_reasoning(smc_analysis, decision)

        result = {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "smc_analysis": smc_analysis,
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "spread": spread,
        }

        self.analysis_history.append(result)
        self.memory_manager.store_memory("smc_analysis", result)
        return result

    # ------------------------------------------------------------------
    def _analyze_market_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        structure_analysis = {
            "trend_direction": self._determine_trend_direction(data),
            "structure_breaks": self._detect_structure_breaks(data),
            "key_levels": self._identify_key_levels(data),
            "market_phase": self._determine_market_phase(data),
        }

        return {
            "decision": "analyze",
            "confidence": 0.8,
            "reasoning": "Market structure analysis completed",
            "smc_analysis": structure_analysis,
            "agent_id": self.config.agent_id,
        }

    # ------------------------------------------------------------------
    def _analyze_market_structure_break(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "break_detected": False,
            "break_type": None,
            "break_level": None,
            "confirmation": False,
        }

    # ------------------------------------------------------------------
    def _identify_liquidity_zones(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "buy_side_liquidity",
                "level": data.get("ask", 0) + 0.0010,
                "strength": 0.8,
                "timeframe": "H1",
            },
            {
                "type": "sell_side_liquidity",
                "level": data.get("bid", 0) - 0.0010,
                "strength": 0.7,
                "timeframe": "H1",
            },
        ]

    # ------------------------------------------------------------------
    def _identify_order_blocks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "bullish_order_block",
                "high": data.get("ask", 0) + 0.0005,
                "low": data.get("bid", 0),
                "strength": 0.9,
                "timeframe": "H1",
            }
        ]

    # ------------------------------------------------------------------
    def _identify_fair_value_gaps(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    def _identify_inducement_levels(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    def _make_smc_decision(self, smc_analysis: Dict[str, Any], data: Dict[str, Any]) -> str:
        structure = smc_analysis.get("market_structure", {})
        order_blocks = smc_analysis.get("order_blocks", [])

        if structure.get("break_detected") and structure.get("confirmation"):
            if structure.get("break_type") == "BOS":
                has_bullish = any(ob["type"] == "bullish_order_block" for ob in order_blocks)
                return "buy" if has_bullish else "hold"
            if structure.get("break_type") == "CHoCH":
                has_bearish = any(ob["type"] == "bearish_order_block" for ob in order_blocks)
                return "sell" if has_bearish else "hold"
        return "hold"

    # ------------------------------------------------------------------
    def _calculate_smc_confidence(self, smc_analysis: Dict[str, Any]) -> float:
        confidence_factors: List[float] = []
        structure = smc_analysis.get("market_structure", {})
        if structure.get("break_detected") and structure.get("confirmation"):
            confidence_factors.append(0.8)

        order_blocks = smc_analysis.get("order_blocks", [])
        if order_blocks:
            avg_strength = sum(ob.get("strength", 0) for ob in order_blocks) / len(order_blocks)
            confidence_factors.append(avg_strength)

        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        if liquidity_zones:
            avg_strength = sum(lz.get("strength", 0) for lz in liquidity_zones) / len(liquidity_zones)
            confidence_factors.append(avg_strength * 0.7)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    # ------------------------------------------------------------------
    def _generate_smc_reasoning(self, smc_analysis: Dict[str, Any], decision: str) -> str:
        parts: List[str] = []
        structure = smc_analysis.get("market_structure", {})
        if structure.get("break_detected"):
            parts.append(f"Structure break detected: {structure.get('break_type')}")

        order_blocks = smc_analysis.get("order_blocks", [])
        if order_blocks:
            parts.append(f"{len(order_blocks)} order blocks identified")

        liquidity_zones = smc_analysis.get("liquidity_zones", [])
        if liquidity_zones:
            parts.append(f"{len(liquidity_zones)} liquidity zones mapped")

        if not parts:
            parts.append("No significant SMC signals detected")

        return f"SMC Analysis: {'; '.join(parts)}. Decision: {decision}"

    # ------------------------------------------------------------------
    def _determine_trend_direction(self, data: Dict[str, Any]) -> str:
        return "neutral"

    # ------------------------------------------------------------------
    def _detect_structure_breaks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    def _identify_key_levels(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    def _determine_market_phase(self, data: Dict[str, Any]) -> str:
        return "consolidation"

    # ------------------------------------------------------------------
    def _default_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "decision": "hold",
            "confidence": 0.5,
            "reasoning": "Default SMC analysis - no specific signals",
            "smc_analysis": {},
            "agent_id": self.config.agent_id,
        }

    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        return self.active

    # ------------------------------------------------------------------
    def emergency_stop(self) -> None:
        self.active = False
        self.logger.error("\U0001F6D1 %s emergency stopped", self.config.agent_id)

    # ------------------------------------------------------------------
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.analysis_history[-limit:]
