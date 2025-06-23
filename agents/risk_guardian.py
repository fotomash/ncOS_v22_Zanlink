"""
RiskGuardian - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class RiskGuardianConfig:
    """Configuration for RiskGuardian"""
    agent_id: str = "risk_guardian"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    max_drawdown: Optional[float] = 0.05
    position_limit: Optional[float] = 0.1
    var_threshold: Optional[float] = 0.02
    stop_loss_pct: Optional[float] = 0.03
    custom_params: Dict[str, Any] = field(default_factory=dict)


class RiskGuardian:
    """
    RiskGuardian - Risk management and position monitoring
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = RiskGuardianConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "risk_checks": 0,
            "alerts_generated": 0
        }

        self.risk_limits = {
            "max_drawdown": self.config.max_drawdown,
            "position_limit": self.config.position_limit,
            "var_threshold": self.config.var_threshold
        }
        self.position_monitor = {}
        self.alerts = []

        self.logger.info(f"{self.config.agent_id} initialized")

    async def initialize(self):
        """Initialize the agent"""
        try:
            self.logger.info(f"Initializing {self.config.agent_id}")
            await self._setup()
            self.status = "ready"
            self.logger.info(f"{self.config.agent_id} ready for operation")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.agent_id}: {e}")
            self.status = "error"
            raise

    async def _setup(self):
        """Agent-specific setup logic"""
        # Initialize risk parameters
        self.logger.info(f"Risk limits: {self.risk_limits}")

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """Background risk monitoring loop"""
        while self.status == "ready":
            try:
                await self._monitor_positions()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                break

    async def _monitor_positions(self):
        """Monitor all positions for risk violations"""
        for position_id, position_data in self.position_monitor.items():
            try:
                risk_assessment = await self._assess_position_risk(position_data)
                if risk_assessment["risk_level"] == "high":
                    await self._generate_alert(position_id, risk_assessment)
            except Exception as e:
                self.logger.error(f"Error monitoring position {position_id}: {e}")

    async def _assess_risk(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a position"""
        self.metrics["risk_checks"] += 1

        position_id = position.get("id", "unknown")
        symbol = position.get("symbol", "UNKNOWN")
        size = position.get("size", 0)
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", entry_price)

        # Calculate current P&L
        pnl = (current_price - entry_price) * size
        pnl_pct = (pnl / (entry_price * abs(size))) if entry_price and size else 0

        # Assess risk level
        risk_level = "low"
        risk_factors = []

        # Check drawdown
        if pnl_pct < -self.config.max_drawdown:
            risk_level = "high"
            risk_factors.append(f"Drawdown exceeds limit: {pnl_pct:.2%} > {self.config.max_drawdown:.2%}")

        # Check position size
        portfolio_value = position.get("portfolio_value", 100000)  # Default portfolio value
        position_value = abs(size * current_price)
        position_pct = position_value / portfolio_value if portfolio_value else 0

        if position_pct > self.config.position_limit:
            risk_level = "high" if risk_level != "high" else risk_level
            risk_factors.append(f"Position size exceeds limit: {position_pct:.2%} > {self.config.position_limit:.2%}")

        # Update position monitor
        self.position_monitor[position_id] = {
            **position,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "risk_level": risk_level,
            "last_check": datetime.now()
        }

        return {
            "position_id": position_id,
            "symbol": symbol,
            "risk_level": risk_level,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "risk_factors": risk_factors,
            "recommendations": self._get_risk_recommendations(risk_level, risk_factors)
        }

    def _get_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []

        if risk_level == "high":
            recommendations.append("Consider reducing position size")
            recommendations.append("Implement stop-loss order")
            recommendations.append("Monitor position closely")
        elif risk_level == "medium":
            recommendations.append("Monitor position")
            recommendations.append("Consider partial profit taking")

        return recommendations

    async def _set_risk_limit(self, limit_type: str, value: float) -> Dict[str, Any]:
        """Set a risk limit"""
        if limit_type in self.risk_limits:
            old_value = self.risk_limits[limit_type]
            self.risk_limits[limit_type] = value
            self.logger.info(f"Updated {limit_type}: {old_value} -> {value}")

            return {
                "limit_type": limit_type,
                "old_value": old_value,
                "new_value": value,
                "status": "updated"
            }
        else:
            return {"error": f"Unknown limit type: {limit_type}"}

    async def _generate_alert(self, position_id: str, risk_assessment: Dict[str, Any]):
        """Generate a risk alert"""
        alert = {
            "id": len(self.alerts) + 1,
            "timestamp": datetime.now().isoformat(),
            "position_id": position_id,
            "risk_level": risk_assessment["risk_level"],
            "risk_factors": risk_assessment["risk_factors"],
            "recommendations": risk_assessment["recommendations"]
        }

        self.alerts.append(alert)
        self.metrics["alerts_generated"] += 1

        self.logger.warning(f"Risk alert generated for position {position_id}: {risk_assessment['risk_level']}")

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        try:
            self.metrics["messages_processed"] += 1
            self.logger.debug(f"Processing message: {message.get('type', 'unknown')}")

            result = await self._handle_message(message)

            return {
                "status": "success",
                "agent_id": self.config.agent_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "agent_id": self.config.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_message(self, message: Dict[str, Any]) -> Any:
        """Agent-specific message handling"""
        # Handle risk management
        msg_type = message.get("type")
        if msg_type == "check_risk":
            return await self._assess_risk(message.get("position"))
        elif msg_type == "set_limit":
            return await self._set_risk_limit(message.get("limit_type"), message.get("value"))
        elif msg_type == "get_alerts":
            return {"alerts": self.alerts}
        elif msg_type == "get_limits":
            return {"limits": self.risk_limits}

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "monitored_positions": len(self.position_monitor),
            "active_alerts": len([a for a in self.alerts if a.get("resolved", False) == False]),
            "risk_limits": self.risk_limits
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> RiskGuardian:
    """Factory function to create RiskGuardian instance"""
    return RiskGuardian(config)


# Export the agent class
__all__ = ["RiskGuardian", "RiskGuardianConfig", "create_agent"]
