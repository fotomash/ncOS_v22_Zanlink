"""
PortfolioManager - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class PortfolioManagerConfig:
    """Configuration for PortfolioManager"""
    agent_id: str = "portfolio_manager"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    rebalance_frequency: Optional[str] = "daily"
    rebalance_threshold: Optional[float] = 0.05
    max_positions: Optional[int] = 20
    target_allocation: Optional[Dict[str, float]] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


class PortfolioManager:
    """
    PortfolioManager - Portfolio optimization and allocation management
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = PortfolioManagerConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "rebalances_performed": 0,
            "positions_managed": 0
        }

        self.positions = {}
        self.allocations = self.config.target_allocation or {}
        self.performance_metrics = {}

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
        # Initialize portfolio tracking
        self.logger.info(f"Max positions: {self.config.max_positions}")
        self.logger.info(f"Rebalance frequency: {self.config.rebalance_frequency}")

        # Start rebalancing loop if configured
        if self.config.rebalance_frequency == "continuous":
            asyncio.create_task(self._rebalance_loop())

    async def _rebalance_loop(self):
        """Background rebalancing loop"""
        while self.status == "ready":
            try:
                await self._check_rebalance_needed()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Rebalance loop error: {e}")
                break

    async def _check_rebalance_needed(self):
        """Check if rebalancing is needed"""
        current_allocation = self._calculate_current_allocation()

        for symbol, target_pct in self.allocations.items():
            current_pct = current_allocation.get(symbol, 0)
            deviation = abs(current_pct - target_pct)

            if deviation > self.config.rebalance_threshold:
                self.logger.info(f"Rebalancing needed for {symbol}: {current_pct:.2%} vs {target_pct:.2%}")
                await self._rebalance_portfolio(self.allocations)
                break

    def _calculate_current_allocation(self) -> Dict[str, float]:
        """Calculate current portfolio allocation"""
        total_value = sum(pos.get("value", 0) for pos in self.positions.values())

        if total_value == 0:
            return {}

        allocation = {}
        for symbol, position in self.positions.items():
            allocation[symbol] = position.get("value", 0) / total_value

        return allocation

    async def _rebalance_portfolio(self, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Rebalance portfolio to target allocation"""
        self.logger.info("Starting portfolio rebalance")

        current_allocation = self._calculate_current_allocation()
        rebalance_orders = []

        total_value = sum(pos.get("value", 0) for pos in self.positions.values())

        for symbol, target_pct in target_allocation.items():
            current_pct = current_allocation.get(symbol, 0)
            target_value = total_value * target_pct
            current_value = self.positions.get(symbol, {}).get("value", 0)

            if abs(target_value - current_value) > 100:  # Minimum rebalance amount
                order = {
                    "symbol": symbol,
                    "action": "buy" if target_value > current_value else "sell",
                    "amount": abs(target_value - current_value),
                    "reason": "rebalance"
                }
                rebalance_orders.append(order)

        self.metrics["rebalances_performed"] += 1

        return {
            "status": "rebalanced",
            "orders": rebalance_orders,
            "target_allocation": target_allocation,
            "current_allocation": current_allocation
        }

    async def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        total_value = sum(pos.get("value", 0) for pos in self.positions.values())
        total_cost = sum(pos.get("cost_basis", 0) for pos in self.positions.values())

        if total_cost == 0:
            return {"error": "No cost basis available"}

        total_return = (total_value - total_cost) / total_cost

        performance = {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "positions_count": len(self.positions),
            "timestamp": datetime.now().isoformat()
        }

        self.performance_metrics = performance
        return performance

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
        # Handle portfolio operations
        msg_type = message.get("type")
        if msg_type == "rebalance":
            return await self._rebalance_portfolio(message.get("target_allocation"))
        elif msg_type == "get_performance":
            return await self._calculate_performance()
        elif msg_type == "get_positions":
            return {"positions": self.positions}
        elif msg_type == "get_allocation":
            return {"allocation": self._calculate_current_allocation()}

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "positions_count": len(self.positions),
            "target_allocation": self.allocations
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> PortfolioManager:
    """Factory function to create PortfolioManager instance"""
    return PortfolioManager(config)


# Export the agent class
__all__ = ["PortfolioManager", "PortfolioManagerConfig", "create_agent"]
