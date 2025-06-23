"""
CoreSystemAgent - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class CoreSystemAgentConfig:
    """Configuration for CoreSystemAgent"""
    agent_id: str = "core_system_agent"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    heartbeat_interval: Optional[int] = 30
    system_check_interval: Optional[int] = 60
    auto_recovery: Optional[bool] = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


class CoreSystemAgent:
    """
    CoreSystemAgent - Core system orchestration and lifecycle management
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = CoreSystemAgentConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "system_checks": 0,
            "heartbeats": 0
        }

        self.registered_agents = {}
        self.system_health = {}

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
        # Initialize core system components
        self.subsystems = []
        self.health_checks = {}

        # Start background tasks
        if self.config.heartbeat_interval:
            asyncio.create_task(self._heartbeat_loop())

        if self.config.system_check_interval:
            asyncio.create_task(self._system_check_loop())

    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent with the core system"""
        self.registered_agents[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")

    async def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while self.status == "ready":
            try:
                self.metrics["heartbeats"] += 1
                self.logger.debug("System heartbeat")
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break

    async def _system_check_loop(self):
        """Background system health check loop"""
        while self.status == "ready":
            try:
                await self._perform_system_check()
                self.metrics["system_checks"] += 1
                await asyncio.sleep(self.config.system_check_interval)
            except Exception as e:
                self.logger.error(f"System check error: {e}")
                break

    async def _perform_system_check(self):
        """Perform system health check"""
        for agent_name, agent in self.registered_agents.items():
            try:
                if hasattr(agent, 'get_status'):
                    status = agent.get_status()
                    self.system_health[agent_name] = status
                else:
                    self.system_health[agent_name] = {"status": "unknown"}
            except Exception as e:
                self.system_health[agent_name] = {"status": "error", "error": str(e)}

    async def _execute_system_command(self, command: str) -> Dict[str, Any]:
        """Execute system-level command"""
        if command == "status":
            return {
                "system_status": self.status,
                "registered_agents": len(self.registered_agents),
                "health": self.system_health
            }
        elif command == "restart":
            return await self._restart_system()
        elif command == "shutdown":
            return await self._shutdown_system()
        else:
            return {"error": f"Unknown command: {command}"}

    async def _restart_system(self):
        """Restart the system"""
        self.logger.info("System restart initiated")
        # Implementation for system restart
        return {"status": "restart_initiated"}

    async def _shutdown_system(self):
        """Shutdown the system"""
        self.logger.info("System shutdown initiated")
        # Implementation for system shutdown
        return {"status": "shutdown_initiated"}

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
        # Handle system-level commands
        msg_type = message.get("type")
        if msg_type == "health_check":
            return self.get_status()
        elif msg_type == "system_command":
            return await self._execute_system_command(message.get("command"))
        elif msg_type == "register_agent":
            agent_name = message.get("agent_name")
            agent_instance = message.get("agent_instance")
            self.register_agent(agent_name, agent_instance)
            return {"registered": agent_name}

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "registered_agents": list(self.registered_agents.keys()),
            "system_health": self.system_health
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> CoreSystemAgent:
    """Factory function to create CoreSystemAgent instance"""
    return CoreSystemAgent(config)


# Export the agent class
__all__ = ["CoreSystemAgent", "CoreSystemAgentConfig", "create_agent"]
