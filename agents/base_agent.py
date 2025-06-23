#!/usr/bin/env python3
"""
NCOS v24 - Base Agent
This module defines the abstract base class for all agents in the system.
It establishes a common interface and lifecycle for every agent, ensuring
that the orchestrator can manage them in a standardized way.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from loguru import logger

from ncos.core.base import BaseComponent
from ncos.core.memory.manager import MemoryManager

class BaseAgent(BaseComponent, ABC):
    """
    The abstract base class for all specialized agents in the NCOS.

    Each agent must have a unique ID, a defined set of capabilities, and
    the logic to handle specific task types.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any], memory_manager: MemoryManager):
        """
        Initializes the agent.

        Args:
            agent_id: A unique identifier for this agent instance.
            config: The configuration dictionary for this agent.
            memory_manager: A reference to the central MemoryManager.
        """
        super().__init__(config)
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        
        # Core attributes defined from config
        self.capabilities: Set[str] = set(config.get("capabilities", []))
        self.task_types: Set[str] = set(config.get("task_types", ["general"]))
        
        # State and performance tracking
        self.state: Dict[str, Any] = {"status": "uninitialized"}
        self.performance_metrics = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "avg_execution_time_ms": 0.0,
            "last_active_timestamp": None
        }

    async def initialize(self) -> bool:
        """Initializes the agent and loads its persistent state."""
        self.logger.info(f"Initializing agent '{self.agent_id}'...")
        # Load persistent state from memory manager
        saved_state = await self.memory_manager.get_state({"key": f"agent_state_{self.agent_id}"})
        if saved_state:
            self.state = saved_state
            self.logger.info(f"Loaded persistent state for agent '{self.agent_id}'.")
        
        self.state["status"] = "idle"
        self.is_initialized = True
        self.logger.success(f"Agent '{self.agent_id}' initialized successfully.")
        return True

    def can_handle(self, task_type: str) -> bool:
        """
        Checks if the agent is capable of handling a given task type.

        Args:
            task_type: The type of the task to check.

        Returns:
            True if the agent can handle the task, False otherwise.
        """
        return "general" in self.task_types or task_type in self.task_types

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        The main processing logic for the agent. This method must be implemented
        by all concrete agent classes.

        Args:
            data: The input data payload for the task.

        Returns:
            The result of the agent's processing.
        """
        pass

    async def save_state(self):
        """Saves the agent's current state to the memory manager."""
        self.logger.debug(f"Saving state for agent '{self.agent_id}'...")
        await self.memory_manager.set_state({
            "key": f"agent_state_{self.agent_id}",
            "value": self.state
        })

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status and performance metrics of the agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "is_initialized": self.is_initialized,
            "state": self.state,
            "capabilities": list(self.capabilities),
            "handled_task_types": list(self.task_types),
            "performance": self.performance_metrics
        }
    
    async def cleanup(self):
        """Saves the final state before shutting down."""
        self.logger.info(f"Cleaning up agent '{self.agent_id}'...")
        self.state['status'] = 'shutdown'
        await self.save_state()
        await super().cleanup()

