#!/usr/bin/env python3
"""
NCOS v24 - Base Execution Strategy
Defines the abstract base class for all execution strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from loguru import logger

# Forward reference for type hinting
class MultiAgentOrchestrator:
    pass

class Strategy(ABC):
    """
    Abstract base class for defining how a task is executed across one
    or more agents.
    """

    def __init__(self, orchestrator: "MultiAgentOrchestrator"):
        """
        Initializes the strategy with a reference to the orchestrator.

        Args:
            orchestrator: The main orchestrator instance.
        """
        self.orchestrator = orchestrator
        self.logger = logger.bind(name=self.__class__.__name__)

    @abstractmethod
    async def execute(self, task: Dict[str, Any], agent_ids: List[str]) -> Any:
        """
        The core execution logic for the strategy.

        Args:
            task: The task dictionary to be executed.
            agent_ids: A list of agent IDs selected to perform the task.

        Returns:
            The result of the execution.
        """
        pass

    def validate_agents(self, agent_ids: List[str]) -> List[str]:
        """
        Filters a list of agent IDs to ensure they are registered and available.

        Args:
            agent_ids: The list of agent IDs to validate.

        Returns:
            A list of valid and available agent IDs.
        """
        valid_agents = [
            agent_id for agent_id in agent_ids 
            if agent_id in self.orchestrator.agents
        ]
        
        if len(valid_agents) != len(agent_ids):
            invalid_ids = set(agent_ids) - set(valid_agents)
            self.logger.warning(f"Could not find registered agents: {invalid_ids}")
            
        return valid_agents
