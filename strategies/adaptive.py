#!/usr/bin/env python3
"""
NCOS v24 - Adaptive Execution Strategy
This module contains the master strategy that dynamically selects the best
execution approach (sequential, parallel, or collaborative) based on the
characteristics of the task and the available agents.
"""
from typing import Dict, Any, List
from .base import Strategy
from .sequential import SequentialStrategy
from .parallel import ParallelStrategy
from .collaborative import CollaborativeStrategy
from ncos.agents.base_agent import BaseAgent


class AdaptiveStrategy(Strategy):
    """
    Dynamically selects and executes the best strategy based on an analysis
    of the task's requirements and the agents' capabilities.
    """

    def __init__(self, orchestrator: "MultiAgentOrchestrator"):
        super().__init__(orchestrator)
        # Initialize sub-strategies, passing the orchestrator reference
        self.sequential = SequentialStrategy(orchestrator)
        self.parallel = ParallelStrategy(orchestrator)
        self.collaborative = CollaborativeStrategy(orchestrator)

    async def execute(self, task: Dict[str, Any], agent_ids: List[str]) -> Any:
        """
        Analyzes the task and routes it to the most appropriate sub-strategy.

        Args:
            task: The task to be executed.
            agents: The list of agent IDs capable of handling the task.

        Returns:
            The result of the execution from the selected strategy.
        """
        selected_strategy = self._select_strategy(task, agent_ids)
        self.logger.info(f"Adaptive strategy selected: '{selected_strategy.__class__.__name__}' for task '{task.get('id', 'N/A')}'.")
        return await selected_strategy.execute(task, agent_ids)

    def _select_strategy(self, task: Dict[str, Any], agent_ids: List[str]) -> Strategy:
        """
        Decision logic for selecting the best execution strategy.

        Args:
            task: The task containing hints about its nature.
            agent_ids: The list of available agents.

        Returns:
            An instance of the selected strategy.
        """
        # --- Strategy Selection Logic ---

        # Rule 1: If the task explicitly requires collaboration or consensus.
        if task.get("requires_consensus", False) or task.get("execution_mode") == "collaborative":
            self.logger.debug("Selecting 'CollaborativeStrategy' due to consensus requirement.")
            return self.collaborative

        # Rule 2: If the task has defined dependencies, it's a pipeline.
        if task.get("dependencies"):
            self.logger.debug("Selecting 'SequentialStrategy' due to task dependencies.")
            return self.sequential
        
        # Rule 3: For time-sensitive tasks with multiple agents, run in parallel.
        if task.get("time_sensitive", False) and len(agent_ids) > 1:
            self.logger.debug("Selecting 'ParallelStrategy' for time-sensitive task.")
            return self.parallel

        # Rule 4: If there is only one agent, it must be sequential.
        if len(agent_ids) == 1:
            self.logger.debug("Selecting 'SequentialStrategy' for single agent task.")
            return self.sequential
            
        # Rule 5: Default to parallel for simple fan-out tasks across many agents.
        if len(agent_ids) > 3:
            self.logger.debug("Defaulting to 'ParallelStrategy' for high agent count.")
            return self.parallel
            
        # Default Fallback: A simple sequential execution is the safest default.
        self.logger.debug("Defaulting to 'SequentialStrategy' as a fallback.")
        return self.sequential

