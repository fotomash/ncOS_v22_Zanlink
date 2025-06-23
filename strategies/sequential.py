#!/usr/bin/env python3
"""
NCOS v24 - Sequential Execution Strategy
A strategy for executing a task as a pipeline, where the output of one
agent becomes the input for the next.
"""

from typing import Dict, Any, List
from .base import Strategy

class SequentialStrategy(Strategy):
    """
    Executes a task sequentially across a list of agents, forming a data pipeline.
    """

    async def execute(self, task: Dict[str, Any], agent_ids: List[str]) -> Any:
        """
        Executes the task by passing the result from one agent to the next.

        Args:
            task: The initial task to execute.
            agent_ids: An ordered list of agent IDs defining the pipeline.

        Returns:
            The result from the final agent in the pipeline.
        """
        valid_agents = self.validate_agents(agent_ids)
        if not valid_agents:
            raise ValueError("No valid agents available for sequential execution.")

        self.logger.info(f"Executing task sequentially across {len(valid_agents)} agents: {valid_agents}")

        current_data = task.get("payload", {})
        
        for agent_id in valid_agents:
            agent = self.orchestrator.agents[agent_id]
            self.logger.debug(f"Pipeline step: Passing data to agent '{agent_id}'")
            try:
                current_data = await agent.process(current_data)
            except Exception as e:
                self.logger.error(f"Sequential execution failed at agent '{agent_id}': {e}")
                raise  # Re-raise the exception to halt the pipeline

        self.logger.success(f"Sequential execution completed. Final result from '{valid_agents[-1]}'.")
        return current_data
