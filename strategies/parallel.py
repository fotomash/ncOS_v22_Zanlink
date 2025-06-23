#!/usr/bin/env python3
"""
NCOS v24 - Parallel Execution Strategy
A strategy for processing a task concurrently across multiple agents.
"""

import asyncio
from typing import Dict, Any, List
from .base import Strategy

class ParallelStrategy(Strategy):
    """
    Executes a task in parallel across multiple agents. All agents process
    the same task simultaneously, and their results are aggregated.
    """

    async def execute(self, task: Dict[str, Any], agent_ids: List[str]) -> Any:
        """
        Executes the task in parallel across the specified agents.

        Args:
            task: The task to execute.
            agent_ids: A list of agent IDs to run in parallel.

        Returns:
            An aggregated dictionary of results from all agents.
        """
        valid_agents = self.validate_agents(agent_ids)
        if not valid_agents:
            raise ValueError("No valid agents available for parallel execution.")

        self.logger.info(f"Executing task in parallel across {len(valid_agents)} agents.")

        # Create a list of coroutines to be executed concurrently
        tasks_to_run = []
        for agent_id in valid_agents:
            agent = self.orchestrator.agents[agent_id]
            task_payload = task.get("payload", {})
            tasks_to_run.append(agent.process(task_payload))

        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        # Collect and structure the results
        agent_results = {}
        for agent_id, result in zip(valid_agents, results):
            if isinstance(result, Exception):
                self.logger.error(f"Agent '{agent_id}' failed during parallel execution: {result}")
                agent_results[agent_id] = {"error": str(result)}
            else:
                agent_results[agent_id] = result

        return self._aggregate_results(agent_results, task)

    def _aggregate_results(self, results: Dict[str, Any], task: Dict[str, Any]) -> Any:
        """
        Aggregates results from multiple agents based on the task's configuration.

        Args:
            results: A dictionary of results, keyed by agent ID.
            task: The original task, which may contain aggregation settings.

        Returns:
            The aggregated result.
        """
        aggregation_method = task.get("aggregation", "all") # Default to returning all results

        if aggregation_method == "all":
            return results
        elif aggregation_method == "first_success":
            for agent_id, result in results.items():
                if "error" not in result:
                    return {agent_id: result}
            return {"error": "All agents failed to produce a successful result."}
        else:
            self.logger.warning(f"Unknown aggregation method '{aggregation_method}'. Returning all results.")
            return results
