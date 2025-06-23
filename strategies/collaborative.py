#!/usr/bin/env python3
"""
NCOS v24 - Collaborative Execution Strategy
A strategy where agents work together, review each other's work,
and build upon it to reach a consensus or a more refined result.
"""

import asyncio
from typing import Dict, Any, List, Optional
from .base import Strategy

class CollaborativeStrategy(Strategy):
    """
    Executes a task through agent collaboration over multiple rounds.
    Agents can generate proposals, review them, and refine their work based on feedback.
    """

    async def execute(self, task: Dict[str, Any], agent_ids: List[str]) -> Any:
        """
        Manages a collaborative session to solve a complex task.

        Args:
            task: The task dictionary, which should include collaboration settings.
            agent_ids: List of agent IDs participating in the collaboration.

        Returns:
            The final result of the collaboration, ideally a consensus.
        """
        valid_agents = self.validate_agents(agent_ids)
        if not valid_agents:
            raise ValueError("No valid agents for collaborative execution.")

        max_rounds = task.get("max_collaboration_rounds", 3)
        self.logger.info(f"Starting collaborative session for task '{task.get('id')}' with {len(valid_agents)} agents for {max_rounds} rounds.")

        collaboration_history = []
        task_payload = task.get("payload", {})
        
        for round_num in range(max_rounds):
            self.logger.debug(f"Collaboration Round {round_num + 1}/{max_rounds}")

            # 1. Proposal Phase: Each agent generates an initial proposal.
            proposals = await self._generate_proposals(valid_agents, task_payload, collaboration_history)
            collaboration_history.append({"round": round_num, "proposals": proposals})

            # 2. Review & Feedback Phase: Agents review others' proposals.
            # This is a simplified version. A full implementation would pass all proposals
            # to all agents for review.
            
            # For this example, we'll consider the last set of proposals as the final result.
            # A more advanced implementation would follow.
            
        # In a more advanced version, a consensus mechanism would be applied here.
        final_result = self._reach_consensus(proposals)
        
        self.logger.success("Collaborative session concluded.")
        return final_result

    async def _generate_proposals(self, agent_ids: List[str], task_data: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:
        """Asks each agent to generate a proposal for the task."""
        
        proposal_tasks = []
        for agent_id in agent_ids:
            agent = self.orchestrator.agents[agent_id]
            # Augment task data with history for context
            contextual_task_data = {
                **task_data,
                "collaboration_history": history
            }
            proposal_tasks.append(agent.process(contextual_task_data))
            
        results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
        
        proposals = {}
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                self.logger.error(f"Agent '{agent_id}' failed during proposal phase: {result}")
                proposals[agent_id] = {"error": str(result)}
            else:
                proposals[agent_id] = result
        return proposals

    def _reach_consensus(self, proposals: Dict[str, Any]) -> Any:
        """
        A simple consensus mechanism. In a real system, this could involve
        voting, scoring, or another agent dedicated to synthesizing results.
        
        For now, it returns the result from the first successful agent.
        """
        for agent_id, result in proposals.items():
            if "error" not in result:
                return {"consensus_result": result, "chosen_agent": agent_id}
                
        return {"error": "Failed to reach a consensus. All agents returned errors."}

