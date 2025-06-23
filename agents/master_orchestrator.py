"""Master Orchestrator Agent - Production Ready.

This module coordinates registered agents and orchestrates asynchronous
workflows across the ncOS platform. The orchestrator expects a
configuration dictionary where each workflow is described under the
``workflows`` key. A workflow is an ordered list of steps and each step
defines the name of an agent and the action method to execute.

Example configuration::

    config = {
        "workflows": {
            "startup": {
                "steps": [
                    {"agent": "logger", "action": "initialize"},
                    {"agent": "risk_engine", "action": "warm_up"}
                ]
            }
        }
    }

Agents are registered at runtime and then referenced by name in the
workflow definition. After initialization, ``execute_workflow`` can be
used to process each step asynchronously.
"""
import logging
from datetime import datetime
from typing import Dict, Any


class MasterOrchestrator:
    """Coordinate agent workflows.

    **Responsibilities**
    - Manage the lifecycle of registered agents.
    - Execute workflows defined in the provided configuration.
    - Track internal state and expose status information.

    **Lifecycle**
    1. Construct with a configuration dictionary.
    2. Call :meth:`initialize` to prepare the orchestrator for use.
    3. Register agents with :meth:`register_agent`.
    4. Run workflows through :meth:`execute_workflow`.

    **Configuration Expectations**
    ``config`` must contain a ``workflows`` mapping where each workflow has a
    ``steps`` list. Each step specifies ``agent`` and ``action`` keys referring
    to a registered agent and the coroutine method to invoke.

    Example usage::

        orchestrator = MasterOrchestrator(config)
        await orchestrator.initialize()
        orchestrator.register_agent('logger', LoggerAgent())
        orchestrator.register_agent('risk_engine', RiskEngine())
        results = await orchestrator.execute_workflow('startup', {})
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the orchestrator with runtime configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents = {}
        self.workflows = config.get('workflows', {})
        self.state = {'status': 'initialized', 'start_time': datetime.now()}

    async def initialize(self):
        """Finalize setup before orchestrating workflows."""
        self.logger.info("MasterOrchestrator initialized")
        self.state['status'] = 'ready'

    def register_agent(self, name: str, agent: Any):
        """Register an agent instance for use in workflows."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")

    async def execute_workflow(self, workflow_name: str, context: Dict[str, Any]):
        """Run a workflow by name using the provided context."""
        if workflow_name not in self.workflows:
            return {'error': f'Unknown workflow: {workflow_name}'}

        workflow = self.workflows[workflow_name]
        results = {}

        for step in workflow.get('steps', []):
            agent_name = step.get('agent')
            action = step.get('action')

            if agent_name in self.agents:
                agent = self.agents[agent_name]
                if hasattr(agent, action):
                    result = await getattr(agent, action)(context)
                    results[f"{agent_name}.{action}"] = result

        return results

    def get_status(self):
        """Return current orchestrator status and registered components."""
        return {
            'state': self.state,
            'registered_agents': list(self.agents.keys()),
            'workflows': list(self.workflows.keys())
        }
