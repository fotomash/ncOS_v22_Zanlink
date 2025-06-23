"""Execution agent for SMC precision entries."""
from typing import Any, Dict

from .ncos_base_agent import NCOSBaseAgent


class EntryExecutorSMCAgent(NCOSBaseAgent):
    """Submit entries when signalled by strategies."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        super().__init__(orchestrator, config)
        self.agent_id = "entry_executor_smc"
        self.register_trigger("precision_entry", self._execute_entry)

    async def _execute_entry(self, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        await self.orchestrator.route_trigger(
            "execution.entry.submitted",
            payload,
            session_state,
        )
