"""Simple liquidity sniper agent."""
from typing import Any, Dict

from .ncos_base_agent import NCOSBaseAgent


class LiquiditySniperAgent(NCOSBaseAgent):
    """Detect liquidity pools and forward events."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        super().__init__(orchestrator, config)
        self.agent_id = "liquidity_sniper"
        self.register_trigger("liquidity_pool_identified", self._on_pool)

    async def _on_pool(self, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        await self.orchestrator.route_trigger(
            "liquidity_sniper.pool_identified",
            payload,
            session_state,
        )
