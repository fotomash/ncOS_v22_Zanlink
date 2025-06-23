"""NCOS base agent with async trigger support."""
import logging
from typing import Any, Awaitable, Callable, Dict


class NCOSBaseAgent:
    """Minimal asynchronous agent base class."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.agent_id = self.config.get("agent_id", self.__class__.__name__.lower())
        self.logger = logging.getLogger(self.agent_id)
        self.trigger_handlers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[None]]] = {}

    async def handle_trigger(self, trigger_name: str, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        """Dispatch trigger to registered handler if available."""
        handler = self.trigger_handlers.get(trigger_name)
        if handler:
            await handler(payload, session_state)
        else:
            self.logger.debug("No handler for trigger %s", trigger_name)

    def register_trigger(self, trigger_name: str,
                         handler: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a handler for a trigger name."""
        self.trigger_handlers[trigger_name] = handler
