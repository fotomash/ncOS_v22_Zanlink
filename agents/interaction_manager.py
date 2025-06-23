"""Simple Interaction Manager agent for handling user interactions."""
import logging
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class InteractionManagerConfig:
    agent_id: str = "interaction_manager"
    enabled: bool = True
    log_level: str = "INFO"


class InteractionManager:
    """Minimal interaction management agent."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = InteractionManagerConfig(**(config or {}))
        self.logger = logging.getLogger(self.config.agent_id)
        self.logger.setLevel(getattr(logging, self.config.log_level))

    async def handle(self, message: str) -> Dict[str, Any]:
        """Return a simple acknowledgement for the provided message."""
        self.logger.debug("Handling message: %s", message)
        return {"ack": True, "message": message}
