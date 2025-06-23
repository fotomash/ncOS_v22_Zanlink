"""Knowledge Intelligence agent for simple knowledge lookups."""
import logging
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class KnowledgeIntelligenceConfig:
    agent_id: str = "knowledge_intelligence"
    enabled: bool = True
    log_level: str = "INFO"


class KnowledgeIntelligence:
    """Provide minimal knowledge base lookup functionality."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = KnowledgeIntelligenceConfig(**(config or {}))
        self.logger = logging.getLogger(self.config.agent_id)
        self.logger.setLevel(getattr(logging, self.config.log_level))

    async def query(self, question: str) -> Dict[str, Any]:
        """Return a canned response for a question."""
        self.logger.debug("Received question: %s", question)
        return {"question": question, "answer": "unknown"}
