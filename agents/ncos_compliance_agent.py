import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ComplianceAgent:
    """Log critical events for regulatory compliance."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.agent_id = "compliance_agent"
        self.log_file = Path("logs/manipulation_compliance_log.jsonl")
        self.logger = logging.getLogger(self.agent_id)

    async def handle_trigger(self, trigger_name: str, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        if trigger_name.startswith("analysis.manipulation."):
            await self.log_compliance_event(trigger_name, payload)

    async def log_compliance_event(self, trigger_name: str, event_payload: Dict[str, Any]) -> None:
        record = {
            "log_timestamp": datetime.now().isoformat(),
            "event_type": trigger_name,
            "event_details": event_payload,
            "agent_id": self.agent_id,
        }
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.logger.info("Compliance event logged: %s", trigger_name)
        except Exception as exc:  # pragma: no cover - logging
            self.logger.error("Failed to write compliance log: %s", exc)
