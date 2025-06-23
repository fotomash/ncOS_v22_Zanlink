import logging
from typing import Any, Dict


class RiskMonitorAgent:
    """Monitor manipulation analysis events and take risk actions."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        self.orchestrator = orchestrator
        self.config = config or {}
        self.agent_id = "risk_monitor_agent"
        self.response_actions = self.config.get("response_actions", {})
        self.logger = logging.getLogger(self.agent_id)

    async def handle_trigger(self, trigger_name: str, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        if trigger_name == "analysis.manipulation.spread_detected":
            await self.assess_risk(payload)

    async def assess_risk(self, event_payload: Dict[str, Any]) -> None:
        severity = event_payload.get("severity", 0)
        action = "log_event"
        if severity >= self.response_actions.get("critical_level", 9):
            action = self.response_actions.get("critical", "emergency_exit")
        elif severity >= self.response_actions.get("high_level", 7):
            action = self.response_actions.get("high", "reduce_position")
        elif severity >= self.response_actions.get("medium_level", 4):
            action = self.response_actions.get("medium", "alert_trader")
        self.logger.warning(
            "RISK MONITOR: Severity %s detected. Action: %s", severity, action.upper()
        )
        await self.orchestrator.route_trigger(
            "action.risk.execution",
            {"action": action, "source_event": event_payload},
            {},
        )
