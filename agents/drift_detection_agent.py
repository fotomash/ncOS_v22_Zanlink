"""Embedding drift detection agent."""

from __future__ import annotations

import logging
from math import sqrt
from typing import Any, Dict, List

from agents.ncos_base_agent import NCOSBaseAgent


class DriftDetectionAgent(NCOSBaseAgent):
    """Detect drift in incoming vector embeddings."""

    def __init__(self, orchestrator: Any, config: Dict[str, Any] | None = None) -> None:
        super().__init__(orchestrator, config)
        self.agent_id = "drift_detection_agent"
        self.drift_threshold: float = float(self.config.get("drift_threshold", 1.0))
        self.history_size: int = int(self.config.get("history_size", 20))
        self.embedding_history: List[List[float]] = []
        self.logger = logging.getLogger(self.agent_id)
        self.register_trigger("embedding.generated", self._on_embedding)

    async def _on_embedding(self, payload: Dict[str, Any], session_state: Dict[str, Any]) -> None:
        embedding = payload.get("embedding")
        if embedding is None:
            return
        self.embedding_history.append(embedding)
        if len(self.embedding_history) > self.history_size:
            self.embedding_history.pop(0)
        if len(self.embedding_history) < 2:
            return
        prev = self.embedding_history[-2]
        drift_value = self._calculate_drift(prev, embedding)
        if drift_value > self.drift_threshold:
            await self.orchestrator.route_trigger(
                "drift.detected",
                {"drift": drift_value, "source": payload.get("key")},
                session_state,
            )

    @staticmethod
    def _calculate_drift(a: List[float], b: List[float]) -> float:
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def get_status(self) -> Dict[str, Any]:
        return {
            "history_size": len(self.embedding_history),
            "threshold": self.drift_threshold,
        }
