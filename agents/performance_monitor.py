from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import psutil  # pragma: no cover - optional
except Exception:  # pragma: no cover - fallback when psutil unavailable
    psutil = None  # type: ignore
import resource


def _summarize_workspace_memory(state: Any) -> Dict[str, Any]:
    memory_mb = getattr(state, "memory_usage_mb", 0.0)
    processed_files = getattr(state, "processed_files", [])
    active_agents = getattr(state, "active_agents", [])
    trading_signals = getattr(state, "trading_signals", [])

    return {
        "memory_usage_mb": float(memory_mb),
        "processed_files": len(processed_files),
        "active_agents": len(active_agents),
        "trading_signals": len(trading_signals),
    }


async def _optimize_memory(state: Any) -> None:
    files = getattr(state, "processed_files", [])
    max_files = 100
    if len(files) > max_files:
        del files[: len(files) - max_files]
    usage = getattr(state, "memory_usage_mb", 0.0)
    if usage > 1024 and hasattr(state, "memory_usage_mb"):
        state.memory_usage_mb = 1024.0


class PerformanceMonitor:
    """Collect runtime performance statistics."""

    def __init__(self, orchestrator: Any, session_state: Any, interval: int = 300) -> None:
        self.orchestrator = orchestrator
        self.session_state = session_state
        self.interval = interval
        self.active = False
        self.task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_report: Dict[str, Any] = {}

    async def start(self) -> None:
        """Start periodic monitoring."""
        if not self.active:
            self.active = True
            self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop periodic monitoring."""
        self.active = False
        if self.task:
            self.task.cancel()
            self.task = None

    async def _loop(self) -> None:
        while self.active:
            try:
                await self.collect_report()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - resilience
                self.logger.error("Performance monitor error: %s", exc)
                await asyncio.sleep(self.interval)

    async def collect_report(self) -> Dict[str, Any]:
        """Collect and store a performance report."""
        memory_mb = self._get_process_memory_mb()
        workspace = _summarize_workspace_memory(self.session_state)
        workspace["memory_usage_mb"] = memory_mb

        vector_stats = {}
        if getattr(self.orchestrator, "vector_engine", None):
            try:
                vector_stats = self.orchestrator.vector_engine.get_vector_store_stats()
            except Exception as exc:  # pragma: no cover - best effort
                vector_stats = {"error": str(exc)}

        agent_stats = {}
        if getattr(self.orchestrator, "agents", None):
            for name, agent in self.orchestrator.agents.items():
                if hasattr(agent, "get_status"):
                    try:
                        agent_stats[name] = agent.get_status()
                    except Exception as exc:  # pragma: no cover - individual errors
                        agent_stats[name] = {"error": str(exc)}

        self.last_report = {
            "timestamp": datetime.now().isoformat(),
            "memory": workspace,
            "vector_store": vector_stats,
            "agents": agent_stats,
        }

        await _optimize_memory(self.session_state)
        return self.last_report

    def get_report(self) -> Dict[str, Any]:
        """Return the last collected report."""
        return self.last_report

    def _get_process_memory_mb(self) -> float:
        if psutil:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage) / 1024


__all__ = ["PerformanceMonitor"]
