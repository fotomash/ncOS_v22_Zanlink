import asyncio

from agents.drift_detection_agent import DriftDetectionAgent
from agents.entry_executor_smc import EntryExecutorSMCAgent
from agents.liquidity_sniper import LiquiditySniperAgent


class DummyOrchestrator:
    def __init__(self):
        self.calls = []

    async def route_trigger(self, name, payload, state):
        self.calls.append((name, payload, state))


def test_liquidity_sniper_trigger():
    orch = DummyOrchestrator()
    agent = LiquiditySniperAgent(orch, {})
    asyncio.run(agent.handle_trigger("liquidity_pool_identified", {"level": 1}, {}))
    assert orch.calls == [("liquidity_sniper.pool_identified", {"level": 1}, {})]


def test_entry_executor_trigger():
    orch = DummyOrchestrator()
    agent = EntryExecutorSMCAgent(orch, {})
    asyncio.run(agent.handle_trigger("precision_entry", {"symbol": "TEST"}, {}))
    assert orch.calls == [("execution.entry.submitted", {"symbol": "TEST"}, {})]


def test_drift_detection_trigger():
    orch = DummyOrchestrator()
    agent = DriftDetectionAgent(orch, {"drift_threshold": 0.5, "history_size": 2})
    asyncio.run(agent.handle_trigger("embedding.generated", {"embedding": [0.0, 0.0]}, {}))
    asyncio.run(agent.handle_trigger("embedding.generated", {"embedding": [1.0, 0.0]}, {}))
    assert orch.calls == [("drift.detected", {"drift": 1.0, "source": None}, {})]
