import pytest

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initializes correctly"""
    from core.orchestrators import UnifiedOrchestrator

    config = {"engines": {"market_structure": {"enabled": True}}}
    orch = UnifiedOrchestrator(config)

    assert orch is not None
    assert orch.active_signals == []
