import logging
import os
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _mock_pd_np(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pandas', types.ModuleType('pandas'))
    monkeypatch.setitem(sys.modules, 'numpy', types.ModuleType('numpy'))


from production_logging import configure_production_logging
from risk_guardian_agent import RiskGuardianAgent


def test_contribution_logging(tmp_path):
    configure_production_logging(log_dir=str(tmp_path), log_level="INFO")
    agent = RiskGuardianAgent({})
    risk_assessment = {
        'position_size_risk': {'risk_score': 100},
        'exposure_risk': {'risk_score': 80},
        'correlation_risk': {'risk_score': 0},
        'drawdown_risk': {'risk_score': 0},
        'daily_loss_risk': {'risk_score': 0},
        'market_risk': {'risk_score': 0},
    }
    score = agent._calculate_overall_risk_score(risk_assessment)
    assert score > 20
    logging.shutdown()
    log_path = os.path.join(str(tmp_path), "ncos_app.log")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        contents = f.read()
    assert "position_size_risk contribution" in contents
