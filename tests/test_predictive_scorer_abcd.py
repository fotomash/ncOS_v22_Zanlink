import importlib.util
from pathlib import Path
import pytest

SCORER_PATH = Path(__file__).resolve().parents[1] / "NCOS_Phoenix_Ultimate_v21.7" / "core" / "engines" / "predictive_scorer.py"
spec = importlib.util.spec_from_file_location("predictive_scorer", SCORER_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PredictiveScorer = module.PredictiveScorer

sample_config = {
    "enabled": True,
    "min_score_to_emit": 0.65,
    "grade_thresholds": {"A": 0.90, "B": 0.75, "C": 0.65},
    "factor_weights": {
        "htf_bias": 0.20,
        "idm_detected": 0.10,
        "sweep_validated": 0.15,
        "choch_confirmed": 0.15,
        "poi_validated": 0.20,
        "tick_density": 0.10,
        "spread_status": 0.10,
    },
    "conflict_alerts": {"enabled": False},
    "audit_trail": {},
}

@pytest.fixture(scope="module")
def scorer():
    return PredictiveScorer(sample_config)

def test_grade_a(scorer):
    features = {k: 1.0 for k in sample_config["factor_weights"]}
    result = scorer.score(features)
    assert result.grade == "A"

def test_grade_b(scorer):
    features = {k: 0.75 for k in sample_config["factor_weights"]}
    result = scorer.score(features)
    assert result.grade == "B"

def test_grade_c(scorer):
    features = {k: 0.66 for k in sample_config["factor_weights"]}
    result = scorer.score(features)
    assert result.grade == "C"

def test_grade_d(scorer):
    features = {k: 0.4 for k in sample_config["factor_weights"]}
    result = scorer.score(features)
    assert result.grade == "D"
