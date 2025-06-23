import pytest
from zanflow.modules.predictive_engine.predictive_scorer import PredictiveScorer

# Sample config simulating YAML agent profile
sample_config = {
    "enabled": True,
    "min_score_to_emit": 0.65,
    "grade_thresholds": {
        "A": 0.90,
        "B": 0.75,
        "C": 0.65
    },
    "factor_weights": {
        "htf_bias": 0.20,
        "idm_detected": 0.10,
        "sweep_validated": 0.15,
        "choch_confirmed": 0.15,
        "poi_validated": 0.20,
        "tick_density": 0.10,
        "spread_status": 0.10
    },
    "conflict_alerts": {
        "enabled": True,
        "min_conflict_score": 0.72
    },
    "audit_trail": {}
}

def test_high_quality_features_should_score_high():
    features = {
        "htf_bias": 1.0,
        "idm_detected": 1.0,
        "sweep_validated": 0.9,
        "choch_confirmed": 0.8,
        "poi_validated": 1.0,
        "tick_density": 0.9,
        "spread_status": 0.95
    }
    scorer = PredictiveScorer(sample_config)
    result = scorer.score(features)

    assert result.maturity_score >= 0.75
    assert result.grade in ("A", "B")
    assert result.potential_entry is True
    assert isinstance(result.rejection_risks, list)

def test_conflict_detection_should_tag_if_direction_opposes():
    features = {
        "htf_bias": -1.0,  # current setup is bearish
        "idm_detected": 0.7,
        "sweep_validated": 0.8,
        "choch_confirmed": 0.8,
        "poi_validated": 0.9,
        "tick_density": 0.85,
        "spread_status": 0.8
    }
    context = {
        "htf_bias": 1.0  # active trade is bullish
    }

    scorer = PredictiveScorer(sample_config)
    result = scorer.score(features, context)

    assert result.conflict_tag is True
