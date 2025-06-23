import json
from pathlib import Path
import types
import pytest

import sys
sys.modules.setdefault('trait_engine', types.ModuleType('trait_engine'))
sys.modules['trait_engine'].merge_config = lambda *a, **k: None
sys.modules.setdefault('copilot_orchestrator', types.ModuleType('copilot_orchestrator'))
sys.modules['copilot_orchestrator'].run_full_analysis = lambda *a, **k: {}
sys.modules.setdefault('advanced_smc_orchestrator', types.ModuleType('advanced_smc_orchestrator'))
sys.modules['advanced_smc_orchestrator'].AdvancedSMCOrchestrator = lambda *a, **k: types.SimpleNamespace(run=lambda: {'pois': []})
sys.modules.setdefault('liquidity_vwap_detector', types.ModuleType('liquidity_vwap_detector'))
sys.modules['liquidity_vwap_detector'].detect_liquidity_sweeps = lambda *a, **k: {}
sys.modules.setdefault('optimizer_loop', types.ModuleType('optimizer_loop'))
sys.modules['optimizer_loop'].run_optimizer_loop = lambda *a, **k: None
sys.modules.setdefault('feedback_analysis_engine', types.ModuleType('feedback_analysis_engine'))
sys.modules['feedback_analysis_engine'].analyze_feedback = lambda *a, **k: {}
sys.modules.setdefault('poi_quality_predictor', types.ModuleType('poi_quality_predictor'))
sys.modules['poi_quality_predictor'].predict_poi_quality = lambda *a, **k: {}

import sys
sys.modules.setdefault('core.intermarket_sentiment', types.ModuleType('core.intermarket_sentiment'))
sys.modules['core.intermarket_sentiment'].snapshot_sentiment = lambda: {}
sys.modules.setdefault('runtime.data_pipeline', types.ModuleType('runtime.data_pipeline'))
sys.modules['runtime.data_pipeline'].DataPipeline = lambda config=None: types.SimpleNamespace(fetch_pairs=lambda: None, resample_htf=lambda: None)
sys.modules.setdefault('pandas', types.ModuleType('pandas'))
sys.modules['pandas'].read_csv = lambda *a, **k: None

import src.ncos.engines.run_zanalytics_session as rz


def test_load_configs_missing(tmp_path):
    cfg = rz.load_configs(tmp_path)
    assert cfg['copilot'] == {}
    assert cfg['chart'] == {}
    assert cfg['strategy'] == {}


def test_initialize_data_failure(monkeypatch):
    class DummyPipeline:
        def fetch_pairs(self):
            raise RuntimeError('fetch fail')
        def resample_htf(self):
            pass
    monkeypatch.setattr(rz, 'DataPipeline', lambda config=None: DummyPipeline())
    result = rz.initialize_data()
    assert result == {}
