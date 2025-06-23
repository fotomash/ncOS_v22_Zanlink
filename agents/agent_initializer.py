# agent_initializer.py

from agents.agent_microstrategist import MicroStrategistAgent
from agents.agent_macroanalyzer import MacroAnalyzerAgent
from agents.agent_riskmanager import RiskManagerAgent
from agents.agent_journalist import TradeJournalistAgent
from agents.agent_htfanalyst import HTFPhaseAnalystAgent
from agents.agent_reputationauditor import ReputationAuditorAgent
from core.agent_semanticdss import SemanticDecisionSupportAgent

def initialize_agents(config):
    """
    Initialize all AI agents with proper shared state context from ZANALYTICS core.
    This includes live Wyckoff phase analysis, microstructure detection, macro snapshot,
    indicator profiles, and any real-time signals from orchestration.
    """

    shared_context = {
        "wyckoff_result": config.get("wyckoff_result"),
        "micro_context": config.get("micro_context"),
        "indicator_profiles": config.get("indicator_profiles"),
        "macro_snapshot": config.get("macro_snapshot"),
        "scalp_signal": config.get("scalp_signal"),
        "symbol": config.get("symbol"),
    }

    agents = {}
    if config.get("agents", {}).get("micro_strategist", {}).get("active", True):
        agents["micro_strategist"] = MicroStrategistAgent(context=shared_context)
        print("[INIT] micro_strategist agent initialized.")
    else:
        print("[SKIP] micro_strategist agent deactivated via config.")
    if config.get("agents", {}).get("macro_analyzer", {}).get("active", True):
        agents["macro_analyzer"] = MacroAnalyzerAgent(context=shared_context)
        print("[INIT] macro_analyzer agent initialized.")
    else:
        print("[SKIP] macro_analyzer agent deactivated via config.")
    if config.get("agents", {}).get("risk_manager", {}).get("active", True):
        agents["risk_manager"] = RiskManagerAgent(context=shared_context)
        print("[INIT] risk_manager agent initialized.")
    else:
        print("[SKIP] risk_manager agent deactivated via config.")
    if config.get("agents", {}).get("trade_journalist", {}).get("active", True):
        agents["trade_journalist"] = TradeJournalistAgent(context=shared_context)
        print("[INIT] trade_journalist agent initialized.")
    else:
        print("[SKIP] trade_journalist agent deactivated via config.")
    if config.get("agents", {}).get("htf_phase_analyst", {}).get("active", True):
        agents["htf_phase_analyst"] = HTFPhaseAnalystAgent(context=shared_context)
        print("[INIT] htf_phase_analyst agent initialized.")
    else:
        print("[SKIP] htf_phase_analyst agent deactivated via config.")
    if config.get("agents", {}).get("reputation_auditor", {}).get("active", True):
        agents["reputation_auditor"] = ReputationAuditorAgent(context=shared_context)
        print("[INIT] reputation_auditor agent initialized.")
    else:
        print("[SKIP] reputation_auditor agent deactivated via config.")
    if config.get("agents", {}).get("semantic_dss", {}).get("active", True):
        agents["semantic_dss"] = SemanticDecisionSupportAgent(context=shared_context)
        print("[INIT] semantic_dss agent initialized.")
    else:
        print("[SKIP] semantic_dss agent deactivated via config.")

    return agents