

from datetime import datetime

class TradeJournalistAgent:
    def __init__(self, context=None):
        self.context = context or {}
        self.entries = []

    def log_decision(self, strategist_result, macro_result=None, risk_result=None, semantic_result=None):
        """
        Log a decision event including microstructure triggers, macro bias, and risk profile.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": strategist_result.get("symbol"),
            "entry_signal": strategist_result.get("entry_signal"),
            "trigger": strategist_result.get("trigger"),
            "confidence": strategist_result.get("confidence"),
            "reason": strategist_result.get("reason"),
            "phase_context": strategist_result.get("phase_context"),
            "macro_bias": macro_result.get("bias") if macro_result else None,
            "macro_reason": macro_result.get("reason") if macro_result else None,
            "risk_level": risk_result.get("risk") if risk_result else None,
            "volatility": risk_result.get("volatility") if risk_result else None,
            "semantic_summary": semantic_result.get("combined_interpretation") if semantic_result else None,
            "semantic_bias": semantic_result.get("macro_bias") if semantic_result else None,
            "notes": []
        }

        self.entries.append(entry)
        return entry

    def get_latest_entry(self):
        return self.entries[-1] if self.entries else None

    def export_log(self):
        return self.entries

from datetime import datetime


class TradeJournalistAgent:
    def __init__(self):
        self.entries = []

    def summary(self, entry):
        """
        Generate a one-line summary of the decision entry.
        """
        return f"[{entry['timestamp']}] {entry['symbol']} | {entry['trigger']} | Risk: {entry['risk_level']} | Conf: {entry['confidence']} | Bias: {entry['macro_bias']}"

    def log_decision(self, strategist_result, macro_result=None, risk_result=None):
        """
        Log a decision event including microstructure triggers, macro bias, and risk profile.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": strategist_result.get("symbol"),
            "entry_signal": strategist_result.get("entry_signal"),
            "trigger": strategist_result.get("trigger"),
            "confidence": strategist_result.get("confidence"),
            "reason": strategist_result.get("reason"),
            "phase_context": strategist_result.get("phase_context"),
            "macro_bias": macro_result.get("bias") if macro_result else None,
            "macro_reason": macro_result.get("reason") if macro_result else None,
            "risk_level": risk_result.get("risk") if risk_result else None,
            "volatility": risk_result.get("volatility") if risk_result else None,
            "notes": []
        }
        entry["summary"] = self.summary(entry)
        self.entries.append(entry)
        return entry