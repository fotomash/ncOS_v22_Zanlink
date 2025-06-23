


class SemanticDecisionSupportAgent:
    def __init__(self, context=None):
        self.context = context or {}

    def analyze(self):
        symbol = self.context.get("symbol", "UNKNOWN")
        wyckoff_phase = self.context.get("wyckoff_result", {}).get("phase_class", "N/A")
        macro_bias = self.context.get("macro_snapshot", {}).get(symbol, {}).get("bias", "neutral")
        return {
            "symbol": symbol,
            "wyckoff_phase": wyckoff_phase,
            "macro_bias": macro_bias,
            "combined_interpretation": f"{symbol} is in Wyckoff phase {wyckoff_phase} with macro bias {macro_bias}."
        }

    def __repr__(self):
        return "<SemanticDecisionSupportAgent>"