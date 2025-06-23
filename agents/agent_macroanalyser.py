

class MacroAnalyzerAgent:
    def __init__(self, context=None):
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.wyckoff_result = self.context.get("wyckoff_result", {})
        self.macro_snapshot = self.context.get("macro_snapshot", {})

    def evaluate_macro_bias(self):
        """
        Evaluate the macro and higher timeframe Wyckoff structure for directional bias.
        """
        result = {
            "symbol": self.symbol,
            "bias": "neutral",
            "phase": None,
            "macro_bias": None,
            "reason": ""
        }

        # Extract Wyckoff phase if available
        phase_class = self.wyckoff_result.get("phase_class", None)
        if phase_class:
            result["phase"] = phase_class

        # Infer macro sentiment
        if self.macro_snapshot:
            dxy = self.macro_snapshot.get("DXY", {}).get("bias")
            gold = self.macro_snapshot.get("XAUUSD", {}).get("bias")
            if gold:
                result["macro_bias"] = gold
            elif dxy:
                # If dollar is strong, gold might be weak (inverse)
                result["macro_bias"] = "bearish" if dxy == "bullish" else "bullish"

        # Aggregate directional bias
        if result["phase"] in ["C", "D"] and result["macro_bias"] == "bullish":
            result["bias"] = "bullish"
            result["reason"] = "HTF Phase C/D and macro alignment"
        elif result["phase"] == "E":
            result["bias"] = "neutral"
            result["reason"] = "Late phase - monitor for reversal"
        elif result["macro_bias"] == "bearish":
            result["bias"] = "bearish"
            result["reason"] = "Macro bearish pressure dominates"

        return result