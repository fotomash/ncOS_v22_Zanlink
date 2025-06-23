# agent_riskmanager.py

class RiskManagerAgent:
    def __init__(self, context=None):
        self.context = context or {}
        self.symbol = self.context.get("symbol", "XAUUSD")
        self.micro_context = self.context.get("micro_context", {})
        self.confidence = self.context.get("confidence", 0.0)
        self.spread = None
        self.ret = None
        self.tick_count = len(self.micro_context) if hasattr(self.micro_context, "__len__") else 0

        if hasattr(self.micro_context, "empty") and not self.micro_context.empty:
            last = self.micro_context.iloc[-1]
            self.spread = last.get("SPREAD", None)
            self.ret = last.get("RET", None)

    def evaluate_risk_profile(self):
        """
        Evaluate risk classification based on current tick volatility and spread.
        """
        result = {
            "symbol": self.symbol,
            "risk": "medium",
            "volatility": "normal",
            "spread": self.spread,
            "ret": self.ret,
            "reason": ""
        }

        if self.spread is None or self.ret is None:
            result["reason"] = "No tick context available"
            return result

        # Spread-based risk
        if self.spread > 0.5:
            result["risk"] = "high"
            result["reason"] += "Spread above 0.5. "
        elif self.spread < 0.2:
            result["risk"] = "low"
            result["reason"] += "Spread under 0.2. "

        # Volatility assessment
        if abs(self.ret) > 0.001:
            result["volatility"] = "spiky"
            result["risk"] = "high"
            result["reason"] += "High volatility detected. "
        elif abs(self.ret) < 0.0003:
            result["volatility"] = "compressed"
            if result["risk"] != "high":
                result["risk"] = "low"
            result["reason"] += "Low volatility zone. "

        return result