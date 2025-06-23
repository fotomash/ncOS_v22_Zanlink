    def evaluate_microstructure_phase_trigger(self):
        """
        Evaluate the current microstructure context using a composite scoring model
        to determine if a trade entry signal should be triggered based on 
        Spring, CHoCH, BOS, and volatility patterns.
        """
        result = {
            "entry_signal": False,
            "trigger": None,
            "confidence": 0.0,
            "reason": "",
            "symbol": self.symbol,
            "phase_context": None,
            "volatility": None,
            "execution_mode": "scalp"
        }

        if not self.phase_info or getattr(self.phase_info, "empty", False):
            result["reason"] = "No microstructure data available"
            return result

        last = self.phase_info.iloc[-1]

        # Signals
        spring = last.get("SPRING", False)
        choch = last.get("CHoCH", False)
        bos = last.get("BOS", False)
        phase = last.get("PHASE", "")
        spread = last.get("SPREAD", 0)
        ret = last.get("RET", 0)

        # Weights for scoring logic
        weights = {
            "spring": 1.2,
            "choch_trap": 0.9,
            "bos_confirm": 0.5,
            "spread_compression": 0.3,
            "reversal_tick": 0.2
        }

        score = 0.0
        reason_list = []

        # Scoring logic
        if spring:
            score += weights["spring"]
            reason_list.append("Spring detected")
        if choch and not bos:
            score += weights["choch_trap"]
            reason_list.append("CHoCH without BOS (trap zone)")
        if bos:
            score += weights["bos_confirm"]
            reason_list.append("BOS confirms structure shift")
        if spread < 0.3:
            score += weights["spread_compression"]
            result["volatility"] = "compressed"
        if abs(ret) > 0.0004:
            score += weights["reversal_tick"]
            reason_list.append("strong reversal tick")

        # Entry decision based on cumulative score
        if score >= 1.5:
            result["entry_signal"] = True
            result["confidence"] = min(score / sum(weights.values()), 1.0)
            result["trigger"] = "+".join(reason_list)
            result["reason"] = " + ".join(reason_list)
            result["phase_context"] = phase
        else:
            result["reason"] = "Score too low: " + ", ".join(reason_list)

        result["raw_score"] = score
        if ret > 0:
            result["tick_bias"] = "bullish"
        elif ret < 0:
            result["tick_bias"] = "bearish"
        else:
            result["tick_bias"] = "neutral"

        return result