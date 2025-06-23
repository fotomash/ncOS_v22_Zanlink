from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ScoringResult:
    maturity_score: float
    grade: str
    potential_entry: bool
    rejection_risks: list[str]
    confidence_factors: Dict[str, str]
    next_killzone_check: str | None
    conflict_tag: bool

class PredictiveScorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get("factor_weights", {})
        self.thresholds = config.get("grade_thresholds", {})
        self.min_score_to_emit = config.get("min_score_to_emit", 0.65)
        self.conflict_config = config.get("conflict_alerts", {})
        self.audit_config = config.get("audit_trail", {})

    def score(self, features: Dict[str, float], context: Dict[str, Any] = None) -> ScoringResult:
        score = 0.0
        for key, weight in self.weights.items():
            score += weight * features.get(key, 0.0)

        maturity_score = round(score, 4)
        grade = self._grade(maturity_score)
        potential_entry = maturity_score >= self.min_score_to_emit
        conflict_tag = self._detect_conflict(maturity_score, features, context or {})

        return ScoringResult(
            maturity_score=maturity_score,
            grade=grade,
            potential_entry=potential_entry,
            rejection_risks=self._infer_risks(features),
            confidence_factors=self._infer_confidence(features),
            next_killzone_check=None,
            conflict_tag=conflict_tag
        )

    def _grade(self, score: float) -> str:
        for level in ("A", "B", "C"):
            if score >= self.thresholds.get(level, 1.1):
                return level
        return "D"

    def _detect_conflict(self, score: float, features: Dict[str, float], context: Dict[str, Any]) -> bool:
        if not self.conflict_config.get("enabled", False) or not context:
            return False
        direction_now = features.get("htf_bias")
        direction_active = context.get("htf_bias")
        if direction_now * direction_active < 0:  # opposing sign
            return score >= self.conflict_config.get("min_conflict_score", 0.72)
        return False

    def _infer_risks(self, features: Dict[str, float]) -> list:
        risks = []
        if features.get("choch_confirmed", 0) < 0.3:
            risks.append("weak_choch")
        if features.get("tick_density", 1.0) < 0.2:
            risks.append("low_density")
        return risks

    def _infer_confidence(self, features: Dict[str, float]) -> Dict[str, str]:
        return {
            "fvg_quality": self._qual_tag(features.get("poi_validated", 0.0)),
            "liquidity_clarity": self._qual_tag(features.get("sweep_validated", 0.0)),
            "volatility_stability": self._qual_tag(1.0 - features.get("spread_status", 0.0)),
        }

    def _qual_tag(self, val: float) -> str:
        if val >= 0.75:
            return "high"
        elif val >= 0.5:
            return "medium"
        return "low"
