# strategy_match_engine.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.0.0
# Description:
#   Evaluates the current market situation (macro context, indicators, SMC tags)
#   against predefined strategy rules to find the best match and confidence score.

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import traceback

# --- Load Strategy Rules ---
RULES_CONFIG_PATH = Path("config/strategy_rules.json")
STRATEGY_RULES = {}
if RULES_CONFIG_PATH.is_file():
    try:
        with open(RULES_CONFIG_PATH, 'r') as f:
            STRATEGY_RULES = json.load(f)
        print(f"[INFO][StrategyMatcher] Loaded {len(STRATEGY_RULES)} strategy rule sets from {RULES_CONFIG_PATH}")
    except Exception as e:
        print(f"[ERROR][StrategyMatcher] Failed to load strategy rules: {e}")
        STRATEGY_RULES = {} # Use empty rules if loading fails
else:
    print(f"[WARN][StrategyMatcher] Strategy rules file not found: {RULES_CONFIG_PATH}")


# --- Rule Evaluation Logic ---
def check_condition(value: Any, condition: str, target: Any) -> bool:
    """ Checks if a value meets a specific condition against a target. """
    if value is None or pd.isna(value): # Cannot evaluate None/NaN
        return False
    try:
        if condition == "equals": return value == target
        elif condition == ">": return float(value) > float(target)
        elif condition == "<": return float(value) < float(target)
        elif condition == ">=": return float(value) >= float(target)
        elif condition == "<=": return float(value) <= float(target)
        elif condition == "!=": return value != target
        elif condition == "is_not_nan": return pd.notna(value) # Target value is ignored here
        elif condition == "is_nan": return pd.isna(value) # Target value is ignored here
        elif condition == "is_above": return value > target # Simple comparison, target is likely another series name
        elif condition == "is_below": return value < target
        elif condition == "squeeze": return value is True # Check for Bollinger Band Squeeze flag
        elif condition == "is_rising": return value > 0 # Check if a value (like ATR diff) is positive
        elif condition == "is_falling": return value < 0 # Check if a value (like ATR diff) is negative
        elif condition == "divergence_bullish": return value == 'Bullish' or value == 'HiddenBullish'
        elif condition == "divergence_bearish": return value == 'Bearish' or value == 'HiddenBearish'
        elif condition == "is_oversold": return value <= float(target if target is not None else 30) # Default OS level 30
        elif condition == "is_overbought": return value >= float(target if target is not None else 70) # Default OB level 70
        elif condition == "price_above": return value is True # Assumes a boolean flag like price > BB_Upper
        elif condition == "price_below": return value is True # Assumes a boolean flag like price < BB_Lower
        # Add more conditions as needed (e.g., 'crosses_above', 'in_range')
        else:
            print(f"[WARN][StrategyMatcher] Unknown condition type: {condition}")
            return False
    except (ValueError, TypeError) as e:
        # Handle cases where comparison fails (e.g., comparing string with number)
        # print(f"[DEBUG][StrategyMatcher] Condition check failed for value '{value}' {condition} '{target}': {e}")
        return False
    except Exception as e:
        print(f"[ERROR][StrategyMatcher] Unexpected error in check_condition: {e}")
        traceback.print_exc()
        return False


def get_value_from_situation(situation: Dict, source: str, name: Optional[str], metric: str):
    """ Safely retrieves a specific value from the situation report. """
    try:
        if source == "macro":
            return situation.get("macro_context", {}).get(metric)
        elif source == "indicator":
            # Assumes indicators are stored like: situation['indicators_snapshot']['H1']['RSI_14']
            # This needs refinement based on actual structure of indicators_snapshot
            # For now, let's assume a flat structure for simplicity in testing
            # Example: situation['indicators']['RSI_14']
            return situation.get("indicators", {}).get(name, {}).get(metric) # Placeholder structure
        elif source == "smc":
            # Example: situation['smc_tags']['M15']['liq_sweep_low']
            return situation.get("smc_tags", {}).get(name, {}).get(metric) # Placeholder structure
        elif source == "price":
             # Example: situation['price_data']['M15']['Close']
             # Needs access to the actual price data, potentially the latest candle
             # This requires passing price data into the situation report or accessing it here
             # Placeholder: return latest close price if metric is 'Close'
             if metric.lower() == 'close':
                  # Find the first available TF data and get last close
                  for tf_data in situation.get('enriched_tf_data', {}).values():
                       if tf_data is not None and not tf_data.empty:
                            return tf_data['Close'].iloc[-1]
             return None # Other price metrics need specific implementation
        else:
            return situation.get(source, {}).get(metric)
    except Exception as e:
        print(f"[ERROR][StrategyMatcher] Error retrieving value for {source}/{name}/{metric}: {e}")
        return None


def evaluate_strategy_fit(situation_report: Dict, rules: List[Dict]) -> float:
    """ Calculates the confidence score for a single strategy based on its rules. """
    if not rules: return 0.0 # No rules means no match

    match_count = 0
    total_rules = len(rules)

    for rule in rules:
        source = rule.get("source")
        name = rule.get("name") # Optional (e.g., for specific indicator)
        metric = rule.get("metric")
        condition = rule.get("condition")
        target = rule.get("value") # Target value for comparison

        if not all([source, metric, condition]):
            print(f"[WARN][StrategyMatcher] Skipping invalid rule: {rule}")
            total_rules -= 1 # Don't penalize score for bad rule
            continue

        # Retrieve the actual value from the situation report
        # This part needs refinement based on the actual structure of situation_report
        # How are indicator values, macro states, SMC tags stored?
        actual_value = get_value_from_situation(situation_report, source, name, metric)

        # Check if the rule condition is met
        if check_condition(actual_value, condition, target):
            match_count += 1
            # print(f"[DEBUG] Rule MET: {source}/{name}/{metric} ({actual_value}) {condition} {target}")
        # else:
            # print(f"[DEBUG] Rule NOT MET: {source}/{name}/{metric} ({actual_value}) {condition} {target}")


    # Calculate confidence score
    confidence = match_count / total_rules if total_rules > 0 else 0.0
    return round(confidence, 3)


def match_strategy(situation_report: Dict, confidence_threshold: float = 0.75) -> Dict:
    """
    Finds the best matching strategy based on the current situation report.

    Args:
        situation_report (Dict): Compiled data including macro context, indicator states, etc.
        confidence_threshold (float): Minimum score required to declare a match.

    Returns:
        Dict: {'strategy': strategy_name|None, 'confidence': score}
    """
    print(f"[INFO][StrategyMatcher] Matching situation against {len(STRATEGY_RULES)} strategies...")
    if not STRATEGY_RULES:
        print("[WARN][StrategyMatcher] No strategy rules loaded. Cannot match.")
        return {"strategy": None, "confidence": 0.0}

    scores = {}
    for strategy_name, rules in STRATEGY_RULES.items():
        score = evaluate_strategy_fit(situation_report, rules)
        scores[strategy_name] = score
        print(f"[DEBUG][StrategyMatcher] Score for {strategy_name}: {score:.3f}")

    # Select highest scoring strategy
    if not scores: # Handle case where no strategies were evaluated
         return {"strategy": None, "confidence": 0.0}

    best_fit_strategy = max(scores, key=scores.get)
    best_score = scores[best_fit_strategy]

    print(f"[INFO][StrategyMatcher] Best match: {best_fit_strategy} (Confidence: {best_score:.3f})")

    # Return match only if confidence exceeds threshold
    if best_score >= confidence_threshold:
        return {
            "strategy": best_fit_strategy,
            "confidence": best_score
        }
    else:
        print(f"[INFO][StrategyMatcher] Best match confidence ({best_score:.3f}) below threshold ({confidence_threshold}). No strategy triggered.")
        return {
            "strategy": None,
            "confidence": best_score # Return the score even if below threshold
        }

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Strategy Match Engine ---")

    # Create a dummy situation report (needs realistic data structure)
    dummy_situation = {
        "asset": "OANDA:EUR_USD",
        "macro_context": {
            "risk_state": "Risk ON",
            "vix_level": 14.5,
            "dxy_trend": "down"
        },
        "indicators": { # Simplified flat structure for testing rule evaluation
            "RSI_14": {"value": 65.0, "divergence_bullish": False, "divergence_bearish": False, "is_overbought": False, "is_oversold": False},
            "OBV": {"trend": "up"},
            "Bollinger_Bands": {"squeeze": True}, # Assuming BBands calc adds this boolean
            "ATR_14": {"value": 0.00050, "is_rising": True}, # Assuming ATR calc adds trend flag
            "EMA_50": {"value": 1.1010},
            "EMA_200": {"value": 1.0980},
            "ADX_14": {"value": 28.0}
        },
        "smc_tags": { # Simplified structure
             "M15": {"liq_sweep_low": 1.1000, "fvg_bullish_nearby": True}
        },
        "price_data": { # Need a way to access latest price if rules require it
             "M15": {"Close": 1.1050}
        }
        # Add enriched_tf_data if needed by get_value_from_situation
        # "enriched_tf_data": { ... }
    }

    print("\nDummy Situation Report:")
    print(json.dumps(dummy_situation, indent=2, default=str))

    print("\nMatching Strategy...")
    match_result = match_strategy(dummy_situation, confidence_threshold=0.6) # Lower threshold for testing

    print("\n--- Match Result ---")
    print(json.dumps(match_result, indent=2))
    print("--------------------")

    print("\n--- Test Complete ---")
