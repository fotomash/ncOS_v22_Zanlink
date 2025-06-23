
"""
ncOS Intelligent Prompt Template System
Dynamic prompt generation for consistent LLM interactions
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from enum import Enum

class PromptType(Enum):
    ANALYSIS = "analysis"
    DECISION = "decision"
    SUMMARY = "summary"
    ALERT = "alert"
    EDUCATION = "education"

class PromptTemplate:
    """Base prompt template with dynamic variable injection"""

    def __init__(self, template: str, variables: List[str], description: str = ""):
        self.template = template
        self.variables = variables
        self.description = description

    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        missing = [var for var in self.variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)

class NCOSPromptLibrary:
    """Library of optimized prompts for trading scenarios"""

    def __init__(self):
        self.templates = {
            # Market Analysis Prompts
            "market_analysis": PromptTemplate(
                template="""Analyze {symbol} on {timeframe} timeframe:

Current Price: {price}
Trend: {trend}
Key Resistance: {resistance}
Key Support: {support}

Recent Patterns:
{patterns}

Volume Profile: {volume_profile}

Provide: 1) Market bias, 2) Entry strategy, 3) Risk levels""",
                variables=["symbol", "timeframe", "price", "trend", "resistance", "support", "patterns", "volume_profile"]
            ),

            # Trade Decision Prompts
            "trade_decision": PromptTemplate(
                template="""Trade Decision for {symbol}:

Setup: {setup_name}
Confluence Score: {confluence_score}/10
Patterns Detected: {patterns}
Risk/Reward: {risk_reward}

Current Market Context:
- Trend: {trend}
- Volatility: {volatility}
- Session: {session}

Should I take this trade? Provide reasoning and position sizing recommendation.""",
                variables=["symbol", "setup_name", "confluence_score", "patterns", "risk_reward", "trend", "volatility", "session"]
            ),

            # Session Summary Prompts
            "session_summary": PromptTemplate(
                template="""Trading Session Summary - {session_id}:

Performance Metrics:
- Total Trades: {total_trades}
- Win Rate: {win_rate}%
- P&L: ${pnl}
- Best Trade: {best_trade}
- Worst Trade: {worst_trade}

Patterns Used:
{patterns_summary}

Market Conditions:
{market_conditions}

Provide: 1) Key lessons, 2) Improvements needed, 3) Tomorrow's focus""",
                variables=["session_id", "total_trades", "win_rate", "pnl", "best_trade", "worst_trade", "patterns_summary", "market_conditions"]
            ),

            # Pattern Recognition Prompts
            "pattern_alert": PromptTemplate(
                template="""🎯 Pattern Alert on {symbol}:

Pattern: {pattern_name}
Confidence: {confidence}%
Location: {price_level}
Timeframe: {timeframe}

Supporting Factors:
{factors}

Recommended Action: {action}
Entry Zone: {entry_zone}
Stop Loss: {stop_loss}
Targets: {targets}""",
                variables=["symbol", "pattern_name", "confidence", "price_level", "timeframe", "factors", "action", "entry_zone", "stop_loss", "targets"]
            ),

            # Risk Assessment Prompts
            "risk_check": PromptTemplate(
                template="""Risk Assessment for {symbol} position:

Position Size: {position_size} lots
Entry: {entry_price}
Stop Loss: {stop_loss}
Risk Amount: ${risk_amount}
Risk Percentage: {risk_percent}%

Market Conditions:
- Volatility: {volatility}
- Upcoming Events: {events}
- Correlation Risk: {correlations}

Is this risk appropriate? Provide adjustment recommendations if needed.""",
                variables=["symbol", "position_size", "entry_price", "stop_loss", "risk_amount", "risk_percent", "volatility", "events", "correlations"]
            )
        }

        # Meta-prompts for chaining
        self.meta_templates = {
            "pre_analysis": "Before analyzing, confirm: 1) Data freshness, 2) Timeframe alignment, 3) No conflicting signals",
            "post_trade": "After trade execution, log: 1) Entry rationale, 2) Pattern screenshots, 3) Risk parameters",
            "context_check": "Verify market context: 1) Session (Asian/London/NY), 2) Economic calendar, 3) Correlation status"
        }

    def get_prompt(self, prompt_type: str, **variables) -> str:
        """Get rendered prompt by type"""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        template = self.templates[prompt_type]
        return template.render(**variables)

    def create_prompt_chain(self, prompts: List[Dict[str, Any]]) -> str:
        """Create a chain of prompts for complex analysis"""
        chain_parts = []

        for i, prompt_config in enumerate(prompts):
            prompt_type = prompt_config.get("type")
            variables = prompt_config.get("variables", {})

            # Add meta-prompt if specified
            if "meta" in prompt_config:
                chain_parts.append(self.meta_templates.get(prompt_config["meta"], ""))

            # Add main prompt
            prompt = self.get_prompt(prompt_type, **variables)
            chain_parts.append(f"Step {i+1}: {prompt}")

            # Add separator
            chain_parts.append("\n" + "="*50 + "\n")

        return "\n".join(chain_parts)

    def create_adaptive_prompt(self, context: Dict[str, Any]) -> str:
        """Create adaptive prompt based on current context"""
        # Determine best prompt type based on context
        if context.get("action") == "analyze":
            if context.get("patterns_detected", 0) > 2:
                prompt_type = "pattern_alert"
            else:
                prompt_type = "market_analysis"
        elif context.get("action") == "review":
            prompt_type = "session_summary"
        elif context.get("risk_check_needed"):
            prompt_type = "risk_check"
        else:
            prompt_type = "trade_decision"

        # Auto-fill variables from context
        template = self.templates[prompt_type]
        variables = {}

        for var in template.variables:
            # Try to auto-populate from context
            if var in context:
                variables[var] = context[var]
            else:
                # Provide sensible defaults
                variables[var] = self._get_default_value(var)

        return self.get_prompt(prompt_type, **variables)

    def _get_default_value(self, variable: str) -> Any:
        """Get default value for missing variables"""
        defaults = {
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "price": "current",
            "trend": "analyzing",
            "patterns": "detecting",
            "volume_profile": "normal",
            "session": "London",
            "volatility": "moderate",
            "confidence": 0,
            "risk_reward": "1:2"
        }
        return defaults.get(variable, "N/A")

    def export_for_chatgpt(self) -> Dict[str, Any]:
        """Export prompt templates for ChatGPT configuration"""
        export_data = {
            "templates": {},
            "usage_examples": {},
            "variables_guide": {}
        }

        for name, template in self.templates.items():
            export_data["templates"][name] = {
                "template": template.template,
                "required_variables": template.variables,
                "description": template.description
            }

            # Add usage example
            example_vars = {var: f"<{var}>" for var in template.variables}
            export_data["usage_examples"][name] = template.render(**example_vars)

        return export_data

# Utility functions for LLM integration
def create_context_aware_prompt(market_data: Dict, user_query: str) -> str:
    """Create context-aware prompt combining market data and user query"""
    library = NCOSPromptLibrary()

    # Analyze user query to determine intent
    query_lower = user_query.lower()

    if any(word in query_lower for word in ["analyze", "analysis", "look at"]):
        context = {
            "action": "analyze",
            **market_data
        }
    elif any(word in query_lower for word in ["risk", "position size", "safe"]):
        context = {
            "risk_check_needed": True,
            **market_data
        }
    elif any(word in query_lower for word in ["summary", "review", "performance"]):
        context = {
            "action": "review",
            **market_data
        }
    else:
        context = market_data

    # Generate adaptive prompt
    base_prompt = library.create_adaptive_prompt(context)

    # Append user query
    full_prompt = f"{base_prompt}\n\nUser Query: {user_query}"

    return full_prompt

# Example configuration for ChatGPT
CHATGPT_SYSTEM_PROMPT = """You are an expert trading assistant using the ncOS system. You analyze markets using Wyckoff and Smart Money Concepts.

When responding:
1. Use the provided data structure
2. Be concise but thorough
3. Always include risk management
4. Highlight pattern confluences
5. Provide actionable insights

Format responses with:
- **Summary**: One-line overview
- **Analysis**: Key findings
- **Action**: Recommended steps
- **Risk**: Important warnings"""
