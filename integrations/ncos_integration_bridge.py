
"""
ncOS Integration Bridge
Connects existing ncOS system with new LLM-optimized components
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from pathlib import Path

# Import new components
from ncos_llm_gateway import NCOSLLMGateway, LLMRequest, LLMResponse
from ncos_data_package_manager import DataPackageManager
from ncos_prompt_templates import NCOSPromptLibrary, create_context_aware_prompt

# Import existing ncOS components (adjust imports based on your structure)
# from engine import VectorEngine
# from zbar_bridge import ZBARProcessor
# from unified_mt4_processor import MT4Processor
# from llm_assistant import JournalLLMAssistant

class NCOSIntegrationBridge:
    """
    Main integration bridge that connects:
    - Existing ncOS components
    - New LLM-optimized gateway
    - Data package manager
    - Prompt templates
    """

    def __init__(self, config_path: str = "config/ncos_config.json"):
        self.config = self._load_config(config_path)

        # Initialize new components
        self.llm_gateway = NCOSLLMGateway()
        self.data_manager = DataPackageManager()
        self.prompt_library = NCOSPromptLibrary()

        # Connection status
        self.components_status = {
            "llm_gateway": "initialized",
            "data_manager": "initialized",
            "prompt_library": "initialized",
            "vector_engine": "pending",
            "zbar_processor": "pending",
            "mt4_processor": "pending"
        }

        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "api_endpoints": {
                "journal": "http://localhost:8000",
                "mt4": "http://localhost:8001",
                "llm": "http://localhost:8002"
            },
            "cache_settings": {
                "ttl_minutes": 5,
                "max_size": 100
            }
        }

    async def process_llm_request(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for LLM requests
        Routes to appropriate processor and returns formatted response
        """
        try:
            # Create LLM request
            request = LLMRequest(
                action=action,
                context=context,
                filters=context.get("filters", {})
            )

            # Check cache first
            cache_key = f"{action}_{json.dumps(context, sort_keys=True)}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if datetime.now().timestamp() - cached_data["timestamp"] < self.cache_ttl:
                    return cached_data["data"]

            # Process through gateway
            response = await self.llm_gateway.process_llm_request(request)

            # Create data package for enhanced response
            package = self.data_manager.get_or_create_package(
                data_type=self._map_action_to_package_type(action),
                **context
            )

            # Combine gateway response with data package
            enhanced_response = {
                "summary": response.summary,
                "insights": response.insights,
                "recommendations": response.recommendations,
                "data": response.data,
                "package": self.data_manager.export_for_llm(package),
                "prompt_suggestion": self._generate_followup_prompt(action, response)
            }

            # Cache the response
            self.cache[cache_key] = {
                "timestamp": datetime.now().timestamp(),
                "data": enhanced_response
            }

            return enhanced_response

        except Exception as e:
            return {
                "error": str(e),
                "summary": "Error processing request",
                "insights": ["An error occurred during processing"],
                "recommendations": ["Please check your request and try again"]
            }

    def _map_action_to_package_type(self, action: str) -> str:
        """Map LLM action to data package type"""
        mapping = {
            "analyze_session": "session_replay",
            "market_overview": "market_analysis",
            "pattern_detection": "pattern_detection",
            "trade_recommendation": "trade_summary",
            "performance_summary": "trade_summary"
        }
        return mapping.get(action, "market_analysis")

    def _generate_followup_prompt(self, action: str, response: LLMResponse) -> str:
        """Generate intelligent follow-up prompt based on response"""
        if action == "market_overview" and response.data.get("trend") == "bullish":
            return "What are the best entry points for a long position?"
        elif action == "pattern_detection" and len(response.data.get("patterns_detected", [])) > 2:
            return "Which pattern has the highest probability of success?"
        elif action == "performance_summary" and response.data.get("win_rate", 0) < 0.6:
            return "How can I improve my win rate?"
        else:
            return "What additional analysis would be helpful?"

    async def get_formatted_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Get formatted prompt with market context"""
        # Get current market data
        market_data = await self._get_current_market_data(context.get("symbol", "XAUUSD"))

        # Create context-aware prompt
        prompt = create_context_aware_prompt(market_data, query)

        return prompt

    async def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data (simplified for example)"""
        # In production, this would fetch real data from your MT4 processor
        return {
            "symbol": symbol,
            "price": 1948.75,
            "trend": "bullish",
            "resistance": 1950.50,
            "support": 1945.30,
            "patterns": "Wyckoff Spring, SMC Order Block",
            "volume_profile": "increasing",
            "timeframe": "H1"
        }

    def create_chatgpt_response(self, raw_data: Dict[str, Any]) -> str:
        """Format response specifically for ChatGPT display"""
        response_parts = []

        # Summary section
        response_parts.append(f"**Summary**: {raw_data.get('summary', 'Analysis complete')}")
        response_parts.append("")

        # Key insights
        if insights := raw_data.get("insights", []):
            response_parts.append("**Key Insights**:")
            for insight in insights[:3]:
                response_parts.append(f"• {insight}")
            response_parts.append("")

        # Data highlights
        if data := raw_data.get("data", {}):
            response_parts.append("**Market Data**:")
            if "trend" in data:
                response_parts.append(f"• Trend: {data['trend']}")
            if "key_levels" in data:
                levels = data['key_levels']
                response_parts.append(f"• Resistance: {levels.get('resistance', 'N/A')}")
                response_parts.append(f"• Support: {levels.get('support', 'N/A')}")
            response_parts.append("")

        # Recommendations
        if recommendations := raw_data.get("recommendations", []):
            response_parts.append("**Recommendations**:")
            for i, rec in enumerate(recommendations[:3], 1):
                response_parts.append(f"{i}. {rec}")
            response_parts.append("")

        # Follow-up prompt
        if followup := raw_data.get("prompt_suggestion"):
            response_parts.append(f"💡 *Suggested follow-up: {followup}*")

        return "\n".join(response_parts)

    async def batch_process_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple requests in parallel for efficiency"""
        tasks = []
        for req in requests:
            task = self.process_llm_request(req["action"], req["context"])
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def export_state_for_llm(self) -> Dict[str, Any]:
        """Export current system state in LLM-friendly format"""
        return {
            "timestamp": datetime.now().isoformat(),
            "components_status": self.components_status,
            "cache_size": len(self.cache),
            "available_actions": list(self.llm_gateway.processors.keys()),
            "prompt_templates": list(self.prompt_library.templates.keys())
        }

# FastAPI integration endpoints
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ncOS Integration Bridge API")
bridge = NCOSIntegrationBridge()

class BridgeRequest(BaseModel):
    action: str
    context: Dict[str, Any]
    format: str = "chatgpt"  # or "raw"

class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]]

@app.post("/bridge/process")
async def process_request(request: BridgeRequest):
    """Process single request through integration bridge"""
    result = await bridge.process_llm_request(request.action, request.context)

    if request.format == "chatgpt":
        return {"response": bridge.create_chatgpt_response(result)}
    else:
        return result

@app.post("/bridge/batch")
async def process_batch(request: BatchRequest):
    """Process multiple requests in batch"""
    results = await bridge.batch_process_requests(request.requests)
    return {"results": results}

@app.get("/bridge/prompt")
async def get_prompt(query: str, symbol: str = "XAUUSD"):
    """Get formatted prompt for given query"""
    prompt = await bridge.get_formatted_prompt(query, {"symbol": symbol})
    return {"prompt": prompt}

@app.get("/bridge/status")
async def get_bridge_status():
    """Get current bridge status"""
    return bridge.export_state_for_llm()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
