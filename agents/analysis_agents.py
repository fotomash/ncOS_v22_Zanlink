"""
NCOS v11.6 - Analysis Agents
Specialized agents for data analysis and pattern detection
"""
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np
from datetime import datetime
from .base_agent import BaseAgent
from ..core.base import logger

class TickAnalysisAgent(BaseAgent):
    """Tick data analysis and manipulation detection agent"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("tick_analysis_agent", config)
        self.task_types = {"tick_analysis", "manipulation_detection", "pattern_recognition"}
        self.capabilities = {"high_frequency_analysis", "anomaly_detection", "statistical_analysis"}
        self.detection_thresholds = config.get("thresholds", {
            "volume_spike": 3.0,
            "price_deviation": 2.5,
            "time_clustering": 0.1
        })

    async def _execute(self, data: Any) -> Any:
        """Execute tick analysis"""
        if isinstance(data, dict):
            task_type = data.get("type", "general")

            if task_type == "tick_analysis":
                return await self._analyze_ticks(data)
            elif task_type == "manipulation_detection":
                return await self._detect_manipulation(data)
            elif task_type == "pattern_recognition":
                return await self._recognize_patterns(data)

        return {"status": "processed", "agent": self.agent_id}

    async def _analyze_ticks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tick data"""
        ticks = data.get("ticks", [])
        symbol = data.get("symbol", "UNKNOWN")

        if not ticks:
            return {"error": "No tick data provided"}

        # Calculate basic statistics
        prices = [tick.get("price", 0) for tick in ticks]
        volumes = [tick.get("volume", 0) for tick in ticks]

        analysis = {
            "symbol": symbol,
            "tick_count": len(ticks),
            "price_range": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0
            },
            "volume_stats": {
                "total": sum(volumes),
                "avg": sum(volumes) / len(volumes) if volumes else 0,
                "max": max(volumes) if volumes else 0
            },
            "analysis_timestamp": datetime.now().isoformat()
        }

        return analysis

    async def _detect_manipulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential market manipulation"""
        ticks = data.get("ticks", [])

        manipulation_signals = []

        # Volume spike detection
        volumes = [tick.get("volume", 0) for tick in ticks]
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            for i, volume in enumerate(volumes):
                if avg_volume > 0 and volume > avg_volume * self.detection_thresholds["volume_spike"]:
                    manipulation_signals.append({
                        "type": "volume_spike",
                        "tick_index": i,
                        "severity": volume / avg_volume,
                        "timestamp": ticks[i].get("timestamp", "unknown")
                    })

        # Price deviation detection
        prices = [tick.get("price", 0) for tick in ticks]
        if len(prices) > 1:
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            avg_change = sum(price_changes) / len(price_changes)

            for i, change in enumerate(price_changes):
                if avg_change > 0 and change > avg_change * self.detection_thresholds["price_deviation"]:
                    manipulation_signals.append({
                        "type": "price_deviation",
                        "tick_index": i + 1,
                        "severity": change / avg_change,
                        "timestamp": ticks[i + 1].get("timestamp", "unknown")
                    })

        return {
            "manipulation_detected": len(manipulation_signals) > 0,
            "signal_count": len(manipulation_signals),
            "signals": manipulation_signals,
            "confidence": min(1.0, len(manipulation_signals) / 10.0)
        }

    async def _recognize_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize trading patterns"""
        ticks = data.get("ticks", [])

        patterns = []

        if len(ticks) >= 3:
            prices = [tick.get("price", 0) for tick in ticks]

            # Simple trend detection
            if prices[-1] > prices[0]:
                patterns.append({"type": "uptrend", "strength": (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0})
            elif prices[-1] < prices[0]:
                patterns.append({"type": "downtrend", "strength": (prices[0] - prices[-1]) / prices[0] if prices[0] != 0 else 0})
            else:
                patterns.append({"type": "sideways", "strength": 0.0})

        return {
            "patterns_found": len(patterns),
            "patterns": patterns,
            "analysis_quality": "basic"
        }

class VectorDBAgent(BaseAgent):
    """Vector database and similarity search agent"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("vectordb_agent", config)
        self.task_types = {"vector_search", "data_ingestion", "similarity_analysis"}
        self.capabilities = {"vector_operations", "embedding_generation", "similarity_search"}
        self.vector_store: Dict[str, List[float]] = {}

    async def _execute(self, data: Any) -> Any:
        """Execute vector operations"""
        if isinstance(data, dict):
            operation = data.get("operation", "unknown")

            if operation == "ingest":
                return await self._ingest_data(data)
            elif operation == "search":
                return await self._search_similar(data)
            elif operation == "analyze":
                return await self._analyze_vectors(data)

        return {"status": "unknown_operation"}

    async def _ingest_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data into vector store"""
        vectors = data.get("vectors", {})
        self.vector_store.update(vectors)

        return {
            "status": "ingested",
            "count": len(vectors),
            "total_vectors": len(self.vector_store)
        }

    async def _search_similar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar vectors"""
        query_vector = data.get("query_vector", [])
        top_k = data.get("top_k", 5)

        if not query_vector or not self.vector_store:
            return {"results": [], "count": 0}

        # Simple cosine similarity calculation
        similarities = []
        for key, vector in self.vector_store.items():
            if len(vector) == len(query_vector):
                similarity = self._cosine_similarity(query_vector, vector)
                similarities.append({"key": key, "similarity": similarity})

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        results = similarities[:top_k]

        return {
            "results": results,
            "count": len(results),
            "query_processed": True
        }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(a * a for a in vec2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0

    async def _analyze_vectors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vector store statistics"""
        if not self.vector_store:
            return {"error": "No vectors in store"}

        vector_lengths = [len(v) for v in self.vector_store.values()]

        return {
            "total_vectors": len(self.vector_store),
            "avg_dimension": sum(vector_lengths) / len(vector_lengths) if vector_lengths else 0,
            "min_dimension": min(vector_lengths) if vector_lengths else 0,
            "max_dimension": max(vector_lengths) if vector_lengths else 0,
            "store_health": "good"
        }
