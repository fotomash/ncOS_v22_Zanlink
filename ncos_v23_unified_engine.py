#!/usr/bin/env python3
"""
ncOS v23 Unified Engine
Merges ncOS vector-native architecture with ZanFlow v17 capabilities
"""

import os
import json
import yaml
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA STRUCTURES ====================

@dataclass
class VectorContext:
    """Vector-native market context"""
    timestamp: datetime
    embeddings: np.ndarray  # 1536-dim vectors
    confidence: float
    signals: Dict[str, Any]
    session_id: str

@dataclass
class UnifiedSignal:
    """Unified trading signal"""
    action: str  # buy, sell, hold
    pair: str
    entry: float
    stop_loss: float
    take_profits: List[float]
    confidence: float
    vectors: np.ndarray
    strategies_aligned: List[str]
    reasoning: Dict[str, str]

# ==================== VECTOR OPERATIONS ====================

class VectorStore:
    """In-memory vector store for session data"""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}

    def add(self, key: str, vector: np.ndarray, metadata: Dict = None):
        """Add vector to store"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {vector.shape[0]} != {self.dimension}")
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Cosine similarity search"""
        if not self.vectors:
            return []

        similarities = []
        query_norm = query_vector / np.linalg.norm(query_vector)

        for key, vector in self.vectors.items():
            vec_norm = vector / np.linalg.norm(vector)
            similarity = np.dot(query_norm, vec_norm)
            similarities.append((key, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# ==================== UNIFIED AGENTS ====================

class UnifiedAgent:
    """Base class for unified agents with vector support"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.vector_store = VectorStore()
        self.session_state = {}

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and return signals with vectors"""
        raise NotImplementedError

    def to_vector(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert analysis to vector representation"""
        # Simplified vector generation - in production use proper embeddings
        features = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)

        # Pad or truncate to dimension
        vector = np.zeros(1536)
        vector[:min(len(features), 1536)] = features[:1536]
        return vector

class SMCMasterAgent(UnifiedAgent):
    """Enhanced SMC analysis with vector operations"""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Perform SMC analysis
        structure = self._analyze_structure(data)
        poi = self._detect_poi(data)
        liquidity = self._find_liquidity(data)

        analysis = {
            "structure": structure,
            "poi": poi,
            "liquidity": liquidity,
            "confidence": self._calculate_confidence(structure, poi, liquidity)
        }

        # Convert to vector and store
        vector = self.to_vector(analysis)
        self.vector_store.add(
            f"smc_{datetime.now().isoformat()}",
            vector,
            {"analysis": analysis}
        )

        return {
            "analysis": analysis,
            "vector": vector,
            "similar_patterns": self.vector_store.search(vector, top_k=3)
        }

    def _analyze_structure(self, data):
        return {"trend": "bullish", "strength": 0.8, "bos": True}

    def _detect_poi(self, data):
        return {"order_blocks": [{"price": 1.1000, "type": "bullish"}]}

    def _find_liquidity(self, data):
        return {"pools": [{"level": 1.1050, "type": "buy_stops"}]}

    def _calculate_confidence(self, structure, poi, liquidity):
        factors = 0
        if structure.get("bos"): factors += 1
        if poi.get("order_blocks"): factors += 1
        if liquidity.get("pools"): factors += 1
        return min(factors * 0.3, 0.9)

class LiquiditySniperAgent(UnifiedAgent):
    """Liquidity hunting with vector similarity"""

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sweeps = self._detect_sweeps(data)
        traps = self._identify_traps(data)

        analysis = {
            "sweeps": sweeps,
            "traps": traps,
            "hunt_score": self._calculate_hunt_score(sweeps, traps)
        }

        vector = self.to_vector(analysis)

        # Find similar historical patterns
        similar = self.vector_store.search(vector, top_k=5)

        return {
            "analysis": analysis,
            "vector": vector,
            "historical_success": self._calculate_historical_success(similar)
        }

    def _detect_sweeps(self, data):
        return [{"level": 1.1050, "swept": True, "volume": "high"}]

    def _identify_traps(self, data):
        return [{"type": "bull_trap", "level": 1.1045}]

    def _calculate_hunt_score(self, sweeps, traps):
        return 0.85 if sweeps and traps else 0.4

    def _calculate_historical_success(self, similar):
        # In production, check actual trade outcomes
        return 0.75

# ==================== STRATEGY IMPLEMENTATIONS ====================

class PrometheusStrategy:
    """Prometheus strategy with vector-based pattern matching"""

    def __init__(self):
        self.name = "Prometheus"
        self.vector_store = VectorStore()
        self.load_historical_patterns()

    def load_historical_patterns(self):
        """Load successful historical patterns as vectors"""
        # In production, load from database
        pass

    async def evaluate(self, market_data: Dict, agent_signals: Dict) -> Optional[UnifiedSignal]:
        """Evaluate for Prometheus setup"""
        # Create context vector from all signals
        context_features = []
        for agent, signal in agent_signals.items():
            if "vector" in signal:
                context_features.append(signal["vector"])

        if not context_features:
            return None

        # Average vectors for context
        context_vector = np.mean(context_features, axis=0)

        # Find similar successful patterns
        similar_patterns = self.vector_store.search(context_vector, top_k=10)

        # Calculate setup probability
        if similar_patterns and similar_patterns[0][1] > 0.85:
            return self._generate_signal(market_data, agent_signals, similar_patterns)

        return None

    def _generate_signal(self, market_data, agent_signals, similar_patterns):
        # Generate trade signal based on pattern matching
        current_price = market_data.get("close", 1.1000)

        return UnifiedSignal(
            action="buy",
            pair=market_data.get("pair", "EURUSD"),
            entry=current_price,
            stop_loss=current_price * 0.997,
            take_profits=[
                current_price * 1.003,
                current_price * 1.006,
                current_price * 1.01
            ],
            confidence=similar_patterns[0][1],
            vectors=np.mean([s[0] for s in similar_patterns], axis=0),
            strategies_aligned=["prometheus", "smc", "liquidity"],
            reasoning={
                "pattern_match": f"95% similarity to historical winner",
                "confluence": "3 factors aligned"
            }
        )

# ==================== UNIFIED ORCHESTRATOR ====================

class UnifiedOrchestrator:
    """Main orchestrator combining ncOS and ZanFlow"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.agents: Dict[str, UnifiedAgent] = {}
        self.strategies: Dict[str, Any] = {}
        self.vector_store = VectorStore()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.external_data_path = os.environ.get("NCOS_DATA_PATH", "/data")

        self._initialize()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Default config
        return {
            "version": "v23.0.0",
            "vector_dimension": 1536,
            "confidence_threshold": 0.7,
            "max_confluence": 5,
            "agents": {
                "smc_master": {"class": "SMCMasterAgent", "weight": 1.5},
                "liquidity_sniper": {"class": "LiquiditySniperAgent", "weight": 1.2},
            },
            "strategies": ["prometheus", "wyckoff", "maz"]
        }

    def _initialize(self):
        """Initialize agents and strategies"""
        # Initialize agents
        self.agents["smc_master"] = SMCMasterAgent("smc_master", self.config.get("agents", {}).get("smc_master", {}))
        self.agents["liquidity_sniper"] = LiquiditySniperAgent("liquidity_sniper", self.config.get("agents", {}).get("liquidity_sniper", {}))

        # Initialize strategies
        self.strategies["prometheus"] = PrometheusStrategy()

        logger.info(f"Initialized {len(self.agents)} agents and {len(self.strategies)} strategies")

    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[UnifiedSignal]:
        """Main processing pipeline"""
        try:
            # 1. Load external data if path provided
            if "data_file" in market_data:
                external_data = self._load_external_data(market_data["data_file"])
                market_data.update(external_data)

            # 2. Run agents in parallel
            agent_results = await self._run_agents(market_data)

            # 3. Build vector context
            context = self._build_vector_context(market_data, agent_results)

            # 4. Evaluate strategies
            for strategy_name, strategy in self.strategies.items():
                signal = await strategy.evaluate(market_data, agent_results)
                if signal and signal.confidence >= self.config["confidence_threshold"]:
                    # Store signal vector
                    self.vector_store.add(
                        f"signal_{self.session_id}_{datetime.now().isoformat()}",
                        signal.vectors,
                        {"signal": signal, "context": context}
                    )
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None

    def _load_external_data(self, filename: str) -> Dict:
        """Load data from external path"""
        filepath = os.path.join(self.external_data_path, filename)
        if os.path.exists(filepath):
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                return {"external_data": df.to_dict('records')}
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    return json.load(f)
        return {}

    async def _run_agents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents and collect results"""
        tasks = []
        for name, agent in self.agents.items():
            tasks.append(self._run_agent(name, agent, data))

        results = await asyncio.gather(*tasks)
        return {name: result for name, result in results}

    async def _run_agent(self, name: str, agent: UnifiedAgent, data: Dict[str, Any]):
        """Run single agent"""
        try:
            result = await agent.analyze(data)
            return (name, result)
        except Exception as e:
            logger.error(f"Agent {name} failed: {e}")
            return (name, {"error": str(e)})

    def _build_vector_context(self, market_data: Dict, agent_results: Dict) -> VectorContext:
        """Build vector context from results"""
        # Combine all vectors
        vectors = []
        confidence_scores = []

        for agent_name, result in agent_results.items():
            if "vector" in result and "error" not in result:
                vectors.append(result["vector"])
                if "confidence" in result.get("analysis", {}):
                    confidence_scores.append(result["analysis"]["confidence"])

        # Create context embedding
        if vectors:
            context_embedding = np.mean(vectors, axis=0)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        else:
            context_embedding = np.zeros(self.config["vector_dimension"])
            avg_confidence = 0.0

        return VectorContext(
            timestamp=datetime.now(),
            embeddings=context_embedding,
            confidence=avg_confidence,
            signals=agent_results,
            session_id=self.session_id
        )

# ==================== MENU SYSTEM ====================

class InteractiveMenu:
    """Interactive menu for ncOS"""

    def __init__(self, orchestrator: UnifiedOrchestrator):
        self.orchestrator = orchestrator
        self.current_menu = "main"

    def display(self) -> Dict:
        """Display current menu"""
        menus = {
            "main": {
                "title": "ncOS v23 Ultimate Trading System",
                "options": {
                    "1": {"title": "🎯 Strategy Management", "action": "strategy_menu"},
                    "2": {"title": "🤖 Agent Operations", "action": "agent_menu"},
                    "3": {"title": "📊 Market Analysis", "action": "analysis_menu"},
                    "4": {"title": "🔧 Configuration", "action": "config_menu"},
                    "5": {"title": "📈 Performance", "action": "performance_menu"},
                    "6": {"title": "📁 Data Management", "action": "data_menu"}
                }
            },
            "strategy_menu": {
                "title": "Strategy Management",
                "options": {
                    "1": {"title": "Prometheus Strategy", "action": "view_prometheus"},
                    "2": {"title": "Wyckoff Scanner", "action": "view_wyckoff"},
                    "3": {"title": "MAZ Levels", "action": "view_maz"},
                    "4": {"title": "Active Signals", "action": "view_signals"},
                    "0": {"title": "Back", "action": "main"}
                }
            }
        }

        return menus.get(self.current_menu, menus["main"])

# ==================== API ENDPOINTS ====================

async def initialize_system():
    """Initialize the unified system"""
    orchestrator = UnifiedOrchestrator()
    menu = InteractiveMenu(orchestrator)

    return {
        "orchestrator": orchestrator,
        "menu": menu,
        "status": "initialized",
        "version": "v23.0.0"
    }

# ==================== MAIN ENTRY ====================

async def main():
    """Main entry point"""
    system = await initialize_system()

    # Example: Process market data
    market_data = {
        "pair": "XAUUSD",
        "timeframe": "M5",
        "close": 3357.50,
        "data_file": "XAUUSD_ticks.csv"  # External data
    }

    signal = await system["orchestrator"].process_market_data(market_data)

    if signal:
        print(f"\n✅ Trade Signal Generated:")
        print(f"  Action: {signal.action}")
        print(f"  Entry: {signal.entry}")
        print(f"  Stop: {signal.stop_loss}")
        print(f"  Targets: {signal.take_profits}")
        print(f"  Confidence: {signal.confidence:.2%}")
    else:
        print("\n❌ No trade signal generated")

if __name__ == "__main__":
    asyncio.run(main())
