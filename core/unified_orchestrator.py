import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from src.session_state import SessionState
from src.budget import TokenBudgetManager
from src.core import SystemConfig
from ncos.core.memory_manager import MemoryManager
from ncos.core.pipeline_models import (
    DataRequest,
    DataResult,
    AnalysisResult,
    StrategyMatch,
    ExecutionResult,
)

class NCOSConfig:
    """Minimal configuration wrapper."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir

    def get(self, *_args, default=None):
        return default


class DataPipelineV24:
    """Simplified data pipeline."""

    def __init__(self, config: NCOSConfig):
        self.config = config

    async def fetch_and_process(self, request: DataRequest) -> DataResult:
        return DataResult(
            status="success",
            symbol=request.symbol,
            data={},
        )


class AnalysisEngineV24:
    """Simplified analysis engine."""

    def __init__(self, config: NCOSConfig):
        self.config = config

    async def run_full_analysis(self, data: DataResult) -> AnalysisResult:
        return AnalysisResult(
            status="success",
            symbol=data.symbol,
            analysis={},
            confluence_score=0.0,
        )


class StrategyEngineV24:
    """Simplified strategy matcher."""

    def __init__(self, config: NCOSConfig):
        self.config = config

    async def match_strategy(self, analysis_result: AnalysisResult) -> StrategyMatch:
        return StrategyMatch(strategy=None, confidence=0.0, status="no_match")
from src.unified_execution_engine import UnifiedExecutionEngine


@dataclass
class TradingSignal:
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    strategy: str
    metadata: Dict[str, Any]


class UnifiedOrchestrator:
    """Unified orchestrator combining analysis and execution layers."""

    def __init__(self, config: Union[Dict[str, Any], str]):
        """Create the orchestrator.

        Parameters
        ----------
        config:
            Either a configuration dictionary or path to a YAML file.
        """

        self.raw_config = config
        self.config: SystemConfig | None = None
        self.logger = logging.getLogger(__name__)
        self.active_signals: List[TradingSignal] = []
        self.engines: Dict[str, Any] = {}

        # Will be initialized later
        self.ncos_config: NCOSConfig | None = None
        self.data_pipeline: DataPipelineV24 | None = None
        self.analysis_engine: AnalysisEngineV24 | None = None
        self.strategy_engine: StrategyEngineV24 | None = None
        self.execution_engine: UnifiedExecutionEngine | None = None
        self.token_manager: TokenBudgetManager | None = None
        self.session_state: SessionState | None = None
        self.memory_manager: MemoryManager | None = None

    async def initialize(self) -> None:
        self.logger.info("Initializing Unified Orchestrator")

        self._load_config()

        self.token_manager = TokenBudgetManager(
            self.config.token_budget.total,
            self.config.token_budget.reserve_percentage,
        )
        self.session_state = SessionState(
            token_budget=self.token_manager.get_budget(),
            config=self.config,
        )

        # Initialize memory manager (vector store configuration can be extended)
        self.memory_manager = MemoryManager({"vector_store_provider": "mock"})
        await self.memory_manager.initialize()

        self.ncos_config = NCOSConfig(self.config.workspace_dir)
        self.data_pipeline = DataPipelineV24(self.ncos_config)
        self.analysis_engine = AnalysisEngineV24(self.ncos_config)
        self.strategy_engine = StrategyEngineV24(self.ncos_config)
        self.execution_engine = UnifiedExecutionEngine(
            self.ncos_config.get("risk_config", {})
        )

        await self._initialize_engines()

    def _load_config(self) -> None:
        """Load configuration from dict or YAML file."""
        try:
            if isinstance(self.raw_config, str):
                with open(self.raw_config, "r") as f:
                    data = yaml.safe_load(f)
            else:
                data = self.raw_config

            self.config = SystemConfig(**data)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.raw_config}")
            raise
        except (yaml.YAMLError, ValidationError) as e:
            self.logger.error(f"Configuration load error: {e}")
            raise

    async def _initialize_engines(self) -> None:
        try:
            from core.engines.market_structure_analyzer_smc import MarketStructureAnalyzer
            from core.engines.liquidity_engine_smc import LiquidityEngine
            from core.engines.wyckoff_phase_engine import WyckoffPhaseEngine
            from core.engines.predictive_scorer import PredictiveScorer
            from core.engines.volatility_engine import VolatilityEngine

            self.engines = {
                "market_structure": MarketStructureAnalyzer(self.config),
                "liquidity": LiquidityEngine(self.config),
                "wyckoff": WyckoffPhaseEngine(self.config),
                "predictive": PredictiveScorer(self.config),
                "volatility": VolatilityEngine(self.config),
            }
        except Exception:  # pragma: no cover - optional deps missing
            self.engines = {}
        for engine in self.engines.values():
            if hasattr(engine, "initialize") and asyncio.iscoroutinefunction(engine.initialize):
                await engine.initialize()

    def score(self, features: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        scorer = self.engines.get("predictive")
        if scorer is None:
            raise RuntimeError("Predictive engine not initialized")
        return scorer.score(features, context)

    async def analyze_symbol(self, symbol: str, timeframes: List[str] | None = None) -> Dict[str, Any]:
        if timeframes is None:
            timeframes = ["m15", "h1", "h4"]

        data_request = DataRequest(symbol=symbol, timeframes=timeframes)
        data_result = await self.data_pipeline.fetch_and_process(data_request)
        if data_result.status != "success":
            return data_result.model_dump()

        analysis_result = await self.analysis_engine.run_full_analysis(data_result)
        if analysis_result.status != "success":
            return analysis_result.model_dump()

        strategy_result = await self.strategy_engine.match_strategy(analysis_result)
        execution_result = None
        if strategy_result.status == "match_found":
            execution_result = await self.execution_engine.execute(strategy_result.model_dump(), analysis_result.model_dump(), symbol)

        return {
            "status": "success",
            "symbol": symbol,
            "timeframes": timeframes,
            "data_status": data_result.status,
            "analysis": analysis_result.model_dump(),
            "strategy_match": strategy_result.model_dump(),
            "execution": execution_result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run_market_scan(self, symbols: List[str]) -> Dict[str, Any]:
        tasks = [self.analyze_symbol(sym) for sym in symbols]
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: Dict[str, Any] = {}
        for symbol, result in zip(symbols, analysis_results):
            if isinstance(result, Exception):
                results[symbol] = {"status": "error", "message": str(result)}
            else:
                results[symbol] = result

        return {
            "status": "success",
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }
