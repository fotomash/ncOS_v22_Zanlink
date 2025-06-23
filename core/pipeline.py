"""
NCOS v11.6 - Core Pipeline Module
Orchestrates data flow through the system
"""
from typing import Dict, Any, List, Optional, Callable
import asyncio
from .base import BaseComponent, DataModel, logger

class PipelineStage(BaseComponent):
    """Individual pipeline stage"""

    def __init__(self, name: str, processor: Callable, config: Dict[str, Any]):
        super().__init__(config)
        self.name = name
        self.processor = processor

    async def initialize(self) -> bool:
        self.is_initialized = True
        return True

    async def process(self, data: Any) -> Any:
        try:
            result = await self.processor(data)
            return result
        except Exception as e:
            self.logger.error(f"Stage {self.name} failed: {e}")
            raise

class Pipeline(BaseComponent):
    """Main processing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stages: List[PipelineStage] = []
        self.metrics = {"processed": 0, "errors": 0}

    async def initialize(self) -> bool:
        for stage in self.stages:
            await stage.initialize()
        self.is_initialized = True
        return True

    def add_stage(self, stage: PipelineStage):
        """Add a processing stage"""
        self.stages.append(stage)

    async def process(self, data: Any) -> Any:
        """Process data through all stages"""
        current_data = data

        for stage in self.stages:
            try:
                current_data = await stage.process(current_data)
                self.logger.debug(f"Completed stage: {stage.name}")
            except Exception as e:
                self.metrics["errors"] += 1
                self.logger.error(f"Pipeline failed at stage {stage.name}: {e}")
                raise

        self.metrics["processed"] += 1
        return current_data

    async def process_batch(self, data_batch: List[Any]) -> List[Any]:
        """Process multiple items concurrently"""
        tasks = [self.process(item) for item in data_batch]
        return await asyncio.gather(*tasks, return_exceptions=True)
