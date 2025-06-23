#!/usr/bin/env python3
"""
NCOS v24 - Core Base Module
This file contains the foundational abstract classes and Pydantic data models
that form the building blocks of the entire NCOS system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

class NCOSConfig(BaseModel):
    """
    A Pydantic model for the base system configuration.
    Provides type-hinted, validated configuration for the application.
    """
    version: str = "24.0.0"
    environment: str = "production"
    debug: bool = False
    max_memory_gb: float = Field(8.0, description="Maximum memory allocation in GB.")
    token_limit: int = Field(128000, description="Token limit for LLM interactions.")

class BaseComponent(ABC):
    """
    An abstract base class for all major components in the NCOS system.
    It enforces a common lifecycle for initialization, processing, and cleanup.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the component with its configuration.
        
        Args:
            config: A dictionary containing the component's configuration.
        """
        self.config = config
        self.logger = logger.bind(name=self.__class__.__name__)
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Asynchronously initializes the component. This is where connections
        are made, models are loaded, and the component is prepared for work.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """
        The main processing method for the component. This method will be
        called by the orchestrator or other components to handle data.
        
        Args:
            data: The input data to be processed.
            
        Returns:
            The result of the processing.
        """
        pass

    async def cleanup(self):
        """
        Cleans up resources used by the component (e.g., closing database
        connections, saving state).
        """
        self.is_initialized = False
        self.logger.info("Component cleaned up.")

class DataModel(BaseModel):
    """
    A base Pydantic model for all data structures within NCOS.
    It includes common fields like a timestamp and source identifier.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "ncos_v24"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(DataModel):
    """
    A standardized data model for agent responses, ensuring consistency
    across the system.
    """
    agent_id: str
    task_id: str
    status: str = "completed"
    result: Any = None
    confidence: float = 0.0
    execution_time_ms: float = 0.0

class SystemMetrics(DataModel):
    """
    A Pydantic model for capturing and transporting system performance metrics.
    """
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    active_agents: int = 0
    pending_tasks: int = 0
    error_count: int = 0
