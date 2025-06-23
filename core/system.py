#!/usr/bin/env python3
"""
NCOS v24 - Core System
This module contains the main NCOSSystem class, which acts as the central
nervous system, initializing and managing all core components.
"""

from typing import Dict, Any
from loguru import logger

from ncos.core.base import BaseComponent
from ncos.core.orchestrators.main_orchestrator import MultiAgentOrchestrator
from ncos.core.memory.manager import MemoryManager
from ncos.utils.helpers import FileHelper

class NCOSSystem(BaseComponent):
    """
    The main NCOS system class. It orchestrates all major components,
    including the memory manager, agent orchestrator, data handlers,
    and monitoring services.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NCOS system with a master configuration.
        
        Args:
            config: The root configuration dictionary for the entire system.
        """
        super().__init__(config)
        self.memory_manager: MemoryManager = None
        self.orchestrator: MultiAgentOrchestrator = None
        # Add other managers as they are created (e.g., DataManager, StrategyManager)

    async def initialize(self) -> bool:
        """
        Initializes the system in the correct order:
        1. Ensures necessary directories exist.
        2. Initializes the Memory Manager.
        3. Initializes the Agent Orchestrator.
        """
        self.logger.info("NCOS System is initializing...")

        try:
            # 1. Ensure data and log directories exist
            FileHelper.ensure_directory(self.config.get("data_dir", "data"))
            FileHelper.ensure_directory(self.config.get("log_dir", "logs"))

            # 2. Initialize Memory Manager
            memory_config = self.config.get("memory_manager", {})
            self.memory_manager = MemoryManager(memory_config)
            await self.memory_manager.initialize()
            self.logger.info("Memory Manager initialized successfully.")

            # 3. Initialize Agent Orchestrator
            orchestrator_config = self.config.get("orchestrator", {})
            # Pass the memory_manager instance to the orchestrator
            self.orchestrator = MultiAgentOrchestrator(orchestrator_config, self.memory_manager)
            await self.orchestrator.initialize()
            self.logger.info("Multi-Agent Orchestrator initialized successfully.")

            # (Future initializations for other managers would go here)

            self.is_initialized = True
            self.logger.success("NCOS System initialization complete.")
            return True

        except Exception as e:
            self.logger.critical(f"A critical error occurred during system initialization: {e}", exc_info=True)
            self.is_initialized = False
            return False

    async def process(self, data: Any) -> Any:
        """
        The main entry point for processing data through the NCOS system.
        This will typically route tasks to the orchestrator.
        """
        if not self.is_initialized:
            raise RuntimeError("NCOS System is not initialized. Cannot process data.")
        
        # For now, we pass data directly to the orchestrator to be handled as a task.
        # This can be made more sophisticated later.
        self.logger.debug(f"Routing data to orchestrator: {data}")
        return await self.orchestrator.process(data)

    async def cleanup(self):
        """Gracefully cleans up all system components."""
        self.logger.info("Initiating system cleanup...")
        if self.orchestrator:
            await self.orchestrator.cleanup()
        if self.memory_manager:
            await self.memory_manager.cleanup()
        
        await super().cleanup()
        self.logger.success("System cleanup complete.")

