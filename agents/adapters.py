"""
NCOS v11.6 - Agent Adapters
Interface adapters for different agent types and external systems
"""
from typing import Dict, Any, List, Optional, Type
import asyncio
from datetime import datetime
from .base_agent import BaseAgent
from ..core.base import logger

class AgentAdapter:
    """Base adapter for agent integration"""

    def __init__(self, agent_class: Type[BaseAgent], config: Dict[str, Any]):
        self.agent_class = agent_class
        self.config = config
        self.agent_instance: Optional[BaseAgent] = None

    async def create_agent(self) -> BaseAgent:
        """Create and initialize agent instance"""
        if self.agent_instance is None:
            self.agent_instance = self.agent_class(self.config)
            await self.agent_instance.initialize()
        return self.agent_instance

    async def execute_task(self, task_data: Any) -> Any:
        """Execute task through the agent"""
        agent = await self.create_agent()
        return await agent.process(task_data)
