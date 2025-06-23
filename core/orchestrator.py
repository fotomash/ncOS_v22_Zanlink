"""
NCOS v11.6 - Core Orchestrator
Manages agent coordination and task distribution
"""
from typing import Dict, Any, List, Optional, Set
import asyncio
from datetime import datetime
from .base import BaseComponent, AgentResponse, logger
from ..agents.base_agent import BaseAgent

class TaskQueue:
    """Async task queue for agent coordination"""

    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.active_tasks: Set[str] = set()

    async def add_task(self, task: Dict[str, Any]) -> str:
        task_id = f"task_{datetime.now().timestamp()}"
        task["task_id"] = task_id
        await self.queue.put(task)
        return task_id

    async def get_task(self) -> Optional[Dict[str, Any]]:
        try:
            task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            self.active_tasks.add(task["task_id"])
            return task
        except asyncio.TimeoutError:
            return None

    def complete_task(self, task_id: str):
        self.active_tasks.discard(task_id)

class Orchestrator(BaseComponent):
    """Main system orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = TaskQueue()
        self.results: Dict[str, AgentResponse] = {}
        self.running = False

    async def initialize(self) -> bool:
        """Initialize orchestrator and agents"""
        for agent in self.agents.values():
            await agent.initialize()
        self.is_initialized = True
        return True

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")

    async def submit_task(self, task_type: str, data: Any, priority: int = 0) -> str:
        """Submit a task for processing"""
        task = {
            "type": task_type,
            "data": data,
            "priority": priority,
            "submitted_at": datetime.now()
        }
        task_id = await self.task_queue.add_task(task)
        logger.info(f"Submitted task: {task_id}")
        return task_id

    async def process(self, data: Any) -> Any:
        """Process incoming data"""
        return await self.submit_task("general", data)

    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("Orchestrator started")

        # Start agent workers
        workers = []
        for agent_id, agent in self.agents.items():
            worker = asyncio.create_task(self._agent_worker(agent))
            workers.append(worker)

        await asyncio.gather(*workers)

    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("Orchestrator stopped")

    async def _agent_worker(self, agent: BaseAgent):
        """Worker loop for an agent"""
        while self.running:
            try:
                task = await self.task_queue.get_task()
                if task is None:
                    continue

                if agent.can_handle(task["type"]):
                    result = await agent.process(task["data"])
                    response = AgentResponse(
                        agent_id=agent.agent_id,
                        task_id=task["task_id"],
                        result=result
                    )
                    self.results[task["task_id"]] = response
                    self.task_queue.complete_task(task["task_id"])

            except Exception as e:
                logger.error(f"Agent {agent.agent_id} error: {e}")
                await asyncio.sleep(1)
