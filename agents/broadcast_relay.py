"""
BroadcastRelay - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class BroadcastRelayConfig:
    """Configuration for BroadcastRelay"""
    agent_id: str = "broadcast_relay"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    max_channels: Optional[int] = 50
    max_queue_size: Optional[int] = 1000
    broadcast_interval: Optional[int] = 1
    message_ttl: Optional[int] = 3600
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BroadcastRelay:
    """
    BroadcastRelay - Message broadcasting and inter-agent communication
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = BroadcastRelayConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "messages_broadcast": 0,
            "channels_created": 0
        }

        self.channels = {}
        self.subscribers = {}
        self.message_queue = []

        self.logger.info(f"{self.config.agent_id} initialized")

    async def initialize(self):
        """Initialize the agent"""
        try:
            self.logger.info(f"Initializing {self.config.agent_id}")
            await self._setup()
            self.status = "ready"
            self.logger.info(f"{self.config.agent_id} ready for operation")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.config.agent_id}: {e}")
            self.status = "error"
            raise

    async def _setup(self):
        """Agent-specific setup logic"""
        # Initialize broadcast channels
        self.logger.info(f"Max channels: {self.config.max_channels}")
        self.logger.info(f"Max queue size: {self.config.max_queue_size}")

        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())

    async def _message_processing_loop(self):
        """Background message processing loop"""
        while self.status == "ready":
            try:
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    await self._process_queued_message(message)

                await asyncio.sleep(self.config.broadcast_interval)
            except Exception as e:
                self.logger.error(f"Message processing loop error: {e}")
                break

    async def _process_queued_message(self, message: Dict[str, Any]):
        """Process a queued message"""
        channel = message.get("channel")
        data = message.get("data")

        if channel in self.subscribers:
            for subscriber in self.subscribers[channel]:
                try:
                    # Simulate message delivery
                    self.logger.debug(f"Delivering message to {subscriber} on channel {channel}")
                    await asyncio.sleep(0.001)  # Simulate delivery time
                except Exception as e:
                    self.logger.error(f"Error delivering message to {subscriber}: {e}")

    async def _broadcast_message(self, channel: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to a channel"""
        if channel not in self.channels:
            return {"error": f"Channel {channel} does not exist"}

        message = {
            "id": len(self.message_queue) + 1,
            "channel": channel,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "ttl": self.config.message_ttl
        }

        # Add to queue if not full
        if len(self.message_queue) < self.config.max_queue_size:
            self.message_queue.append(message)
            self.metrics["messages_broadcast"] += 1

            subscriber_count = len(self.subscribers.get(channel, []))

            return {
                "status": "queued",
                "message_id": message["id"],
                "channel": channel,
                "subscribers": subscriber_count
            }
        else:
            return {"error": "Message queue is full"}

    async def _subscribe_channel(self, channel: str, agent_id: str) -> Dict[str, Any]:
        """Subscribe an agent to a channel"""
        # Create channel if it doesn't exist
        if channel not in self.channels:
            if len(self.channels) >= self.config.max_channels:
                return {"error": "Maximum number of channels reached"}

            self.channels[channel] = {
                "created": datetime.now().isoformat(),
                "message_count": 0
            }
            self.subscribers[channel] = []
            self.metrics["channels_created"] += 1
            self.logger.info(f"Created new channel: {channel}")

        # Add subscriber if not already subscribed
        if agent_id not in self.subscribers[channel]:
            self.subscribers[channel].append(agent_id)
            self.logger.info(f"Agent {agent_id} subscribed to channel {channel}")

            return {
                "status": "subscribed",
                "channel": channel,
                "agent_id": agent_id,
                "subscriber_count": len(self.subscribers[channel])
            }
        else:
            return {
                "status": "already_subscribed",
                "channel": channel,
                "agent_id": agent_id
            }

    async def _unsubscribe_channel(self, channel: str, agent_id: str) -> Dict[str, Any]:
        """Unsubscribe an agent from a channel"""
        if channel in self.subscribers and agent_id in self.subscribers[channel]:
            self.subscribers[channel].remove(agent_id)
            self.logger.info(f"Agent {agent_id} unsubscribed from channel {channel}")

            return {
                "status": "unsubscribed",
                "channel": channel,
                "agent_id": agent_id,
                "subscriber_count": len(self.subscribers[channel])
            }
        else:
            return {"error": f"Agent {agent_id} not subscribed to channel {channel}"}

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message"""
        try:
            self.metrics["messages_processed"] += 1
            self.logger.debug(f"Processing message: {message.get('type', 'unknown')}")

            result = await self._handle_message(message)

            return {
                "status": "success",
                "agent_id": self.config.agent_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "agent_id": self.config.agent_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_message(self, message: Dict[str, Any]) -> Any:
        """Agent-specific message handling"""
        # Handle broadcast operations
        msg_type = message.get("type")
        if msg_type == "broadcast":
            return await self._broadcast_message(message.get("channel"), message.get("data"))
        elif msg_type == "subscribe":
            return await self._subscribe_channel(message.get("channel"), message.get("agent_id"))
        elif msg_type == "unsubscribe":
            return await self._unsubscribe_channel(message.get("channel"), message.get("agent_id"))
        elif msg_type == "get_channels":
            return {"channels": list(self.channels.keys())}
        elif msg_type == "get_subscribers":
            channel = message.get("channel")
            return {"subscribers": self.subscribers.get(channel, [])}

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "active_channels": len(self.channels),
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "queue_size": len(self.message_queue)
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> BroadcastRelay:
    """Factory function to create BroadcastRelay instance"""
    return BroadcastRelay(config)


# Export the agent class
__all__ = ["BroadcastRelay", "BroadcastRelayConfig", "create_agent"]
