"""
MarketDataCaptain - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class MarketDataCaptainConfig:
    """Configuration for MarketDataCaptain"""
    agent_id: str = "market_data_captain"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    cache_size: Optional[int] = 1000
    feeds: Optional[List[str]] = field(default_factory=lambda: ["yahoo", "alpha_vantage"])
    update_interval: Optional[int] = 60
    symbols: Optional[List[str]] = field(default_factory=list)
    custom_params: Dict[str, Any] = field(default_factory=dict)


class MarketDataCaptain:
    """
    MarketDataCaptain - Market data ingestion and real-time feed management
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = MarketDataCaptainConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "quotes_fetched": 0,
            "subscriptions": 0
        }

        self.data_feeds = {}
        self.symbols = self.config.symbols or []
        self.last_update = None
        self.quote_cache = {}

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
        # Initialize market data connections
        for feed in self.config.feeds:
            self.data_feeds[feed] = {
                "status": "connected",
                "last_update": datetime.now()
            }
            self.logger.info(f"Initialized feed: {feed}")

        # Start background update loop
        if self.config.update_interval:
            asyncio.create_task(self._update_loop())

    async def _update_loop(self):
        """Background data update loop"""
        while self.status == "ready":
            try:
                await self._update_all_quotes()
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                break

    async def _update_all_quotes(self):
        """Update quotes for all subscribed symbols"""
        for symbol in self.symbols:
            try:
                quote = await self._fetch_quote(symbol)
                self.quote_cache[symbol] = quote
                self.metrics["quotes_fetched"] += 1
            except Exception as e:
                self.logger.error(f"Error updating quote for {symbol}: {e}")

        self.last_update = datetime.now()

    async def _fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch quote for a symbol"""
        # Simulate fetching quote data
        await asyncio.sleep(0.01)  # Simulate network delay

        return {
            "symbol": symbol,
            "price": 100.0,  # Mock price
            "volume": 1000,  # Mock volume
            "timestamp": datetime.now().isoformat(),
            "source": self.config.feeds[0] if self.config.feeds else "mock"
        }

    async def _subscribe_symbol(self, symbol: str) -> Dict[str, Any]:
        """Subscribe to a symbol"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.metrics["subscriptions"] += 1
            self.logger.info(f"Subscribed to symbol: {symbol}")

            # Fetch initial quote
            quote = await self._fetch_quote(symbol)
            self.quote_cache[symbol] = quote

            return {"status": "subscribed", "symbol": symbol, "quote": quote}
        else:
            return {"status": "already_subscribed", "symbol": symbol}

    async def _get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol"""
        if symbol in self.quote_cache:
            return self.quote_cache[symbol]
        elif symbol in self.symbols:
            # Fetch fresh quote
            quote = await self._fetch_quote(symbol)
            self.quote_cache[symbol] = quote
            return quote
        else:
            return {"error": f"Symbol {symbol} not subscribed"}

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
        # Handle market data requests
        msg_type = message.get("type")
        if msg_type == "subscribe":
            return await self._subscribe_symbol(message.get("symbol"))
        elif msg_type == "get_quote":
            return await self._get_quote(message.get("symbol"))
        elif msg_type == "get_subscriptions":
            return {"symbols": self.symbols}
        elif msg_type == "update_feeds":
            return await self._update_all_quotes()

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "subscribed_symbols": len(self.symbols),
            "active_feeds": list(self.data_feeds.keys()),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")
        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> MarketDataCaptain:
    """Factory function to create MarketDataCaptain instance"""
    return MarketDataCaptain(config)


# Export the agent class
__all__ = ["MarketDataCaptain", "MarketDataCaptainConfig", "create_agent"]
