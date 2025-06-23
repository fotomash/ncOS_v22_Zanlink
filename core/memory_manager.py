"""
NCOS v11.6 - Memory Manager
Handles system memory, caching, and state management
"""
from typing import Dict, Any, Optional, List
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from .base import BaseComponent, logger

class MemoryStore:
    """In-memory store with persistence"""

    def __init__(self, db_path: str = "data/memory.db"):
        self.cache: Dict[str, Any] = {}
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            timestamp REAL,
            expires_at REAL
        )
        """)
        conn.commit()
        conn.close()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in memory"""
        self.cache[key] = value

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        expires_at = datetime.now().timestamp() + ttl if ttl else None
        conn.execute(
            "INSERT OR REPLACE INTO memory (key, value, timestamp, expires_at) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value), datetime.now().timestamp(), expires_at)
        )
        conn.commit()
        conn.close()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory"""
        # Check cache first
        if key in self.cache:
            return self.cache[key]

        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value, expires_at FROM memory WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            value, expires_at = row
            if expires_at is None or expires_at > datetime.now().timestamp():
                parsed_value = json.loads(value)
                self.cache[key] = parsed_value
                return parsed_value

        return None

    async def delete(self, key: str):
        """Delete a value from memory"""
        self.cache.pop(key, None)
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()

class MemoryManager(BaseComponent):
    """Main memory management system"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.store = MemoryStore(config.get("db_path", "data/memory.db"))
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.system_state: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize memory manager"""
        await self.cleanup_expired()
        self.is_initialized = True
        return True

    async def process(self, data: Any) -> Any:
        """Process memory operations"""
        if isinstance(data, dict) and "operation" in data:
            op = data["operation"]
            if op == "set":
                await self.store.set(data["key"], data["value"], data.get("ttl"))
            elif op == "get":
                return await self.store.get(data["key"])
            elif op == "delete":
                await self.store.delete(data["key"])
        return data

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Save agent state"""
        self.agent_states[agent_id] = state
        await self.store.set(f"agent_state_{agent_id}", state)

    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get agent state"""
        if agent_id in self.agent_states:
            return self.agent_states[agent_id]
        return await self.store.get(f"agent_state_{agent_id}") or {}

    async def cleanup_expired(self):
        """Clean up expired memory entries"""
        conn = sqlite3.connect(self.store.db_path)
        now = datetime.now().timestamp()
        conn.execute("DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        conn.commit()
        conn.close()
