"""
SessionStateManager - NCOS v21 Agent
Fixed version with proper config fields
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class SessionStateManagerConfig:
    """Configuration for SessionStateManager"""
    agent_id: str = "session_state_manager"
    enabled: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    checkpoint_interval: Optional[int] = 300
    max_checkpoints: Optional[int] = 10
    state_file: Optional[str] = "session_state.json"
    backup_interval: Optional[int] = 3600
    custom_params: Dict[str, Any] = field(default_factory=dict)


class SessionStateManager:
    """
    SessionStateManager - Session state persistence and recovery
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = SessionStateManagerConfig(**(config or {}))
        self.logger = logging.getLogger(f"NCOS.{self.config.agent_id}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        self.status = "initialized"
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "uptime_start": datetime.now(),
            "checkpoints_created": 0,
            "states_saved": 0,
            "states_restored": 0
        }

        self.session_data = {}
        self.checkpoints = {}
        self.recovery_points = []
        self.state_file_path = Path(self.config.state_file)

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
        # Initialize state management
        self.logger.info(f"State file: {self.state_file_path}")
        self.logger.info(f"Checkpoint interval: {self.config.checkpoint_interval}s")

        # Load existing state if available
        await self._load_existing_state()

        # Start checkpoint loop
        if self.config.checkpoint_interval:
            asyncio.create_task(self._checkpoint_loop())

        # Start backup loop
        if self.config.backup_interval:
            asyncio.create_task(self._backup_loop())

    async def _checkpoint_loop(self):
        """Background checkpoint creation loop"""
        while self.status == "ready":
            try:
                await self._create_checkpoint()
                await asyncio.sleep(self.config.checkpoint_interval)
            except Exception as e:
                self.logger.error(f"Checkpoint loop error: {e}")
                break

    async def _backup_loop(self):
        """Background backup creation loop"""
        while self.status == "ready":
            try:
                await self._create_backup()
                await asyncio.sleep(self.config.backup_interval)
            except Exception as e:
                self.logger.error(f"Backup loop error: {e}")
                break

    async def _load_existing_state(self):
        """Load existing state from file"""
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r') as f:
                    loaded_data = json.load(f)

                self.session_data = loaded_data.get("session_data", {})
                self.checkpoints = loaded_data.get("checkpoints", {})
                self.recovery_points = loaded_data.get("recovery_points", [])

                self.logger.info(f"Loaded existing state with {len(self.session_data)} sessions")

            except Exception as e:
                self.logger.error(f"Error loading existing state: {e}")
                # Initialize with empty state
                self.session_data = {}
                self.checkpoints = {}
                self.recovery_points = []

    async def _create_checkpoint(self):
        """Create a checkpoint of current state"""
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_data = {
            "id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "session_data": self.session_data.copy(),
            "metrics": self.metrics.copy()
        }

        self.checkpoints[checkpoint_id] = checkpoint_data
        self.metrics["checkpoints_created"] += 1

        # Clean up old checkpoints
        if len(self.checkpoints) > self.config.max_checkpoints:
            oldest_checkpoint = min(self.checkpoints.keys())
            del self.checkpoints[oldest_checkpoint]
            self.logger.debug(f"Removed old checkpoint: {oldest_checkpoint}")

        self.logger.debug(f"Created checkpoint: {checkpoint_id}")

    async def _create_backup(self):
        """Create a backup of the state file"""
        if self.state_file_path.exists():
            backup_path = self.state_file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            try:
                import shutil
                shutil.copy2(self.state_file_path, backup_path)
                self.logger.debug(f"Created backup: {backup_path}")

                # Clean up old backups (keep only last 5)
                backup_files = sorted(self.state_file_path.parent.glob(f"{self.state_file_path.stem}.backup_*.json"))
                if len(backup_files) > 5:
                    for old_backup in backup_files[:-5]:
                        old_backup.unlink()

            except Exception as e:
                self.logger.error(f"Error creating backup: {e}")

    async def _save_session_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Save session state"""
        session_id = state.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Update session data
        self.session_data[session_id] = {
            "session_id": session_id,
            "data": state.get("data", {}),
            "timestamp": datetime.now().isoformat(),
            "version": state.get("version", 1)
        }

        # Save to file
        await self._persist_state()

        self.metrics["states_saved"] += 1

        return {
            "status": "saved",
            "session_id": session_id,
            "timestamp": self.session_data[session_id]["timestamp"]
        }

    async def _restore_session_state(self, session_id: str) -> Dict[str, Any]:
        """Restore session state"""
        if session_id in self.session_data:
            session_data = self.session_data[session_id]
            self.metrics["states_restored"] += 1

            return {
                "status": "restored",
                "session_id": session_id,
                "data": session_data["data"],
                "timestamp": session_data["timestamp"],
                "version": session_data["version"]
            }
        else:
            return {"error": f"Session {session_id} not found"}

    async def _persist_state(self):
        """Persist current state to file"""
        state_data = {
            "session_data": self.session_data,
            "checkpoints": self.checkpoints,
            "recovery_points": self.recovery_points,
            "last_updated": datetime.now().isoformat()
        }

        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_file = self.state_file_path.with_suffix('.tmp')

            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            temp_file.rename(self.state_file_path)

        except Exception as e:
            self.logger.error(f"Error persisting state: {e}")
            raise

    async def _restore_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore state from a checkpoint"""
        if checkpoint_id in self.checkpoints:
            checkpoint_data = self.checkpoints[checkpoint_id]

            # Restore session data
            self.session_data = checkpoint_data["session_data"].copy()

            # Save restored state
            await self._persist_state()

            return {
                "status": "restored_from_checkpoint",
                "checkpoint_id": checkpoint_id,
                "timestamp": checkpoint_data["timestamp"],
                "sessions_restored": len(self.session_data)
            }
        else:
            return {"error": f"Checkpoint {checkpoint_id} not found"}

    async def _list_sessions(self) -> Dict[str, Any]:
        """List all active sessions"""
        sessions = []

        for session_id, session_data in self.session_data.items():
            sessions.append({
                "session_id": session_id,
                "timestamp": session_data["timestamp"],
                "version": session_data["version"],
                "data_keys": list(session_data["data"].keys()) if isinstance(session_data["data"], dict) else []
            })

        return {
            "sessions": sessions,
            "total_count": len(sessions)
        }

    async def _cleanup_old_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []

        for session_id, session_data in self.session_data.items():
            try:
                session_time = datetime.fromisoformat(session_data["timestamp"])
                if session_time < cutoff_time:
                    sessions_to_remove.append(session_id)
            except Exception as e:
                self.logger.error(f"Error parsing timestamp for session {session_id}: {e}")

        # Remove old sessions
        for session_id in sessions_to_remove:
            del self.session_data[session_id]

        if sessions_to_remove:
            await self._persist_state()

        return {
            "status": "cleaned",
            "removed_sessions": sessions_to_remove,
            "removed_count": len(sessions_to_remove)
        }

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
        # Handle state operations
        msg_type = message.get("type")
        if msg_type == "save_state":
            return await self._save_session_state(message.get("state"))
        elif msg_type == "restore_state":
            return await self._restore_session_state(message.get("session_id"))
        elif msg_type == "list_sessions":
            return await self._list_sessions()
        elif msg_type == "create_checkpoint":
            await self._create_checkpoint()
            return {"status": "checkpoint_created"}
        elif msg_type == "restore_checkpoint":
            return await self._restore_from_checkpoint(message.get("checkpoint_id"))
        elif msg_type == "list_checkpoints":
            return {"checkpoints": list(self.checkpoints.keys())}
        elif msg_type == "cleanup_sessions":
            max_age = message.get("max_age_hours", 24)
            return await self._cleanup_old_sessions(max_age)

        return {"processed": True, "agent": self.config.agent_id}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = datetime.now() - self.metrics["uptime_start"]
        return {
            "agent_id": self.config.agent_id,
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": self.metrics.copy(),
            "active_sessions": len(self.session_data),
            "available_checkpoints": len(self.checkpoints),
            "state_file": str(self.state_file_path)
        }

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down {self.config.agent_id}")

        # Save final state
        try:
            await self._persist_state()
            self.logger.info("Final state saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving final state: {e}")

        self.status = "shutdown"


# Agent factory function
def create_agent(config: Dict[str, Any] = None) -> SessionStateManager:
    """Factory function to create SessionStateManager instance"""
    return SessionStateManager(config)


# Export the agent class
__all__ = ["SessionStateManager", "SessionStateManagerConfig", "create_agent"]
