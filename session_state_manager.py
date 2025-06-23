#!/usr/bin/env python3
"""
Session State Manager for ncOS v24
Handles persistent memory across GPT conversations
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

class SessionStateManager:
    """Manages persistent session state for ncOS"""

    def __init__(self, state_file: str = "session_state.json"):
        self.state_file = Path(state_file)
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        """Load previous session state or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    state['resumed_at'] = datetime.now().isoformat()
                    return state
            except:
                pass

        # Create new session
        return self.create_new_session()

    def create_new_session(self) -> Dict[str, Any]:
        """Create a fresh session state"""
        new_state = {
            "session_id": f"ncOS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "user_preferences": {
                "default_symbol": "XAUUSD",
                "preferred_timeframes": ["M15", "M5", "M1"],
                "risk_percentage": 1.0,
                "favorite_strategies": ["SMC_Structural_Flip_POI_v12"],
                "alert_preferences": {
                    "structure_breaks": True,
                    "poi_approaches": True,
                    "session_opens": True
                }
            },
            "market_context": {
                "XAUUSD": {
                    "htf_bias": None,
                    "last_price": None,
                    "key_levels": {"resistance": [], "support": [], "poi_zones": []},
                    "active_setups": [],
                    "last_analysis": None
                }
            },
            "active_monitoring": {
                "setups": [],
                "alerts": [],
                "risk_exposure": 0.0,
                "open_positions": []
            },
            "conversation_context": {
                "last_command": "",
                "last_analysis_type": "",
                "pending_actions": [],
                "important_notes": []
            },
            "trading_journal": {
                "setups_identified": 0,
                "setups_triggered": 0,
                "win_rate": 0.0,
                "total_rr_captured": 0.0
            }
        }
        self.state = new_state
        self.save_state()
        return new_state

    def save_state(self):
        """Save current state to file"""
        self.state['last_active'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update_market_context(self, symbol: str, context: Dict[str, Any]):
        """Update market context for a symbol"""
        if symbol not in self.state['market_context']:
            self.state['market_context'][symbol] = {
                "htf_bias": None,
                "last_price": None,
                "key_levels": {"resistance": [], "support": [], "poi_zones": []},
                "active_setups": [],
                "last_analysis": None
            }

        self.state['market_context'][symbol].update(context)
        self.state['market_context'][symbol]['last_analysis'] = datetime.now().isoformat()
        self.save_state()

    def add_setup(self, setup: Dict[str, Any]):
        """Add a new trading setup to monitor"""
        setup['identified_at'] = datetime.now().isoformat()
        setup['status'] = 'monitoring'
        self.state['active_monitoring']['setups'].append(setup)
        self.save_state()

    def get_welcome_message(self) -> str:
        """Generate personalized welcome message"""
        try:
            last_active = datetime.fromisoformat(self.state['last_active'].replace('Z', '+00:00'))
            time_away = datetime.now() - last_active.replace(tzinfo=None)

            if time_away < timedelta(hours=1):
                time_str = f"{int(time_away.seconds / 60)} minutes ago"
            elif time_away < timedelta(days=1):
                time_str = f"{int(time_away.seconds / 3600)} hours ago"
            else:
                time_str = f"{time_away.days} days ago"
        except:
            time_str = "a while ago"

        symbol = self.state.get('user_preferences', {}).get('default_symbol', 'XAUUSD')
        market_ctx = self.state.get('market_context', {}).get(symbol, {})

        welcome = f"""Welcome back! Last session: {time_str}

📊 {symbol} Update:"""

        if market_ctx.get('last_price'):
            welcome += f"""
• Last tracked price: {market_ctx['last_price']}
• HTF Bias: {market_ctx.get('htf_bias', 'Not analyzed')}"""

        active_setups = self.state.get('active_monitoring', {}).get('setups', [])
        if active_setups:
            welcome += f"""

🎯 Active Monitoring:
• {len(active_setups)} setups being tracked
• Risk exposure: {self.state.get('active_monitoring', {}).get('risk_exposure', 0.0)}%"""

        welcome += """

Ready to continue where we left off!"""

        return welcome

    def checkpoint(self, note: str = None):
        """Create a checkpoint with optional note"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "market_snapshot": self.state.get('market_context', {}),
            "active_setups": len(self.state.get('active_monitoring', {}).get('setups', []))
        }

        if 'checkpoints' not in self.state:
            self.state['checkpoints'] = []

        self.state['checkpoints'].append(checkpoint)
        self.save_state()

        return "Checkpoint saved!"
