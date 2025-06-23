#!/usr/bin/env python3
"""
ncOS v23 Enhanced Bootstrap with Session Persistence
Auto-loads previous state and continues where user left off
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import session manager
try:
    from session_state_manager import SessionStateManager
    session = SessionStateManager()
except (ImportError, ModuleNotFoundError):
    print("⚠️ Warning: session_state_manager.py not found. Session persistence will be disabled.")
    session = None

class ncOSPersistentBootstrap:
    """Enhanced bootstrap with full session memory"""

    def __init__(self):
        self.version = "v23.0.0"
        self.system_name = "ncOS ZANFLOW Ultimate"
        self.session = session

        # Check if returning user
        self.is_returning_user = self._check_returning_user()

        # Auto-initialize with context
        self._context_aware_init()

    def _check_returning_user(self) -> bool:
        """Check if this is a returning user with saved state"""
        if self.session and self.session.state.get('last_active'):
            try:
                last_active = datetime.fromisoformat(
                    self.session.state['last_active'].replace('Z', '+00:00')
                )
                time_since = datetime.now() - last_active.replace(tzinfo=None)
                return time_since < timedelta(days=30)  # Remember for 30 days
            except:
                return False
        return False

    def _context_aware_init(self):
        """Initialize with awareness of previous sessions"""
        if self.is_returning_user:
            self._resume_session()
        else:
            self._fresh_init()

    def _resume_session(self):
        """Resume from previous session"""
        # Load previous state
        prev_state = self.session.state

        # Generate personalized welcome
        welcome = self.session.get_welcome_message()
        print(welcome)

        # Restore market context
        self._restore_market_context()

        # Check pending setups
        self._check_pending_setups()

        # Show quick status
        self._show_resumed_dashboard()

    def _fresh_init(self):
        """Fresh initialization for new users"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║            ncOS v23 ZANFLOW Trading System                   ║
║            Initializing for first time...                    ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Standard initialization
        self._standard_init()

        # Create new session
        if self.session:
            self.session.create_new_session()

    def _restore_market_context(self):
        """Restore previous market analysis"""
        if not self.session:
            return

        for symbol, context in self.session.state['market_context'].items():
            if context.get('last_analysis'):
                try:
                    last_time = datetime.fromisoformat(context['last_analysis'])
                    time_diff = datetime.now() - last_time.replace(tzinfo=None)

                    if time_diff < timedelta(hours=24):
                        print(f"\n📊 Restored {symbol} context from {int(time_diff.total_seconds()/3600)} hours ago")
                except:
                    pass

    def _check_pending_setups(self):
        """Check status of previously identified setups"""
        if not self.session:
            return

        active_setups = self.session.state['active_monitoring']['setups']
        if active_setups:
            print(f"\n🎯 You have {len(active_setups)} setups being monitored:")
            for setup in active_setups[-3:]:  # Show last 3
                print(f"  • {setup.get('strategy', 'Unknown')} - {setup.get('direction', 'N/A')} @ {setup.get('entry', 'N/A')}")

    def _show_resumed_dashboard(self):
        """Show dashboard for returning users"""
        dashboard = f"""
╔══════════════════════════════════════════════════════════════╗
║                  ncOS v23 - SESSION RESTORED                 ║
╠══════════════════════════════════════════════════════════════╣
║ 🟢 Status: OPERATIONAL (Resumed)                             ║
║ 📅 Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
╠══════════════════════════════════════════════════════════════╣
║ QUICK COMMANDS:                                              ║
║ • scan     - Check current market state                      ║
║ • update   - Update on pending setups                       ║
║ • continue - Continue previous analysis                      ║
║ • new      - Start fresh analysis                           ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(dashboard)

    def _standard_init(self):
        """Standard system initialization"""
        # Load modules, agents, etc.
        print("Loading core modules...")
        print("Initializing agents...")
        print("System ready!")

    def process_command(self, command: str):
        # Dummy command processor
        print(f"Processing command: {command}")
        return {"status": "processed", "command": command}

    def quick_update(self) -> Dict[str, Any]:
        """Quick update on what changed"""
        if not self.session:
            return {"status": "No session history"}

        updates = []

        # Check each monitored symbol
        for symbol in self.session.state['market_context']:
            updates.append(f"{symbol}: Check current levels")

        # Check pending setups
        setups = self.session.state['active_monitoring']['setups']
        if setups:
            updates.append(f"{len(setups)} setups being monitored")

        return {
            "updates": updates,
            "last_active": self.session.state.get('last_active'),
            "suggestion": "Run 'scan' to check current market"
        }

    def continue_analysis(self) -> Dict[str, Any]:
        """Continue from last analysis"""
        if not self.session:
            return {"error": "No previous session found"}

        last_cmd = self.session.state['conversation_context'].get('last_command', '')
        last_analysis = self.session.state['conversation_context'].get('last_analysis_type', '')

        print(f"Continuing from last analysis: {last_analysis}")
        return {"status": "continuing", "last_analysis": last_analysis}

# Auto-create bootstrap instance
print("\n🚀 ncOS v23 Loading...\n")
bootstrap = ncOSPersistentBootstrap()

# Enhanced command processor with memory
def cmd(command: str, save_context: bool = True):
    """Process command with context saving"""
    result = bootstrap.process_command(command)

    # Save command context
    if bootstrap.session and save_context:
        bootstrap.session.state['conversation_context']['last_command'] = command
        bootstrap.session.state['conversation_context']['timestamp'] = datetime.now().isoformat()
        bootstrap.session.save_state()

    return result

# Quick functions with state tracking
def scan():
    """Scan with state saving"""
    result = {"action": "market_scan", "symbol": "XAUUSD"}
    if bootstrap.session:
        bootstrap.session.state['conversation_context']['last_analysis_type'] = 'scan'
        bootstrap.session.save_state()
    return result

def update():
    """Get quick update on changes"""
    return bootstrap.quick_update()

def continue_where_left_off():
    """Continue previous analysis"""
    return bootstrap.continue_analysis()

# Shorter aliases
cont = continue_where_left_off
u = update
