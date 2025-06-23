## 2. Fixed Menu System

```python
# ncos_v5_ultimate_menu.py
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

class UltimateMenuSystem:
    def __init__(self):
        self.state = "main"
        self.context = {}
        self.history = []
        
    def get_main_menu(self) -> Dict:
        return {
            "title": "🎯 ncOS Trading System v5",
            "ascii_art": """
    ███╗   ██╗ ██████╗ ██████╗ ███████╗
    ████╗  ██║██╔════╝██╔═══██╗██╔════╝
    ██╔██╗ ██║██║     ██║   ██║███████╗
    ██║╚██╗██║██║     ██║   ██║╚════██║
    ██║ ╚████║╚██████╗╚██████╔╝███████║
    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝
            """,
            "options": [
                {"id": "1", "label": "📊 Market Analysis", "icon": "📊"},
                {"id": "2", "label": "🔍 Pattern Scanner", "icon": "🔍"},
                {"id": "3", "label": "⚡ Quick Analysis", "icon": "⚡"},
                {"id": "4", "label": "📈 Strategy Execution", "icon": "📈"},
                {"id": "5", "label": "📁 Data Management", "icon": "📁"},
                {"id": "6", "label": "🎨 Chart Marking", "icon": "🎨"},
                {"id": "7", "label": "🔧 System Config", "icon": "🔧"},
                {"id": "8", "label": "📚 Help & Docs", "icon": "📚"}
            ],
            "prompt": "Select option (1-8):",
            "footer": "💡 Tip: Type 'quick XAUUSD' for instant analysis"
        }
    
    def get_pair_menu(self) -> Dict:
        return {
            "title": "💱 Select Trading Pair",
            "options": [
                {"id": "1", "label": "XAUUSD (Gold/USD)", "value": "XAUUSD"},
                {"id": "2", "label": "EURUSD", "value": "EURUSD"},
                {"id": "3", "label": "GBPUSD", "value": "GBPUSD"},
                {"id": "4", "label": "BTCUSD", "value": "BTCUSD"},
                {"id": "5", "label": "USDJPY", "value": "USDJPY"},
                {"id": "6", "label": "AUDUSD", "value": "AUDUSD"},
                {"id": "7", "label": "Custom Pair...", "action": "custom_pair"},
                {"id": "0", "label": "← Back to Main", "action": "main"}
            ],
            "prompt": "Select pair or 0 to go back:"
        }
    
    def get_timeframe_menu(self) -> Dict:
        return {
            "title": f"⏰ Timeframe for {self.context.get('pair', 'Unknown')}",
            "options": [
                {"id": "1", "label": "M1 (1 Minute)", "value": "M1"},
                {"id": "2", "label": "M5 (5 Minutes)", "value": "M5"},
                {"id": "3", "label": "M15 (15 Minutes)", "value": "M15"},
                {"id": "4", "label": "M30 (30 Minutes)", "value": "M30"},
                {"id": "5", "label": "H1 (1 Hour)", "value": "H1"},
                {"id": "6", "label": "H4 (4 Hours)", "value": "H4"},
                {"id": "7", "label": "D1 (Daily)", "value": "D1"},
                {"id": "8", "label": "Multi-Timeframe Analysis", "action": "multi_tf"},
                {"id": "0", "label": "← Back", "action": "back"}
            ],
            "prompt": "Select timeframe:"
        }
    
    def get_analysis_menu(self) -> Dict:
        pair = self.context.get('pair', 'Unknown')
        tf = self.context.get('timeframe', 'Unknown')
        
        return {
            "title": f"🔬 Analysis Type for {pair} {tf}",
            "options": [
                {"id": "1", "label": "📊 Full Market Analysis", "value": "full"},
                {"id": "2", "label": "🏛️ Smart Money Concepts", "value": "smc"},
                {"id": "3", "label": "📈 Wyckoff Method", "value": "wyckoff"},
                {"id": "4", "label": "🌊 Market Structure", "value": "structure"},
                {"id": "5", "label": "💧 Liquidity Analysis", "value": "liquidity"},
                {"id": "6", "label": "📉 Volume Profile", "value": "volume"},
                {"id": "7", "label": "🔄 Combined Analysis", "value": "combined"},
                {"id": "0", "label": "← Back", "action": "back"}
            ],
            "prompt": "Select analysis type:"
        }
    
    def process_selection(self, choice: str) -> Dict:
        """Process user menu selection"""
        
        # Quick commands
        if choice.lower().startswith("quick "):
            pair = choice.split()[1].upper()
            return self._quick_analysis(pair)
        
        # State-based processing
        if self.state == "main":
            return self._process_main_menu(choice)
        elif self.state == "pair_select":
            return self._process_pair_menu(choice)
        elif self.state == "timeframe_select":
            return self._process_timeframe_menu(choice)
        elif self.state == "analysis_select":
            return self._process_analysis_menu(choice)
        
        return {"error": "Unknown state"}
    
    def _process_main_menu(self, choice: str) -> Dict:
        if choice == "1":
            self.state = "pair_select"
            return {"menu": self.get_pair_menu()}
        elif choice == "3":  # Quick Analysis
            return self._quick_analysis("XAUUSD")
        elif choice == "5":  # Data Management
            return {"action": "scan_data"}
        else:
            return {"menu": self.get_main_menu()}
    
    def _process_pair_menu(self, choice: str) -> Dict:
        if choice == "0":
            self.state = "main"
            return {"menu": self.get_main_menu()}
        
        pairs = {
            "1": "XAUUSD",
            "2": "EURUSD",
            "3": "GBPUSD",
            "4": "BTCUSD",
            "5": "USDJPY",
            "6": "AUDUSD"
        }
        
        if choice in pairs:
            self.context["pair"] = pairs[choice]
            self.state = "timeframe_select"
            return {"menu": self.get_timeframe_menu()}
        
        return {"menu": self.get_pair_menu()}
    
    def _process_timeframe_menu(self, choice: str) -> Dict:
        if choice == "0":
            self.state = "pair_select"
            return {"menu": self.get_pair_menu()}
        
        timeframes = {
            "1": "M1", "2": "M5", "3": "M15", 
            "4": "M30", "5": "H1", "6": "H4", "7": "D1"
        }
        
        if choice in timeframes:
            self.context["timeframe"] = timeframes[choice]
            self.state = "analysis_select"
            return {"menu": self.get_analysis_menu()}
        
        return {"menu": self.get_timeframe_menu()}
    
    def _process_analysis_menu(self, choice: str) -> Dict:
        if choice == "0":
            self.state = "timeframe_select"
            return {"menu": self.get_timeframe_menu()}
        
        analysis_types = {
            "1": "full", "2": "smc", "3": "wyckoff",
            "4": "structure", "5": "liquidity", 
            "6": "volume", "7": "combined"
        }
        
        if choice in analysis_types:
            self.context["analysis_type"] = analysis_types[choice]
            return {
                "action": "execute_analysis",
                "params": self.context,
                "reset": True
            }
        
        return {"menu": self.get_analysis_menu()}
    
    def _quick_analysis(self, pair: str) -> Dict:
        return {
            "action": "quick_analysis",
            "params": {"pair": pair},
            "message": f"🚀 Running quick analysis for {pair}..."
        }