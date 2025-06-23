"""
ncOS Journal API - Phoenix Edition
Focused on journaling without voice dependencies
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="ncOS Journal API",
    description="Trading journal and analysis API",
    version="21.7"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class TradeEntry(BaseModel):
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: Optional[str] = None
    notes: Optional[str] = None
    patterns: Optional[List[str]] = []
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


class JournalEntry(BaseModel):
    title: str
    content: str
    category: Optional[str] = "general"
    tags: Optional[List[str]] = []
    timestamp: Optional[str] = None


class AnalysisEntry(BaseModel):
    symbol: str
    analysis_type: str
    content: Dict[str, Any]
    timestamp: Optional[str] = None


# Data storage paths
DATA_DIR = Path("../data")
TRADES_FILE = DATA_DIR / "trades.jsonl"
JOURNAL_FILE = DATA_DIR / "journal.jsonl"
ANALYSIS_FILE = DATA_DIR / "analysis.jsonl"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)


# Helper functions
def append_jsonl(file_path: Path, data: dict):
    """Append data to JSONL file"""
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')


def read_jsonl(file_path: Path) -> List[dict]:
    """Read all entries from JSONL file"""
    if not file_path.exists():
        return []

    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


# API Routes
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "ncOS Journal API - Phoenix Edition",
        "version": "21.7",
        "endpoints": {
            "trades": "/trades",
            "journal": "/journal",
            "analysis": "/analysis",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_files": {
            "trades": TRADES_FILE.exists(),
            "journal": JOURNAL_FILE.exists(),
            "analysis": ANALYSIS_FILE.exists()
        }
    }


# Trade endpoints
@app.post("/trades")
def create_trade(trade: TradeEntry):
    """Create a new trade entry"""
    trade_data = trade.dict()
    trade_data["timestamp"] = trade_data.get("timestamp") or datetime.now().isoformat()
    trade_data["id"] = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    append_jsonl(TRADES_FILE, trade_data)
    return {"message": "Trade created", "id": trade_data["id"]}


@app.get("/trades")
def get_trades(symbol: Optional[str] = None, session_id: Optional[str] = None):
    """Get all trades with optional filtering"""
    trades = read_jsonl(TRADES_FILE)

    if symbol:
        trades = [t for t in trades if t.get("symbol") == symbol]

    if session_id:
        trades = [t for t in trades if t.get("session_id") == session_id]

    return {"trades": trades, "count": len(trades)}


# Journal endpoints
@app.post("/journal")
def create_journal_entry(entry: JournalEntry):
    """Create a new journal entry"""
    entry_data = entry.dict()
    entry_data["timestamp"] = entry_data.get("timestamp") or datetime.now().isoformat()
    entry_data["id"] = f"journal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    append_jsonl(JOURNAL_FILE, entry_data)
    return {"message": "Journal entry created", "id": entry_data["id"]}


@app.get("/journal")
def get_journal_entries(category: Optional[str] = None, limit: int = 100):
    """Get journal entries with optional filtering"""
    entries = read_jsonl(JOURNAL_FILE)

    if category:
        entries = [e for e in entries if e.get("category") == category]

    # Sort by timestamp (newest first) and limit
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    entries = entries[:limit]

    return {"entries": entries, "count": len(entries)}


# Analysis endpoints
@app.post("/analysis")
def create_analysis(analysis: AnalysisEntry):
    """Create a new analysis entry"""
    analysis_data = analysis.dict()
    analysis_data["timestamp"] = analysis_data.get("timestamp") or datetime.now().isoformat()
    analysis_data["id"] = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    append_jsonl(ANALYSIS_FILE, analysis_data)
    return {"message": "Analysis created", "id": analysis_data["id"]}


@app.get("/analysis")
def get_analysis(symbol: Optional[str] = None, analysis_type: Optional[str] = None):
    """Get analysis entries with optional filtering"""
    analyses = read_jsonl(ANALYSIS_FILE)

    if symbol:
        analyses = [a for a in analyses if a.get("symbol") == symbol]

    if analysis_type:
        analyses = [a for a in analyses if a.get("analysis_type") == analysis_type]

    return {"analyses": analyses, "count": len(analyses)}


# Statistics endpoint
@app.get("/stats")
def get_statistics():
    """Get journal statistics"""
    trades = read_jsonl(TRADES_FILE)
    journal_entries = read_jsonl(JOURNAL_FILE)
    analyses = read_jsonl(ANALYSIS_FILE)

    # Calculate trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get("exit_price", 0) > t.get("entry_price", 0))

    # Get unique symbols
    symbols = list(set(t.get("symbol", "") for t in trades if t.get("symbol")))

    # Get pattern statistics
    all_patterns = []
    for trade in trades:
        all_patterns.extend(trade.get("patterns", []))

    pattern_counts = {}
    for pattern in all_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    return {
        "trades": {
            "total": total_trades,
            "winning": winning_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0
        },
        "journal_entries": len(journal_entries),
        "analyses": len(analyses),
        "symbols": symbols,
        "patterns": pattern_counts,
        "last_update": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
