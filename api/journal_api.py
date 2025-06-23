#!/usr/bin/env python3
"""
Journal API Routes
Handles all journal-related operations for NCOS Voice Journal System
"""

import json
import logging
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Pydantic models
class JournalEntry(BaseModel):
    """Model for journal entries"""
    symbol: str = Field(..., description="Trading symbol (e.g., XAUUSD)")
    timeframe: Optional[str] = Field("H4", description="Timeframe (M15, H1, H4, D1)")
    bias: Optional[str] = Field(None, description="Market bias (bullish/bearish/neutral)")
    action: Optional[str] = Field(None, description="Action taken (mark/analyze/check)")
    notes: Optional[str] = Field(None, description="Additional notes or context")
    session_id: Optional[str] = Field(None, description="Trading session identifier")
    maturity_score: Optional[float] = Field(None, description="Setup maturity score (0-1)")
    entry_price: Optional[float] = Field(None, description="Entry price if trade taken")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class JournalQuery(BaseModel):
    """Model for journal queries"""
    symbol: Optional[str] = None
    session_id: Optional[str] = None
    bias: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_maturity: Optional[float] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class JournalStats(BaseModel):
    """Model for journal statistics"""
    total_entries: int
    unique_symbols: List[str]
    unique_sessions: List[str]
    avg_maturity_score: Optional[float]
    date_range: Dict[str, str]


# Helper functions
def get_journal_path() -> Path:
    """Get the journal file path from configuration"""
    # In production, load from config
    return Path("../logs/trade_journal.jsonl")


def read_journal_entries(filters: Optional[JournalQuery] = None) -> List[Dict]:
    """Read and filter journal entries"""
    journal_path = get_journal_path()

    if not journal_path.exists():
        return []

    entries = []
    try:
        with open(journal_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line}")
    except Exception as e:
        logger.error(f"Error reading journal: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading journal: {str(e)}")

    # Apply filters if provided
    if filters:
        filtered = entries

        if filters.symbol:
            filtered = [e for e in filtered if e.get("symbol", "").upper() == filters.symbol.upper()]

        if filters.session_id:
            filtered = [e for e in filtered if e.get("session_id") == filters.session_id]

        if filters.bias:
            filtered = [e for e in filtered if e.get("bias", "").lower() == filters.bias.lower()]

        if filters.start_date:
            filtered = [e for e in filtered if datetime.fromisoformat(e.get("timestamp", "")) >= filters.start_date]

        if filters.end_date:
            filtered = [e for e in filtered if datetime.fromisoformat(e.get("timestamp", "")) <= filters.end_date]

        if filters.min_maturity is not None:
            filtered = [e for e in filtered if e.get("maturity_score", 0) >= filters.min_maturity]

        # Apply pagination
        start = filters.offset
        end = start + filters.limit
        return filtered[start:end]

    return entries


def append_journal_entry(entry_data: Dict) -> Dict:
    """Append a new entry to the journal"""
    journal_path = get_journal_path()

    # Ensure journal directory exists
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    entry = {
        "trace_id": f"journal_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.utcnow().isoformat(),
        **entry_data
    }

    try:
        with open(journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Added journal entry: {entry['trace_id']}")
        return entry
    except Exception as e:
        logger.error(f"Error appending to journal: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing to journal: {str(e)}")


# API Routes

@router.post("/append", response_model=Dict)
async def append_entry(entry: JournalEntry):
    """Add a new entry to the journal"""
    entry_data = entry.dict(exclude_none=True)
    result = append_journal_entry(entry_data)
    return result


@router.get("/query", response_model=List[Dict])
async def query_entries(
        symbol: Optional[str] = Query(None, description="Filter by symbol"),
        session_id: Optional[str] = Query(None, description="Filter by session ID"),
        bias: Optional[str] = Query(None, description="Filter by bias"),
        start_date: Optional[datetime] = Query(None, description="Start date filter"),
        end_date: Optional[datetime] = Query(None, description="End date filter"),
        min_maturity: Optional[float] = Query(None, description="Minimum maturity score"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
        offset: int = Query(0, ge=0, description="Result offset")
):
    """Query journal entries with filters"""
    filters = JournalQuery(
        symbol=symbol,
        session_id=session_id,
        bias=bias,
        start_date=start_date,
        end_date=end_date,
        min_maturity=min_maturity,
        limit=limit,
        offset=offset
    )

    entries = read_journal_entries(filters)
    return entries


@router.get("/recap/{session_id}", response_model=Dict)
async def get_session_recap(session_id: str):
    """Get a recap/summary of a specific trading session"""
    filters = JournalQuery(session_id=session_id, limit=1000)
    entries = read_journal_entries(filters)

    if not entries:
        raise HTTPException(status_code=404, detail=f"No entries found for session: {session_id}")

    # Calculate statistics
    symbols = list(set(e.get("symbol", "") for e in entries if e.get("symbol")))
    maturity_scores = [e.get("maturity_score", 0) for e in entries if e.get("maturity_score") is not None]

    recap = {
        "session_id": session_id,
        "entry_count": len(entries),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "avg_maturity_score": sum(maturity_scores) / len(maturity_scores) if maturity_scores else None,
        "bias_distribution": {},
        "timeframe_distribution": {},
        "first_entry": entries[0].get("timestamp") if entries else None,
        "last_entry": entries[-1].get("timestamp") if entries else None,
        "high_maturity_setups": len([e for e in entries if e.get("maturity_score", 0) >= 0.8]),
        "entries": entries
    }

    # Calculate bias distribution
    for entry in entries:
        bias = entry.get("bias", "unknown")
        recap["bias_distribution"][bias] = recap["bias_distribution"].get(bias, 0) + 1

    # Calculate timeframe distribution
    for entry in entries:
        tf = entry.get("timeframe", "unknown")
        recap["timeframe_distribution"][tf] = recap["timeframe_distribution"].get(tf, 0) + 1

    return recap


@router.get("/stats", response_model=JournalStats)
async def get_journal_stats():
    """Get overall journal statistics"""
    entries = read_journal_entries()

    if not entries:
        return JournalStats(
            total_entries=0,
            unique_symbols=[],
            unique_sessions=[],
            avg_maturity_score=None,
            date_range={"start": None, "end": None}
        )

    # Calculate statistics
    symbols = list(set(e.get("symbol", "") for e in entries if e.get("symbol")))
    sessions = list(set(e.get("session_id", "") for e in entries if e.get("session_id")))
    maturity_scores = [e.get("maturity_score", 0) for e in entries if e.get("maturity_score") is not None]

    # Get date range
    timestamps = [e.get("timestamp") for e in entries if e.get("timestamp")]
    timestamps.sort()

    return JournalStats(
        total_entries=len(entries),
        unique_symbols=sorted(symbols),
        unique_sessions=sorted(sessions),
        avg_maturity_score=sum(maturity_scores) / len(maturity_scores) if maturity_scores else None,
        date_range={
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None
        }
    )


@router.delete("/entry/{trace_id}")
async def delete_entry(trace_id: str):
    """Delete a specific journal entry by trace_id"""
    journal_path = get_journal_path()

    if not journal_path.exists():
        raise HTTPException(status_code=404, detail="Journal file not found")

    # Read all entries
    entries = []
    found = False

    try:
        with open(journal_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("trace_id") != trace_id:
                        entries.append(entry)
                    else:
                        found = True
    except Exception as e:
        logger.error(f"Error reading journal: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading journal: {str(e)}")

    if not found:
        raise HTTPException(status_code=404, detail=f"Entry not found: {trace_id}")

    # Write back remaining entries
    try:
        with open(journal_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return {"message": f"Entry {trace_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error writing journal: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing journal: {str(e)}")


@router.get("/export/csv")
async def export_csv(
        session_id: Optional[str] = Query(None, description="Filter by session ID"),
        symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """Export journal entries as CSV"""
    filters = JournalQuery(
        session_id=session_id,
        symbol=symbol,
        limit=10000  # Large limit for export
    )

    entries = read_journal_entries(filters)

    if not entries:
        raise HTTPException(status_code=404, detail="No entries found for export")

    # Convert to DataFrame
    df = pd.DataFrame(entries)

    # Reorder columns for better readability
    column_order = [
        "timestamp", "trace_id", "symbol", "timeframe", "bias", "action",
        "maturity_score", "entry_price", "stop_loss", "take_profit",
        "session_id", "notes"
    ]

    # Only include columns that exist
    columns = [col for col in column_order if col in df.columns]
    df = df[columns]

    # Convert to CSV
    output = StringIO()
    df.to_csv(output, index=False)

    # Create filename
    filename_parts = ["journal_export"]
    if session_id:
        filename_parts.append(f"session_{session_id}")
    if symbol:
        filename_parts.append(f"symbol_{symbol}")
    filename_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    filename = "_".join(filename_parts) + ".csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/search")
async def search_entries(
        q: str = Query(..., description="Search query"),
        limit: int = Query(50, ge=1, le=200)
):
    """Search journal entries by text"""
    entries = read_journal_entries()

    # Simple text search across all string fields
    results = []
    search_term = q.lower()

    for entry in entries:
        # Search in various fields
        searchable_text = " ".join([
            str(entry.get("symbol", "")),
            str(entry.get("notes", "")),
            str(entry.get("bias", "")),
            str(entry.get("action", "")),
            str(entry.get("session_id", ""))
        ]).lower()

        if search_term in searchable_text:
            results.append(entry)

        if len(results) >= limit:
            break

    return {
        "query": q,
        "count": len(results),
        "results": results
    }


@router.post("/backup")
async def create_backup():
    """Create a backup of the journal"""
    journal_path = get_journal_path()

    if not journal_path.exists():
        raise HTTPException(status_code=404, detail="Journal file not found")

    # Create backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = journal_path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"journal_{timestamp}.jsonl"

    try:
        # Copy journal to backup
        import shutil
        shutil.copy2(journal_path, backup_path)

        return {
            "message": "Backup created successfully",
            "backup_file": str(backup_path),
            "timestamp": timestamp
        }
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating backup: {str(e)}")
