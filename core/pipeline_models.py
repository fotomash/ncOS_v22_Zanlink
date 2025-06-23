from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataRequest(BaseModel):
    """Request to fetch data for a symbol and set of timeframes."""

    symbol: str
    timeframes: List[str]


class DataResult(BaseModel):
    """Output from the data pipeline."""

    status: str
    symbol: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalysisResult(BaseModel):
    """Result from the analysis engine."""

    status: str
    symbol: str
    analysis: Dict[str, Any] = Field(default_factory=dict)
    confluence_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StrategyMatch(BaseModel):
    """Result from the strategy engine."""

    strategy: Optional[str] = None
    confidence: float = 0.0
    status: str = "no_match"


class ExecutionResult(BaseModel):
    """Result from the execution engine."""

    status: str
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    direction: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
