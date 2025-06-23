# /xanflow/quarry_tools/swing_engine.py
"""
SwingEngine - Utility for identifying swing points and structural breaks (BoS/CHoCH).

This module consolidates logic for robust swing detection and market structure
analysis, designed to be used by other core modules like StructureValidator.
Logic is primarily derived and refactored from 
GEMINI/_py_repo/confirmation_engine_smc.py.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field # type: ignore

# --- Constants & Pydantic Models ---

class SwingType:
    HIGH = "High"
    LOW = "Low"

class StructureBreakType:
    BOS = "BoS"  # Break of Structure (in the direction of the prevailing trend)
    CHOCH = "CHoCH" # Change of Character (a break against the recent micro-trend, potentially signaling reversal)

class SwingPoint(BaseModel):
    timestamp: pd.Timestamp
    price: float
    type: Literal[SwingType.HIGH, SwingType.LOW]
    iloc: int # Integer location in the source DataFrame slice (relative to the analyzed window)

    class Config:
        arbitrary_types_allowed = True # For pd.Timestamp

class StructureBreak(BaseModel):
    break_timestamp: pd.Timestamp       # Timestamp of the candle that closed beyond the swing
    break_price: float                  # Closing price of the breaking candle
    break_type: Literal[StructureBreakType.BOS, StructureBreakType.CHOCH]
    broken_swing: SwingPoint            # The swing point that was broken
    breaking_candle_iloc: int           # iloc of the breaking candle in the analyzed window
    
    class Config:
        arbitrary_types_allowed = True


# --- Configuration Model ---
class SwingEngineConfig(BaseModel):
    swing_n: int = Field(default=1, ge=1, description="Number of bars on each side for strict fractal swing point definition (e.g., n=1 means H[i] > H[i-1] & H[i] > H[i+1]).")
    break_on_close: bool = Field(default=True, description="If True, a structural break is confirmed by candle close. If False, logic would need to consider wick penetration (currently uses close for break_price, but checks High/Low for break detection).")

# --- SwingEngine Class ---
class SwingEngine:
    MODULE = "SwingEngine"

    def __init__(self, config: SwingEngineConfig, logger: Optional[Any] = None):
        self.config = config
        self._log = logger or (lambda *args, **kwargs: None) 
        self._log_event("initialized", {"config": self.config.model_dump()})

    def _log_event(self, event_name: str, details: Optional[Dict[str, Any]] = None):
        payload = {"module": self.MODULE, "event": event_name}
        if details:
            payload.update(details)
        if self._log and callable(self._log):
             try: 
                self._log(self.MODULE, payload) 
             except TypeError:
                pass


    def find_swing_points(self, price_series: pd.Series, swing_type: Literal[SwingType.HIGH, SwingType.LOW]) -> List[SwingPoint]:
        """
        Identifies local swing points (highs or lows) in a price series
        using a strict fractal definition: a point is higher/lower than 'n' bars on each side.

        Args:
            price_series (pd.Series): Price series (typically 'High' or 'Low' column of a DataFrame).
                                      The series must have a DatetimeIndex.
            swing_type (Literal[SwingType.HIGH, SwingType.LOW]): Type of swings to find.

        Returns:
            List[SwingPoint]: A list of identified SwingPoint objects, sorted by timestamp.
        """
        n = self.config.swing_n
        if not isinstance(price_series, pd.Series) or price_series.empty or len(price_series) < (2 * n + 1):
            self._log_event("find_swing_points_insufficient_data", {"series_len": len(price_series), "required_for_strict_fractal": 2 * n + 1, "type": swing_type})
            return []
        if not isinstance(price_series.index, pd.DatetimeIndex):
            self._log_event("find_swing_points_invalid_index", {"index_type": type(price_series.index)})
            return []

        swings_data: List[SwingPoint] = []
        # Iterate from n-th element to (len - n - 1)-th element to allow for n bars on each side
        for i in range(n, len(price_series) - n):
            current_price = price_series.iloc[i]
            is_strict_swing = True
            if swing_type == SwingType.HIGH:
                for j in range(1, n + 1): # Check n bars to the left and n to the right
                    if not (current_price > price_series.iloc[i - j] and current_price > price_series.iloc[i + j]):
                        is_strict_swing = False
                        break
            elif swing_type == SwingType.LOW:
                for j in range(1, n + 1):
                    if not (current_price < price_series.iloc[i - j] and current_price < price_series.iloc[i + j]):
                        is_strict_swing = False
                        break
            else: # Should not be reached due to Literal typing
                is_strict_swing = False 

            if is_strict_swing:
                swings_data.append(SwingPoint(
                    timestamp=price_series.index[i],
                    price=current_price,
                    type=swing_type,
                    iloc=i 
                ))
        
        self._log_event("find_swing_points_result", {"type": swing_type, "count": len(swings_data)})
        return swings_data # Already sorted by nature of iteration if input series is time-sorted

    def _get_last_relevant_swings(
        self, 
        current_candle_ts: pd.Timestamp, 
        all_sorted_swings: List[SwingPoint] 
    ) -> Tuple[Optional[SwingPoint], Optional[SwingPoint]]:
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None
        for swing in reversed(all_sorted_swings): 
            if swing.timestamp < current_candle_ts:
                if swing.type == SwingType.HIGH and last_high is None:
                    last_high = swing
                elif swing.type == SwingType.LOW and last_low is None:
                    last_low = swing
            if last_high and last_low: 
                break
        return last_high, last_low

    def label_structural_breaks(
        self,
        price_df: pd.DataFrame, 
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[StructureBreak]:
        breaks: List[StructureBreak] = []
        if price_df.empty or not isinstance(price_df.index, pd.DatetimeIndex):
            self._log_event("label_structural_breaks_invalid_price_df")
            return breaks
        if not swing_highs and not swing_lows: # No swings to break
            self._log_event("label_structural_breaks_no_swings_provided")
            return breaks

        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.timestamp)
        if not all_swings: # Should be redundant if above check passes
            return breaks
            
        broken_swing_timestamps: set[pd.Timestamp] = set() 
        current_micro_trend: Optional[Literal["bullish", "bearish"]] = None 
        
        if len(all_swings) >= 2:
            s1, s2 = all_swings[0], all_swings[1]
            if s1.type == SwingType.LOW and s2.type == SwingType.HIGH and s2.price > s1.price:
                current_micro_trend = "bullish"
            elif s1.type == SwingType.HIGH and s2.type == SwingType.LOW and s2.price < s1.price:
                current_micro_trend = "bearish"

        for i in range(len(price_df)):
            current_candle_ts = price_df.index[i]
            current_candle = price_df.iloc[i]
            
            # Determine price to check against swing based on config
            price_for_break_check: float
            if self.config.break_on_close:
                price_for_break_check = current_candle["Close"]
            # Else, if break_on_close is False, specific logic for wick breaks is needed
            # For bullish break (of SH), use current High. For bearish (of SL), use current Low.
            # This part of the logic is simplified; a full wick-break implementation would be more explicit here.
            # The break_price in StructureBreak is still recorded as Close.
            
            last_relevant_high, last_relevant_low = self._get_last_relevant_swings(current_candle_ts, all_swings)

            # Check for bullish break (break of a prior swing high)
            if last_relevant_high and last_relevant_high.timestamp not in broken_swing_timestamps:
                comparison_price = current_candle["High"] if not self.config.break_on_close else current_candle["Close"]
                if comparison_price > last_relevant_high.price:
                    break_type = StructureBreakType.CHOCH if current_micro_trend == "bearish" else StructureBreakType.BOS
                    
                    breaks.append(StructureBreak(
                        break_timestamp=current_candle_ts,
                        break_price=current_candle["Close"], # Break is always confirmed by close for this field
                        break_type=break_type,
                        broken_swing=last_relevant_high,
                        breaking_candle_iloc=i
                    ))
                    broken_swing_timestamps.add(last_relevant_high.timestamp)
                    current_micro_trend = "bullish" 
                    self._log_event("bullish_structural_break_added", {"type": break_type.value, "broken_swing_price": last_relevant_high.price, "break_price": current_candle["Close"], "timestamp": str(current_candle_ts)})

            # Check for bearish break (break of a prior swing low)
            if last_relevant_low and last_relevant_low.timestamp not in broken_swing_timestamps:
                comparison_price = current_candle["Low"] if not self.config.break_on_close else current_candle["Close"]
                if comparison_price < last_relevant_low.price:
                    break_type = StructureBreakType.CHOCH if current_micro_trend == "bullish" else StructureBreakType.BOS
                        
                    breaks.append(StructureBreak(
                        break_timestamp=current_candle_ts,
                        break_price=current_candle["Close"], # Break is always confirmed by close for this field
                        break_type=break_type,
                        broken_swing=last_relevant_low,
                        breaking_candle_iloc=i
                    ))
                    broken_swing_timestamps.add(last_relevant_low.timestamp)
                    current_micro_trend = "bearish" 
                    self._log_event("bearish_structural_break_added", {"type": break_type.value, "broken_swing_price": last_relevant_low.price, "break_price": current_candle["Close"], "timestamp": str(current_candle_ts)})
        
        breaks.sort(key=lambda x: x.break_timestamp) 
        return breaks
