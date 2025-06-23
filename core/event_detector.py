# zanzibar/analysis/wyckoff/event_detector.py
# Version: 5.0 (ZBar includes Heuristic Delta Calculation)

# ... (imports and MarketOrderData remain the same) ...
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Any # Added List, Any
import pandas as pd # Added pandas
import numpy as np # Added numpy
import logging # Added logging

log = logging.getLogger(__name__)

@dataclass
class MarketOrderData:
    """ Represents aggregated order flow at a price level inside a ZBar. """
    bid_volume: int = 0
    ask_volume: int = 0
    total_volume: int = 0
    delta: int = 0  # ask_volume - bid_volume

    def update_from_tick_aggression(self, volume: int, is_buyer_initiated: bool):
        """Updates volumes based on inferred tick aggression."""
        self.total_volume += volume
        if is_buyer_initiated:
            self.ask_volume += volume
            self.delta += volume
        else:
            self.bid_volume += volume
            self.delta -= volume

@dataclass
class ZBar:
    """
    ZBAR™ - Proprietary Market Data Object
    Captures enhanced volume & delta metrics per bar for VSA/Wyckoff analysis.
    V5: Includes heuristic delta calculation from OHLCV.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int # Total bar volume from sum of ticks or source
    price_ladder: Dict[float, MarketOrderData] = field(default_factory=dict)

    # Derived metrics
    bar_delta: Optional[int] = None # Can be accurate (from ladder) or heuristic
    poc_price: Optional[float] = None
    poi_price: Optional[float] = None
    bid_volume_total: Optional[int] = None
    ask_volume_total: Optional[int] = None
    # _bar_delta: Optional[int] = field(init=False, default=None) # V4 internal cache removed for simplicity

    # @property - Removed property to allow direct setting by heuristics/ladder
    # def bar_delta(self) -> Optional[int]:
    #     """Calculates or retrieves the bar's total delta."""
    #     if self._bar_delta is None:
    #         self.calculate_derived_metrics() # Calculate if not already done
    #     return self._bar_delta

    def calculate_derived_metrics_from_ladder(self) -> Dict[str, Any]:
        """
        Populate metrics (delta, POC, POI, bid/ask totals) from the price_ladder.
        Call this AFTER the price_ladder is fully populated (e.g., by Tick Processor).
        """
        log.debug(f"Calculating derived metrics from LADDER for ZBar at {self.timestamp}")
        if not self.price_ladder:
            log.warning(f"Price ladder is empty for ZBar at {self.timestamp}. Cannot calculate metrics from ladder.")
            # Keep existing values or set defaults? Let's keep existing for now.
            # self.bar_delta = 0
            # self.bid_volume_total = 0
            # self.ask_volume_total = 0
            # self.poc_price = None
            # self.poi_price = None
            return {}

        current_bar_delta = 0
        current_bid_total = 0
        current_ask_total = 0
        max_vol_at_price = -1
        min_vol_at_price = float('inf')
        calculated_poc = None
        calculated_poi = None

        for price, mo in self.price_ladder.items():
            current_bid_total += mo.bid_volume
            current_ask_total += mo.ask_volume
            current_bar_delta += mo.delta
            tv = mo.total_volume
            if tv > max_vol_at_price:
                max_vol_at_price = tv
                calculated_poc = price
            if 0 < tv < min_vol_at_price:
                min_vol_at_price = tv
                calculated_poi = price

        self.bid_volume_total = current_bid_total
        self.ask_volume_total = current_ask_total
        self.bar_delta = current_bar_delta # Accurate delta from ladder
        self.poc_price = calculated_poc
        self.poi_price = calculated_poi

        log.debug(f"ZBar {self.timestamp} (Ladder): Delta={self.bar_delta}, POC={self.poc_price}, POI={self.poi_price}")
        return { "bar_delta": self.bar_delta, "poc_price": self.poc_price, "poi_price": self.poi_price,
                 "bid_volume_total": self.bid_volume_total, "ask_volume_total": self.ask_volume_total }

    def calculate_heuristic_delta(self) -> Optional[int]:
        """
        Calculates a heuristic delta based on OHLC and Volume if price_ladder is empty.
        Rule: Assigns volume portion to delta based on close location.
              Close near high -> positive delta. Close near low -> negative delta.
        Returns:
            int: Estimated bar delta.
        """
        if self.price_ladder: # If we have real ladder data, use that delta
             if self.bar_delta is None: self.calculate_derived_metrics_from_ladder()
             return self.bar_delta

        log.debug(f"Calculating HEURISTIC delta for ZBar at {self.timestamp}")
        spread = self.high - self.low
        if pd.isna(spread) or spread < 1e-9 or self.volume == 0:
            self.bar_delta = 0 # Assign zero delta if no range or no volume
            return self.bar_delta

        close_location = (self.close - self.low) / spread # 0 = close at low, 1 = close at high

        # Simple linear heuristic: delta = volume * (2 * close_location - 1)
        # This maps close_location [0, 1] to delta [-volume, +volume]
        heuristic_delta = int(round(self.volume * (2 * close_location - 1)))
        self.bar_delta = heuristic_delta
        log.debug(f"ZBar {self.timestamp} (Heuristic): CloseLoc={close_location:.2f} -> Delta={self.bar_delta}")
        return self.bar_delta


    def classify_volume_signature(self, volume_window: List[int], threshold_z: float = 2.0) -> str:
        """
        Classifies this ZBar's volume signature as 'climax', 'absorption', or 'normal'.
        Uses z-score threshold on volume.
        Args:
            volume_window (List[int]): Recent volume values for z-score comparison.
            threshold_z (float): Z-score threshold to classify as anomaly.
        Returns:
            str: 'climax', 'absorption', or 'normal'
        """
        if len(volume_window) < 10 or self.volume is None:
            return "normal"

        mean_vol = np.mean(volume_window)
        std_vol = np.std(volume_window)
        if std_vol == 0:
            return "normal"

        z_score = (self.volume - mean_vol) / std_vol

        if z_score >= threshold_z:
            return "climax"
        elif z_score <= -threshold_z:
            return "absorption"
        else:
            return "normal"


    # --- update_from_tick and other methods would go here if building ZBar from ticks ---
    # For now, assume ZBar is created from bar data by the mapper


# --- Keep the rest of event_detector.py (helpers, detector functions) as is (V4) ---

# ... (calculate_volume_stats, calculate_spread_stats, is_down_trend, ...)
# ... (detect_stopping_action, detect_automatic_rally_reaction, ...)
# ... (detect_secondary_test, detect_spring, detect_upthrust, ...)

from typing import Optional, Dict, Any

def find_initial_wyckoff_events(ohlcv: List[ZBar], config: Optional[Dict[str, Any]] = None):
    """
    Main event detection entrypoint. Returns a list of (bar_idx, event_label).
    Accepts config for lookback windows and threshold settings.
    """
    if config is None:
        config = {}

    events = []
    n = len(ohlcv)
    for i in range(n):
        # Example: use config-driven lookback for stopping action
        stopping_action_window = config.get('stopping_action_window', 5)
        recent = ohlcv[max(0, i-stopping_action_window+1):i+1]
        # Example logic for detecting stopping action
        # Use enriched indicator, e.g., bar_delta, volume, etc.
        curr_bar = ohlcv[i]
        bar_delta = getattr(curr_bar, 'bar_delta', None)
        if pd.isna(bar_delta):
            bar_delta = None
        volume = getattr(curr_bar, 'volume', None)
        if pd.isna(volume):
            volume = None
        # Example event detection (replace with actual logic as needed)
        if volume is not None and bar_delta is not None:
            # Dummy example: if current volume > max in recent window AND delta flips sign
            max_prev_vol = max([getattr(z, 'volume', 0) for z in recent[:-1]] or [0])
            prev_delta = getattr(recent[-2], 'bar_delta', None) if len(recent) > 1 else None
            if volume > max_prev_vol and prev_delta is not None and not pd.isna(prev_delta):
                # Stopping action: large volume, delta sign change
                if np.sign(bar_delta) != np.sign(prev_delta):
                    events.append((i, "stopping_action"))

        # Example: automatic rally/reaction with config-driven window
        rally_window = config.get('automatic_rally_window', 10)
        rally_lookback = ohlcv[max(0, i-rally_window+1):i+1]
        # Dummy logic for rally: price closes above prior highs in window
        close = getattr(curr_bar, 'close', None)
        if not pd.isna(close):
            max_prev_high = max([getattr(z, 'high', float('-inf')) for z in rally_lookback[:-1]] or [float('-inf')])
            if close > max_prev_high:
                events.append((i, "automatic_rally"))

        # Add more event detections as needed, using config.get('<event>_window', <default>)

    return events


