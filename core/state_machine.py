from typing import Optional
# zanzibar/analysis/wyckoff/state_machine.py
# Author: Tomasz Laskowski (& Gemini Co-pilot)
# License: Proprietary / Private
# Created: 2025-05-07
# Version: 3.0 (Context-Aware Phase Transition Logic)
# Description: Manages Wyckoff phase transitions based on detected event sequences and ZBar context.

import logging
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum, auto
import pandas as pd # For timestamp handling in logs
from datetime import datetime, timezone # For transition log timestamp

# Assuming ZBar is defined elsewhere and will be passed for context
# For development, using the ZBar definition that includes heuristic delta
try:
    # If ZBar is in a central models.py
    # from zanzibar.data_management.models import ZBar, MarketOrderData
    # For now, assuming it's accessible from where event_detector defines it
    from zanzibar.analysis.wyckoff.event_detector import ZBar, MarketOrderData
except ImportError: # pragma: no cover
    # Fallback placeholder if imports fail during isolated development/testing
    from dataclasses import dataclass, field
    @dataclass
    class MarketOrderData: # Minimal
        bid_volume: int = 0; ask_volume: int = 0; total_volume: int = 0; delta: int = 0
    @dataclass
    class ZBar: # Minimal
        timestamp: datetime; open: float; high: float; low: float; close: float; volume: int
        price_ladder: Dict[float, MarketOrderData] = field(default_factory=dict)
        bar_delta: Optional[int] = None; poc_price: Optional[float] = None; poi_price: Optional[float] = None
        bid_volume_total: Optional[int] = None; ask_volume_total: Optional[int] = None
        def calculate_heuristic_delta(self): self.bar_delta = 0 # Dummy
        def calculate_derived_metrics_from_ladder(self): pass # Dummy
        def __post_init__(self): # Ensure delta is calculated if not provided
            if self.bar_delta is None and not self.price_ladder: self.calculate_heuristic_delta()
            elif self.bar_delta is None and self.price_ladder: self.calculate_derived_metrics_from_ladder()

# --- VOLUME SIGNATURE ANNOTATION PATCH ---
# If detect_events is defined elsewhere, this is a placeholder for context.
def detect_events(zbars: List[ZBar]) -> List[dict]:
    """
    Example event detection function that processes ZBars and annotates events with volume signature.
    """
    events = []
    for i, bar in enumerate(zbars):
        # Dummy event detection logic for illustration
        event = {
            "index": i,
            "event_type": "DummyEvent",
        }
        # Precompute volume window (last 20 bars, not including current)
        volume_window = [b.volume for b in zbars[max(0, i-20):i]]
        # Classify volume signature
        if hasattr(bar, "classify_volume_signature"):
            volume_class = bar.classify_volume_signature(volume_window)
        else:
            # Fallback: always normal
            volume_class = "normal"
        event["volume_signature"] = volume_class
        events.append(event)
    return events


log = logging.getLogger(__name__)
if not log.hasHandlers(): # pragma: no cover
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')


class WyckoffPhase(Enum):
    UNKNOWN = auto()
    ACCUMULATION_A = auto()
    ACCUMULATION_B = auto()
    ACCUMULATION_C = auto()
    ACCUMULATION_D = auto()
    ACCUMULATION_E = auto()
    DISTRIBUTION_A = auto()
    DISTRIBUTION_B = auto()
    DISTRIBUTION_C = auto()
    DISTRIBUTION_D = auto()
    DISTRIBUTION_E = auto()
    # REACCUMULATION = auto() # Future
    # REDISTRIBUTION = auto() # Future

    def __str__(self):
        return self.name # Keep it simple for logging, title case for display later if needed

class WyckoffStateMachine:
    """
    Manages Wyckoff phase transitions based on a sequence of detected events
    and contextual ZBar data. V3 implements context updates and smarter transitions.
    """
    def __init__(self, config: Optional[Dict] = None):
        if config is None: config = {}
        self._load_transition_rules(config.get("phase_transitions", {}))

        self.current_phase: WyckoffPhase = WyckoffPhase.UNKNOWN
        self.detected_events_indices: Dict[str, List[int]] = {}
        self.event_sequence: List[Tuple[int, str]] = [] # Chronological (index, event_type)
        self.transition_log: List[Dict[str, Any]] = []
        self.event_timeframes: List[Optional[str]] = []
        
        self.schematic_type: Optional[str] = None # "Accumulation" or "Distribution"
        self.trading_range: Dict[str, Optional[float]] = {"support": None, "resistance": None}
        self.last_significant_low_idx: Optional[int] = None
        self.last_significant_high_idx: Optional[int] = None
        
        self.zbars_context: List[ZBar] = [] # To store the ZBars list passed on first event

        log.info(f"WyckoffStateMachine initialized. Phase: {self.current_phase}")

    def _load_transition_rules(self, rules_config: Dict):
        """Loads transition rules. Keys are current phases (str), values are dicts of event_type: next_phase (str)."""
        self.rules: Dict[WyckoffPhase, Dict[str, WyckoffPhase]] = {}
        # Default rules define basic event -> next_phase mappings
        default_rules = {
            "UNKNOWN": {"PS": "ACCUMULATION_A", "SC": "ACCUMULATION_A", "PSY": "DISTRIBUTION_A", "BC": "DISTRIBUTION_A"},
            "ACCUMULATION_A": {"AR_acc": "ACCUMULATION_B"},
            "ACCUMULATION_B": {"ST_Acc": "ACCUMULATION_B", "ST_Acc_Weak": "ACCUMULATION_B", "Spring": "ACCUMULATION_C", "Spring_Weak": "ACCUMULATION_C"},
            "ACCUMULATION_C": {"Test": "ACCUMULATION_D", "LPS": "ACCUMULATION_D", "SOS": "ACCUMULATION_D"}, # Test of Spring/LPS
            "ACCUMULATION_D": {"BU": "ACCUMULATION_E", "SOS": "ACCUMULATION_E", "LPS": "ACCUMULATION_D"}, # BU = Back-up to creek
            "ACCUMULATION_E": {"LPS": "ACCUMULATION_E"}, # Trend continuation
            "DISTRIBUTION_A": {"AR_dist": "DISTRIBUTION_B"},
            "DISTRIBUTION_B": {"ST_dist": "DISTRIBUTION_B", "UT": "DISTRIBUTION_C", "UT_Weak": "DISTRIBUTION_C"}, # Upthrust
            "DISTRIBUTION_C": {"UTAD": "DISTRIBUTION_D", "Test_UTAD": "DISTRIBUTION_D", "LPSY": "DISTRIBUTION_D", "SOW": "DISTRIBUTION_D"}, # UTAD = Upthrust After Distribution
            "DISTRIBUTION_D": {"SOW_break": "DISTRIBUTION_E", "LPSY": "DISTRIBUTION_D"}, # SOW_break for clarity
            "DISTRIBUTION_E": {"LPSY": "DISTRIBUTION_E"},
        }
        
        effective_rules = rules_config or default_rules
        for phase_str, transitions in effective_rules.items():
            try:
                current_phase_enum = WyckoffPhase[phase_str.upper()]
                self.rules[current_phase_enum] = {}
                for event_type, next_phase_str in transitions.items():
                    self.rules[current_phase_enum][event_type] = WyckoffPhase[next_phase_str.upper()]
            except KeyError as e: # pragma: no cover
                log.warning(f"Invalid phase name '{e}' in transition rules config for phase '{phase_str}'. Skipping rule part.")
        log.debug(f"Loaded transition rules: {self.rules}")

    def _update_context(self, event_type: str, index: int):
        """Updates internal context like TR boundaries based on events and ZBar data."""
        if not self.zbars_context or index >= len(self.zbars_context): # pragma: no cover
            log.warning(f"Cannot update context for event {event_type} at index {index}: ZBars context not available or index out of bounds.")
            return

        current_event_bar = self.zbars_context[index]

        # Determine Schematic Type and Initial TR
        if event_type == "SC":
            self.schematic_type = "Accumulation"
            self.trading_range["support"] = current_event_bar.low
            self.last_significant_low_idx = index
            log.info(f"Context Update: Schematic set to Accumulation. SC at {index} defines TR.Support={self.trading_range['support']:.4f}")
        elif event_type == "PS" and self.schematic_type is None:
            self.schematic_type = "Accumulation"
            self.trading_range["support"] = current_event_bar.low
            self.last_significant_low_idx = index
            log.info(f"Context Update: Schematic tentatively Accumulation. PS at {index} sets potential TR.Support={self.trading_range['support']:.4f}")
        elif event_type == "AR_acc" and self.schematic_type == "Accumulation":
            if self.trading_range["support"] is not None: # Ensure support was set by SC/PS
                self.trading_range["resistance"] = current_event_bar.high
                self.last_significant_high_idx = index
                log.info(f"Context Update: Accumulation TR defined. AR_acc at {index} sets TR.Resistance={self.trading_range['resistance']:.4f}. TR: [{self.trading_range['support']:.4f} - {self.trading_range['resistance']:.4f}]")
            else: log.warning(f"AR_acc at {index} but TR.Support not yet defined.")

        # Similar logic for Distribution
        elif event_type == "BC":
            self.schematic_type = "Distribution"
            self.trading_range["resistance"] = current_event_bar.high
            self.last_significant_high_idx = index
            log.info(f"Context Update: Schematic set to Distribution. BC at {index} defines TR.Resistance={self.trading_range['resistance']:.4f}")
        elif event_type == "PSY" and self.schematic_type is None:
            self.schematic_type = "Distribution"
            self.trading_range["resistance"] = current_event_bar.high
            self.last_significant_high_idx = index
            log.info(f"Context Update: Schematic tentatively Distribution. PSY at {index} sets potential TR.Resistance={self.trading_range['resistance']:.4f}")
        elif event_type == "AR_dist" and self.schematic_type == "Distribution":
            if self.trading_range["resistance"] is not None:
                self.trading_range["support"] = current_event_bar.low
                self.last_significant_low_idx = index
                log.info(f"Context Update: Distribution TR defined. AR_dist at {index} sets TR.Support={self.trading_range['support']:.4f}. TR: [{self.trading_range['support']:.4f} - {self.trading_range['resistance']:.4f}]")
            else: log.warning(f"AR_dist at {index} but TR.Resistance not yet defined.")

        # Refine context based on STs, Springs, UTs
        elif event_type == "ST_Acc" and self.schematic_type == "Accumulation":
            if self.trading_range["support"] is not None and current_event_bar.low >= self.trading_range["support"]:
                self.last_significant_low_idx = index # ST can be a new reference low if it holds well
                log.info(f"Context Update: ST_Acc at {index} confirms/holds TR.Support {self.trading_range['support']:.4f}. New ref low at {current_event_bar.low:.4f}")
        
        elif event_type == "Spring" and self.schematic_type == "Accumulation":
            # Spring often redefines the support or marks the absolute low of the TR for Phase C.
            # The 'support' might be the low of the break_bar (index-1) or recovery_bar (index)
            # For simplicity, let's use the recovery bar's low if it's lower than existing support,
            # or the break bar's low if that's the true extreme.
            # This needs careful consideration of the spring definition.
            # Let's assume the event_detector gives the recovery bar index for "Spring".
            break_bar_low = self.zbars_context[index-1].low
            if self.trading_range["support"] is None or break_bar_low < self.trading_range["support"]:
                 # self.trading_range["support"] = break_bar_low # The actual penetration
                 pass # Support is the original SC/ST low that was broken
            self.last_significant_low_idx = index # The spring recovery is a new significant point
            log.info(f"Context Update: Spring at {index}. Original Support {self.trading_range.get('support', 'N/A')} tested. Break low: {break_bar_low:.4f}")

        # TODO: Add context updates for ST_dist, UT, UTAD, LPS, LPSY, SOS, SOW, BU, Test

    def process_event(self, event_type: str, index: int, zbars: List[ZBar], timeframe: Optional[str] = None):
        """
        Feed an event into the state machine, update context, and attempt phase transition.
        V3: Uses zbars for context and refined transition logic.
        """
        if not self.zbars_context: # Store zbars on first call
            self.zbars_context = zbars
            log.info(f"ZBars context set for State Machine ({len(self.zbars_context)} bars).")
        
        if index >= len(self.zbars_context): # pragma: no cover
            log.error(f"Event index {index} is out of bounds for zbars_context (len {len(self.zbars_context)}).")
            return

        log.info(f"Processing event: {event_type} at index {index} (Current Phase: {self.current_phase})")

        # 1. Record the event
        self.event_sequence.append((index, event_type))
        self.event_timeframes.append(timeframe)
        log.info(f"[StateMachine] Event '{event_type}' on TF '{timeframe}' at index {index}")
        if event_type not in self.detected_events_indices: self.detected_events_indices[event_type] = []
        self.detected_events_indices[event_type].append(index)

        # 2. Update internal context (TR boundaries, schematic type, etc.)
        self._update_context(event_type, index)

        # 3. Determine potential next phase
        prev_phase = self.current_phase
        next_phase_candidate = None

        # --- Phase Transition Logic ---
        if self.current_phase == WyckoffPhase.UNKNOWN:
            if event_type == "SC" and self.schematic_type == "Accumulation": next_phase_candidate = WyckoffPhase.ACCUMULATION_A
            elif event_type == "PS" and self.schematic_type == "Accumulation": next_phase_candidate = WyckoffPhase.ACCUMULATION_A
            # TODO: Add PSY, BC for Distribution_A

        elif self.current_phase == WyckoffPhase.ACCUMULATION_A:
            if event_type == "AR_acc" and self.schematic_type == "Accumulation":
                # Condition: AR must follow SC or PS
                if self.detected_events_indices.get("SC") or self.detected_events_indices.get("PS"):
                    # Condition: AR high must be above the low of SC/PS
                    ref_low_idx = (self.detected_events_indices.get("SC", []) + self.detected_events_indices.get("PS", []))[-1]
                    if self.zbars_context[index].high > self.zbars_context[ref_low_idx].low:
                        next_phase_candidate = WyckoffPhase.ACCUMULATION_B
                    else: log.warning(f"AR_acc at {index} too weak to confirm ACC_B.")
                else: log.warning(f"AR_acc at {index} but no prior SC/PS for ACC_B.")
        
        elif self.current_phase == WyckoffPhase.ACCUMULATION_B:
            if event_type == "ST_Acc" or event_type == "ST_Acc_Weak":
                # Condition: ST must test the support of the TR
                if self.trading_range["support"] is not None and \
                   self.zbars_context[index].low >= self.trading_range["support"] - (self.trading_range.get("resistance", self.zbars_context[index].high) - self.trading_range["support"])*0.1: # Allow minor poke
                    next_phase_candidate = WyckoffPhase.ACCUMULATION_B # Remain in B
                    self.last_significant_low_idx = index # Update last significant low
                else: log.warning(f"{event_type} at {index} not a valid test of current TR support for ACC_B.")
            elif event_type == "Spring" or event_type == "Spring_Weak":
                # Condition: Spring must break established TR support and recover
                if self.trading_range["support"] is not None and \
                   index > 0 and self.zbars_context[index-1].low < self.trading_range["support"] and \
                   self.zbars_context[index].close > self.trading_range["support"]:
                    next_phase_candidate = WyckoffPhase.ACCUMULATION_C
                else: log.warning(f"{event_type} at {index} not a valid Spring for ACC_C transition. TR Support: {self.trading_range['support']}")
        
        elif self.current_phase == WyckoffPhase.ACCUMULATION_C:
            # Test of Spring, LPS, or SOS moves to D
            if event_type == "Test": # Assuming 'Test' is a test of a Spring
                if self.detected_events_indices.get("Spring") or self.detected_events_indices.get("Spring_Weak"):
                    # Test should hold above the Spring's low
                    spring_idx = (self.detected_events_indices.get("Spring", []) + self.detected_events_indices.get("Spring_Weak", []))[-1]
                    if self.zbars_context[index].low >= self.zbars_context[spring_idx].low:
                        next_phase_candidate = WyckoffPhase.ACCUMULATION_D
                    else: log.warning(f"Test at {index} failed to hold above Spring low for ACC_D.")
                else: log.warning(f"Test event at {index} in ACC_C but no prior Spring detected.")
            elif event_type == "LPS": # Last Point of Support
                # LPS should be a higher low after Phase C events
                if self.last_significant_low_idx is not None and self.zbars_context[index].low > self.zbars_context[self.last_significant_low_idx].low:
                    next_phase_candidate = WyckoffPhase.ACCUMULATION_D
                else: log.warning(f"LPS at {index} not a clear higher low for ACC_D.")
            elif event_type == "SOS": # Sign of Strength
                # SOS should ideally break above TR resistance
                if self.trading_range["resistance"] is not None and self.zbars_context[index].high > self.trading_range["resistance"]:
                    next_phase_candidate = WyckoffPhase.ACCUMULATION_D
                else: log.warning(f"SOS at {index} did not clearly break TR resistance for ACC_D.")
        
        # TODO: Implement transitions for ACC_D -> ACC_E
        # TODO: Implement all Distribution phase transitions (DIST_A -> E)

        # Fallback to basic rule if no specific contextual logic matched but event is in rules
        if next_phase_candidate is None and self.current_phase in self.rules:
            if event_type in self.rules[self.current_phase]:
                log.debug(f"Applying basic rule for {event_type} from {self.current_phase}.")
                next_phase_candidate = self.rules[self.current_phase][event_type]


        # --- Apply Transition ---
        if next_phase_candidate and next_phase_candidate != self.current_phase:
            log.info(f"Phase Transition: {prev_phase} -> {next_phase_candidate} (Triggered by: {event_type} at {index})")
            self.transition_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), # Log actual processing time
                "from_phase": str(prev_phase),
                "trigger_event": event_type,
                "trigger_index": index,
                "to_phase": str(next_phase_candidate),
                "timeframe": timeframe,
            })
            self.current_phase = next_phase_candidate
        elif next_phase_candidate and next_phase_candidate == self.current_phase:
             log.debug(f"Event {event_type} occurred at {index}, remaining in phase {self.current_phase}.")
        else:
            log.debug(f"No valid phase transition from {self.current_phase} on event {event_type} at {index}.")

    def get_current_phase(self) -> str:
        return str(self.current_phase)

    def summarize(self): # pragma: no cover
        print("\n--- Wyckoff State Machine Summary (v3 - Context Aware) ---")
        print(f"Final Determined Phase: {self.get_current_phase()}")
        print(f"Schematic Type: {self.schematic_type}")
        print(f"Trading Range: Support={self.trading_range['support']}, Resistance={self.trading_range['resistance']}")
        print("\nDetected Events (Indices):")
        sorted_event_types = sorted(self.detected_events_indices.keys(),
                                   key=lambda et: min(self.detected_events_indices[et]) if self.detected_events_indices.get(et) else float('inf'))
        for event_type in sorted_event_types:
            indices = self.detected_events_indices[event_type]
            if indices: print(f"  {event_type}: {sorted(indices)}")
        print("\nPhase Transition Log:")
        if self.transition_log:
            for log_entry in self.transition_log: print(f"  - {log_entry}")
        else: print("  (No phase transitions logged)")

# --- Example Usage ---
if __name__ == "__main__": # pragma: no cover
    # Create dummy ZBars for testing context
    dummy_zbars_data = [ # Timestamp, O, H, L, C, V
        (datetime(2023,1,1,9,0), 105,105.5,104,104.5,1000), #0 Downtrend
        (datetime(2023,1,1,9,1), 104.5,104.8,102,102.2,1500), #1 PS
        (datetime(2023,1,1,9,2), 102.2,102.5,100,100.5,3000), #2 SC -> Support = 100
        (datetime(2023,1,1,9,3), 100.5,103.5,100.5,103.0,2000), #3 AR_acc -> Resistance = 103.5
        (datetime(2023,1,1,9,4), 103.0,103.2,101,101.2,800),   #4 ST_Acc (tests SC low, holds) -> last_sig_low_idx = 4, low = 101
        (datetime(2023,1,1,9,5), 101.2,101.5,99.5,99.8,700),   #5 Break Bar for Spring (breaks ST low of 101, and SC low of 100)
        (datetime(2023,1,1,9,6), 99.8,102,99.7,101.9,1800),    #6 Spring Recovery (closes > ST low of 101)
        (datetime(2023,1,1,9,7), 101.9,102.5,101.5,101.8,600), #7 Test of Spring (holds above Spring low of 99.7)
        (datetime(2023,1,1,9,8), 101.8,104,101.5,103.8,2200),  #8 SOS (breaks AR_acc high of 103.5)
    ]
    dummy_zbars = [ZBar(ts,o,h,l,c,v) for ts,o,h,l,c,v in dummy_zbars_data]
    for zbar in dummy_zbars: zbar.calculate_heuristic_delta()

    sm = WyckoffStateMachine()
    sm.process_event("PS", 1, dummy_zbars)
    sm.process_event("SC", 2, dummy_zbars)
    sm.process_event("AR_acc", 3, dummy_zbars) # UNKNOWN -> A -> B
    sm.process_event("ST_Acc", 4, dummy_zbars) # B -> B
    # For Spring, event_detector gives index of recovery bar (6), break bar is (5)
    sm.process_event("Spring", 6, dummy_zbars) # B -> C
    sm.process_event("Test", 7, dummy_zbars)   # C -> D
    sm.process_event("SOS", 8, dummy_zbars)    # D -> E (if SOS is strong enough to be considered BU) or D->D

    sm.summarize()
