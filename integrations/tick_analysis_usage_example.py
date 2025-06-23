
"""
Example: How to Use the Tick Analysis Engine with Your Existing System
"""

from tick_analysis_engine import AdvancedTickAnalyzer
from tick_bar_integration import TickBarIntegrator, analyze_tick_file
import pandas as pd

# Method 1: Quick analysis of your tick file
# ==========================================
results = analyze_tick_file('XAUUSD_TICKS_5days_20250623.csv')

# This creates:
# - tick_analysis_output/XAUUSD_TICKS_5days_20250623_tick_analysis_summary.json
# - tick_analysis_output/XAUUSD_TICKS_5days_20250623_microstructure_events.json  
# - tick_analysis_output/XAUUSD_TICKS_5days_20250623_enhanced_1T.csv (and other timeframes)


# Method 2: Custom integration with your convert_final_enhanced_smc.py
# ===================================================================

# Step 1: Analyze ticks
analyzer = AdvancedTickAnalyzer()
tick_df = pd.read_csv('XAUUSD_TICKS_5days_20250623.csv', delimiter='\t')
tick_results = analyzer.analyze_tick_data(tick_df)

# Step 2: Check what we found
print(f"Found {len(tick_results['events'])} microstructure events:")
for event in tick_results['events'][:5]:  # Show first 5
    print(f"  - {event.timestamp}: {event.event_type} ({event.direction})")

# Step 3: Get trading signals
for signal in tick_results['trading_signals']:
    print(f"Signal: {signal['type']} - {signal['direction']} @ {signal['entry_price']}")
    print(f"  Reason: {signal['reason']}")

# Step 4: Create enhanced bars with microstructure
integrator = TickBarIntegrator(analyzer)
processed = integrator.process_tick_file('XAUUSD_TICKS_5days_20250623.csv')

# Step 5: Use the enhanced 1-minute bars in your existing system
enhanced_1m_bars = processed['enhanced_bars']['1T']
print(f"\nEnhanced 1M bars columns: {enhanced_1m_bars.columns.tolist()}")

# These bars now include:
# - Standard OHLC
# - Sweep counts (sweep_high_sum, sweep_low_sum)
# - Trap counts (trap_bull_sum, trap_bear_sum)  
# - Absorption scores
# - Order flow imbalance
# - Microstructure score
# - And more...


# Method 3: Real-time tick processing (for live trading)
# =====================================================

class LiveTickProcessor:
    def __init__(self):
        self.analyzer = AdvancedTickAnalyzer()
        self.tick_buffer = []

    def process_new_tick(self, tick_data):
        # Add to buffer
        self.tick_buffer.append(tick_data)

        # Process every 100 ticks
        if len(self.tick_buffer) >= 100:
            df = pd.DataFrame(self.tick_buffer)
            results = self.analyzer.analyze_tick_data(df)

            # Check for immediate signals
            for signal in results['trading_signals']:
                if signal['strength'] > 0.7:
                    print(f"HIGH PRIORITY: {signal['type']} - {signal['direction']}")

            # Clear old ticks
            self.tick_buffer = self.tick_buffer[-50:]  # Keep some for continuity


# Method 4: Combine with your existing indicators
# ==============================================

# After running your convert_final_enhanced_smc.py:
bar_data = pd.read_csv('processed_output/XAUUSD/XAUUSD_1T_enriched.csv')

# Add tick microstructure features
tick_enhanced = integrator.merge_with_bar_data(processed, 'path/to/bar_data.csv')

# Now you have ALL indicators + microstructure in one DataFrame!
print(f"Combined columns: {tick_enhanced.columns.tolist()}")
