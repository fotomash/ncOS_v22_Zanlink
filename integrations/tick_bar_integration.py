
"""
Integration Helper for Tick Analysis with Existing Bar Processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tick_analysis_engine import AdvancedTickAnalyzer, create_smart_tick_aggregator
import logging

logger = logging.getLogger(__name__)


class TickBarIntegrator:
    """
    Integrates tick-level analysis with existing bar data processing pipeline
    """

    def __init__(self, tick_analyzer: AdvancedTickAnalyzer = None):
        self.tick_analyzer = tick_analyzer or AdvancedTickAnalyzer()
        self.aggregator = create_smart_tick_aggregator(self.tick_analyzer)

    def process_tick_file(self, tick_file_path: str) -> Dict:
        """Process tick file and generate enhanced bar data"""

        logger.info(f"Processing tick file: {tick_file_path}")

        # Load tick data
        tick_df = pd.read_csv(tick_file_path, delimiter='\t')

        # Run comprehensive analysis
        analysis_results = self.tick_analyzer.analyze_tick_data(tick_df)

        # Generate various bar types
        bars = {
            '1S': self.aggregator(analysis_results['processed_ticks'], 'time', '1S'),
            '5S': self.aggregator(analysis_results['processed_ticks'], 'time', '5S'),
            '30S': self.aggregator(analysis_results['processed_ticks'], 'time', '30S'),
            '1T': self.aggregator(analysis_results['processed_ticks'], 'time', '1T'),
            '5T': self.aggregator(analysis_results['processed_ticks'], 'time', '5T'),
            'tick_500': self.aggregator(analysis_results['processed_ticks'], 'tick', 500),
            'tick_1000': self.aggregator(analysis_results['processed_ticks'], 'tick', 1000),
            'imbalance_50': self.aggregator(analysis_results['processed_ticks'], 'imbalance', 50),
            'imbalance_100': self.aggregator(analysis_results['processed_ticks'], 'imbalance', 100)
        }

        # Add microstructure features to standard timeframes
        enhanced_bars = {}
        for timeframe, bar_df in bars.items():
            if not bar_df.empty:
                # Add derived features
                bar_df['microstructure_score'] = (
                    bar_df.get('sweeps', 0) * 0.3 +
                    bar_df.get('traps', 0) * 0.3 +
                    bar_df.get('stop_hunt_sum', 0) * 0.4
                )

                # Add trend quality based on microstructure
                if 'flow_imbalance' in bar_df.columns:
                    bar_df['trend_quality'] = bar_df['flow_imbalance'].rolling(5).mean()

                enhanced_bars[timeframe] = bar_df

        return {
            'tick_analysis': analysis_results,
            'enhanced_bars': enhanced_bars,
            'metadata': {
                'total_ticks': len(tick_df),
                'timespan': str(tick_df['timestamp'].max() - tick_df['timestamp'].min()),
                'events_detected': len(analysis_results['events']),
                'signals_generated': len(analysis_results['trading_signals'])
            }
        }

    def merge_with_bar_data(self, tick_results: Dict, bar_file_path: str) -> pd.DataFrame:
        """Merge tick-derived features with existing bar data"""

        # Load existing bar data
        bar_df = pd.read_csv(bar_file_path)
        bar_df['timestamp'] = pd.to_datetime(bar_df['timestamp'])
        bar_df.set_index('timestamp', inplace=True)

        # Get corresponding tick-derived bars
        tick_bars = tick_results['enhanced_bars'].get('1T')  # Or appropriate timeframe

        if tick_bars is not None and not tick_bars.empty:
            # Merge microstructure features
            merge_columns = [
                'sweeps', 'traps', 'avg_absorption_mean', 'flow_imbalance',
                'microstructure_score', 'trend_quality'
            ]

            available_columns = [col for col in merge_columns if col in tick_bars.columns]

            # Perform merge
            merged_df = bar_df.join(tick_bars[available_columns], how='left')

            # Fill NaN values
            for col in available_columns:
                merged_df[col] = merged_df[col].fillna(0)

            return merged_df
        else:
            logger.warning("No tick bars available for merging")
            return bar_df


# Quick start function
def analyze_tick_file(tick_file_path: str, output_dir: str = 'tick_analysis_output'):
    """Quick analysis of tick file with full output"""

    from pathlib import Path
    import json

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize integrator
    integrator = TickBarIntegrator()

    # Process tick file
    results = integrator.process_tick_file(tick_file_path)

    # Save results
    base_name = Path(tick_file_path).stem

    # Save analysis summary
    summary_file = output_path / f"{base_name}_tick_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'summary_stats': results['tick_analysis']['summary_stats'],
            'metadata': results['metadata'],
            'signals': results['tick_analysis']['trading_signals'],
            'volume_profile': results['tick_analysis']['volume_profile']
        }, f, indent=2, default=str)

    # Save enhanced bars
    for timeframe, bars in results['enhanced_bars'].items():
        if not bars.empty:
            bar_file = output_path / f"{base_name}_enhanced_{timeframe}.csv"
            bars.to_csv(bar_file)

    # Save events
    events_data = []
    for event in results['tick_analysis']['events']:
        events_data.append({
            'timestamp': str(event.timestamp),
            'type': event.event_type,
            'direction': event.direction,
            'strength': event.strength,
            'price': event.price_level,
            'metadata': event.metadata
        })

    events_file = output_path / f"{base_name}_microstructure_events.json"
    with open(events_file, 'w') as f:
        json.dump(events_data, f, indent=2)

    print(f"Analysis complete! Results saved to {output_path}")
    print(f"- Summary: {summary_file}")
    print(f"- Events: {events_file}")
    print(f"- Enhanced bars: {len(results['enhanced_bars'])} timeframes")

    return results


if __name__ == "__main__":
    # Example usage
    tick_file = "XAUUSD_TICKS_5days_20250623.csv"
    results = analyze_tick_file(tick_file)
