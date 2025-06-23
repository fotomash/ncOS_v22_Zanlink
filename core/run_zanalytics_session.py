import json
import os
import argparse
import sys
from datetime import datetime
import logging
from trait_engine import merge_config
from core.intermarket_sentiment import snapshot_sentiment
from copilot_orchestrator import run_full_analysis

# Additional modules from zanalytics package
from advanced_smc_orchestrator import AdvancedSMCOrchestrator
from liquidity_vwap_detector import detect_liquidity_sweeps
from optimizer_loop import run_optimizer_loop
from feedback_analysis_engine import analyze_feedback
from poi_quality_predictor import predict_poi_quality

from pathlib import Path
import re
from runtime.data_pipeline import DataPipeline
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_configs(config_dir):
    """Load config JSON files from a directory.

    Missing or malformed files are logged and skipped so the
    application can continue with partial configs.
    """
    files = ['copilot_config.json', 'chart_config.json', 'strategy_profiles.json']
    configs = {}
    for fname in files:
        path = os.path.join(config_dir, fname)
        if not os.path.exists(path):
            logging.error("Config file missing: %s", path)
            configs[fname.split('_')[0]] = {}
            continue
        try:
            with open(path) as f:
                configs[fname.split('_')[0]] = json.load(f)
        except Exception as e:  # pragma: no cover - unexpected read error
            logging.error("Failed loading %s: %s", path, e)
            configs[fname.split('_')[0]] = {}

    try:
        merge_config(
            configs.get('copilot', {}),
            configs.get('chart', {}),
            configs.get('strategy', {})
        )
    except Exception as e:
        logging.error("Config merge failed: %s", e)

    logging.info("[Config] Configurations merged.")
    return configs


def initialize_data():
    """Fetch and load all timeframe data.

    Any errors during fetching or loading will be logged and an
    empty dictionary returned so calling code can decide how to
    proceed.
    """
    try:
        dp = DataPipeline(config=None)  # config not needed here
        dp.fetch_pairs()
        dp.resample_htf()
    except Exception as e:
        logging.error("DataPipeline execution failed: %s", e)
        return {}

    all_tf = {}
    pattern = re.compile(rf"([A-Z]+)_(?P<tf>[A-Za-z0-9]+)_.*\.csv")
    htf_dir = Path("tick_data/htf")
    for path in htf_dir.glob("*.csv"):
        try:
            m = pattern.match(path.name)
            if not m:
                logging.warning("Skipping unexpected file %s", path.name)
                continue
            tf = m.group('tf').lower()
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            all_tf[tf] = df
            logging.info("Loaded %s from %s", tf.upper(), path.name)
        except Exception as e:
            logging.error("Failed loading %s: %s", path.name, e)
    return all_tf


def inject_sentiment(output_path, macro_version):
    """Create and persist a sentiment snapshot.

    If snapshot retrieval or saving fails, an empty snapshot is returned
    so downstream processing can continue.
    """
    try:
        snapshot = snapshot_sentiment()
    except Exception as e:
        logging.error("Snapshot sentiment failed: %s", e)
        snapshot = {}

    snapshot['run_meta'] = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_version': macro_version,
        'macro_bias': snapshot.get('context_overall_bias')
    }

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        logging.error("Failed writing sentiment snapshot: %s", e)
    else:
        logging.info("[Sentiment] Snapshot saved.")

    return snapshot


def run_analysis(multi_tf, sentiment, output_dir, macro_version):
    """Run the full analytics stack with graceful error handling."""
    try:
        smc = AdvancedSMCOrchestrator(multi_tf, sentiment['context_overall_bias'])
        result_smc = smc.run()

        sweeps = detect_liquidity_sweeps(multi_tf)
        poi_scores = predict_poi_quality(result_smc['pois'])

        result_full = run_full_analysis(
            data=multi_tf,
            sentiment_context=sentiment.get('context_overall_bias'),
            sweeps=sweeps,
            poi_scores=poi_scores
        )
    except Exception as e:
        logging.error("Analysis pipeline failed: %s", e)
        return {}

    result_full['run_meta'] = {
        'timestamp': datetime.utcnow().isoformat(),
        'strategy_version': macro_version
    }

    try:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'zanalytics_log.json')
        with open(log_path, 'w') as f:
            json.dump(result_full.get('log', {}), f, indent=2)
        summary_path = os.path.join(output_dir, 'summary_zanalytics.md')
        with open(summary_path, 'w') as f:
            f.write(result_full.get('summary', ''))
    except Exception as e:
        logging.error("Failed writing analysis output: %s", e)
    else:
        logging.info("[Analysis] Full analysis complete.")

    return result_full


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a ZANALYTICS trading session with full pipeline orchestration.'
    )
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Directory containing configuration JSON files.')
    parser.add_argument('--output-dir', type=str, default='journal',
                        help='Directory where logs and summaries will be saved.')
    parser.add_argument('--strategy-version', type=str, default='5.2',
                        help='Strategy version tag for run metadata.')
    return parser.parse_args()


def main():
    logging.info("[ZANALYTICS] Starting session v5.2")
    args = parse_args()
    configs = load_configs(args.config_dir)
    multi_tf = initialize_data()
    sentiment = inject_sentiment(
        output_path=os.path.join(args.output_dir, 'sentiment_snapshot.json'),
        macro_version=args.strategy_version
    )

    analysis_results = run_analysis(
        multi_tf,
        sentiment,
        output_dir=args.output_dir,
        macro_version=args.strategy_version
    )

    if analysis_results:
        try:
            feedback = analyze_feedback(analysis_results)
            run_optimizer_loop(analysis_results, feedback)
        except Exception as e:
            logging.error("Post-analysis processing failed: %s", e)
    logging.info("[ZANALYTICS] Session complete.")


if __name__ == '__main__':
    main()
