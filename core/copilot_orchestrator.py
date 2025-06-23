# Zanzibar v5.1 Core Module
# Version: 5.1.9 (Operational - Integrated Liquidity Sweep Detector)
# Module: copilot_orchestrator.py
# Description: Central routing and orchestration, including liquidity sweep tagging.

# --- (Imports and other initializations remain the same) ---
import sys, os, re, pandas as pd, numpy as np, json, traceback, inspect, time, csv, requests, argparse
from core.scalp_filters import validate_scalp_signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pkgutil
import importlib
import core

# Dynamically import all modules in the core package
def import_all_core_modules():
    """
    Dynamically import every module in the core package.
    """
    for finder, module_name, ispkg in pkgutil.iter_modules(core.__path__):
        try:
            importlib.import_module(f"{core.__name__}.{module_name}")
            log_info(f"[COPILOT_ORCH] Imported core.{module_name}")
        except Exception as e:
            log_info(f"[WARN][COPILOT_ORCH] Failed to import core.{module_name}: {e}", level="WARN")
    log_info("✅ All core modules dynamically imported")
from typing import Dict, Optional, Tuple, Any

# --- Plotly Check ---
try: import plotly.graph_objects as go; from plotly.subplots import make_subplots; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; go=None; make_subplots=None; print("WARN (copilot_orchestrator): Plotly not found.")

# --- Load Agent Names ---
try:
    boot_config_path = Path('zanzibar_boot.yaml');
    if boot_config_path.exists(): import yaml; f=open(boot_config_path,'r', encoding='utf-8'); boot_settings=yaml.safe_load(f); f.close(); AGENT_NAMES=boot_settings.get('agents',{})
    else: AGENT_NAMES = {}; print("[WARN][COPILOT_ORCH] zanzibar_boot.yaml not found.")
except ImportError: AGENT_NAMES = {}; print("[WARN][COPILOT_ORCH] PyYAML not installed.")
except Exception as e: AGENT_NAMES = {}; print(f"[WARN][COPILOT_ORCH] Failed loading agent names: {e}")

# --- Logging Helper ---
def log_info(message, level="INFO"):
    caller_frame = inspect.currentframe().f_back; module_name = "COPILOT"
    try:
        caller_module = inspect.getmodule(caller_frame)
        if caller_module and caller_module.__name__: base_name = caller_module.__name__.split('.')[-1].upper(); module_name = AGENT_NAMES.get('copilot', base_name) if base_name else AGENT_NAMES.get('copilot', 'COPILOT')
    except Exception: module_name = AGENT_NAMES.get('copilot', 'COPILOT')
    timestamp_log = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{level}][{module_name}][{timestamp_log}] {message}")

# --- Load Default Chart Config ---
DEFAULT_CHART_CONFIG_PATH = Path("config/chart_config.json")
default_chart_config_settings = {}
try:
    if DEFAULT_CHART_CONFIG_PATH.is_file():
        with open(DEFAULT_CHART_CONFIG_PATH, "r", encoding="utf-8") as f: chart_config_content = json.load(f)
        if "default_chart_settings" in chart_config_content: default_chart_config_settings = chart_config_content["default_chart_settings"]; log_info(f"Loaded default chart settings from {DEFAULT_CHART_CONFIG_PATH}")
        else: log_info(f"WARN: {DEFAULT_CHART_CONFIG_PATH} missing 'default_chart_settings'.")
    else: log_info(f"WARN: Default chart config not found at {DEFAULT_CHART_CONFIG_PATH}")
except Exception as cfg_err: log_info(f"WARN: Failed loading {DEFAULT_CHART_CONFIG_PATH}: {cfg_err}", "WARN")


# Automatically import all core modules
# Automatically import all core modules
import_all_core_modules()

# --- ZANALYTICS MICRO WYCKOFF PROTOCOL ---
#
# This protocol is used internally by AI agents and onboarding routines
# to interpret microstructure logic consistently across M1/M5/M15.
#
# Purpose: Detect early accumulation/distribution phases, CHoCH, and BOS for scalping
# Timeframes:
#   M1  - Execution Bias
#   M5  - Microstructure Formation
#   M15 - Liquidity Context and Confirmation
#
# Detection Rules:
#   CHoCH: Higher High + Lower Low vs Previous Bar
#   BOS: Current High > Max High in Rolling Window
#   Phase:
#     - Close > Open  → Micro-Accumulation
#     - Close < Open  → Micro-Distribution
#
# Interpretation Matrix:
#   CHoCH + No BOS → Trap Zone (Reversal)
#   CHoCH + BOS    → Momentum Breakout
#   Repeated Accum → Pre-Breakout Loading
#   Distribution Cluster → Breakdown Risk
#
# Signal Triggers:
#   - CHoCH seen in last 2 bars
#   - Spread < 0.3 and Tick Volume rising
#   - BOS missing → Reversal Trap
#   - BOS confirmed → Breakout Entry
#
# Notes for AI Agents:
#   - Always respect HTF bias (H1/H4)
#   - Use RSI, EMA, BB for confidence boosting
#   - Post-BOS entry preferred; fade failed CHoCHs with volume divergence

# --- Dynamic Imports ---
# (Imports remain the same, including the new liquidity_sweep_detector)
try: from core.finnhub_data_fetcher import load_and_aggregate_m1; FETCHER_AVAILABLE = True
except ImportError: log_info("CRITICAL ERROR: core/finnhub_data_fetcher.py not found.", "ERROR"); load_and_aggregate_m1 = None; FETCHER_AVAILABLE = False
try: from core.market_structure_analyzer_smc import analyze_market_structure
except ImportError: log_info("ERROR: core/market_structure_analyzer_smc.py not found.", "ERROR"); analyze_market_structure = None
try: from core.liquidity_engine_smc import detect_inducement_from_structure
except ImportError: log_info("ERROR: core/liquidity_engine_smc.py not found.", "ERROR"); detect_inducement_from_structure = None
try: from core.poi_manager_smc import find_and_validate_smc_pois
except ImportError: log_info("ERROR: core/poi_manager_smc.py not found.", "ERROR"); find_and_validate_smc_pois = None
try: from core.poi_hit_watcher_smc import check_poi_tap_smc
except ImportError: log_info("ERROR: core/poi_hit_watcher_smc.py not found.", "ERROR"); check_poi_tap_smc = None
try: from core.impulse_correction_detector import detect_impulse_correction_phase
except ImportError: log_info("ERROR: core/impulse_correction_detector.py not found.", "ERROR"); detect_impulse_correction_phase = None
try: from core.confirmation_engine_smc import confirm_smc_entry
except ImportError: log_info("ERROR: core/confirmation_engine_smc.py not found.", "ERROR"); confirm_smc_entry = None
try: from core.entry_executor_smc import execute_smc_entry, calculate_trade_risk, generate_pine_payload
except ImportError: log_info("ERROR: core/entry_executor_smc.py or helpers not found.", "ERROR"); execute_smc_entry=None; calculate_trade_risk=None; generate_pine_payload=None
try: from core.advanced_smc_orchestrator import run_advanced_smc_strategy, load_strategy_profile
except ImportError: log_info("ERROR: core/advanced_smc_orchestrator.py not found! Cannot route to advanced strategies.", "ERROR"); run_advanced_smc_strategy = None; load_strategy_profile = None
try: from core.accum_engine import tag_accumulation
except ImportError: log_info("WARN: core/accum_engine.py not found.", "WARN"); tag_accumulation = None
try: from core.mentfx_ici_engine import tag_mentfx_ici
except ImportError: log_info("WARN: core/mentfx_ici_engine.py not found.", "WARN"); tag_mentfx_ici = None
try: from core.phase_detector_wyckoff_v1 import detect_wyckoff_phases_and_events # Corrected path
except ImportError: log_info("WARN: core/phase_detector_wyckoff_v1.py not found.", "WARN"); detect_wyckoff_phases_and_events = None
try: from core.smc_enrichment_engine import tag_smc_zones
except ImportError: log_info("WARN: core/smc_enrichment_engine.py not found.", "WARN"); tag_smc_zones = None
try: from core.indicator_enrichment_engine import calculate_standard_indicators
except ImportError: log_info("WARN: core/indicator_enrichment_engine.py not found.", "WARN"); calculate_standard_indicators = None
try: from core.mentfx_stoploss_model_v2_adaptive import compute_mentfx_stop_loss_adaptive
except ImportError: log_info("WARN: core/mentfx_stoploss_model_v2_adaptive.py not found.", "WARN"); compute_mentfx_stop_loss_adaptive = None
try: from core.fetch_live_tail import fetch_m1_tail
except ImportError: log_info("WARN: core/fetch_live_tail.py not found.", "WARN"); fetch_m1_tail = None
try: from core.strategy_trigger import scan_for_entry_signals
except ImportError: log_info("WARN: core/strategy_trigger.py not found.", "WARN"); scan_for_entry_signals = None
# --- Confluence Engine Imports ---
try: from core.confluence_engine import ConfluenceEngine
except ImportError: log_info("WARN: core/confluence_engine.py not found.", "WARN"); ConfluenceEngine = None
try: from core.confluence_engine import compute_confluence_indicators
except ImportError: log_info("WARN: core/confluence_engine.py compute_confluence_indicators not found.", "WARN"); compute_confluence_indicators = None
# --- NEW: Import Liquidity Sweep Detector ---
try: from core.liquidity_sweep_detector import tag_liquidity_sweeps
except ImportError: log_info("WARN: core/liquidity_sweep_detector.py not found.", "WARN"); tag_liquidity_sweeps = None
# --- Intermarket Sentiment Import (NEW) ---
try: from core.intermarket_sentiment import snapshot_sentiment
except ImportError: log_info("WARN: core/intermarket_sentiment.py not found.", "WARN"); snapshot_sentiment = None


# --- Chart Generation Function (ENHANCED to include Wyckoff & Sweeps) ---
def generate_analysis_chart_json(
    price_df: pd.DataFrame,
    chart_tf: str,
    pair: str,
    target_time: str, # Reference time for title/context
    structure_data: Optional[Dict] = None,
    inducement_result: Optional[Dict] = None, # May become less relevant with new sweep detector
    poi_tap_result: Optional[Dict] = None,
    phase_result: Optional[Dict] = None, # Impulse/Correction Phase
    wyckoff_result: Optional[Dict] = None, # NEW: Wyckoff Phase Result
    confirmation_result: Optional[Dict] = None,
    entry_result: Optional[Dict] = None,
    variant_name: Optional[str] = None,
    chart_options: Optional[Dict] = None
    ) -> Optional[str]:
    """
    Generates a Plotly chart figure JSON visualizing the analysis results.
    Now includes annotations for liquidity sweeps and Wyckoff context.
    """
    # --- (Initial checks and config loading remain the same) ---
    if not PLOTLY_AVAILABLE: log_info("Plotly not available, skipping chart generation.", "WARN"); return None
    if price_df is None or price_df.empty: log_info("No price data for chart.", "WARN"); return None
    if not isinstance(price_df.index, pd.DatetimeIndex): log_info("Price data index not DatetimeIndex.", "WARN"); return None
    try:
        # --- Chart Configuration ---
        if chart_options is None: chart_options = {}
        layout_config = default_chart_config_settings.get('visuals', {})
        elements_config = default_chart_config_settings.get('elements', {})
        layout_config.update(chart_options.get('visuals', {}))
        elements_config.update(chart_options.get('elements', {}))
        paper_bgcolor=layout_config.get('paper_bgcolor','rgb(17,17,17)'); plot_bgcolor=layout_config.get('plot_bgcolor','rgb(17,17,17)'); font_color=layout_config.get('font_color','white'); grid_color=layout_config.get('grid_color','rgba(180,180,180,0.1)'); bull_color=layout_config.get('bullish_candle_color','#26a69a'); bear_color=layout_config.get('bearish_candle_color','#ef5350'); poi_bull_fill=layout_config.get('poi_bullish_fill_color', "rgba(0, 200, 0, 0.1)"); poi_bear_fill=layout_config.get('poi_bearish_fill_color', "rgba(200, 0, 0, 0.1)"); poi_bull_border=layout_config.get('poi_bullish_border_color', "darkgreen"); poi_bear_border=layout_config.get('poi_bearish_border_color', "darkred"); ltf_poi_opacity = layout_config.get('ltf_poi_opacity', 0.3); sl_color = layout_config.get('sl_line_color', 'red'); tp_color = layout_config.get('tp_line_color', 'lime'); entry_marker_style = elements_config.get('entry_marker_style', 'arrow'); entry_marker_size = elements_config.get('entry_marker_size', 12); label_font_size = elements_config.get('chart_label_font_size', 9); price_precision = layout_config.get('price_precision', 5)

        log_info(f"Generating analysis chart JSON for {pair} ({chart_tf}) | Variant: {variant_name}")
        fig = go.Figure()
        # --- Base Layout & Candlesticks ---
        title = f"ZANZIBAR Analysis: {pair} ({chart_tf})"
        if variant_name and chart_options.get('tag_variant', True): title += f" | Variant: {variant_name}"
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", xaxis_rangeslider_visible=False, hovermode="x unified", paper_bgcolor=paper_bgcolor, plot_bgcolor=plot_bgcolor, font=dict(color=font_color, size=label_font_size + 1), xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color), legend_title_text='Analysis Layers')
        fig.add_trace(go.Candlestick(x=price_df.index, open=price_df['Open'], high=price_df['High'], low=price_df['Low'], close=price_df['Close'], name='Candles', increasing_line_color=bull_color, decreasing_line_color=bear_color))

        # --- Annotations ---
        # --- 1. Structure Annotations --- ### SYNTAX FIX APPLIED ###
        if structure_data and elements_config.get('show_market_structure', True):
            htf_poi_type = structure_data.get('htf_bias', 'Unknown'); fig.add_annotation(text=f"HTF Bias: {htf_poi_type}", align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.5)", font_size=label_font_size)
            if elements_config.get('show_bos_choch_labels', True):
                 for point in structure_data.get('structure_points', []):
                     ts_str=point.get('timestamp'); price=point.get('price'); ptype=point.get('type','');
                     # Use try-except for robustness in plotting each point
                     try:
                         ts = pd.Timestamp(ts_str, tz='UTC') if isinstance(ts_str, str) else ts_str
                         if ts and price and isinstance(ts, pd.Timestamp) and ts in price_df.index:
                             # Assign variables on separate lines for clarity
                             color='lime' if 'Low' in ptype else ('red' if 'High' in ptype else 'grey')
                             symbol='circle' if 'Strong' in ptype else ('x' if 'Weak' in ptype else 'circle-open')
                             size=10 if 'Strong' in ptype else 8
                             textpos="bottom center" if 'Low' in ptype else "top center"
                             # Add trace
                             fig.add_trace(go.Scatter(x=[ts], y=[price], mode='markers+text', name=ptype, text=[ptype], textposition=textpos,
                                                      marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1, color='white')),
                                                      textfont_size=label_font_size))
                     except Exception as ts_err:
                         log_info(f"Skipping structure point plot: Error parsing/plotting timestamp for {ptype} at {ts_str}: {ts_err}", "WARN")

            tr=structure_data.get('valid_trading_range'); dp=structure_data.get('discount_premium')
            if tr and dp and isinstance(tr.get('start'), dict) and isinstance(tr.get('end'), dict) and elements_config.get('show_discount_premium_zones', True):
                start_p = tr['start'].get('price'); end_p = tr['end'].get('price')
                if isinstance(start_p, (int, float)) and isinstance(end_p, (int, float)):
                     range_high=max(start_p, end_p); range_low=min(start_p, end_p); midpoint=dp.get('midpoint')
                     if range_high > range_low and midpoint is not None:
                         fig.add_hline(y=range_high, line=dict(color="rgba(255,100,100,0.6)",width=1,dash="dash"), annotation_text=f"Range H {range_high:.{price_precision}f}", annotation_position="bottom right", annotation_font_size=label_font_size)
                         fig.add_hline(y=range_low, line=dict(color="rgba(100,255,100,0.6)",width=1,dash="dash"), annotation_text=f"Range L {range_low:.{price_precision}f}", annotation_position="top right", annotation_font_size=label_font_size)
                         fig.add_hline(y=midpoint, line=dict(color="grey",width=1,dash="dot"), annotation_text="50%", annotation_position="bottom right", annotation_font_size=label_font_size)
                         fig.add_hrect(y0=range_low, y1=midpoint, line_width=0, fillcolor="green", opacity=0.05, layer="below")
                         fig.add_hrect(y0=midpoint, y1=range_high, line_width=0, fillcolor="red", opacity=0.05, layer="below")
        else: htf_poi_type = 'Unknown'
        # --- ### END STRUCTURE ANNOTATIONS ### ---

        # --- 2. Inducement Annotations (Legacy) ---
        # (Keep collapsed or remove if liquidity_sweep_detector replaces it)
        # --- 3. POI Tap Marker ---
        if poi_tap_result and poi_tap_result.get('is_tapped') and chart_options.get('show_tap_marker', True):
            tap_time = poi_tap_result.get('tap_time'); padded_range = poi_tap_result.get('padded_poi_range')
            if tap_time and isinstance(tap_time, pd.Timestamp) and tap_time in price_df.index: fig.add_vline(x=tap_time, line=dict(color="cyan", width=1, dash="dot"), annotation_text="POI Tap", annotation_position="top left", annotation_font_size=label_font_size);
            if padded_range: fig.add_annotation(x=tap_time, y=price_df['High'].max(), text=f"Tapped: [{padded_range[0]:.{price_precision}f}-{padded_range[1]:.{price_precision}f}]", showarrow=False, bgcolor="rgba(0,200,200,0.6)", font_size=label_font_size-1, xanchor="left", yanchor="top", xshift=5)
        # --- 4. Phase Annotation (Impulse/Correction) ---
        if phase_result and chart_options.get('show_phase_label', True): phase = phase_result.get('phase', 'Unknown'); color = "lightgreen" if phase == "Correction" else ("orange" if phase == "Impulse" else "grey"); fig.add_annotation(text=f"Phase: {phase}", align='right', showarrow=False, xref='paper', yref='paper', x=0.99, y=0.98, bgcolor=color, font_size=label_font_size, font_color="black")
        # --- 5. Wyckoff Phase Annotation ---
        if wyckoff_result and chart_options.get('show_wyckoff_phase', True):
            wyckoff_phase = wyckoff_result.get('current_phase', 'Unknown'); color = "lightblue"
            fig.add_annotation(text=f"Wyckoff: {wyckoff_phase}", align='right', showarrow=False, xref='paper', yref='paper', x=0.99, y=0.94, bgcolor=color, font_size=label_font_size, font_color="black")
            if elements_config.get('show_wyckoff_events', True) and wyckoff_result.get('detected_events'):
                 for event_name, event_data in wyckoff_result['detected_events'].items():
                      try:
                           event_ts_str = event_data.get('time'); event_price = event_data.get('price')
                           if not event_ts_str or event_price is None: continue
                           event_ts = pd.Timestamp(event_ts_str, tz='UTC')
                           if event_ts in price_df.index: is_low_event = event_name in ['SC', 'ST', 'Spring', 'Test', 'LPS', 'LowPoint1']; event_symbol = 'triangle-up' if is_low_event else 'triangle-down'; event_color = 'magenta'; text_pos = "bottom center" if is_low_event else "top center"; fig.add_trace(go.Scatter(x=[event_ts], y=[event_price], mode='markers+text', name=f"W:{event_name}", text=[event_name], textposition=text_pos, marker=dict(symbol=event_symbol, size=10, color=event_color, line=dict(width=1, color='white')), textfont_size=label_font_size-1))
                      except Exception as plot_err: log_info(f"WARN: Could not plot Wyckoff event {event_name}: {plot_err}", "WARN")

        # --- 6. Liquidity Sweep Annotations (NEW) --- ### UPDATED ###
        # Now uses columns from liquidity_sweep_detector
        if elements_config.get('show_liquidity_sweeps', True):
            # Define the columns added by tag_liquidity_sweeps
            sweep_cols = {
                'liq_sweep_fractal_high': ('x', 'aqua', 'Sweep H (Fr)'), # Symbol, Color, Name
                'liq_sweep_fractal_low': ('x', 'aqua', 'Sweep L (Fr)'),
                # Add more columns here if detector adds int/ext distinctions later
                # 'liq_sweep_high_ext': ('triangle-ne', 'fuchsia', 'Sweep H (Ext)'),
                # 'liq_sweep_low_ext': ('triangle-se', 'fuchsia', 'Sweep L (Ext)'),
            }
            for col, style in sweep_cols.items():
                if col in price_df.columns:
                    # Find rows where a sweep occurred (column has the price of the swept level)
                    swept_points = price_df[price_df[col].notna()]
                    # Determine whether to plot marker at High or Low of the *sweeping* candle
                    y_col_marker = 'High' if 'high' in col else 'Low'
                    for idx, row in swept_points.iterrows():
                        # Add a marker at the High/Low of the candle that performed the sweep
                        fig.add_trace(go.Scatter(x=[idx], y=[row[y_col_marker]], mode='markers', name=style[2],
                                                 marker=dict(symbol=style[0], size=8, color=style[1], line=dict(width=1))))
                        # Optionally, mark the actual price level that was swept (stored in the column)
                        # fig.add_hline(y=row[col], line=dict(color=style[1], width=1, dash="dot"))
        # --- ### END SWEEP ANNOTATIONS ### ---

        # --- 7. Confirmation Annotations --- ### SYNTAX FIX APPLIED ###
        if confirmation_result and confirmation_result.get('confirmation_status'):
            # Plot CHoCH
            if confirmation_result.get('choch_details') and chart_options.get('highlight_structure_breaks', True):
                choch = confirmation_result['choch_details']
                try: # Moved try block here
                    choch_ts_str = choch.get('timestamp'); choch_price = choch.get('price')
                    if choch_ts_str and isinstance(choch_price, (int, float)):
                        choch_ts = pd.Timestamp(choch_ts_str).tz_convert('UTC') if isinstance(choch_ts_str, str) else pd.Timestamp(choch_ts_str).tz_localize('UTC')
                        if choch_ts in price_df.index:
                            # Plotting logic split for readability
                            fig.add_hline(y=choch['price'], line=dict(color="yellow", width=1, dash="dashdot"),
                                          annotation_text=f"CHoCH ({choch.get('type','?')[0]}) @ {choch['price']:.{price_precision}f}",
                                          annotation_position="bottom right", annotation_font_size=label_font_size)
                            fig.add_trace(go.Scatter(x=[choch_ts], y=[choch['price']], mode='markers', name='CHoCH Break',
                                                     marker=dict(symbol='star', size=10, color='yellow')))
                except Exception as ts_err: # Correctly indented except block
                    log_info(f"WARN: Could not parse/plot CHoCH data {choch}: {ts_err}", "WARN")

            # Plot BOS
            if confirmation_result.get('bos_details') and chart_options.get('highlight_structure_breaks', True):
                bos = confirmation_result['bos_details']
                try: # Moved try block here
                    bos_ts_str = bos.get('timestamp'); bos_price = bos.get('price')
                    if bos_ts_str and isinstance(bos_price, (int, float)):
                        bos_ts = pd.Timestamp(bos_ts_str).tz_convert('UTC') if isinstance(bos_ts_str, str) else pd.Timestamp(bos_ts_str).tz_localize('UTC')
                        if bos_ts in price_df.index:
                            # Plotting logic split for readability
                            fig.add_hline(y=bos['price'], line=dict(color="orange", width=1, dash="dash"),
                                          annotation_text=f"LTF BOS ({bos.get('type','?')[0]}) @ {bos['price']:.{price_precision}f}",
                                          annotation_position="bottom right", annotation_font_size=label_font_size)
                            fig.add_trace(go.Scatter(x=[bos_ts], y=[bos['price']], mode='markers', name='LTF BOS Break',
                                                     marker=dict(symbol='diamond', size=10, color='orange')))
                except Exception as ts_err: # Correctly indented except block
                    log_info(f"WARN: Could not parse/plot BOS data {bos}: {ts_err}", "WARN")

            # Plot LTF POI Box
            if confirmation_result.get('ltf_poi_range') and chart_options.get('show_ltf_poi', True):
                 ltf_poi_range = confirmation_result['ltf_poi_range']; ltf_poi_ts_str = confirmation_result.get('ltf_poi_timestamp'); ltf_poi_type = confirmation_result.get('ltf_poi_type', 'LTF POI')
                 try:
                     ltf_poi_ts = pd.Timestamp(ltf_poi_ts_str, tz='UTC') if ltf_poi_ts_str else None
                     if isinstance(ltf_poi_range, (list, tuple)) and len(ltf_poi_range) == 2 and ltf_poi_ts and ltf_poi_ts in price_df.index:
                         poi_low, poi_high = min(ltf_poi_range), max(ltf_poi_range); poi_color = poi_bull_fill if 'Bullish' in htf_poi_type else poi_bear_fill; poi_border = poi_bull_border if 'Bullish' in htf_poi_type else poi_bear_border
                         # Determine end time for rectangle - moved inside try block
                         try:
                             freq_str = pd.infer_freq(price_df.index); time_delta = pd.Timedelta(minutes=1)
                             if freq_str: time_delta = pd.Timedelta(freq_str)
                             else: diffs = price_df.index.to_series().diff(); time_delta = diffs[diffs > pd.Timedelta(0)].median() or time_delta
                             num_bars_extend = 5; poi_end_time = ltf_poi_ts + (time_delta * num_bars_extend)
                             poi_end_time = min(poi_end_time, price_df.index.max() + time_delta)
                         except Exception: # Fallback if frequency inference fails
                             poi_end_time = ltf_poi_ts + pd.Timedelta(hours=1); poi_end_time = min(poi_end_time, price_df.index.max() + pd.Timedelta(minutes=5))

                         fig.add_shape(type="rect", x0=ltf_poi_ts, y0=poi_low, x1=poi_end_time, y1=poi_high, line=dict(color=poi_border, width=1, dash="dot"), fillcolor=poi_color, layer='below', opacity=ltf_poi_opacity); fig.add_annotation(x=ltf_poi_ts, y=poi_high, align='left', xanchor='left', yanchor='bottom', text=f"{ltf_poi_type}", showarrow=False, bgcolor="rgba(0,0,0,0.6)", font_size=label_font_size-1)
                     else: log_info(f"WARN: Invalid LTF POI range or timestamp for plotting: Range={ltf_poi_range}, TS={ltf_poi_ts_str}", "WARN")
                 except Exception as plot_err: log_info(f"WARN: Failed to plot LTF POI: {plot_err}", "WARN")
        # --- ### END CONFIRMATION ANNOTATIONS ### ---

        # --- 8. Entry/SL/TP Annotations --- ### SYNTAX FIX APPLIED ###
        if entry_result and entry_result.get('entry_confirmed') and chart_options.get('label_entry_sl_tp', True):
            entry_price = entry_result.get('entry_price'); sl_price = entry_result.get('sl'); tp_price = entry_result.get('tp'); entry_ts_str = entry_result.get('entry_candle_timestamp'); direction = entry_result.get('direction')
            final_sl_price = entry_result.get('final_sl_price', sl_price); tp1_price = entry_result.get('tp1', tp_price); tp2_price = entry_result.get('tp2')

            if entry_price and entry_ts_str:
                 try: # Moved try block here
                     entry_ts = pd.Timestamp(entry_ts_str).tz_convert('UTC') if isinstance(entry_ts_str, str) else pd.Timestamp(entry_ts_str).tz_localize('UTC')
                     if entry_ts in price_df.index:
                         # Assign variables on separate lines
                         marker_symbol = 'triangle-up' if direction == 'buy' else 'triangle-down'
                         marker_color = 'lime' if direction == 'buy' else 'red'
                         if entry_marker_style == 'cross': marker_symbol = 'cross'
                         elif entry_marker_style == 'circle': marker_symbol = 'circle'
                         # Add trace
                         fig.add_trace(go.Scatter(x=[entry_ts], y=[entry_price], mode='markers', name='Entry',
                                                  marker=dict(symbol=marker_symbol, size=entry_marker_size, color=marker_color, line=dict(width=1, color='white'))))
                         fig.add_annotation(x=entry_ts, y=entry_price, text=f"Entry @ {entry_price:.{price_precision}f}",
                                            showarrow=True, arrowhead=1, ax=0, ay=-40 if direction=='buy' else 40, font_size=label_font_size)
                 except Exception as ts_err: # Correctly indented except block
                     log_info(f"WARN: Could not parse/plot Entry timestamp {entry_ts_str}: {ts_err}", "WARN")

            if final_sl_price: fig.add_hline(y=final_sl_price, line=dict(color=sl_color, width=2, dash="solid"), annotation_text=f"SL {final_sl_price:.{price_precision}f}", annotation_position="bottom left" if direction=='buy' else "top left", annotation_font_size=label_font_size)
            if tp1_price: fig.add_hline(y=tp1_price, line=dict(color=tp_color, width=2, dash="solid"), annotation_text=f"TP1 {tp1_price:.{price_precision}f}", annotation_position="top left" if direction=='buy' else "bottom left", annotation_font_size=label_font_size)
            if tp2_price: fig.add_hline(y=tp2_price, line=dict(color=tp_color, width=1, dash="dot"), annotation_text=f"TP2 {tp2_price:.{price_precision}f}", annotation_position="top left" if direction=='buy' else "bottom left", annotation_font_size=label_font_size-1)
        # --- ### END ENTRY/SL/TP ANNOTATIONS ### ---


        # --- Convert to JSON ---
        chart_json = fig.to_json()
        log_info("Successfully generated enriched Plotly chart JSON.")
        return chart_json
    except Exception as e:
        log_info(f"Error converting enriched Plotly figure to JSON: {e}", "ERROR")
        traceback.print_exc() # Print stack trace for debugging chart errors
        return None
# --- ### END CHART FUNCTION ### ---

# --- Legacy/Default Analysis Function ---
def run_full_analysis(all_tf_data: dict, pair: str, timestamp_str: str, target_tf_str: str, variant_name: Optional[str] = "Default") -> dict:
    """
    Performs a simplified analysis sequence (Legacy/Default version).
    Now includes Wyckoff phase detection.
    """
    # --- (Function body remains the same - see previous version) ---
    log_info(f"--- Running Full Analysis (Legacy/Default Flow) for {pair} @ {timestamp_str} (Target TF: {target_tf_str}) ---")
    variant_config = load_strategy_profile(variant_name) if load_strategy_profile else {}
    chart_options = variant_config.get("chart_options", {})
    wyckoff_config = variant_config.get("wyckoff_config", {})
    structure_data = {"htf_bias": "Unknown"}
    wyckoff_result = None; wyckoff_tf = variant_config.get("wyckoff_timeframe", "h1")
    if detect_wyckoff_phases_and_events and wyckoff_tf in all_tf_data and not all_tf_data[wyckoff_tf].empty:
        try: log_info(f"Running Wyckoff analysis on {wyckoff_tf.upper()} data..."); wyckoff_result = detect_wyckoff_phases_and_events(df=all_tf_data[wyckoff_tf], timeframe=wyckoff_tf.upper(), config=wyckoff_config); log_info(f"Wyckoff Analysis Result: Current Phase = {wyckoff_result.get('current_phase')}")
        except Exception as ph_err: log_info(f"WARN: Wyckoff analysis call failed: {ph_err}", "WARN"); wyckoff_result = {"error": str(ph_err)}
    else: log_info(f"Skipping Wyckoff analysis: Module or {wyckoff_tf.upper()} data not available.")
    legacy_bos = "N/A"; legacy_poi="N/A"; legacy_entry="N/A"
    analysis_results_for_chart = { "smc_structure": structure_data, "wyckoff_result": wyckoff_result, "legacy_bos": legacy_bos, "legacy_poi": legacy_poi, "legacy_entry": legacy_entry, }
    chart_tf = target_tf_str.lower(); chart_data_to_plot = all_tf_data.get(chart_tf, pd.DataFrame())
    if chart_data_to_plot.empty: chart_tf = 'm15'; chart_data_to_plot = all_tf_data.get(chart_tf, pd.DataFrame())
    if chart_data_to_plot.empty and all_tf_data:
        first_available_tf = next((tf for tf in ['m1', 'm5', 'm15', 'h1', 'h4'] if tf in all_tf_data and not all_tf_data[tf].empty), None)
        if first_available_tf:
            chart_data_to_plot = all_tf_data[first_available_tf].tail(100)
            chart_tf = first_available_tf
        else:
            log_info("No suitable data found for plotting.", "WARN")
    chart_json = None
    if not chart_data_to_plot.empty: chart_json = generate_analysis_chart_json( price_df=chart_data_to_plot.tail(200), chart_tf=chart_tf.upper(), pair=pair, target_time=timestamp_str, structure_data=structure_data, wyckoff_result=wyckoff_result, variant_name=variant_name, chart_options=chart_options )
    final_analysis_results = { "analysis": analysis_results_for_chart, "chart_json": chart_json, "journal": "Analysis pending (legacy)", "error": wyckoff_result.get('error') if wyckoff_result else None }
    log_info(f"--- Full Analysis Complete (Legacy/Default Flow) ---"); return final_analysis_results


# --- Data Fetching, Enrichment, and Routing Function --- ### UPDATED ###
def handle_price_check(pair: str, timestamp_str: str, tf: str = 'm15', strategy_variant: Optional[str] = None) -> dict:
    """ Fetches data, prepares multi-TF data, applies enrichment tags, runs Wyckoff, ROUTES to analysis, and handles errors. """
    log_info(f"--- Handling Price Check Request ---")
    log_info(f"Received: Pair='{pair}', Time='{timestamp_str}', Target TF='{tf}', Variant='{strategy_variant}'")
    try:
        try: target_dt_utc = pd.Timestamp(timestamp_str, tz='UTC')
        except Exception as e: return {"status": "error", "message": f"Invalid timestamp format: '{timestamp_str}'."}

        # --- Data Fetching & Aggregation ---
        fetch_window_delta = timedelta(days=30); start_dt = target_dt_utc - fetch_window_delta; end_dt = target_dt_utc + timedelta(hours=1)
        log_info(f"Task: Fetching data using load_and_aggregate_m1."); log_info(f"Parameters: Symbol={pair}, Start={start_dt}, End={end_dt}")
        if not FETCHER_AVAILABLE or not load_and_aggregate_m1: return {"status": "error", "message": "Data fetcher module not loaded."}
        fetch_agg_result = load_and_aggregate_m1(pair, start_dt, end_dt)
        if fetch_agg_result.get('status') != 'ok': return {"status": "error_live_data", "message": fetch_agg_result.get('message', 'Data fetch/aggregation failed.'), "source": fetch_agg_result.get('source', 'finnhub/aggregator')}
        all_tf_data = fetch_agg_result.get("data")
        if not all_tf_data or not isinstance(all_tf_data, dict): return {"status": "error", "message": "No aggregated TF data returned."}
        log_info(f"Result: Data fetch and aggregation successful. TFs: {list(all_tf_data.keys())}")

        # --- Data Enrichment --- ### UPDATED ###
        log_info("Task: Applying data enrichment tags...")
        enriched_tf_data = {}
        indicator_config = {} # Load config if needed
        liquidity_config = {} # Load config if needed
        for tf_key, df_orig in all_tf_data.items():
            if df_orig is None or df_orig.empty:
                enriched_tf_data[tf_key] = df_orig
                continue
            df_temp = df_orig.copy()
            tf_str_for_tagging = tf_key.upper()
            try:
                # Apply Standard Indicators
                if calculate_standard_indicators:
                    df_temp = calculate_standard_indicators(df_temp, tf=tf_str_for_tagging, config=indicator_config)
                # Apply SMC Zones
                if tag_smc_zones:
                    df_temp = tag_smc_zones(df_temp, tf=tf_str_for_tagging)
                # Apply Liquidity Sweeps (NEW)
                if tag_liquidity_sweeps:
                    df_temp = tag_liquidity_sweeps(df_temp, tf=tf_str_for_tagging, config=liquidity_config)
                # Apply Strategy-Specific Tags (Mentfx ICI)
                if tag_mentfx_ici:
                    df_temp = tag_mentfx_ici(df_temp, tf=tf_str_for_tagging)
                # Apply Accumulation/Distribution Tags
                if tag_accumulation:
                    df_temp = tag_accumulation(df_temp)
                # Wyckoff tagging is now done separately below
            except Exception as enrich_err:
                log_info(f"Error during enrichment for TF {tf_key}: {enrich_err}", "ERROR")
                traceback.print_exc()
            enriched_tf_data[tf_key] = df_temp
        log_info("Result: Data enrichment tagging complete.")
        # --- Confluence Indicators Computation --- ### NEW ###
        if compute_confluence_indicators:
            try:
                confluence_metrics = compute_confluence_indicators(enriched_tf_data)
                # Attach to micro_context for downstream use
                micro_context = enriched_tf_data.get("micro_context", {})
                micro_context["confluence"] = confluence_metrics
                enriched_tf_data["micro_context"] = micro_context
                log_info("Confluence indicators computed and attached.")
            except Exception as conf_err:
                log_info(f"Error computing confluence indicators: {conf_err}", "ERROR")
        # --- ### END UPDATED ### ---

        # --- Micro Wyckoff Trigger Auto-Detection (if tick or M1 data available) ---
        try:
            from micro_wyckoff_phase_engine import detect_micro_wyckoff_phase
            scalp_tf = 'm1'
            if scalp_tf in enriched_tf_data and not enriched_tf_data[scalp_tf].empty:
                df_micro = enriched_tf_data[scalp_tf].copy()
                wyckoff_signal = detect_micro_wyckoff_phase(df_micro)
                if wyckoff_signal and wyckoff_signal.get("trigger") == "micro_spring":
                    print(f"[AUTO MICRO-WYCKOFF] Spring detected — suggesting scalp trigger at {wyckoff_signal['entry_zone']}")
                    context = {"suggested_scalp_mode": True, "micro_wyckoff": wyckoff_signal}
                    enriched_tf_data["micro_context"] = context
        except Exception as wy_err:
            log_info(f"[ERROR] Micro Wyckoff detection failed: {wy_err}", "ERROR")

        # --- Microstructure Scalp Filter Hook (LIVE SIGNAL VALIDATION) ---
        try:
            scalp_ticks_tf = 'm1'
            if scalp_ticks_tf in enriched_tf_data and not enriched_tf_data[scalp_ticks_tf].empty:
                scalp_ticks_df = enriched_tf_data[scalp_ticks_tf][['<BID>', '<ASK>', '<DATE>', '<TIME>']].dropna().tail(6)
                live_scalp_signal = {
                    "symbol": pair,
                    "entry_price": float(scalp_ticks_df['<ASK>'].iloc[-1]),
                    "rr": 2.2,
                    "direction": "short",  # Could be dynamically inferred
                    "trigger_tf": "M1"
                }
                micro_config = {
                    "tick_window": 5,
                    "min_drift_pips": 1.0,
                    "max_spread": 1.8,
                    "min_rr_threshold": 1.8
                }
                micro_result = validate_scalp_signal(live_scalp_signal, scalp_ticks_df, micro_config)
                if not micro_result.get("confirmed"):
                    log_info(f"[MICROSTRUCTURE FILTER] ❌ Rejected scalp: {micro_result.get('reason')}")
                else:
                    log_info(f"[MICROSTRUCTURE FILTER] ✅ Signal Passed: {micro_result}")
        except Exception as micro_err:
            log_info(f"[FILTER ERROR] Microstructure scalp filter exception: {micro_err}", "ERROR")

        # --- Wyckoff Phase Detection ---
        wyckoff_result = None
        variant_config = load_strategy_profile(strategy_variant or "Default") if load_strategy_profile else {}
        wyckoff_config = variant_config.get("wyckoff_config", {})
        wyckoff_tf = variant_config.get("wyckoff_timeframe", "h1")

        if detect_wyckoff_phases_and_events and wyckoff_tf in enriched_tf_data and not enriched_tf_data[wyckoff_tf].empty:
            try:
                log_info(f"Running Wyckoff analysis on {wyckoff_tf.upper()} data...")
                wyckoff_result = detect_wyckoff_phases_and_events( df=enriched_tf_data[wyckoff_tf], timeframe=wyckoff_tf.upper(), config=wyckoff_config )
                log_info(f"Wyckoff Analysis Result: Current Phase = {wyckoff_result.get('current_phase')}")
            except Exception as ph_err: log_info(f"WARN: Wyckoff analysis call failed: {ph_err}", "WARN"); wyckoff_result = {"error": str(ph_err)}
        else: log_info(f"Skipping Wyckoff analysis: Module or {wyckoff_tf.upper()} data not available.")

        # --- Routing ---
        analysis_results = None
        advanced_variants = ["Inv", "MAZ2", "TMC", "Mentfx"]
        use_advanced_flow = (strategy_variant and strategy_variant in advanced_variants) or (strategy_variant is None)

        # Pass micro_context if available
        micro_context = enriched_tf_data.get("micro_context", {})

        if use_advanced_flow:
            effective_variant = strategy_variant or "Inv"
            log_info(f"Routing to Advanced SMC Strategy Orchestrator for variant: {effective_variant}")
            if run_advanced_smc_strategy:
                analysis_results = run_advanced_smc_strategy(
                    enriched_tf_data=enriched_tf_data,
                    strategy_variant=effective_variant,
                    target_timestamp=target_dt_utc,
                    symbol=pair,
                    wyckoff_context=wyckoff_result,
                    context_data=micro_context
                )
            else:
                log_info("Advanced SMC Orchestrator module not loaded. Falling back.", "ERROR")
                analysis_results = run_full_analysis(
                    enriched_tf_data,
                    pair,
                    timestamp_str,
                    tf,
                    "Default_Fallback",
                    context_data=micro_context
                )
                if isinstance(analysis_results.get("analysis"), dict):
                    analysis_results["analysis"]["wyckoff_result"] = wyckoff_result
        else:
            log_info(f"Routing to Legacy/Default Analysis Flow (Variant: {strategy_variant or 'None Specified'}).")
            analysis_results = run_full_analysis(
                enriched_tf_data,
                pair,
                timestamp_str,
                tf,
                strategy_variant or "Default",
                context_data=micro_context
            )

        # --- Format Final Result ---
        final_result = { "status": "ok", "pair": pair, "timeframe": tf, "strategy_variant_used": strategy_variant or "Default", "target_time": timestamp_str, **(analysis_results if isinstance(analysis_results, dict) else {"analysis": None, "error": "Analysis function returned invalid data."}) }
        for key_to_remove in ["variant", "symbol", "target_timestamp", "variant_used", "symbol_processed", "timestamp_target"]: final_result.pop(key_to_remove, None)
        log_info(f"Analysis handling complete for prompt."); return final_result

    except ImportError as imp_err: log_info(f"Import Error: {imp_err}.", "CRITICAL"); return {"status": "error", "message": f"Import Error: {imp_err}."}
    except Exception as e: log_info(f"Unexpected error in handle_price_check: {e}", "ERROR"); traceback.print_exc(); return {"status": "error", "message": f"Unexpected error: {e}"}


# --- (NLP Prompt Parser, Main Prompt Handler, Session Scanner Functions remain the same) ---
def parse_user_prompt(prompt):  # FIXED
    # ... (Function body remains the same) ...
    prompt_lower = prompt.lower(); log_info(f"Parsing prompt: '{prompt}'")
    strategy_keywords = { "mentfx": "Mentfx", "mentfx logic": "Mentfx", "mentfx style": "Mentfx", "tmc": "TMC", "tmc logic": "TMC", "tmc style": "TMC", "tmc confirmation": "TMC", "theory masterclass": "TMC", "inv": "Inv", "inversion": "Inv", "inv setup": "Inv", "maz2": "MAZ2", "maz": "MAZ2", "maz setup": "MAZ2", "maz style": "MAZ2", "wyckoff": "Wyckoff", "smc": "SMC", "institutional": "Institutional", "default": None, "legacy": None }
    detected_variant = None; sorted_keywords = sorted(strategy_keywords.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower): detected_variant = strategy_keywords[keyword]; log_info(f"Detected strategy keyword: '{keyword}' -> Variant: {detected_variant}"); break
    pattern = r""" (?:analyze|show|run|check|give\s+me|what\s*about)?\s* (?P<symbol>[A-Z]{3,6}(?:[-_/.:=]?[A-Z]{1,6})?|gold|btc|eth|eur|gbp|xau|spx|dxy|vix|us10y|oil)\s+ .*? (?: at\s+(?P<hour>\d{1,2})[:.]?(?P<minute>\d{2})? (?:\s*(?P<meridiem>am|pm))? (?:\s*(?P<timezone>[A-Z]{3,}|[+-]\d{1,2}(?::\d{2})?))? | (?P<now>now) ) (?:\s+(?:on|in|tf)?\s*(?P<tf>m1|m5|m15|m30|h1|h4|d1|w1))? """
    match = re.search(pattern, prompt_lower, re.IGNORECASE | re.VERBOSE)
    if not match:
        log_info(f"Could not parse core symbol/time from prompt: '{prompt}'. Attempting simpler parse.", "WARN"); simple_pattern = r"(?P<symbol>[A-Z]{3,6}(?:[-_/.:=]?[A-Z]{1,6})?|gold|btc|eth|eur|gbp|xau|spx|dxy|vix|us10y|oil)\s+(?P<now>now)"; match = re.search(simple_pattern, prompt_lower, re.IGNORECASE)
        if not match: log_info(f"Simple parsing also failed for prompt: '{prompt}'.", "ERROR"); return None
    parsed_dict = match.groupdict(); symbol_raw = parsed_dict.get('symbol')
    symbol_map = { "GBPUSD": "OANDA:GBP_USD", "EURUSD": "OANDA:EUR_USD", "BTC": "BINANCE:BTCUSDT", "BTCUSD": "BINANCE:BTCUSDT", "BTCUSDT": "BINANCE:BTCUSDT", "ETH": "BINANCE:ETHUSDT", "ETHUSD": "BINANCE:ETHUSDT", "ETHUSDT": "BINANCE:ETHUSDT", "XAUUSD": "OANDA:XAU_USD", "XAU": "OANDA:XAU_USD", "GOLD": "OANDA:XAU_USD", "GC": "COMEX:GC", "EUR": "OANDA:EUR_USD", "GBP": "OANDA:GBP_USD", "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL", "SPX": "^GSPC", "DXY": "DX-Y.NYB", "VIX": "^VIX", "US10Y": "^TNX", "OIL": "CL=F" }
    pair = symbol_map.get(symbol_raw.upper(), symbol_raw.upper()); target_timestamp_str = None
    if parsed_dict.get('now'): target_dt = datetime.now(timezone.utc); target_timestamp_str = target_dt.strftime('%Y-%m-%d %H:%M:%S')
    elif parsed_dict.get('hour'):
        try:
            hour = int(parsed_dict['hour'])
            minute = int(parsed_dict['minute']) if parsed_dict.get('minute') else 0
            meridiem = parsed_dict.get('meridiem')
            if meridiem:
                meridiem = meridiem.lower()
                if meridiem == "pm" and 1 <= hour <= 11:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
            if not (0 <= hour <= 23 and 0 <= minute <= 59): raise ValueError("Invalid hour/minute range")
            target_dt_naive = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            target_dt_utc = target_dt_naive.replace(tzinfo=timezone.utc)
            target_timestamp_str = target_dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            log_info(f"Error parsing time: {e}", "ERROR")
            return None
    else: log_info("Could not determine target time.", "ERROR"); return None
    tf = "m15"; tf_raw = parsed_dict.get('tf'); tf_map_internal = {'m1':'m1', 'm5':'m5', 'm15':'m15', 'm30':'m30', 'h1':'h1', 'h4':'h4', 'd1':'d1', 'w1':'w1'}
    if tf_raw: tf = tf_map_internal.get(tf_raw.lower(), "m15")
    final_parsed_args = {"pair": pair, "timestamp_str": target_timestamp_str, "tf": tf, "strategy_variant": detected_variant}; log_info(f"Parsed prompt successfully: {final_parsed_args}"); return final_parsed_args

def handle_prompt(...): # Collapsed
    # ... (Function body remains the same) ...
    log_info(f"Received prompt for handling: '{prompt}'")
    parsed_args = parse_user_prompt(prompt)
    if not parsed_args: log_info("Prompt parsing failed."); return {"status": "error", "message": "Could not parse the prompt. Use format 'SYMBOL at HH:MM [am/pm] [TF] [using VARIANT strategy]' or 'SYMBOL now [TF] [variant]'."}
    result = handle_price_check(**parsed_args); return result

def send_webhook_alert(...): # Collapsed
    # ... (Function body remains the same) ...
    webhook_url = os.environ.get("SCANNER_WEBHOOK_URL");
    if not webhook_url: log_info("No webhook URL found in environment (SCANNER_WEBHOOK_URL). Skipping alert.", "WARN"); return
    try: response = requests.post(webhook_url, json=payload, timeout=10); response.raise_for_status(); log_info(f"Webhook alert sent successfully to {webhook_url}.")
    except requests.exceptions.RequestException as e: log_info(f"Exception sending webhook alert: {e}", "ERROR")

def log_session_result(result: dict, log_dir: str, format: str): # UPDATED for Wyckoff
    """ Logs the result of a session scan to CSV or JSON file. """
    try:
        log_path_obj = Path(log_dir); log_path_obj.mkdir(parents=True, exist_ok=True); log_date = datetime.utcnow().strftime("%Y-%m-%d"); log_file = log_path_obj / f"session_scan_{log_date}.{format}"; result_copy = result.copy(); result_copy["log_timestamp"] = datetime.utcnow().isoformat()
        if format == "csv":
            # Add Wyckoff Phase to header
            header = [ "log_timestamp", "pair", "timeframe", "target_time", "strategy_variant_used", "htf_bias", "phase", "wyckoff_phase", "confirmation_status", "entry_confirmed", "error" ]; # Added wyckoff_phase
            row_data = {h: result_copy.get(h) for h in header}; analysis_data = result_copy.get("analysis", {});
            if isinstance(analysis_data, dict):
                row_data["htf_bias"] = analysis_data.get("smc_structure", {}).get("htf_bias");
                row_data["phase"] = analysis_data.get("phase_result", {}).get("phase")
                row_data["wyckoff_phase"] = analysis_data.get("wyckoff_result", {}).get("current_phase") # Get Wyckoff phase
            conf_data = result_copy.get("confirmation_result", {}); # Assuming confirmation result might be nested
            if isinstance(conf_data, dict): row_data["confirmation_status"] = conf_data.get("confirmation_status")
            entry_data = result_copy.get("final_entry_result", {}); # Check the correct key
            if isinstance(entry_data, dict): row_data["entry_confirmed"] = entry_data.get("entry_confirmed")
            row = [row_data.get(h) for h in header]; write_header = not log_file.exists()
            with open(log_file, "a", newline='', encoding='utf-8') as f: writer = csv.writer(f);
            if write_header: writer.writerow(header); writer.writerow(row)
        elif format == "json":
            with open(log_file, "a", encoding='utf-8') as f: f.write(json.dumps(result_copy, default=str) + "\n")
        else: log_info(f"Unsupported log format: {format}", "ERROR")
    except Exception as e: log_info(f"Failed to log session result to {log_dir}: {e}", "ERROR"); traceback.print_exc()

def run_session_scan(pairs: list, tf: str, log_dir: str, log_format: str, strategy: Optional[str] = None): # UPDATED for Wyckoff
    """ Runs analysis for multiple pairs and logs/alerts results. """
    # ... (Alert payload updated) ...
    log_info(f"--- Running Session Scan @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ---"); log_info(f"Pairs: {pairs}, Timeframe: {tf}, Strategy: {strategy or 'Default/Advanced Routing'}")
    for pair in pairs:
        timestamp_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'); log_info(f"Analyzing {pair} @ {timestamp_str} (TF: {tf})...")
        result = handle_price_check(pair, timestamp_str, tf, strategy_variant=strategy)
        if result.get("status") == "ok":
            log_info(f"Analysis OK for {pair}."); log_session_result(result, log_dir, log_format); entry_confirmed = result.get("final_entry_result", {}).get("entry_confirmed", False); confluence_strength = "weak" # Placeholder
            alert_reason = None;
            if entry_confirmed: alert_reason = "Confirmed Entry Signal"; elif confluence_strength == "strong": alert_reason = "Strong Confluence Detected"
            if alert_reason:
                log_info(f"ALERT condition met for {pair}: {alert_reason}")
                alert_payload = {
                    "alert_reason": alert_reason,
                    "pair": result.get("pair"),
                    "timeframe": result.get("timeframe"),
                    "strategy": result.get("strategy_variant_used"),
                    "timestamp": result.get("target_time"),
                    "htf_bias": result.get("analysis", {}).get("smc_structure", {}).get("htf_bias"),
                    "phase": result.get("analysis", {}).get("phase_result", {}).get("phase"),
                    "wyckoff_phase": result.get("analysis", {}).get("wyckoff_result", {}).get("current_phase"),
                    "entry_details": result.get("final_entry_result")
                }
                send_webhook_alert(alert_payload)
            else: log_info(f"No alert condition met for {pair}.")
        else: error_message = result.get("message", "Unknown error"); log_info(f"Analysis ERROR for {pair}: {error_message}", "ERROR"); log_session_result(result, log_dir, log_format)
    log_info(f"--- Session Scan Complete ---")


# --- (Example Invocation __main__ removed for operational code) ---
# Ensure this script is imported and its functions (like handle_prompt or run_session_scan)
# are called from a main execution script or scheduler.

