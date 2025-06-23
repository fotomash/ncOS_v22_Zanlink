import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from confirmation_engine_smc import detect_structure_break, ConfirmationConstants
import logging
from typing import Optional, Dict, Any
from telegram_alert_engine import send_simple_summary_alert

# --- Import production implementations ---
from core.execution import calculate_trade_risk
from core.pine_connector import generate_pine_payload

log = logging.getLogger(__name__)

# --- Confluence Scanner ---
def scan_confluences(
    structure_break: Any,
    mitigation_candle: Any,
    micro_res: Optional[Dict[str, Any]],
    latest_atr: Optional[float],
    confluence_data: Dict[str, Any],
    risk_model_config: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[bool]]:
    """
    Build a confluence report of indicator checks. Config-driven and handles NaNs gracefully.
    """
    if config is None:
        config = {}
    thresholds = config.get("confluence", {})
    report = {}
    # Structure break present
    report['structure_ok'] = structure_break is not None
    # POI mitigation present
    report['poi_mitigated'] = mitigation_candle is not None
    # Microstructure confirmation
    report['micro_ok'] = bool(micro_res and micro_res.get('confirmed', False))
    # ATR check
    atr_thr = thresholds.get('atr_max', float('inf'))
    report['atr_ok'] = latest_atr is not None and not pd.isna(latest_atr) and latest_atr <= atr_thr
    # VWAP deviation check
    vwap_thr = thresholds.get('vwap_deviation_max', 2.0)
    vwap_dev = confluence_data.get('vwap_deviation')
    report['vwap_ok'] = vwap_dev is not None and not pd.isna(vwap_dev) and abs(vwap_dev) <= vwap_thr
    # MACD H4 histogram check
    macd_thr = thresholds.get('macd_h4_hist_min', 0.0)
    macd_val = confluence_data.get('macd_h4_histogram')
    report['macd_h4_ok'] = macd_val is not None and not pd.isna(macd_val) and macd_val >= macd_thr
    # DSS slope check
    dss_thr = thresholds.get('dss_slope_min', 0.0)
    dss_slope = confluence_data.get('dss_slope')
    report['dss_ok'] = dss_slope is not None and not pd.isna(dss_slope) and dss_slope >= dss_thr
    log.debug(f"Confluence report: {report}")
    return report

# --- Microstructure filter import ---
from scalp_filters import validate_scalp_signal


def find_mitigation_candle(ltf_data: pd.DataFrame, poi_range: List[float], poi_type: str) -> Optional[pd.Series]:
    """
    Finds the first candle that mitigates (taps into) the LTF POI range.

    Args:
        ltf_data: DataFrame of LTF OHLC data.
        poi_range: List [poi_low, poi_high].
        poi_type: 'Bullish' or 'Bearish'.

    Returns:
        The Pandas Series (candle) that first mitigated the POI, or None.
    """
    poi_low, poi_high = min(poi_range), max(poi_range)
    if poi_type == 'Bullish': # Demand POI
        # Look for candle low hitting the POI high or dipping inside
        mitigation_mask = (ltf_data['Low'] <= poi_high) & (ltf_data['High'] >= poi_low) # Check if candle range overlaps POI
    elif poi_type == 'Bearish': # Supply POI
        # Look for candle high hitting the POI low or pushing inside
        mitigation_mask = (ltf_data['High'] >= poi_low) & (ltf_data['Low'] <= poi_high) # Check if candle range overlaps POI
    else:
        return None

    mitigating_candles = ltf_data[mitigation_mask]
    if not mitigating_candles.empty:
        return mitigating_candles.iloc[0] # Return the first candle that mitigated
    return None

def check_engulfing_pattern(candle: pd.Series, previous_candle: pd.Series, direction: str) -> bool:
    """Checks for a simple engulfing pattern."""
    if candle is None or previous_candle is None: return False
    try:
        last_open=float(candle['Open']); last_close=float(candle['Close'])
        prev_open=float(previous_candle['Open']); prev_close=float(previous_candle['Close'])

        is_last_bullish = last_close > last_open
        is_prev_bearish = prev_close < prev_open
        is_bull_engulfing = is_last_bullish and is_prev_bearish and (last_close > prev_open) and (last_open < prev_close)

        is_last_bearish = last_close < last_open
        is_prev_bullish = prev_close > prev_open
        is_bear_engulfing = is_last_bearish and is_prev_bullish and (last_close < prev_open) and (last_open > prev_close)

        if direction == 'buy' and is_bull_engulfing: return True
        if direction == 'sell' and is_bear_engulfing: return True
        return False
    except Exception:
        return False # Error during check


def execute_smc_entry(
    ltf_data: pd.DataFrame,
    confirmation_data: Dict,
    strategy_variant: str = "Inv",
    risk_model_config: Optional[Dict] = None,
    spread_points: int = 0,
    confluence_data: Optional[Dict] = None,
    execution_params: Optional[Dict] = None,
    symbol: str = "XAUUSD" # Added symbol for payload
) -> Dict:
    """
    Executes trade entry logic based on confirmed SMC setup and strategy variant rules.

    Args:
        ltf_data: LTF OHLCV data (e.g., M1/M5) covering the period after confirmation.
        confirmation_data: Output dictionary from confirm_smc_entry().
        strategy_variant: Strategy variant ('Inv', 'MAZ2', 'TMC', 'Mentfx').
        risk_model_config: Dict for SL buffer, TP RR, account size, risk %.
                           e.g., {'sl_buffer_pips': 1, 'tp_rr': 3.0, 'account_size': 100000, 'risk_percent': 1.0}
        spread_points: Instrument spread in points (e.g., 2 points = 0.2 pips for EURUSD).
        confluence_data: Optional results from confluence engine.
        execution_params: Optional overrides {'force_direction': 'buy'/'sell', 'entry_type': 'midpoint'/'wick'}.
        symbol: The trading symbol (e.g., 'XAUUSD', 'OANDA:EUR_USD') for Pine payload.

    Returns:
        Dictionary containing execution result and optional PineConnector payload.
    """
    # Initialize result structure
    result = {
        "entry_confirmed": False,
        "entry_type": "None",
        "entry_candle_timestamp": None,
        "entry_price": None,
        "sl": None,
        "tp": None,
        "direction": None,
        "lot_size": None,
        "r_multiple": None,
        "payout_usd": None,
        "pine_payload": None,
        "error": None
    }

    # --- Input Validation ---
    if not confirmation_data or not confirmation_data.get("confirmation_status"):
        result["error"] = "Invalid or unconfirmed confirmation data provided."
        return result
    if not confirmation_data.get("ltf_poi_range") or not isinstance(confirmation_data["ltf_poi_range"], list) or len(confirmation_data["ltf_poi_range"]) != 2:
        result["error"] = "Missing or invalid LTF POI range in confirmation data."
        return result
    if ltf_data is None or ltf_data.empty:
        result["error"] = "LTF data for execution is missing or empty."
        return result
    if not all(col in ltf_data.columns for col in ['Open', 'High', 'Low', 'Close']):
         result["error"] = "LTF data missing required OHLC columns."
         return result

    # --- Extract Confirmation Details ---
    ltf_poi_range = confirmation_data["ltf_poi_range"]
    ltf_poi_low, ltf_poi_high = min(ltf_poi_range), max(ltf_poi_range)
    htf_poi_type = confirmation_data.get("mitigated_htf_poi", {}).get("type", "Unknown")
    direction = 'buy' if htf_poi_type == 'Bullish' else ('sell' if htf_poi_type == 'Bearish' else None)
    ltf_poi_timestamp_str = confirmation_data.get("ltf_poi_timestamp")

    if not direction:
        result["error"] = "Could not determine trade direction from HTF POI type."
        return result
    result["direction"] = direction # Set direction early

    if execution_params and execution_params.get('force_direction'):
        direction = execution_params['force_direction']
        result["direction"] = direction
        print(f"[INFO] Direction forced to: {direction}")

    # --- Configuration ---
    if risk_model_config is None: risk_model_config = {}
    if execution_params is None: execution_params = {}

    tp_rr = risk_model_config.get('tp_rr', 3.0)
    account_size = risk_model_config.get('account_size', 100000)
    risk_percent = risk_model_config.get('risk_percent', 1.0)

    # Dynamically adjust risk if entry is detected as counter-trend scalp
    is_scalp_mode = confirmation_data.get("suggested_scalp_mode", False)
    if is_scalp_mode:
        print("[SCALP MODE] Activated – Lowering risk threshold.")
        risk_percent = 0.25 if confluence_data and confluence_data.get("risk_tag") == "high" else 0.5
        result["entry_type"] += " [SCALP]"
        result["risk_mode"] = "scalp"
        result["adjusted_risk_percent"] = risk_percent
    # SL buffer needs careful handling - pips vs points vs absolute value
    # Assuming sl_buffer_pips needs conversion based on instrument
    sl_buffer_pips = risk_model_config.get('sl_buffer_pips', 1.0) # e.g., 1 pip
    instrument_point_value = 0.0001 # Example for EURUSD 5-decimal
    sl_buffer_price = sl_buffer_pips * instrument_point_value * 10 # Convert pips to price value

    # --- Ultra-Precision Gating: DSS + VWAP + POI Score ---
    # Optional ultra-precision gating (VWAP, DSS, POI score)
    if confluence_data:
        poi_score = confluence_data.get("poi_score", 0)
        vwap_dev = confluence_data.get("vwap_deviation", 0)
        dss_slope = confluence_data.get("dss_slope", 0)

        if poi_score < 0.55:
            result["error"] = f"POI score too low: {poi_score}"
            print(f"[DEBUG] Ultra Filter: POI score below threshold.")
            # --- Markdown Logging for Rejected Entry ---
            import os
            from datetime import datetime
            log_dir = "journal"
            os.makedirs(log_dir, exist_ok=True)

            entry_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            md_path = os.path.join(log_dir, f"rejected_entry_{symbol}_{entry_time}.md")

            poi_reason = confluence_data.get("poi_reason", "")
            md_content = f"""
## ❌ Rejected Entry – POI Score Below Threshold

- **Symbol**: {symbol}
- **Strategy**: {strategy_variant}
- **POI Score**: {poi_score:.2f}
- **Reason**: {poi_reason}
- **DSS Slope**: {confluence_data.get('dss_slope') if confluence_data else 'N/A'}
- **VWAP Deviation**: {confluence_data.get('vwap_deviation') if confluence_data else 'N/A'}
- **Phase**: {confirmation_data.get('wyckoff_phase') if confirmation_data else 'N/A'}
- **Time**: {entry_time}
"""

            with open(md_path, "w") as f:
                f.write(md_content.strip())

            print(f"[MARKDOWN] Rejected entry logged to {md_path}")
            return result
        if abs(vwap_dev) > 2.5:
            result["error"] = f"VWAP deviation too extreme: {vwap_dev}"
            print(f"[DEBUG] Ultra Filter: VWAP deviation exceeded.")
            return result
        if dss_slope < 0:
            result["error"] = "DSS slope bearish / declining at trigger."
            print(f"[DEBUG] Ultra Filter: DSS slope invalid.")
            return result

    # --- Core Logic ---
    try:
        # 1. Detect Mitigation of LTF POI
        # Filter LTF data to only include candles after the LTF POI was formed
        if not ltf_poi_timestamp_str:
             result["error"] = "Missing LTF POI timestamp in confirmation data."
             return result
        try:
             ltf_poi_ts = pd.Timestamp(ltf_poi_timestamp_str, tz='UTC')
             execution_ltf_data = ltf_data[ltf_data.index > ltf_poi_ts]
        except Exception as ts_err:
             result["error"] = f"Error parsing LTF POI timestamp: {ts_err}"
             return result

        if execution_ltf_data.empty:
            result["error"] = "No LTF data available after LTF POI formation timestamp."
            return result

        mitigation_candle = find_mitigation_candle(execution_ltf_data, ltf_poi_range, direction)

        if mitigation_candle is None:
            result["error"] = "LTF POI was not mitigated in the provided LTF data."
            return result

        print(f"[DEBUG] LTF POI Mitigation detected by candle at: {mitigation_candle.name}")
        result["entry_candle_timestamp"] = mitigation_candle.name.strftime('%Y-%m-%d %H:%M:%S')

        # --- Structure Break Confirmation after Mitigation ---
        structure_break = detect_structure_break(execution_ltf_data, direction=direction, tf_prefix='LTF')
        if not structure_break or structure_break.get('break_type') not in [ConfirmationConstants.CHOCH, ConfirmationConstants.BOS]:
            result["error"] = "Structure break not confirmed after mitigation."
            print(f"[DEBUG] Structure break validation failed. Skipping entry.")
            return result

        # 2. Apply Variant-Specific Entry/Mitigation Rules
        entry_price = None
        sl_price = None
        entry_type_detail = "Mitigated POI"
        mitigation_valid = False

        # Get candle immediately before mitigation candle for engulfing check
        mitigation_candle_index = execution_ltf_data.index.get_loc(mitigation_candle.name)
        prev_mitigation_candle = execution_ltf_data.iloc[mitigation_candle_index - 1] if mitigation_candle_index > 0 else None

        if strategy_variant == 'Inv':
            # "POI mitigation + wick tap allowed"
            mitigation_valid = True # Simple wick mitigation is enough
            entry_price = mitigation_candle['Low'] if direction == 'buy' else mitigation_candle['High'] # Entry at wick extreme
            sl_price = ltf_poi_low - sl_buffer_price if direction == 'buy' else ltf_poi_high + sl_buffer_price # SL below/above POI range
            entry_type_detail = "Inv: Wick Mitigation"
            print(f"[DEBUG] Inv Logic: Wick mitigation accepted. Entry={entry_price}, SL={sl_price}")

        elif strategy_variant == 'MAZ2':
            # "Must re-test refined FVG (no body engulf allowed)"
            # TODO: MAZ2 - Implement FVG re-test check. Placeholder assumes simple mitigation is ok for now.
            # TODO: MAZ2 - Implement 'no body engulf' check.
            if confirmation_data.get("ltf_poi_type") == 'FVG':
                 mitigation_valid = True # Placeholder: Assume FVG mitigation is valid
                 entry_price = (ltf_poi_low + ltf_poi_high) / 2 # Enter FVG midpoint
                 sl_price = ltf_poi_low - sl_buffer_price if direction == 'buy' else ltf_poi_high + sl_buffer_price # SL outside FVG
                 entry_type_detail = "MAZ2: FVG Mitigation (Checks Pending)"
                 print(f"[DEBUG] MAZ2 Logic: FVG mitigation accepted (placeholder). Entry={entry_price}, SL={sl_price}")
                 print("[WARN] MAZ2 specific checks (FVG re-test, no body engulf) not implemented.")
            else:
                 result["error"] = "MAZ2 variant requires FVG type LTF POI (based on summary)."
                 print(f"[DEBUG] {result['error']}")


        elif strategy_variant == 'TMC':
            # "Entry only valid after BOS + confluence"
            # Check if confirmation included BOS
            if confirmation_data.get("confirmation_type") != "CHoCH+BOS":
                 result["error"] = "TMC variant requires BOS confirmation."
                 print(f"[DEBUG] {result['error']}")
            else:
                 # TODO: TMC - Implement actual confluence check using confluence_data
                 tmc_confluence_check = True # Placeholder
                 if tmc_confluence_check:
                     mitigation_valid = True
                     entry_price = mitigation_candle['Close'] # Example: Enter on close of mitigation candle
                     sl_price = mitigation_candle['Low'] - sl_buffer_price if direction == 'buy' else mitigation_candle['High'] + sl_buffer_price # SL below/above mitigation candle
                     entry_type_detail = "TMC: BOS Confirmed + Mitigation (Confluence Placeholder)"
                     print(f"[DEBUG] TMC Logic: BOS confirmed, mitigation accepted (confluence placeholder). Entry={entry_price}, SL={sl_price}")
                     print("[WARN] TMC confluence check not implemented.")
                 else:
                     result["error"] = "TMC variant confluence check failed (placeholder)."
                     print(f"[DEBUG] {result['error']}")


        elif strategy_variant == 'Mentfx':
            # "Requires DSS/RSI + Pinbar or engulfing confirmation"
            # TODO: Mentfx - Implement actual confluence check (DSS/RSI) using confluence_data
            mentfx_confluence_check = True # Placeholder
            if not mentfx_confluence_check:
                 result["error"] = "Mentfx variant confluence check failed (placeholder)."
                 print(f"[DEBUG] {result['error']}")
            else:
                 # Check for engulfing pattern on mitigation candle
                 is_engulfing = check_engulfing_pattern(mitigation_candle, prev_mitigation_candle, direction)
                 # TODO: Mentfx - Implement Pinbar check on mitigation candle
                 is_pinbar = False # Placeholder

                 if is_engulfing or is_pinbar:
                     mitigation_valid = True
                     entry_price = mitigation_candle['Close'] # Enter on close of confirmation candle
                     sl_price = mitigation_candle['Low'] - sl_buffer_price if direction == 'buy' else mitigation_candle['High'] + sl_buffer_price # SL below/above confirmation candle
                     pattern = "Engulfing" if is_engulfing else "Pinbar"
                     entry_type_detail = f"Mentfx: Mitigation + {pattern} (Confluence Placeholder)"
                     print(f"[DEBUG] Mentfx Logic: Mitigation + {pattern} ok (confluence placeholder). Entry={entry_price}, SL={sl_price}")
                     print("[WARN] Mentfx confluence and pinbar checks not fully implemented.")

                 else:
                     result["error"] = "Mentfx variant requires Engulfing or Pinbar at mitigation."
                     print(f"[DEBUG] {result['error']}")

        else: # Default / Unknown variant
             mitigation_valid = True # Default: Simple mitigation is enough
             entry_price = (ltf_poi_low + ltf_poi_high) / 2 # Default: Enter POI midpoint
             sl_price = ltf_poi_low - sl_buffer_price if direction == 'buy' else ltf_poi_high + sl_buffer_price # SL below/above POI range
             entry_type_detail = "Default: POI Midpoint Mitigation"
             print(f"[DEBUG] Default Logic: POI midpoint mitigation accepted. Entry={entry_price}, SL={sl_price}")


        # 3. Calculate Final Parameters if Mitigation is Valid
        if mitigation_valid and entry_price is not None and sl_price is not None:
            result["entry_type"] = entry_type_detail

            # Apply overrides if present
            if execution_params.get('entry_type') == 'midpoint':
                entry_price = (ltf_poi_low + ltf_poi_high) / 2
                print(f"[INFO] Entry price overridden to POI midpoint: {entry_price}")
            elif execution_params.get('entry_type') == 'wick' and mitigation_candle is not None:
                 entry_price = mitigation_candle['Low'] if direction == 'buy' else mitigation_candle['High']
                 print(f"[INFO] Entry price overridden to mitigation wick: {entry_price}")


            # --- Microstructure Validation (Scalp-Grade) ---
            try:
                # Reconstruct recent ticks if available
                if isinstance(ltf_data, pd.DataFrame) and all(k in ltf_data.columns for k in ['<BID>', '<ASK>', '<DATE>', '<TIME>']):
                    tick_df = ltf_data[['<BID>', '<ASK>', '<DATE>', '<TIME>']].dropna().tail(6)
                    scalp_signal = {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "rr": tp_rr,
                        "direction": direction,
                        "trigger_tf": "M1"
                    }
                    default_config = {
                        "tick_window": 5,
                        "min_drift_pips": 1.0,
                        "max_spread": 1.8,
                        "min_rr_threshold": 1.8
                    }
                    micro_res = validate_scalp_signal(scalp_signal, tick_df, default_config)
                    if not micro_res.get("confirmed"):
                        result["error"] = f"Scalp signal rejected: {micro_res['reason']}"
                        result["entry_confirmed"] = False
                        print(f"[MICROSTRUCTURE REJECT] {micro_res['reason']}")
                        return result
            except Exception as ms_err:
                print(f"[ERROR] Microstructure validation error: {ms_err}")


            # 4. Generate Risk + Lot Size
            risk_calcs = calculate_trade_risk(
                entry=entry_price,
                sl=sl_price, # Pass unadjusted SL, buffer/spread applied inside
                r_multiple=tp_rr,
                account_size=account_size,
                risk_pct=risk_percent,
                spread_points=spread_points, # Pass spread points directly
                direction=direction
            )

            if risk_calcs and risk_calcs.get('lot_size', 0) > 0:
                result["entry_confirmed"] = True
                result["entry_price"] = risk_calcs['entry']
                result["sl"] = risk_calcs['sl'] # Adjusted SL from calculation
                result["tp"] = risk_calcs['tp']
                result["lot_size"] = risk_calcs['lot_size']
                result["r_multiple"] = risk_calcs['r_multiple']
                result["payout_usd"] = risk_calcs['payout_usd']

                # 5. Generate Pine Payload
                result["pine_payload"] = generate_pine_payload(
                    symbol=symbol,
                    entry=result["entry_price"],
                    sl=result["sl"],
                    tp=result["tp"],
                    lot_size=result["lot_size"],
                    direction=direction,
                    comment=f"{strategy_variant} Entry" # Add variant to comment
                )
                # Store structure break confirmation details
                result["confirmation_details"] = structure_break
                print(f"[INFO] Entry Execution SUCCESS: Dir={direction}, Entry={result['entry_price']}, SL={result['sl']}, TP={result['tp']}, Lots={result['lot_size']}")

                # --- Confluence Scan ---
                # Compute latest ATR from ltf_data if not already available
                latest_atr = None
                if 'latest_atr' in locals():
                    pass  # Use as is
                else:
                    # Compute ATR from ltf_data if possible (using last 14 candles)
                    try:
                        if isinstance(ltf_data, pd.DataFrame) and all(col in ltf_data.columns for col in ['High','Low','Close']):
                            ltf_atr = ltf_data[['High','Low','Close']].copy()
                            tr = np.maximum(
                                ltf_atr['High'] - ltf_atr['Low'],
                                np.maximum(
                                    abs(ltf_atr['High'] - ltf_atr['Close'].shift(1)),
                                    abs(ltf_atr['Low'] - ltf_atr['Close'].shift(1))
                                )
                            )
                            latest_atr = tr.rolling(window=14).mean().iloc[-1]
                        else:
                            latest_atr = None
                    except Exception:
                        latest_atr = None
                # micro_res may not be defined if microstructure validation not run
                micro_res = locals().get('micro_res', None)
                confluence_report = scan_confluences(
                    structure_break,
                    mitigation_candle,
                    micro_res if 'micro_res' in locals() else None,
                    latest_atr,
                    confluence_data or {},
                    risk_model_config or {}
                )
                result['confluence_report'] = confluence_report

                # --- Markdown Logging for Accepted Entry ---
                from datetime import datetime
                import os
                log_dir = "journal"
                os.makedirs(log_dir, exist_ok=True)

                entry_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                md_path = os.path.join(log_dir, f"accepted_entry_{symbol}_{entry_time}.md")

                tags = ["POI", "accepted", "strategy_live"]
                wyckoff_phase = confirmation_data.get("wyckoff_phase", "").lower()
                dss_slope = confluence_data.get("dss_slope") if confluence_data else None
                vwap_dev = confluence_data.get("vwap_deviation") if confluence_data else None

                if "phase" in wyckoff_phase:
                    tags.append(f"phase_{wyckoff_phase.replace(' ', '_')}")
                if dss_slope and dss_slope > 0.1:
                    tags.append("dss_up")
                if vwap_dev and abs(vwap_dev) < 2.0:
                    tags.append("vwap_near")

                md_content = f"""\
## ✅ Accepted Entry – Trade Triggered

- **Symbol**: {symbol}
- **Strategy**: {strategy_variant}
- **POI Score**: {confirmation_data.get('poi_score')}
- **Reason**: {confirmation_data.get('poi_reason')}
- **DSS Slope**: {dss_slope}
- **VWAP Deviation**: {vwap_dev}
- **Wyckoff Phase**: {wyckoff_phase}
- **Tags**: [{', '.join(tags)}]
- **Time**: {entry_time}
"""

                with open(md_path, "w") as f:
                    f.write(md_content.strip())

                print(f"[MARKDOWN] Accepted entry logged to {md_path}")

                # --- Optional Telegram Alert ---
                try:
                    from telegram_alerts import send_telegram_alert
                    alert_msg = f"✅ ENTRY TRIGGERED [{symbol}] | {result['entry_type']} | R:{result['r_multiple']} | ${round(result['payout_usd'], 2)}"
                    send_telegram_alert(alert_msg)
                except Exception as e:
                    print(f"[WARN] Telegram alert failed: {e}")

                # --- POI Score Verification Hook (Final Entry Filter) ---
                if confirmation_data:
                    poi_score = confirmation_data.get("poi_score", 0)
                    poi_reason = confirmation_data.get("poi_reason", "")
                    if poi_score < 0.55:
                        result["entry_confirmed"] = False
                        result["error"] = f"Entry blocked: POI score {poi_score:.2f} too low. Reason: {poi_reason}"
                        print(f"[ENTRY_EXECUTOR] Entry score gate failed: {poi_score:.2f} | {poi_reason}")
                        # --- Markdown Logging for Rejected Entry ---
                        import os
                        from datetime import datetime
                        log_dir = "journal"
                        os.makedirs(log_dir, exist_ok=True)

                        entry_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                        md_path = os.path.join(log_dir, f"rejected_entry_{symbol}_{entry_time}.md")

                        # Try to get DSS/VWAP/Phase from confluence_data if available, else from confirmation_data or N/A
                        dss_slope = confluence_data.get('dss_slope') if confluence_data else 'N/A'
                        vwap_deviation = confluence_data.get('vwap_deviation') if confluence_data else 'N/A'
                        phase = confirmation_data.get('wyckoff_phase') if confirmation_data else 'N/A'
                        md_content = f"""
## ❌ Rejected Entry – POI Score Below Threshold

- **Symbol**: {symbol}
- **Strategy**: {strategy_variant}
- **POI Score**: {poi_score:.2f}
- **Reason**: {poi_reason}
- **DSS Slope**: {dss_slope}
- **VWAP Deviation**: {vwap_deviation}
- **Phase**: {phase}
- **Time**: {entry_time}
"""

                        with open(md_path, "w") as f:
                            f.write(md_content.strip())

                        print(f"[MARKDOWN] Rejected entry logged to {md_path}")
                        return result

            else:
                result["error"] = "Risk calculation failed or resulted in zero lot size."
                print(f"[ERROR] {result['error']}")
                result["entry_confirmed"] = False # Ensure confirmed is false

        else:
             if not result["error"]: # If no specific error set yet
                 result["error"] = f"Mitigation or entry/SL calculation failed for variant {strategy_variant}."
             print(f"[WARN] Entry execution failed: {result['error']}")
             result["entry_confirmed"] = False


    except Exception as e:
        import traceback
        print(f"[CRITICAL] Exception in execute_smc_entry: {e}\n{traceback.format_exc()}")
        result["error"] = f"Runtime error: {e}"
        result["entry_confirmed"] = False

    # Clean up result if entry not confirmed
    if not result["entry_confirmed"]:
         result["entry_type"] = "None"
         # Keep timestamp if mitigation occurred but failed later
         # result["entry_candle_timestamp"] = None
         result["entry_price"] = None
         result["sl"] = None
         result["tp"] = None
         # Keep direction if determined
         result["lot_size"] = None
         result["r_multiple"] = None
         result["payout_usd"] = None
         result["pine_payload"] = None


    return result


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing Entry Executor (SMC) ---")

    # Dummy confirmation data (output from previous module)
    dummy_confirmation = {
      "confirmation_status": True,
      "confirmation_type": "CHoCH",
      "choch_details": { "price": 1.1015, "timestamp": "2023-10-26 10:25:00", "type": "Bullish", "broken_swing_timestamp": "2023-10-26 10:18:00"},
      "bos_details": None,
      "ltf_poi_range": [1.1008, 1.1013], # Example FVG/OB range
      "ltf_poi_timestamp": "2023-10-26 10:23:00", # Timestamp POI formed
      "ltf_poi_type": "OB",
      "mitigated_htf_poi": {'range': (1.1000, 1.1010), 'type': 'Bullish', 'timestamp': pd.Timestamp('2023-10-26 08:00:00'), 'source_tf': 'H4'},
      "error": None
    }

    # Dummy LTF data covering period *after* LTF POI formation
    timestamps_exec = pd.date_range(start='2023-10-26 10:24:00', periods=20, freq='T', tz='UTC')
    data_exec = {
        'Open': np.ones(20) * 1.1015, 'High': np.ones(20) * 1.1017,
        'Low': np.ones(20) * 1.1013, 'Close': np.ones(20) * 1.1015,
    }
    dummy_ltf_df_exec = pd.DataFrame(data_exec, index=timestamps_exec)
    # Simulate mitigation
    dummy_ltf_df_exec.loc[timestamps_exec[5], 'Low'] = 1.1012 # Wick into POI [1.1008, 1.1013]
    dummy_ltf_df_exec.loc[timestamps_exec[5], 'Close'] = 1.1014 # Close higher
    # Simulate engulfing candle after mitigation
    dummy_ltf_df_exec.loc[timestamps_exec[4], ['Open', 'Close']] = [1.1013, 1.1011] # Prev bearish
    dummy_ltf_df_exec.loc[timestamps_exec[5], ['Open', 'Close']] = [1.1010, 1.1014] # Bullish engulfing


    dummy_risk_config = {'tp_rr': 3.0, 'account_size': 50000, 'risk_percent': 0.5, 'sl_buffer_pips': 1.5}
    dummy_symbol = "EURUSD" # Example symbol

    variants_to_test = ['Inv', 'MAZ2', 'TMC', 'Mentfx', 'Unknown']
    for variant in variants_to_test:
         print(f"\n--- Testing Variant: {variant} ---")
         # Adjust confirmation data slightly if needed for variant test
         if variant == 'TMC': dummy_confirmation['confirmation_type'] = 'CHoCH+BOS' # Ensure BOS for TMC
         if variant == 'MAZ2': dummy_confirmation['ltf_poi_type'] = 'FVG' # Ensure FVG for MAZ2 test

         exec_result = execute_smc_entry(
             ltf_data=dummy_ltf_df_exec,
             confirmation_data=dummy_confirmation,
             strategy_variant=variant,
             risk_model_config=dummy_risk_config,
             spread_points=2, # Example spread
             symbol=dummy_symbol
         )
         import json
         print(json.dumps(exec_result, indent=2, default=str))

    print("\n--- Testing Complete ---")

