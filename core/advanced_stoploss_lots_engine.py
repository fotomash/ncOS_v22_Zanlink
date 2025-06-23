# advanced_stoploss_lots_engine.py
# Author: ZANZIBAR LLM Assistant (Generated for Tomasz)
# Date: 2025-04-28
# Version: 1.2.1 (Pip/Point Map Finalized for FTMO/FTP + Syntax Fix)
# Description:
#   Calculates blended stop-loss, volatility-adjusted risk, and lot size.
#   Includes refined Pip/Point value mapping for common prop firm instruments.

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import traceback
import math # For ceiling function in lot size calculation
from datetime import datetime, timezone # Added timezone for awareness
import re # For cleaning symbols
from core.microstructure_filter import MicrostructureFilter
from core.advanced_stoploss_lots_engine import calculate_atr  # or adjust import if calculate_atr is in this module

# --- Define Dummy Fallback Functions ---
# These are defined first, outside the try/except blocks.

def _dummy_mentfx_sl(**kwargs):
    """Dummy function if mentfx_stoploss_model import fails."""
    print("[WARN][AdvSLRiskEngine] Using dummy structural SL function.")
    return {"computed_stop_loss": np.nan, "error": "Module not available"}

def _dummy_volatility_profile(**kwargs):
    """Dummy function if volatility_engine import fails."""
    print("[WARN][AdvSLRiskEngine] Using dummy volatility profile function.")
    return {"volatility_regime": "Normal", "error": "Module not available"}

# --- Import Dependencies ---
# Structural SL Model (Adaptive)
try:
    from core.mentfx_stoploss_model_v2_adaptive import compute_mentfx_stop_loss_adaptive
    MENTFX_SL_AVAILABLE = True
    print("[INFO][AdvSLRiskEngine] Imported mentfx_stoploss_model_v2_adaptive.")
except ImportError:
    print("[WARN][AdvSLRiskEngine] mentfx_stoploss_model_v2_adaptive.py not found. Structural SL calculation disabled.")
    MENTFX_SL_AVAILABLE = False
    # Assign the dummy function if import failed
    compute_mentfx_stop_loss_adaptive = _dummy_mentfx_sl

# Volatility Regime Engine
try:
    from core.volatility_engine import get_volatility_profile
    VOLATILITY_ENGINE_AVAILABLE = True
    print("[INFO][AdvSLRiskEngine] Imported volatility_engine.")
except ImportError:
    print("[WARN][AdvSLRiskEngine] volatility_engine.py not found. Volatility regime adjustment disabled.")
    VOLATILITY_ENGINE_AVAILABLE = False
    # Assign the dummy function if import failed
    get_volatility_profile = _dummy_volatility_profile

# ATR Calculation (using TA-Lib if available, otherwise pandas)
try:
    import talib
    TALIB_AVAILABLE = True
    print("[INFO][AdvSLRiskEngine] TA-Lib found for ATR calculation.")
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARN][AdvSLRiskEngine] TA-Lib not found. Using pandas for ATR calculation.")

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ Calculates Average True Range (ATR)."""
    min_len = period + 1
    if high.isnull().all() or low.isnull().all() or close.isnull().all() or len(high) < min_len:
        return pd.Series(np.nan, index=high.index)
    if TALIB_AVAILABLE:
        try:
            # Ensure inputs are float type for TA-Lib
            return talib.ATR(high.astype(float), low.astype(float), close.astype(float), timeperiod=period)
        except Exception as e:
            print(f"[ERROR][AdvSLRiskEngine] TA-Lib ATR calculation failed: {e}. Falling back.")
            # Fallback to pandas if TA-Lib fails unexpectedly
            pass
    # Pandas fallback implementation
    high_low = high - low
    high_close_prev = abs(high - close.shift(1))
    low_close_prev = abs(low - close.shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close_prev, 'lc': low_close_prev}).max(axis=1)
    # Using EWM (closer to Wilder's smoothing used in TA-Lib's default ATR)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

# --- Helper Functions ---

# --- ### START PIP/POINT MAP FINALIZATION ### ---
# Incorporating the user-provided function structure and map
def get_pip_point_value(symbol: str, account_currency: str = "USD") -> Optional[Tuple[float, int]]:
    """
    Provides pip/point value and decimals for given trading symbol.
    Now patched for FTMO and FundedTradingPlus environments.

    Args:
        symbol (str): Trading symbol, e.g., 'OANDA:EUR_USD', 'FXCM:GBPJPY', 'BTCUSD'.
        account_currency (str): The currency of the trading account (e.g., "USD", "EUR"). Default is "USD".


    Returns:
        Optional[Tuple[float, int]]: (value per point per lot, decimal places)
                                       Point value is for the smallest price increment.
    """
    if not isinstance(symbol, str) or not symbol:
        print(f"[ERROR][AdvSLRiskEngine] Invalid symbol provided to get_pip_point_value: {symbol}")
        return None

    # Clean symbol input
    cleaned_symbol = symbol.upper()
    prefixes_to_remove = ["OANDA:", "FXCM:", "PEPPERSTONE:", "ICMARKETS:", "FTMO/", "FUNDEDTRADINGPLUS/"]
    for prefix in prefixes_to_remove:
        if cleaned_symbol.startswith(prefix):
            cleaned_symbol = cleaned_symbol[len(prefix):]
    cleaned_symbol = re.sub(r'[_/\-\.]', '', cleaned_symbol) # Remove separators and dots

    # Handle specific aliases AFTER removing prefixes/separators
    symbol_aliases = {
        "SPX500": "US500", "SPX500USD": "US500", "US500CASH": "US500", "S&P500": "US500", "ES": "US500",
        "NAS100": "USTEC", "NAS100USD": "USTEC", "USTEC100": "USTEC", "NASDAQ100": "USTEC", "NQ": "USTEC",
        "GER40": "DE40", "DE30": "DE40", "DAX40": "DE40", "DAX": "DE40",
        "US30": "US30", "DOWJONES": "US30", "DJ30": "US30", "YM": "US30",
        "UK100": "UK100", "FTSE100": "UK100",
        "XAUUSD": "XAUUSD", "GOLD": "XAUUSD",
        "XAGUSD": "XAGUSD", "SILVER": "XAGUSD",
        "BTCUSD": "BTCUSD", "BTC": "BTCUSD",
        "ETHUSD": "ETHUSD", "ETH": "ETHUSD",
        "USOIL": "WTI", "OILUSD": "WTI", "OIL": "WTI", "CL": "WTI",
        "UKOIL": "BRENT", "OILBRENT": "BRENT",
    }
    cleaned_symbol = symbol_aliases.get(cleaned_symbol, cleaned_symbol)

    print(f"[DEBUG][AdvSLRiskEngine] Cleaned symbol for Pip/Point lookup: {cleaned_symbol}")

    # Predefined Pip/Point Value Map for FTMO and FTP (Standard Lot Sizes)
    # Format: {Canonical Symbol: (Point Value in QUOTE CURRENCY, Price Decimals)}
    # Point = Smallest price increment (e.g., 0.00001 for EURUSD, 0.01 for XAUUSD)
    # Value needs conversion to account currency later if quote != account currency.
    pip_point_map = {
        # --- FX (Standard Lot = 100,000 base currency) ---
        # Value = 100,000 * point_size
        "EURUSD": (1.0, 5), # 100k * 0.00001 = 1 USD per point
        "GBPUSD": (1.0, 5), # 1 USD per point
        "AUDUSD": (1.0, 5), # 1 USD per point
        "NZDUSD": (1.0, 5), # 1 USD per point
        "USDCAD": (1.0, 5), # 1 CAD per point
        "USDCHF": (1.0, 5), # 1 CHF per point
        "USDJPY": (100.0, 3), # 100k * 0.001 = 100 JPY per point
        "EURJPY": (100.0, 3), # 100 JPY per point
        "GBPJPY": (100.0, 3), # 100 JPY per point
        "AUDJPY": (100.0, 3), # 100 JPY per point
        "NZDJPY": (100.0, 3), # 100 JPY per point
        "CADJPY": (100.0, 3), # 100 JPY per point
        "CHFJPY": (100.0, 3), # 100 JPY per point
        "EURGBP": (1.0, 5), # 1 GBP per point
        "EURCHF": (1.0, 5), # 1 CHF per point
        "EURAUD": (1.0, 5), # 1 AUD per point
        "EURCAD": (1.0, 5), # 1 CAD per point
        "EURNZD": (1.0, 5), # 1 NZD per point
        "GBPAUD": (1.0, 5), # 1 AUD per point
        "GBPCAD": (1.0, 5), # 1 CAD per point
        "GBPCHF": (1.0, 5), # 1 CHF per point
        "GBPNZD": (1.0, 5), # 1 NZD per point
        "AUDCAD": (1.0, 5), # 1 CAD per point
        "AUDCHF": (1.0, 5), # 1 CHF per point
        "AUDNZD": (1.0, 5), # 1 NZD per point
        "CADCHF": (1.0, 5), # 1 CHF per point
        "NZDCAD": (1.0, 5), # 1 CAD per point
        "NZDCHF": (1.0, 5), # 1 CHF per point
        # Add other common crosses...

        # --- Metals (Check FTMO/FTP Contract Specs - Lot Size Varies!) ---
        # Assuming 1 Lot = 100 oz for Gold, 5000 oz for Silver (Common but VERIFY)
        "XAUUSD": (1.0, 2), # 100oz * $0.01 price change = $1 USD per point
        "XAGUSD": (5.0, 3), # 5000oz * $0.001 price change = $5 USD per point - VERIFY DECIMALS (often 3)

        # --- Indices (CFDs - Check FTMO/FTP Specs - Lot Size = 1 Index usually) ---
        # Value = Point size ($1) * Contract Multiplier (often 1 for CFDs quoted in USD)
        "US500": (1.0, 2), # $1 USD per 1.0 index move
        "USTEC": (1.0, 2), # $1 USD per 1.0 index move
        "US30": (1.0, 2),  # $1 USD per 1.0 index move
        "DE40": (1.0, 2),  # €1 EUR per 1.0 index move - Needs conversion
        "UK100": (1.0, 2), # £1 GBP per 1.0 index move - Needs conversion
        "JP225": (100.0, 2), # JPY based index - Value often 100 JPY per 1.0 move - Needs conversion
        "AUS200": (1.0, 1), # AUD based index - Value often 1 AUD per 1.0 move - Needs conversion

        # --- Crypto (CFDs - Check FTMO/FTP Specs - Lot Size = 1 Coin usually) ---
        "BTCUSD": (1.0, 2), # $1 USD per $1 price change
        "ETHUSD": (1.0, 2), # $1 USD per $1 price change

        # --- Commodities (CFDs - Check FTMO/FTP Specs - Lot Size Varies!) ---
        # WTI/BRENT often 1 lot = 100 or 1000 barrels. Point = 0.01.
        # If 100 barrel lot, value = 100 * $0.01 = $1 USD per point. VERIFY!
        "WTI": (1.0, 2),
        "BRENT": (1.0, 2),
    }

    # --- Lookup and Return ---
    if cleaned_symbol in pip_point_map:
        value_quote_ccy, decimals = pip_point_map[cleaned_symbol]
        quote_currency = "USD" # Assume USD unless determined otherwise

        # --- Determine Quote Currency (Basic) ---
        # This logic needs refinement based on broker conventions
        if len(cleaned_symbol) >= 6 and cleaned_symbol.endswith(("JPY", "CHF", "CAD", "AUD", "NZD", "GBP", "EUR")):
            quote_currency = cleaned_symbol[-3:]
        elif cleaned_symbol in ["DE40"]: quote_currency = "EUR"
        elif cleaned_symbol in ["UK100"]: quote_currency = "GBP"
        elif cleaned_symbol in ["JP225"]: quote_currency = "JPY"
        elif cleaned_symbol in ["AUS200"]: quote_currency = "AUD"
        # Assume USD for others (XAUUSD, Indices like US500, Crypto, WTI/BRENT)

        # --- Apply Conversion if Needed ---
        value_acct_ccy = value_quote_ccy
        if account_currency.upper() != quote_currency.upper():
            print(f"[WARN][AdvSLRiskEngine] Account currency ({account_currency}) differs from quote currency ({quote_currency}) for {symbol}. Point value requires live rate conversion for accuracy. Using approximate value.")
            # --- TODO: Implement Live Rate Conversion ---
            # Example:
            # try:
            #     rate = get_live_fx_rate(f"{quote_currency}{account_currency}") # e.g., get_live_fx_rate("JPYUSD")
            #     if rate is not None and rate > 0:
            #         value_acct_ccy = value_quote_ccy * rate
            #     else:
            #         print(f"[ERROR] Invalid rate received for {quote_currency}{account_currency}")
            # except Exception as rate_err:
            #     print(f"[ERROR] Failed rate conversion for {symbol}: {rate_err}")
            # --- End TODO ---

        # --- Final Warnings ---
        if any(cmd in cleaned_symbol for cmd in ["WTI", "BRENT", "XAGUSD"]):
             print(f"[WARN][AdvSLRiskEngine] Point value for commodity/metal {symbol} depends heavily on contract size. Verify value.")

        return value_acct_ccy, decimals
    else:
        # Fallback for unknown symbols (assume 5-decimal FX Major)
        print(f"[WARN][AdvSLRiskEngine] Pip/Point value mapping not found for symbol '{symbol}' (Cleaned: '{cleaned_symbol}'). Using default FX Major (Point Value=1.0, Decimals=5).")
        return 1.0, 5
# --- ### END PIP/POINT MAP FINALIZATION ### ---


def map_conviction_to_risk(score: int) -> float:
    """ Maps conviction score (1-5) to risk percentage (0.25% - 1.0%). """
    # (Function body remains the same)
    if not isinstance(score, int) or not 1 <= score <= 5: print(f"[WARN][AdvSLRiskEngine] Invalid conviction score '{score}'. Defaulting to minimum risk (0.25%)."); score = 1
    mapping = { 5: 1.00, 4: 0.75, 3: 0.50, 2: 0.35, 1: 0.25 }; return mapping.get(score, 0.25) / 100.0

# --- Main Calculation Engine ---
def calculate_sl_and_risk(
    account_balance: float,
    conviction_score: int,
    entry_price: float,
    entry_time: datetime,
    trade_type: str,
    symbol: str,
    strategy_variant: str,
    ohlc_data: pd.DataFrame, # For ATR & Volatility Regime (e.g., M15 or H1)
    tick_data: Optional[pd.DataFrame] = None, # For Structural SL (Optional)
    mentfx_sl_config: Optional[Dict] = None,
    atr_config: Optional[Dict] = None,
    risk_config: Optional[Dict] = None, # General risk settings + Volatility settings
    volatility_config: Optional[Dict] = None # Specific config for volatility engine
) -> Dict[str, Any]:
    """
    Calculates blended stop-loss, volatility-adjusted risk percentage, and lot size.
    Uses updated get_pip_point_value. Incorporates volatility regime.
    """
    # (Function body remains the same as v1.2.1 - see previous version)
    print(f"[INFO][AdvSLRiskEngine] Calculating SL & Risk for {symbol} ({trade_type}) Entry @ {entry_price:.5f}")
    output: Dict[str, Any] = { "status": "error", "risk_percent_base": None, "volatility_regime": None, "risk_percent_final": None, "structural_sl": None, "atr_value": None, "atr_sl": None, "final_sl": None, "sl_distance_price": None, "sl_distance_points": None, "point_value": None, "lot_size": None, "risk_amount_usd": None, "log_details": {}, "error": None }
    # --- Parameter Defaults --- (Corrected Syntax)
    if mentfx_sl_config is None: mentfx_sl_config = {}
    if atr_config is None: atr_config = {}
    if risk_config is None: risk_config = {}
    if volatility_config is None: volatility_config = {}
    # --- End Parameter Defaults ---
    atr_period = atr_config.get('period', 14); atr_multiplier = atr_config.get('multiplier', 1.5)
    try:
        # Validate Inputs
        if not isinstance(entry_time, datetime) or entry_time.tzinfo is None:
             if entry_time.tzinfo is None: print("[WARN][AdvSLRiskEngine] entry_time is timezone-naive. Assuming UTC."); entry_time = entry_time.replace(tzinfo=timezone.utc)
             else: raise ValueError("entry_time must be a timezone-aware datetime object.")
        if not isinstance(ohlc_data.index, pd.DatetimeIndex) or ohlc_data.index.tz is None: raise ValueError("ohlc_data must have a timezone-aware DatetimeIndex.")
        if tick_data is not None and (not isinstance(tick_data.index, pd.DatetimeIndex) or tick_data.index.tz is None): raise ValueError("tick_data must have a timezone-aware DatetimeIndex if provided.")

        # 1. Base Risk % from Conviction
        risk_percent_base_decimal = map_conviction_to_risk(conviction_score); output["risk_percent_base"] = round(risk_percent_base_decimal * 100, 2); output["log_details"]["conviction_score"] = conviction_score; output["log_details"]["risk_percent_base"] = output["risk_percent_base"]; print(f"[INFO][AdvSLRiskEngine] Conviction: {conviction_score}/5 => Base Risk: {output['risk_percent_base']}%")

        # 2. Get Volatility Regime
        vol_profile = {"volatility_regime": "Normal", "error": "Skipped"};
        if VOLATILITY_ENGINE_AVAILABLE:
            print(f"[INFO][AdvSLRiskEngine] Getting volatility profile..."); ohlc_data_tz = ohlc_data;
            if ohlc_data.index.tz != entry_time.tzinfo:
                try: ohlc_data_tz = ohlc_data.tz_convert(entry_time.tzinfo)
                except Exception as tz_err: print(f"[WARN][AdvSLRiskEngine] Could not convert OHLC data timezone for Volatility check: {tz_err}.")
            vol_profile = get_volatility_profile(ohlc_data_tz, config=volatility_config); output["volatility_regime"] = vol_profile.get('volatility_regime', 'Normal'); output["log_details"]["volatility_profile"] = vol_profile; print(f"[INFO][AdvSLRiskEngine] Detected Volatility Regime: {output['volatility_regime']}");
            if vol_profile.get("error"): print(f"[WARN][AdvSLRiskEngine] Error getting volatility profile: {vol_profile['error']}")
        else: output["volatility_regime"] = "Normal"; output["log_details"]["volatility_profile_error"] = "Volatility engine not available"; print(f"[WARN][AdvSLRiskEngine] Volatility engine not available. Using '{output['volatility_regime']}' regime.")

        # 3. Adjust Risk % by Volatility
        regime = output["volatility_regime"]; adjusted_risk_percent_decimal = risk_percent_base_decimal; vol_adjust_quiet = risk_config.get('vol_adjustment_quiet', 1.0); vol_adjust_normal = risk_config.get('vol_adjustment_normal', 1.0); vol_adjust_explosive = risk_config.get('vol_adjustment_explosive', 0.75); max_risk_allowed_pct = risk_config.get('max_risk_percent', 1.0); max_risk_allowed_decimal = max_risk_allowed_pct / 100.0; adjustment_factor = vol_adjust_normal;
        if regime == 'Explosive': adjustment_factor = vol_adjust_explosive; print(f"[INFO][AdvSLRiskEngine] Explosive regime detected. Applying risk factor: {adjustment_factor:.2f}")
        elif regime == 'Quiet': adjustment_factor = vol_adjust_quiet; print(f"[INFO][AdvSLRiskEngine] Quiet regime detected. Applying risk factor: {adjustment_factor:.2f}")
        adjusted_risk_percent_decimal *= adjustment_factor; adjusted_risk_percent_decimal = min(adjusted_risk_percent_decimal, max_risk_allowed_decimal); adjusted_risk_percent_decimal = max(0.0001, adjusted_risk_percent_decimal);
        output["risk_percent_final"] = round(adjusted_risk_percent_decimal * 100, 2); output["log_details"]["risk_percent_vol_factor"] = adjustment_factor; output["log_details"]["risk_percent_final"] = output["risk_percent_final"]; print(f"[INFO][AdvSLRiskEngine] Final Risk Percent Used (after vol adjust): {output['risk_percent_final']:.2f}%")

        # 4. Calculate Structural SL
        structural_sl = np.nan;
        if MENTFX_SL_AVAILABLE and tick_data is not None and not tick_data.empty:
            print("[INFO][AdvSLRiskEngine] Calculating Structural SL...");
            try:
                if tick_data.index.tz != entry_time.tzinfo:
                     try: tick_data = tick_data.tz_convert(entry_time.tzinfo)
                     except Exception as tz_err: print(f"[WARN][AdvSLRiskEngine] Could not convert tick data timezone for SL calc: {tz_err}")
                sl_struct_result = compute_mentfx_stop_loss_adaptive(tick_df=tick_data, entry_time=entry_time, entry_price=entry_price, trade_type=trade_type, **mentfx_sl_config); structural_sl = sl_struct_result.get("computed_stop_loss"); output["structural_sl"] = structural_sl if pd.notna(structural_sl) else None; output["log_details"]["structural_sl_raw"] = structural_sl; output["log_details"]["structural_sl_params"] = sl_struct_result.get("params");
                if pd.notna(structural_sl): print(f"[INFO][AdvSLRiskEngine] Structural SL calculated: {structural_sl:.5f}")
                else: print(f"[WARN][AdvSLRiskEngine] Structural SL calculation returned None/NaN. Reason: {sl_struct_result.get('error')}")
            except Exception as struct_e: print(f"[ERROR][AdvSLRiskEngine] Exception during structural SL calculation: {struct_e}"); traceback.print_exc(); output["log_details"]["structural_sl_error"] = str(struct_e)
        else: reason = "Module not available" if not MENTFX_SL_AVAILABLE else "Tick data not provided"; print(f"[WARN][AdvSLRiskEngine] Skipping Structural SL calculation: {reason}."); output["log_details"]["structural_sl_skipped_reason"] = reason

        # 5. Calculate ATR SL
        atr_sl = np.nan; atr_value = np.nan;
        if not ohlc_data.empty and all(c in ohlc_data for c in ['High', 'Low', 'Close']):
            print(f"[INFO][AdvSLRiskEngine] Calculating ATR({atr_period}) SL...");
            try:
                ohlc_data_tz = ohlc_data;
                if ohlc_data.index.tz != entry_time.tzinfo:
                     try: ohlc_data_tz = ohlc_data.tz_convert(entry_time.tzinfo)
                     except Exception as tz_err: print(f"[WARN][AdvSLRiskEngine] Could not convert OHLC data timezone for ATR: {tz_err}")
                atr_series = calculate_atr(ohlc_data_tz['High'], ohlc_data_tz['Low'], ohlc_data_tz['Close'], period=atr_period); atr_value_series = atr_series.loc[:entry_time].dropna();
                if not atr_value_series.empty:
                    atr_value = atr_value_series.iloc[-1]; output["atr_value"] = round(atr_value, 5); atr_buffer = atr_value * atr_multiplier; atr_sl = entry_price - atr_buffer if trade_type == 'buy' else entry_price + atr_buffer; output["atr_sl"] = round(atr_sl, 5); output["log_details"]["atr_period"] = atr_period; output["log_details"]["atr_multiplier"] = atr_multiplier; output["log_details"]["atr_value_raw"] = atr_value; output["log_details"]["atr_sl_raw"] = atr_sl; print(f"[INFO][AdvSLRiskEngine] ATR({atr_period}) = {atr_value:.5f}, Multiplier = {atr_multiplier}, ATR SL = {atr_sl:.5f}")
                else: print("[WARN][AdvSLRiskEngine] Could not get valid ATR value at entry time."); output["log_details"]["atr_error"] = "ATR value unavailable at entry time"
            except Exception as atr_e: print(f"[ERROR][AdvSLRiskEngine] Exception during ATR calculation: {atr_e}"); traceback.print_exc(); output["log_details"]["atr_error"] = str(atr_e)
        else: print("[WARN][AdvSLRiskEngine] Skipping ATR SL calculation: OHLC data missing or incomplete."); output["log_details"]["atr_skipped_reason"] = "OHLC data missing/incomplete"

        # 6. Blend SL
        blend_logic = risk_config.get("sl_blend_logic", "wider"); final_sl = np.nan; sl_choice_reason = "N/A"; valid_structural = pd.notna(structural_sl); valid_atr = pd.notna(atr_sl);
        if valid_structural and valid_atr:
            if blend_logic == "wider":
                 if trade_type == 'buy': final_sl = min(structural_sl, atr_sl); sl_choice_reason = f"Wider (Min of Struct={structural_sl:.5f}, ATR={atr_sl:.5f})"
                 else: final_sl = max(structural_sl, atr_sl); sl_choice_reason = f"Wider (Max of Struct={structural_sl:.5f}, ATR={atr_sl:.5f})"
            elif blend_logic == "structural_priority": final_sl = structural_sl; sl_choice_reason = "Structural Priority"
            elif blend_logic == "atr_priority": final_sl = atr_sl; sl_choice_reason = "ATR Priority"
            else: if trade_type == 'buy': final_sl = min(structural_sl, atr_sl); else: final_sl = max(structural_sl, atr_sl); sl_choice_reason = f"Wider (Unknown Blend Logic: {blend_logic})"
        elif valid_structural: final_sl = structural_sl; sl_choice_reason = "Structural Only (ATR Invalid)"
        elif valid_atr: final_sl = atr_sl; sl_choice_reason = "ATR Only (Structural Invalid)"
        else: output["error"] = "Both Structural and ATR SL calculations failed."; print(f"[ERROR][AdvSLRiskEngine] {output['error']}"); return output
        output["final_sl"] = round(final_sl, 5); output["log_details"]["sl_blend_logic_used"] = blend_logic; output["log_details"]["sl_choice_reason"] = sl_choice_reason; print(f"[INFO][AdvSLRiskEngine] Final Blended SL: {final_sl:.5f} (Reason: {sl_choice_reason})")

        # 7. Calculate SL Distance
        sl_distance_price = abs(entry_price - final_sl); output["sl_distance_price"] = round(sl_distance_price, 5);
        point_info = get_pip_point_value(symbol) # USING UPDATED FUNCTION
        if point_info is None: output["error"] = f"Cannot calculate lot size: Point value unknown for {symbol}."; print(f"[ERROR][AdvSLRiskEngine] {output['error']}"); output["status"] = "partial_success"; return output
        point_value_per_lot, price_decimals = point_info; output["point_value"] = point_value_per_lot; output["log_details"]["price_decimals"] = price_decimals;
        point_increment = 1 / (10**price_decimals) if price_decimals >= 0 else 1; sl_distance_points = sl_distance_price / point_increment if point_increment > 1e-9 else 0; output["sl_distance_points"] = round(sl_distance_points, 2); print(f"[INFO][AdvSLRiskEngine] SL Distance: {sl_distance_price:.{price_decimals}f} (Price) / {sl_distance_points:.2f} (Points)")

        # 8. Calculate Lot Size
        if sl_distance_points <= 1e-9 or point_value_per_lot <= 1e-9: output["error"] = "Invalid SL distance or point value (zero or negative) for lot size calculation."; print(f"[ERROR][AdvSLRiskEngine] {output['error']} (Dist={sl_distance_points}, PointVal={point_value_per_lot})"); output["status"] = "partial_success"; return output
        risk_amount_usd = account_balance * adjusted_risk_percent_decimal; output["risk_amount_usd"] = round(risk_amount_usd, 2);
        denominator = sl_distance_points * point_value_per_lot;
        if abs(denominator) < 1e-9: output["error"] = "Cannot calculate lot size: Zero denominator."; print(f"[ERROR][AdvSLRiskEngine] {output['error']}"); output["status"] = "partial_success"; return output
        raw_lot_size = risk_amount_usd / denominator;
        min_lot_step = risk_config.get('min_lot_step', 0.01);
        if min_lot_step <= 0: print(f"[WARN][AdvSLRiskEngine] Invalid min_lot_step ({min_lot_step}). Defaulting to 0.01."); min_lot_step = 0.01
        # Round UP to nearest step using ceiling division trick, handle potential precision issues
        lot_size = math.ceil(raw_lot_size / min_lot_step - 1e-9) * min_lot_step # Subtract small epsilon before ceiling
        lot_size = max(min_lot_step, lot_size); # Ensure minimum lot size
        max_lot_size = risk_config.get('max_lot_size');
        if max_lot_size and lot_size > max_lot_size: print(f"[WARN][AdvSLRiskEngine] Calculated lot size {lot_size:.2f} exceeds max cap {max_lot_size}. Clamping."); lot_size = max_lot_size; output["log_details"]["lot_size_clamped_max"] = True
        output["lot_size"] = round(lot_size, 2); output["log_details"]["lot_size_raw"] = raw_lot_size; print(f"[INFO][AdvSLRiskEngine] Risk Amount: ${risk_amount_usd:.2f}"); print(f"[INFO][AdvSLRiskEngine] Calculated Lot Size: {lot_size:.2f} (Raw: {raw_lot_size:.4f})")

        # Final Status
        output["status"] = "success"; output["error"] = None

    except Exception as e:
        output["error"] = f"Unexpected error in calculate_sl_and_risk: {e}"; print(f"[ERROR][AdvSLRiskEngine] {output['error']}"); traceback.print_exc(); output["status"] = "error"

    return output


    # --- Example Usage ---
    if __name__ == '__main__':
    # --- Example Usage ---
    if __name__ == '__main__':
    print("\n--- Testing Advanced StopLoss & Risk Engine (Volatility Aware + Patched Pip Map + Syntax Fix) ---")

    # --- Create Dummy Data ---
    ohlc_dates = pd.date_range(start='2024-04-28 10:00', periods=50, freq='15T', tz='UTC')
    ohlc_data = pd.DataFrame({ 'Open': np.linspace(1.1000, 1.1050, 50), 'High': np.linspace(1.1005, 1.1055, 50) + np.random.rand(50)*0.001, 'Low': np.linspace(1.0995, 1.1045, 50) - np.random.rand(50)*0.001, 'Close': np.linspace(1.1003, 1.1053, 50) + np.random.randn(50)*0.0005, 'Volume': np.random.randint(100, 1000, 50).astype(float) }, index=ohlc_dates)
    ohlc_data['High'] = ohlc_data[['Open','Close','High']].max(axis=1); ohlc_data['Low'] = ohlc_data[['Open','Close','Low']].min(axis=1)
    tick_dates = pd.date_range(start=ohlc_dates[-1] - timedelta(minutes=5), periods=60, freq='5S', tz='UTC'); last_close = ohlc_data['Close'].iloc[-1]
    tick_data = pd.DataFrame({'bid': last_close + np.random.randn(60)*0.00005 - 0.00005, 'ask': last_close + np.random.randn(60)*0.00005 + 0.00005,}, index=tick_dates)

    # --- Test Parameters ---
    test_account_balance = 100000; test_conviction = 4; test_entry_price = ohlc_data['Close'].iloc[-1]; test_entry_time = ohlc_data.index[-1]; test_trade_type = 'buy'; test_symbol = 'OANDA:EUR_USD'; test_strategy = 'Inv'
    test_mentfx_config = {'min_buffer_base': 0.00020, 'max_buffer': 0.00150, 'spread_buffer_base': 0.00002, 'use_adaptive_buffer': False}
    test_atr_config = {'period': 14, 'multiplier': 1.5}
    test_volatility_config = {'atr_period': 14, 'bb_period': 20, 'bb_stddev': 2.0, 'atr_ma_period': 10, 'bbw_ma_period': 10}
    test_risk_config = { 'max_risk_percent': 1.0, 'vol_adjustment_quiet': 1.10, 'vol_adjustment_normal': 1.00, 'vol_adjustment_explosive': 0.70, 'min_lot_step': 0.01, 'max_lot_size': 10.0, 'sl_blend_logic': 'wider', 'volatility_config': test_volatility_config }

    print(f"\n--- Running Test Calculation (EURUSD - Normal Vol Expected) ---")
    risk_result_eurusd = calculate_sl_and_risk( account_balance=test_account_balance, conviction_score=test_conviction, entry_price=test_entry_price, entry_time=test_entry_time, trade_type=test_trade_type, symbol=test_symbol, strategy_variant=test_strategy, ohlc_data=ohlc_data, tick_data=tick_data, mentfx_sl_config=test_mentfx_config, atr_config=test_atr_config, risk_config=test_risk_config, volatility_config=test_volatility_config )
    print("\n--- Calculation Result (EURUSD) ---"); print(json.dumps(risk_result_eurusd, indent=2, default=str)); print("-" * 35)

    # --- Test Gold ---
    print(f"\n--- Running Test Calculation (XAUUSD) ---")
    test_symbol_xau = "XAUUSD"; test_entry_price_xau = 2350.50; test_trade_type_xau = 'sell'; test_conviction_xau = 5
    test_mentfx_config_xau = {'min_buffer_base': 0.30, 'max_buffer': 2.50, 'spread_buffer_base': 0.10, 'use_adaptive_buffer': False}
    print("WARN: Using EURUSD OHLC/Tick data for XAUUSD test - ATR/Vol/Struct SL will be inaccurate.")
    risk_result_xau = calculate_sl_and_risk( account_balance=test_account_balance, conviction_score=test_conviction_xau, entry_price=test_entry_price_xau, entry_time=test_entry_time, trade_type=test_trade_type_xau, symbol=test_symbol_xau, strategy_variant=test_strategy, ohlc_data=ohlc_data, tick_data=tick_data, mentfx_sl_config=test_mentfx_config_xau, atr_config=test_atr_config, risk_config=test_risk_config, volatility_config=test_volatility_config )
    print("\n--- Calculation Result (XAUUSD) ---"); print(json.dumps(risk_result_xau, indent=2, default=str)); print("-" * 35)

     # --- Test Index ---
    print(f"\n--- Running Test Calculation (SPX500) ---")
    test_symbol_spx = "SPX500"; test_entry_price_spx = 5100.50; test_trade_type_spx = 'buy'; test_conviction_spx = 3
    test_mentfx_config_spx = {'min_buffer_base': 2.0, 'max_buffer': 15.0, 'spread_buffer_base': 0.5, 'use_adaptive_buffer': False}
    print("WARN: Using EURUSD OHLC/Tick data for SPX500 test - ATR/Vol/Struct SL will be inaccurate.")
    risk_result_spx = calculate_sl_and_risk( account_balance=test_account_balance, conviction_score=test_conviction_spx, entry_price=test_entry_price_spx, entry_time=test_entry_time, trade_type=test_trade_type_spx, symbol=test_symbol_spx, strategy_variant=test_strategy, ohlc_data=ohlc_data, tick_data=tick_data, mentfx_sl_config=test_mentfx_config_spx, atr_config=test_atr_config, risk_config=test_risk_config, volatility_config=test_volatility_config )
    print("\n--- Calculation Result (SPX500) ---"); print(json.dumps(risk_result_spx, indent=2, default=str)); print("-" * 35)

    # --- Test Unknown Symbol ---
    print(f"\n--- Running Test Calculation (UNKNOWN:XYZ) ---")
    test_symbol_unk = "UNKNOWN:XYZ"
    risk_result_unk = calculate_sl_and_risk( account_balance=test_account_balance, conviction_score=3, entry_price=100.0, entry_time=test_entry_time, trade_type='buy', symbol=test_symbol_unk, strategy_variant=test_strategy, ohlc_data=ohlc_data, tick_data=tick_data, mentfx_sl_config={}, atr_config={}, risk_config=test_risk_config, volatility_config=test_volatility_config )
    print("\n--- Calculation Result (UNKNOWN) ---"); print(json.dumps(risk_result_unk, indent=2, default=str)); print("-" * 35)


    print("\n--- Engine Testing Complete ---")

    # --- Example Usage ---
    if __name__ == '__main__':


# --- Example class containing calculate_stop method ---
class ExampleStopLossEngine:
    def calculate_stop(self, entry_price: float) -> float:
        # 1. Build M5 bars and compute ATR
        m5_bars = self._resampler.get_ohlc('M5')  # DataFrame with columns ['high','low','close']
        atr_value = calculate_atr(m5_bars['high'], m5_bars['low'], m5_bars['close'], period=14)
        atr_stop = entry_price - atr_value * self._config.atr_multiplier

        # 2. Fetch last M5 microstructure pivot
        pivot_price = MicrostructureFilter(self._tick_provider).get_last_m5_pivot()
        pivot_stop = pivot_price - self._config.pivot_buffer

        # 3. Use the safer (higher) long stop
        stop_price = max(atr_stop, pivot_stop)
        return stop_price