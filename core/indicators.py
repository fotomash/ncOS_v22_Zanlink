def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Momentum: current close minus close n periods ago."""
    mom = series.diff(period)
    mom.name = f"MOM_{period}"
    return mom

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100."""
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    wr = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    wr.name = f"WILLR_{period}"
    return wr

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(1), 0.0)
    negative = money_flow.where(typical < typical.shift(1), 0.0)
    mf_pos = positive.rolling(window=period, min_periods=period).sum()
    mf_neg = negative.rolling(window=period, min_periods=period).sum()
    mfi_val = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))
    mfi_val.name = f"MFI_{period}"
    return mfi_val

def volume_oscillator(volume: pd.Series, fast_period: int = 5, slow_period: int = 20) -> pd.Series:
    """Volume Oscillator: (EMA_fast - EMA_slow)/EMA_slow * 100."""
    ema_fast = volume.ewm(span=fast_period, adjust=False).mean()
    ema_slow = volume.ewm(span=slow_period, adjust=False).mean()
    vo = (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100
    vo.name = f"VO_{fast_period}_{slow_period}"
    return vo

def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Donchian Channels: upper = max high, lower = min low, middle = average."""
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({
        f"DC_Upper_{period}": upper,
        f"DC_Middle_{period}": middle,
        f"DC_Lower_{period}": lower
    }, index=high.index)

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner Channels: EMA of close ± multiplier * ATR."""
    ema_close = close.ewm(span=period, adjust=False).mean()
    atr_series = atr(high, low, close, atr_period)
    upper = ema_close + multiplier * atr_series
    lower = ema_close - multiplier * atr_series
    return pd.DataFrame({
        f"KC_Upper_{period}": upper,
        f"KC_Middle_{period}": ema_close,
        f"KC_Lower_{period}": lower
    }, index=close.index)

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Classic Pivot Point: (high+low+close)/3."""
    pp = (high + low + close) / 3.0
    pp.name = "PIVOT_Point"
    return pp

def hull_moving_average(series: pd.Series, period: int = 16) -> pd.Series:
    """Hull Moving Average."""
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma1 = series.rolling(window=half_length, min_periods=half_length).mean()
    wma2 = series.rolling(window=period, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    hma = diff.rolling(window=sqrt_length, min_periods=sqrt_length).mean()
    hma.name = f"HMA_{period}"
    return hma

def relative_vigor_index(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    """Relative Vigor Index."""
    num = (close - open_) + 2*(close.shift(1) - open_.shift(1)) + 2*(close.shift(2) - open_.shift(2)) + (close.shift(3) - open_.shift(3))
    den = (high - low) + 2*(high.shift(1) - low.shift(1)) + 2*(high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))
    rv = num / (den + 1e-10)
    rvi = rv.rolling(window=period, min_periods=period).mean()
    rvi.name = f"RVI_{period}"
    return rvi

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Ichimoku Kinko Hyo lines."""
    conversion = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return pd.DataFrame({
        "Ichimoku_Conversion": conversion,
        "Ichimoku_Base": base,
        "Ichimoku_SpanA": span_a,
        "Ichimoku_SpanB": span_b
    }, index=close.index)
# zanzibar/utils/indicators.py
# Author: Tomasz Laskowski 
# License: Proprietary / Private
# Version: 5.2 (Implemented DSS, OBV, CVD)
# Description: Common financial indicator calculation utilities.

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

# Assuming ZBar is defined and importable for type hints if needed
# from zanzibar.data_management.models import ZBar

# Setup logger for this module
# In a larger application, logging might be configured centrally.
log = logging.getLogger(__name__)
# Basic config if run standalone or not configured upstream
# This prevents 'No handlers could be found for logger "__name__"'
if not log.hasHandlers(): # pragma: no cover
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')


# --- Standard Indicator Implementations (SMA, EMA, RSI, MACD, BB, VWAP - from V3) ---

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    if series.empty or window <= 0:
        log.warning(f"SMA calculation skipped: Empty series or invalid window ({window}).")
        return pd.Series(dtype=float, index=series.index)
    # Ensure window size is not larger than the series length to avoid all NaN result initially
    min_p = max(1, int(window * 0.8)) # Allow calculation with slightly less than full window initially
    log.debug(f"Calculating SMA with window={window}, min_periods={min_p}")
    return series.rolling(window=window, min_periods=min_p).mean()

def ema(series: pd.Series, span: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average"""
    if series.empty or span <= 0:
        log.warning(f"EMA calculation skipped: Empty series or invalid span ({span}).")
        return pd.Series(dtype=float, index=series.index)
    # For consistency with TA-Lib and common usage, adjust=False is often preferred.
    # min_periods=span ensures output only starts when enough data is available.
    # Using max(1, span) for min_periods to handle span=1 case and avoid errors.
    log.debug(f"Calculating EMA with span={span}, adjust={adjust}, min_periods={max(1, span)}")
    return series.ewm(span=span, adjust=adjust, min_periods=max(1, span)).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (using Wilder's smoothing via EMA)."""
    if series.empty or period <= 1:
        log.warning(f"RSI calculation skipped: Empty series or invalid period ({period}).")
        return pd.Series(dtype=float, index=series.index)
    log.debug(f"Calculating RSI with period={period}")
    delta = series.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)

    # Use Wilder's smoothing (equivalent to EMA with alpha = 1 / period)
    # Use com (center of mass) = period - 1 which is equivalent for Wilder's RSI
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10) # prevent div zero
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    # RSI is undefined until the first full period average is calculated
    rsi_val.iloc[:period] = np.nan # Set initial NaNs correctly using iloc
    return rsi_val.rename(f"RSI_{period}")


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD, Signal line, and Histogram"""
    if series.empty or fast <= 0 or slow <= fast or signal <= 0:
        log.warning(f"MACD calculation skipped: Empty series or invalid periods (fast={fast}, slow={slow}, signal={signal}).")
        return pd.DataFrame({'MACD': np.nan, 'Signal': np.nan, 'Histogram': np.nan}, index=series.index)
    log.debug(f"Calculating MACD with fast={fast}, slow={slow}, signal={signal}")

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    # Add suffixes for clarity
    macd_col = f'MACD_{fast}_{slow}'
    signal_col = f'Signal_{signal}'
    hist_col = f'Histogram_{signal}'
    return pd.DataFrame({macd_col: macd_line, signal_col: signal_line, hist_col: histogram})

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Bollinger Bands"""
    if series.empty or window <= 0 or num_std <= 0:
        log.warning(f"Bollinger Bands calculation skipped: Empty series or invalid parameters (window={window}, num_std={num_std}).")
        return pd.DataFrame({f'BB_Upper_{window}_{num_std}': np.nan, f'BB_SMA_{window}': np.nan, f'BB_Lower_{window}_{num_std}': np.nan}, index=series.index)
    log.debug(f"Calculating Bollinger Bands with window={window}, num_std={num_std}")

    sma_ = sma(series, window)
    # Use ddof=0 for population standard deviation if matching trading platforms, pandas default is ddof=1 (sample)
    std = series.rolling(window=window, min_periods=max(1, int(window*0.8))).std(ddof=0)
    upper = sma_ + num_std * std
    lower = sma_ - num_std * std
    # Renamed for clarity and parameter inclusion
    return pd.DataFrame({f'BB_Upper_{window}_{num_std}': upper, f'BB_SMA_{window}': sma_, f'BB_Lower_{window}_{num_std}': lower})

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    Volume Weighted Average Price (Rolling or Cumulative).
    NOTE: True session VWAP requires session start/end detection. This is a simple cumulative version if window is None.
    """
    if high.empty or low.empty or close.empty or volume.empty:
        log.warning("VWAP calculation skipped: Input series are empty.")
        return pd.Series(dtype=float, index=high.index)
    log.debug(f"Calculating VWAP (Window: {'Cumulative' if window is None else window})...")

    typical_price = (high + low + close) / 3.0
    # Ensure volume is numeric and handle potential NaNs
    volume_numeric = pd.to_numeric(volume, errors='coerce').fillna(0)
    tpv = typical_price * volume_numeric

    if window is not None and window > 0:
        # Rolling VWAP
        min_p = max(1, int(window * 0.8))
        cumulative_tpv = tpv.rolling(window=window, min_periods=min_p).sum()
        cumulative_volume = volume_numeric.rolling(window=window, min_periods=min_p).sum()
    else:
        # Simple cumulative - reset daily would be better for true session VWAP
        # This version accumulates indefinitely unless reset externally.
        cumulative_tpv = tpv.cumsum()
        cumulative_volume = volume_numeric.cumsum()

    vwap_val = cumulative_tpv / (cumulative_volume + 1e-10) # Avoid division by zero
    return vwap_val.rename("VWAP" if window is None else f"VWAP_{window}")


# --- MT5-Style ATR, ADX, and Stochastic Oscillator Implementations ---

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period, min_periods=period).mean()
    atr_val.name = f"ATR_{period}"
    return atr_val


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(window=period, min_periods=period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(window=period, min_periods=period).mean() / tr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.rolling(window=period, min_periods=period).mean()
    adx_val.name = f"ADX_{period}"
    return adx_val



def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k_val = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_val = k_val.rolling(window=d_period, min_periods=d_period).mean()
    k_name = f"%K_{k_period}"
    d_name = f"%D_{d_period}"
    return pd.DataFrame({k_name: k_val, d_name: d_val}, index=close.index)


def fractals(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Fractals indicator: bullish fractal (low) and bearish fractal (high).
    A bullish fractal is when low[i] is the lowest among lows[i-2:i+3].
    A bearish fractal is when high[i] is the highest among highs[i-2:i+3].
    """
    length = len(high)
    fractal_high = pd.Series(index=high.index, dtype=float, name="Fractal_High")
    fractal_low  = pd.Series(index=low.index, dtype=float, name="Fractal_Low")
    for i in range(2, length-2):
        window_high = high.iloc[i-2:i+3]
        window_low  = low.iloc[i-2:i+3]
        if high.iloc[i] == window_high.max():
            fractal_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == window_low.min():
            fractal_low.iloc[i] = low.iloc[i]
    return pd.concat([fractal_high, fractal_low], axis=1)


# --- CCI Indicator ---
def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=period, min_periods=period).mean()
    md = tp.rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_val = (tp - ma) / (0.015 * md.replace(0, np.nan))
    cci_val.name = f"CCI_{period}"
    return cci_val


# --- Parabolic SAR Indicator ---

def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Parabolic SAR."""
    length = len(high)
    sar = pd.Series(index=high.index, dtype=float, name="SAR")
    up_trend = True
    ep = low.iloc[0]
    af = step
    sar.iloc[0] = low.iloc[0]
    for i in range(1, length):
        prev_sar = sar.iloc[i-1]
        if up_trend:
            sar_val = prev_sar + af * (ep - prev_sar)
            if low.iloc[i] < sar_val:
                up_trend = False
                sar_val = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar_val = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > sar_val:
                up_trend = True
                sar_val = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
        sar.iloc[i] = sar_val
    return sar

# --- Bill Williams' Awesome Oscillator and Accelerator/Decelerator Oscillator ---
def awesome_oscillator(high: pd.Series, low: pd.Series, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """
    Awesome Oscillator: difference between the simple moving averages
    of (high+low)/2 over fast and slow periods.
    """
    median_price = (high + low) / 2.0
    fast_sma = sma(median_price, fast_period)
    slow_sma = sma(median_price, slow_period)
    ao = fast_sma - slow_sma
    ao.name = f"AO_{fast_period}_{slow_period}"
    return ao

def accelerator_decelerator_oscillator(high: pd.Series, low: pd.Series, ao_series: pd.Series = None) -> pd.Series:
    """
    Accelerator/Decelerator Oscillator: AO difference and its 5-period SMA.
    """
    if ao_series is None:
        ao_series = awesome_oscillator(high, low)
    ac = ao_series - sma(ao_series, 5)
    ac.name = "AC"
    return ac


# --- DSS Bressert Implementation ---
def dss_bressert(high: pd.Series, low: pd.Series, close: pd.Series, config: Dict[str, int]) -> pd.DataFrame:
    """
    Dynamic Smart Smoother (DSS) by Walter Bressert.

    Args:
        high, low, close (pd.Series): Price series.
        config (Dict): Dictionary with keys 'k_period', 'd_period', 'smooth_period'.

    Returns:
        pd.DataFrame: DataFrame with 'DSS_K' (smoothed %K) and 'DSS_D' (%D) columns.
    """
    k_period = config.get('k_period', 13)
    d_period = config.get('d_period', 8)
    smooth_period = config.get('smooth_period', 5)
    log.info(f"Calculating DSS Bressert (k={k_period}, d={d_period}, smooth={smooth_period})...")

    if high.empty or k_period <= 0 or d_period <= 0 or smooth_period <= 0:
        log.warning("DSS calculation skipped: Empty series or invalid periods.")
        return pd.DataFrame({'DSS_K': np.nan, 'DSS_D': np.nan}, index=close.index)

    # 1. Calculate Stochastic %K
    lowest_low_k = low.rolling(window=k_period, min_periods=max(1, k_period)).min()
    highest_high_k = high.rolling(window=k_period, min_periods=max(1, k_period)).max()
    price_range_k = highest_high_k - lowest_low_k
    # Prevent division by zero or NaN if range is zero
    # Use .replace(0, np.nan) before division, then fillna
    stoch_k_raw = (close - lowest_low_k) / (price_range_k.replace(0, np.nan))
    stoch_k = 100 * stoch_k_raw
    stoch_k.fillna(50, inplace=True) # Fill NaNs (initial periods or zero range) with neutral 50
    stoch_k.clip(0, 100, inplace=True) # Ensure %K is within [0, 100] bounds

    # 2. Smooth %K with EMA(smooth_period) -> DSS_K (%K smoothed)
    # Use span = smooth_period for EMA calculation
    # Ensure the span is at least 1
    dss_k_span = max(1, smooth_period)
    # Use the ema function defined above for consistency
    dss_k = ema(stoch_k, span=dss_k_span)

    # 3. Calculate %D as EMA(d_period) of DSS_K -> DSS_D
    dss_d_span = max(1, d_period)
    dss_d = ema(dss_k, span=dss_d_span)

    return pd.DataFrame({'DSS_K': dss_k, 'DSS_D': dss_d}, index=close.index)

# --- OBV Implementation ---
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""
    log.info("Calculating OBV...")
    if close.empty or volume.empty:
        log.warning("OBV calculation skipped: Empty series.")
        return pd.Series(dtype=float, index=close.index)

    # Ensure volume is numeric
    volume_numeric = pd.to_numeric(volume, errors='coerce').fillna(0).clip(lower=0)
    close_diff = close.diff()

    # Assign volume based on price change sign
    # np.sign returns 0 for no change, 1 for positive, -1 for negative
    signed_volume = volume_numeric * np.sign(close_diff)
    # Handle first value (diff is NaN) - OBV typically starts at 0
    # or with the first day's volume if close > previous_close (which is unknown for day 0)
    # A common convention is to set the first signed_volume to 0.
    if len(signed_volume) > 0:
        signed_volume.iloc[0] = 0

    obv_series = signed_volume.cumsum()
    obv_series.name = "OBV"
    return obv_series

# --- CVD Implementation ---
def cvd(delta: Optional[pd.Series]) -> pd.Series:
    """
    Cumulative Volume Delta (CVD).

    Args:
        delta (pd.Series, optional): Series of bar delta values (Ask Vol - Bid Vol).
                                     This series MUST be provided, ideally calculated from ZBars.

    Returns:
        pd.Series: Cumulative Volume Delta, or NaN series if delta input is invalid.
    """
    # Check if delta series is provided and valid
    if delta is None or not isinstance(delta, pd.Series) or delta.empty:
        log.warning("CVD calculation requires a valid bar 'delta' Series. Returning NaN series.")
        # Create a NaN series with the same index as delta if possible
        index_ref = delta.index if isinstance(delta, pd.Series) else None
        # Make sure index is not None before creating Series
        if index_ref is None: # pragma: no cover
             log.error("Cannot determine index for NaN CVD series as input delta was None or not a Series.")
             return pd.Series(dtype=float, name="CVD") # Return empty Series
        return pd.Series(np.nan, index=index_ref, name="CVD")

    log.info("Calculating CVD (Cumulative Volume Delta)...")
    # Ensure delta is numeric, fill potential NaNs with 0 before cumsum
    delta_numeric = pd.to_numeric(delta, errors='coerce').fillna(0)

    cvd_series = delta_numeric.cumsum()
    cvd_series.name = "CVD"
    return cvd_series


# --- Enrichment Function (Updated to call new implementations) ---
def add_indicators_to_df(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Adds a configured set of indicators to the input DataFrame.
    V4: Calls implemented DSS, OBV, CVD based on config.
    """
    if config is None: config = {} # pragma: no cover
    indicator_config = config.get("indicators", {})
    df_out = df.copy()
    log.info("Starting indicator enrichment...")

    required_cols = ['High', 'Low', 'Close', 'Volume'] # Standard internal names
    if not all(col in df_out.columns for col in required_cols):
        log.error(f"Input DataFrame missing required columns for indicators: Need {required_cols}. Available: {df_out.columns.tolist()}")
        return df_out # Return original df if essential columns missing

    # --- Access price/volume series ---
    # Ensure columns are numeric, coercing errors - safer approach
    close = pd.to_numeric(df_out['Close'], errors='coerce')
    volume = pd.to_numeric(df_out['Volume'], errors='coerce')
    high = pd.to_numeric(df_out['High'], errors='coerce')
    low = pd.to_numeric(df_out['Low'], errors='coerce')
    # Attempt to get 'Delta' column if it exists (e.g., pre-calculated from ZBars)
    delta = pd.to_numeric(df_out.get('Delta'), errors='coerce') # Returns None if 'Delta' column doesn't exist or all NaN

    # Drop rows where essential inputs became NaN after coercion
    essential_cols_check = ['High', 'Low', 'Close', 'Volume']
    # Create temporary DF for check to avoid modifying df_out inplace yet
    df_check = pd.DataFrame({'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=df_out.index)
    initial_len = len(df_check)
    df_check.dropna(subset=essential_cols_check, inplace=True)
    if len(df_check) < initial_len:
        log.warning(f"Dropped {initial_len - len(df_check)} rows due to NaN in essential OHLCV columns during enrichment.")
        # Realign all series to the filtered index
        valid_index = df_check.index
        close, volume, high, low = close.loc[valid_index], volume.loc[valid_index], high.loc[valid_index], low.loc[valid_index]
        if delta is not None: delta = delta.loc[valid_index] # Realign delta too if it exists
        df_out = df_out.loc[valid_index].copy() # Keep df_out aligned with valid data
        log.info(f"DataFrame shape after NaN drop: {df_out.shape}")

    if close.empty: # pragma: no cover
        log.error("No valid data remaining after NaN checks in essential columns. Cannot calculate indicators.")
        return df_out


    # --- Calculate configured indicators ---
    # Standard MAs
    if indicator_config.get("sma", {}).get("active", False):
        sma_cfg = indicator_config["sma"]
        window = sma_cfg.get("window", 50)
        if window > 0: df_out[f'SMA_{window}'] = sma(close, window)

    if indicator_config.get("ema_fast", {}).get("active", True): # Default active
        ema_f_cfg = indicator_config.get("ema_fast", {})
        span = ema_f_cfg.get("span", 20)
        if span > 0: df_out[f'EMA_{span}'] = ema(close, span)

    if indicator_config.get("ema_slow", {}).get("active", True): # Default active
        ema_s_cfg = indicator_config.get("ema_slow", {})
        span = ema_s_cfg.get("span", 50)
        if span > 0: df_out[f'EMA_{span}'] = ema(close, span)

    # Oscillators
    if indicator_config.get("rsi", {}).get("active", True):
        rsi_cfg = indicator_config.get("rsi", {})
        period = rsi_cfg.get("period", 14)
        if period > 1: df_out[f'RSI_{period}'] = rsi(close, period)

    if indicator_config.get("macd", {}).get("active", True):
        macd_cfg = indicator_config.get("macd", {})
        macd_df = macd(close, macd_cfg.get("fast", 12), macd_cfg.get("slow", 26), macd_cfg.get("signal", 9))
        df_out = pd.concat([df_out, macd_df], axis=1)

    if indicator_config.get("dss", {}).get("active", True): # Default active=True
        dss_cfg = indicator_config.get("dss", {})
        # Pass only the params needed by dss_bressert
        dss_params = {
            'k_period': dss_cfg.get('k', dss_cfg.get('k_period', 13)), # Allow short names
            'd_period': dss_cfg.get('d', dss_cfg.get('d_period', 8)),
            'smooth_period': dss_cfg.get('smooth', dss_cfg.get('smooth_period', 5))
        }
        dss_df = dss_bressert(high, low, close, config=dss_params)
        df_out = pd.concat([df_out, dss_df], axis=1)

    # Volatility / Bands
    if indicator_config.get("bbands", {}).get("active", True):
        bb_cfg = indicator_config.get("bbands", {})
        window = bb_cfg.get("window", 20)
        num_std = bb_cfg.get("std_dev", 2)
        if window > 0 and num_std > 0:
            bb_df = bollinger_bands(close, window, num_std)
            df_out = pd.concat([df_out, bb_df], axis=1)

    # ATR
    if indicator_config.get("atr", {}).get("active", False):
        atr_cfg = indicator_config.get("atr", {})
        period = atr_cfg.get("period", 14)
        df_out[f'ATR_{period}'] = atr(high, low, close, period)

    # ADX
    if indicator_config.get("adx", {}).get("active", False):
        adx_cfg = indicator_config.get("adx", {})
        period = adx_cfg.get("period", 14)
        df_out[f'ADX_{period}'] = adx(high, low, close, period)


    # Stochastic Oscillator
    if indicator_config.get("stochastic", {}).get("active", False):
        sto_cfg = indicator_config.get("stochastic", {})
        kp = sto_cfg.get("k_period", 14)
        dp = sto_cfg.get("d_period", 3)
        sto_df = stochastic_oscillator(high, low, close, k_period=kp, d_period=dp)
        df_out = pd.concat([df_out, sto_df], axis=1)

    # Fractals
    if indicator_config.get("fractals", {}).get("active", False):
        fractals_df = fractals(high, low)
        df_out = pd.concat([df_out, fractals_df], axis=1)

    # Volume-Based
    # Awesome Oscillator
    if indicator_config.get("awesome", {}).get("active", False):
        ao_cfg = indicator_config.get("awesome", {})
        fast = ao_cfg.get("fast", 5)
        slow = ao_cfg.get("slow", 34)
        df_out[f"AO_{fast}_{slow}"] = awesome_oscillator(high, low, fast_period=fast, slow_period=slow)

    # Accelerator/Decelerator Oscillator
    if indicator_config.get("ac", {}).get("active", False):
        ac_series = accelerator_decelerator_oscillator(high, low, df_out.get(f"AO_{fast}_{slow}"))
        df_out["AC"] = ac_series

    if indicator_config.get("vwap", {}).get("active", True):
        vwap_cfg = indicator_config.get("vwap", {})
        # Ensure window is None or positive int
        vwap_window = vwap_cfg.get("window")
        if vwap_window is not None: vwap_window = int(vwap_window)
        # Ensure window is not zero if specified
        if vwap_window is None or vwap_window > 0:
            vwap_series = vwap(high, low, close, volume, window=vwap_window)
            df_out[vwap_series.name] = vwap_series # Use dynamic name from function

    # CCI
    if indicator_config.get("cci", {}).get("active", False):
        cci_cfg = indicator_config.get("cci", {})
        period = cci_cfg.get("period", 20)
        df_out[f'CCI_{period}'] = cci(high, low, close, period)

    # Parabolic SAR
    if indicator_config.get("sar", {}).get("active", False):
        sar_cfg = indicator_config.get("sar", {})
        step = sar_cfg.get("step", 0.02)
        max_step = sar_cfg.get("max_step", 0.2)
        df_out['SAR'] = parabolic_sar(high, low, step=step, max_step=max_step)

    if indicator_config.get("obv", {}).get("active", True): # Default active=True
        df_out['OBV'] = obv(close, volume)

    if indicator_config.get("cvd", {}).get("active", True): # Default active=True
        # Pass the 'Delta' column if it exists in the input df_out
        delta_series = df_out.get('Delta') # Use get() safely
        df_out['CVD'] = cvd(delta=delta_series) # cvd handles None input gracefully

    log.info("Finished indicator enrichment.")
    return df_out


# --- Multi-Timeframe Indicator Enrichment Utility ---
from typing import Dict
def add_indicators_multi_tf(
    all_tf_data: Dict[str, pd.DataFrame],
    config: Optional[Dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply configured indicators to each timeframe DataFrame in all_tf_data.
    
    Args:
        all_tf_data: dict mapping timeframe names to OHLCV DataFrames.
        config: indicator configuration dict (same as for add_indicators_to_df).
    
    Returns:
        Dict mapping each timeframe to its enriched DataFrame.
    """
    enriched_data: Dict[str, pd.DataFrame] = {}
    for tf, df in all_tf_data.items():
        try:
            enriched_df = add_indicators_to_df(df, config)
            enriched_data[tf] = enriched_df
            log.info(f"[Indicators] Enriched timeframe '{tf}' with indicators.")
        except Exception as e:
            log.error(f"[Indicators] Failed to enrich timeframe '{tf}': {e}")
    return enriched_data

    # Momentum
    if indicator_config.get("mom", {}).get("active", False):
        mom_cfg = indicator_config.get("mom", {})
        per = mom_cfg.get("period", 10)
        df_out[f"MOM_{per}"] = momentum(close, per)

    # Williams %R
    if indicator_config.get("williams_r", {}).get("active", False):
        wr_cfg = indicator_config.get("williams_r", {})
        per = wr_cfg.get("period", 14)
        df_out[f"WILLR_{per}"] = williams_r(high, low, close, per)

    # Money Flow Index
    if indicator_config.get("mfi", {}).get("active", False):
        mfi_cfg = indicator_config.get("mfi", {})
        per = mfi_cfg.get("period", 14)
        df_out[f"MFI_{per}"] = mfi(high, low, close, volume, per)

    # Volume Oscillator
    if indicator_config.get("vo", {}).get("active", False):
        vo_cfg = indicator_config.get("vo", {})
        fast = vo_cfg.get("fast", 5)
        slow = vo_cfg.get("slow", 20)
        df_out[f"VO_{fast}_{slow}"] = volume_oscillator(volume, fast, slow)

    # Donchian Channels
    if indicator_config.get("donchian", {}).get("active", False):
        dc_cfg = indicator_config.get("donchian", {})
        per = dc_cfg.get("period", 20)
        dc_df = donchian_channels(high, low, per)
        df_out = pd.concat([df_out, dc_df], axis=1)

    # Keltner Channels
    if indicator_config.get("keltner", {}).get("active", False):
        kc_cfg = indicator_config.get("keltner", {})
        per = kc_cfg.get("period", 20)
        atrp = kc_cfg.get("atr_period", 10)
        mult = kc_cfg.get("multiplier", 2.0)
        kc_df = keltner_channels(high, low, close, per, atrp, mult)
        df_out = pd.concat([df_out, kc_df], axis=1)

    # Pivot Points
    if indicator_config.get("pivot", {}).get("active", False):
        df_out["PIVOT_Point"] = pivot_points(high, low, close)

    # Hull Moving Average
    if indicator_config.get("hma", {}).get("active", False):
        hma_cfg = indicator_config.get("hma", {})
        per = hma_cfg.get("period", 16)
        df_out[f"HMA_{per}"] = hull_moving_average(close, per)

    # Relative Vigor Index
    if indicator_config.get("rvi", {}).get("active", False):
        rvi_cfg = indicator_config.get("rvi", {})
        per = rvi_cfg.get("period", 10)
        df_out[f"RVI_{per}"] = relative_vigor_index(df_out['Open'], high, low, close, per)

    # Ichimoku
    if indicator_config.get("ichimoku", {}).get("active", False):
        ichi_df = ichimoku(high, low, close)
        df_out = pd.concat([df_out, ichi_df], axis=1)