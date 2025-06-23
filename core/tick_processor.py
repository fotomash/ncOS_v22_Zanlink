# core/tick_processor.py

import pandas as pd

def parse_tick_csv(
    path: str,
    timestamp_col: str = 'Date',
    bid_col: str = 'Bid',
    ask_col: str = 'Ask',
    last_col: str = 'Last',
    vol_col: str = 'Volume',
    sep: str = '\t',
    tz: str = None
) -> pd.DataFrame:
    """
    Load a tick CSV/TSV with columns for bid & ask (and optionally last & volume),
    normalize column names, compute missing fields, and drop bad rows.
    Returns a DataFrame with columns: [Date, Bid, Ask, Last, Volume].
    """
    # 1) Read the file
    df = pd.read_csv(
        path,
        sep=sep,
        parse_dates=[timestamp_col],
        dayfirst=False,  # adjust if your dates are D/M/Y
        infer_datetime_format=True
    )

    # 2) Ensure required columns exist
    # compute mid-price if Last is missing
    if last_col not in df.columns:
        df[last_col] = (df[bid_col] + df[ask_col]) / 2.0

    # assign 1 tick as proxy volume if missing
    if vol_col not in df.columns:
        df[vol_col] = 1

    # 3) Drop any rows lacking bid/ask/last
    df.dropna(subset=[bid_col, ask_col, last_col], inplace=True)

    # 4) (Optional) localize timezone
    if tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(tz)

    # 5) Sort by time and reset index
    df.sort_values(by=timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[[timestamp_col, bid_col, ask_col, last_col, vol_col]]


def ticks_to_bars(
    df_ticks: pd.DataFrame,
    timeframe: str = '1T',
    timestamp_col: str = 'Date',
    last_col: str = 'Last',
    vol_col: str = 'Volume',
    keep_empty_bars: bool = False
) -> pd.DataFrame:
    """
    Convert a ticks DataFrame (with a datetime index or column) into OHLCV bars.
    timeframe examples: '1T' (1min), '5T', '15T', '1H', '1D', etc.
    Returns DataFrame with columns [Open, High, Low, Close, Volume].
    """
    # 1) Ensure the timestamp is the index
    df = df_ticks.set_index(timestamp_col).copy()

    # 2) Resample for price OHLC
    ohlc = df[last_col].resample(timeframe, label='right', closed='right').ohlc()

    # 3) Sum volumes
    vol = df[vol_col].resample(timeframe, label='right', closed='right').sum().rename('Volume')

    # 4) Merge and drop empty bars
    bars = ohlc.join(vol)
    if keep_empty_bars:
        # ensure continuous index and forward-fill prices, zero-fill volume
        bars = bars.asfreq(timeframe)
        for col in ['open','high','low','close']:
            bars[col].ffill(inplace=True)
        bars['Volume'].fillna(0, inplace=True)
    else:
        bars.dropna(subset=['open','high','low','close'], inplace=True)

    # 5) Standardize column names
    bars.columns = ['Open','High','Low','Close','Volume']
    return bars