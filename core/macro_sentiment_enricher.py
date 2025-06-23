import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_ticker_history(ticker: str, period: str = '1d', interval: str = '1h') -> pd.Series:
    """
    Fetches historical close prices for the given ticker using yfinance.
    Args:
        ticker: e.g. '^VIX', 'DX-Y.NYB', '^TNX'
        period: data lookback period (e.g. '5d', '1mo')
        interval: sampling interval (e.g. '1h', '1d')
    Returns:
        pd.Series of closing prices indexed by timestamp.
    """
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    return data['Close']


def compute_macro_signals(config: dict = None) -> dict:
    """
    Pulls key macro indicators (VIX, DXY, US 10Y yield) and computes a simple sentiment bias.
    Args:
        config: optional dict to override tickers or thresholds.
    Returns:
        dict with raw values and interpreted bias.
    """
    # Default tickers
    tickers = {
        'vix': config.get('vix_ticker', '^VIX') if config else '^VIX',
        'dxy': config.get('dxy_ticker', 'DX-Y.NYB') if config else 'DX-Y.NYB',
        '10y': config.get('yield10_ticker', '^TNX') if config else '^TNX'
    }
    # Fetch latest
    now = datetime.utcnow()
    out = {}
    for key, ticker in tickers.items():
        try:
            series = fetch_ticker_history(ticker, period='5d', interval='1h')
            latest = float(series.dropna().iloc[-1])
            prev = float(series.dropna().iloc[-2])
            change = latest - prev
            out[f'{key}_value'] = latest
            out[f'{key}_change'] = change
        except Exception as e:
            out[f'{key}_value'] = None
            out[f'{key}_change'] = None
            out[f'{key}_error'] = str(e)
    # Interpret bias
    bias = 'neutral'
    if out.get('vix_change') and out['vix_change'] > 0:
        bias = 'risk-off'
    if out.get('dxy_change') and out['dxy_change'] > 0 and bias == 'neutral':
        bias = 'dollar-strength'
    if out.get('10y_change') and out['10y_change'] < 0:
        bias = 'yield-dip'
    out['macro_bias'] = bias
    return out


def enrich_with_macro_sentiment(data_context: dict, config: dict = None) -> dict:
    """
    Injects macro sentiment into an existing context dict.
    Returns updated dict.
    """
    signals = compute_macro_signals(config)
    data_context['macro_sentiment'] = signals
    return data_context
