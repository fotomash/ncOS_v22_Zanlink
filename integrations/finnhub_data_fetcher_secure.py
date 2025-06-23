# Finnhub Data Fetcher - Secure Version
import finnhub
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from secure_config import config
import logging

logger = logging.getLogger(__name__)

class FinnhubDataFetcher:
    """Fetch market data from Finnhub API"""

    def __init__(self):
        # Get API key from environment - NEVER hardcode
        api_key = config.get_api_key("finnhub")
        if not api_key:
            raise ValueError(
                "Finnhub API key not found. "
                "Please set FINNHUB_API_KEY in your .env file"
            )

        self.client = finnhub.Client(api_key=api_key)
        logger.info("Finnhub client initialized")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            return self.client.quote(symbol)
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}

    def get_candles(self, symbol: str, resolution: str, start: int, end: int) -> pd.DataFrame:
        """Get historical candles"""
        try:
            data = self.client.stock_candles(symbol, resolution, start, end)
            if data['s'] == 'ok':
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                return df.set_index('timestamp')
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Get company news"""
        try:
            return self.client.company_news(symbol, from_date, to_date)
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def get_pattern(self, symbol: str, resolution: str) -> Dict[str, Any]:
        """Get technical pattern recognition"""
        try:
            return self.client.pattern_recognition(symbol, resolution)
        except Exception as e:
            logger.error(f"Error fetching patterns for {symbol}: {e}")
            return {}

    def get_crypto_candles(self, symbol: str, resolution: str, start: int, end: int) -> pd.DataFrame:
        """Get crypto candles"""
        try:
            data = self.client.crypto_candles(symbol, resolution, start, end)
            if data['s'] == 'ok':
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                return df.set_index('timestamp')
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching crypto candles for {symbol}: {e}")
            return pd.DataFrame()

# Usage example
if __name__ == "__main__":
    # This will only work if FINNHUB_API_KEY is set in environment
    try:
        fetcher = FinnhubDataFetcher()
        quote = fetcher.get_quote("AAPL")
        print(f"AAPL Quote: {quote}")
    except ValueError as e:
        print(f"Error: {e}")
        print("To use Finnhub, add your API key to .env file:")
        print("FINNHUB_API_KEY=your_api_key_here")
