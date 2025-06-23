import pandas as pd

class Resampler:
    def __init__(self, tick_provider):
        """
        tick_provider: object with method get_m1_bars() returning a DataFrame
                       indexed by 'timestamp' (datetime) and columns
                       ['open','high','low','close','volume'] or tick data.
        """
        self._tp = tick_provider

    def get_ohlc(self, timeframe: str) -> pd.DataFrame:
        """
        Resample raw M1 bars into any higher timeframe.
        Args:
          timeframe: pandas offset alias (e.g. '5T','1H','4H','D','W').
        Returns:
          DataFrame with ['open','high','low','close','volume'] indexed by timestamp.
        """
        # 1. Fetch raw M1 bars
        df = self._tp.get_m1_bars()
        # 2. Ensure datetime index
        df = df.set_index('timestamp').sort_index()
        # 3. Define aggregation rules
        ohlc_dict = {
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum'
        }
        # 4. Resample and drop empty bars
        return df.resample(timeframe).agg(ohlc_dict).dropna()
