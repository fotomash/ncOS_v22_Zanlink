import os
import unittest

import numpy as np
import pandas as pd

from utils.chart_engine import ChartEngine


class TestChartEngine(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        self.df = pd.DataFrame(
            {
                "close": np.linspace(100, 110, 10),
                "volume": np.random.randint(1000, 5000, 10),
            },
            index=dates,
        )
        self.df["sma_20"] = self.df["close"].rolling(window=2).mean()
        self.df["sma_50"] = self.df["close"].rolling(window=3).mean()
        self.engine = ChartEngine()

    def tearDown(self):
        # Remove any generated png files
        for fname in os.listdir('.'):
            if fname.endswith('.png'):
                try:
                    os.remove(fname)
                except OSError:
                    pass

    def test_generate_price_chart(self):
        fname = self.engine.generate_price_chart(self.df, title="Test Chart")
        self.assertTrue(os.path.exists(fname))

    def test_generate_heatmap(self):
        data = np.corrcoef(np.random.rand(5, 5))
        corr_df = pd.DataFrame(data, columns=list('ABCDE'), index=list('ABCDE'))
        fname = self.engine.generate_heatmap(corr_df)
        self.assertTrue(os.path.exists(fname))


if __name__ == "__main__":
    unittest.main()
