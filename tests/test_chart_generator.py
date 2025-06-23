import os
import unittest

import numpy as np
import pandas as pd

from utils.chart_generator import ChartGenerator


class TestChartGenerator(unittest.TestCase):
    def setUp(self):
        # simple correlation matrix
        data = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.corr_df = pd.DataFrame(data, columns=["A", "B"], index=["A", "B"])
        self.gen = ChartGenerator()

    def tearDown(self):
        # clean up generated png files in charts directory
        charts_dir = os.path.join(os.getcwd(), "charts")
        if os.path.isdir(charts_dir):
            for fname in os.listdir(charts_dir):
                if fname.endswith('.png'):
                    try:
                        os.remove(os.path.join(charts_dir, fname))
                    except OSError:
                        pass

    def test_create_correlation_heatmap(self):
        fname = self.gen.create_correlation_heatmap(self.corr_df)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(fname.endswith('.png'))


if __name__ == "__main__":
    unittest.main()
