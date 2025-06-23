import unittest

import numpy as np
import pandas as pd

from engines.confirmation_engine_smc import _find_ltf_swing_points, confirm_smc_entry


class TestFractalDetection(unittest.TestCase):
    def test_fractal_high_low_detection(self):
        series = pd.Series([1, 2, 3, 4, 3, 2, 3, 4, 5, 4])
        swings = _find_ltf_swing_points(series, n=2)
        swings = swings.dropna()
        expected = {0: 1.0, 3: 4.0, 5: 2.0, 8: 5.0, 9: 4.0}
        self.assertEqual(swings.to_dict(), expected)


class TestStructureAssignment(unittest.TestCase):
    def test_structure_after_fractal_break(self):
        htf_poi = {"type": "Bullish", "range": [1.0990, 1.0995]}
        base_time = pd.Timestamp("2024-04-28 14:00:00", tz="UTC")
        periods = 40
        timestamps = pd.date_range(start=base_time, periods=periods, freq="15T")

        data = {
            "Open": np.linspace(1.1005, 1.0998, periods),
            "Close": np.linspace(1.1003, 1.0996, periods),
        }
        data["High"] = np.maximum(data["Open"], data["Close"]) + 0.0003
        data["Low"] = np.minimum(data["Open"], data["Close"]) - 0.0003
        df = pd.DataFrame(data, index=timestamps)

        swing_high_iloc = 20
        break_iloc = 30
        df.iloc[swing_high_iloc, df.columns.get_loc("High")] = 1.1002
        df.iloc[break_iloc, df.columns.get_loc("Close")] = 1.1005
        df.iloc[break_iloc - 1, df.columns.get_loc("Open")] = df.iloc[break_iloc - 1]["Close"] + 0.0001
        df.iloc[break_iloc - 1, df.columns.get_loc("Low")] = df.iloc[break_iloc - 1]["Close"] - 0.0002
        df.iloc[break_iloc - 1, df.columns.get_loc("High")] = df.iloc[break_iloc - 1]["Open"] + 0.00005

        result = confirm_smc_entry(htf_poi, df, "Inv")
        self.assertFalse(result["confirmation_status"])


if __name__ == "__main__":
    unittest.main()
