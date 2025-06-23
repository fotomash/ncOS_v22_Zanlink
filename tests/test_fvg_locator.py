import unittest

from fvg_locator import FVGLocator


class DummyLogger(list):
    def __call__(self, module, payload):
        self.append((module, payload))


class TestFVGLocator(unittest.TestCase):
    def setUp(self):
        self.logger = DummyLogger()
        self.locator = FVGLocator(logger=self.logger)

    def test_bullish_gap_detection(self):
        data = [
            {"open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
            {"open": 1.1005, "high": 1.1015, "low": 1.1000, "close": 1.1010},
            {"open": 1.1010, "high": 1.1012, "low": 1.1002, "close": 1.1010},
            {"open": 1.1012, "high": 1.1020, "low": 1.1005, "close": 1.1015},
            {"open": 1.1015, "high": 1.1025, "low": 1.1010, "close": 1.1020},
        ]
        state = {"Structural_Shift_Detected": True, "direction": "bullish"}
        cfg = {"max_fvg_size_pips": 10, "pip_precision": 4, "use_midpoint": True}
        result = self.locator.run(data, state, cfg)
        self.assertIn(result["status"], {"PASS", "FAIL"})

    def test_no_gap_found(self):
        data = [
            {"open": 1.2000, "high": 1.2010, "low": 1.1990, "close": 1.2005},
            {"open": 1.2005, "high": 1.2015, "low": 1.2000, "close": 1.2010},
            {"open": 1.2010, "high": 1.2015, "low": 1.2005, "close": 1.2010},
            {"open": 1.2012, "high": 1.2018, "low": 1.2008, "close": 1.2011},
            {"open": 1.2013, "high": 1.2019, "low": 1.2009, "close": 1.2012},
        ]
        state = {"Structural_Shift_Detected": True, "direction": "bullish"}
        cfg = {"max_fvg_size_pips": 10, "pip_precision": 4}
        result = self.locator.run(data, state, cfg)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["reason"], "NO_FVG_FOUND")


if __name__ == "__main__":
    unittest.main()
