import unittest

import pandas as pd

from backtesting.engine import BacktestEngine


class BuyHoldStrategy:
    """Buy at first bar and sell at last bar."""

    def generate_signals(self, data):
        if data["index"] == 0:
            return [{"action": "buy"}]
        if data["index"] == data["last_index"]:
            return [{"action": "sell"}]
        return []


class NoTradeStrategy:
    def generate_signals(self, data):
        return []


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        prices = [100, 101, 102, 103]
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        self.df = pd.DataFrame({"close": prices}, index=dates)

    def test_basic_backtest(self):
        engine = BacktestEngine([BuyHoldStrategy()], initial_capital=1000)
        metrics = engine.run(self.df)
        self.assertEqual(len(engine.trades), 1)
        self.assertAlmostEqual(engine.trades[0]["pnl"], 30.0, places=2)
        self.assertGreater(metrics["profit_factor"], 1)

    def test_no_trades(self):
        engine = BacktestEngine([NoTradeStrategy()], initial_capital=1000)
        metrics = engine.run(self.df)
        self.assertEqual(len(engine.trades), 0)
        self.assertEqual(metrics["profit_factor"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0)

    def test_constant_prices_sharpe(self):
        df = pd.DataFrame({"close": [100, 100, 100]}, index=pd.date_range("2024", periods=3))
        engine = BacktestEngine([BuyHoldStrategy()], initial_capital=1000)
        metrics = engine.run(df)
        self.assertEqual(metrics["sharpe_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
