import unittest

from agents.market_maker import MarketMaker


class TestMarketMaker(unittest.TestCase):
    def test_generate_signals_raises_error_when_spread_too_wide(self):
        mm = MarketMaker({})
        spreads = {
            "optimal_spread": 0.003,
            "base_spread": 0.0002,
        }
        microstructure = {"order_flow": 0, "volatility": 0, "liquidity": "high"}
        with self.assertRaises(ValueError):
            mm._generate_signals(spreads, microstructure)


if __name__ == "__main__":
    unittest.main()
