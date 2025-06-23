import unittest

from agents.smc_master_agent import SMCMasterAgent


class TestSMCMasterAgent(unittest.TestCase):
    def test_basic_trade_decision(self):
        agent = SMCMasterAgent({})
        request = {
            "type": "trade_decision",
            "data": {"symbol": "EURUSD", "bid": 1.1, "ask": 1.1005},
        }
        result = agent.analyze(request)
        self.assertIn("decision", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["decision"], "hold")
        self.assertGreaterEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
