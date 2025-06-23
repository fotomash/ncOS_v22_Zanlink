import unittest


class TestCrossDomainRiskAnalyzerIntegration(unittest.TestCase):
    def setUp(self):
        # Clear any existing history by reloading the module
        from importlib import reload
        import ncos_risk_engine as engine
        reload(engine)
        self.engine = engine

    def test_unified_risk_score_and_recommendations(self):
        high_risk = self.engine.RiskFactor(
            domain="deployment",
            factor_name="api_changes",
            severity=0.9,
            impact_radius=["order_execution"],
            mitigation_available=True,
        )
        self.engine.add_cross_domain_risk_factor(high_risk)

        score = self.engine.get_unified_risk_score()
        self.assertIn("overall_risk", score)
        self.assertGreater(score["overall_risk"], 0)
        recs = self.engine.get_mitigation_recommendations()
        self.assertTrue(any(r.get("domain") == "deployment" for r in recs))


if __name__ == "__main__":
    unittest.main()
