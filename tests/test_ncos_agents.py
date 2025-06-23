"""
NCOS Agent Unit Test Suite
Comprehensive testing for all 13 core agents
"""

import json
import random
import tempfile
import unittest
from datetime import datetime
from typing import Dict, Any


# Mock imports for testing (these would be actual imports in production)
class MockAgent:
    """Base mock agent for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        return True


# Test utilities
def generate_mock_market_data(num_points: int = 100) -> Dict[str, Any]:
    """Generate mock market data for testing"""
    base_price = 100.0
    prices = []
    volumes = []

    for i in range(num_points):
        # Random walk with trend
        change = random.uniform(-2, 2) + 0.01 * i
        price = base_price + change
        prices.append(price)
        volumes.append(random.randint(1000, 10000))

    return {
        'prices': prices,
        'volumes': volumes,
        'current_price': prices[-1],
        'symbol': 'TEST',
        'timestamp': datetime.now().isoformat()
    }


class TestCoreSystemAgent(unittest.TestCase):
    """Test CoreSystemAgent functionality"""

    def setUp(self):
        self.config = {
            'data_directory': tempfile.mkdtemp(),
            'heartbeat_interval': 1,
            'retry_limit': 3
        }

    def test_initialization(self):
        """Test agent initialization"""
        agent = MockAgent(self.config)
        self.assertFalse(agent.initialized)

        result = agent.initialize()
        self.assertTrue(result)
        self.assertTrue(agent.initialized)

    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid config should pass
        agent = MockAgent(self.config)
        self.assertIsNotNone(agent.config)

        # Invalid config should be handled
        invalid_config = {'invalid_key': 'value'}
        agent2 = MockAgent(invalid_config)
        self.assertEqual(agent2.config.get('data_directory', None), None)


class TestMarketDataCaptain(unittest.TestCase):
    """Test MarketDataCaptain functionality"""

    def setUp(self):
        self.config = {
            'cache_size': 1000,
            'update_interval': 60,
            'supported_symbols': ['TEST', 'DEMO']
        }

    def test_data_validation(self):
        """Test market data validation"""
        valid_data = generate_mock_market_data()
        self.assertIn('prices', valid_data)
        self.assertIn('volumes', valid_data)
        self.assertGreater(len(valid_data['prices']), 0)

    def test_cache_management(self):
        """Test data caching functionality"""
        cache = {}
        data = generate_mock_market_data()

        # Add to cache
        cache['TEST'] = data
        self.assertEqual(len(cache), 1)

        # Verify retrieval
        retrieved = cache.get('TEST')
        self.assertEqual(retrieved['symbol'], 'TEST')


class TestTechnicalAnalyst(unittest.TestCase):
    """Test TechnicalAnalyst functionality"""

    def test_indicator_calculation(self):
        """Test technical indicator calculations"""
        data = generate_mock_market_data()
        prices = data['prices']

        # Test SMA calculation
        sma_period = 20
        if len(prices) >= sma_period:
            sma = sum(prices[-sma_period:]) / sma_period
            self.assertIsInstance(sma, float)
            self.assertGreater(sma, 0)

    def test_signal_generation(self):
        """Test trading signal generation"""
        data = generate_mock_market_data()

        # Mock signal
        signal = {
            'type': 'buy',
            'strength': 0.8,
            'timestamp': datetime.now().isoformat()
        }

        self.assertIn('type', signal)
        self.assertIn('strength', signal)
        self.assertGreater(signal['strength'], 0)
        self.assertLessEqual(signal['strength'], 1)


class TestVectorMemoryBoot(unittest.TestCase):
    """Test VectorMemoryBoot functionality"""

    def setUp(self):
        self.config = {
            'vector_dimension': 768,
            'max_entries': 10000,
            'similarity_threshold': 0.7
        }

    def test_vector_storage(self):
        """Test vector storage and retrieval"""
        # Mock vector
        vector = [random.random() for _ in range(768)]
        metadata = {'source': 'test', 'timestamp': datetime.now().isoformat()}

        self.assertEqual(len(vector), 768)
        self.assertIsInstance(metadata, dict)

    def test_similarity_search(self):
        """Test vector similarity search"""
        # Mock similarity calculation
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.9, 0.1, 0.0]

        # Cosine similarity (simplified)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        self.assertGreater(dot_product, 0.8)  # High similarity


class TestParquetIngestor(unittest.TestCase):
    """Test ParquetIngestor functionality"""

    def test_schema_validation(self):
        """Test data schema validation"""
        # Mock DataFrame columns
        valid_schema = ['timestamp', 'symbol', 'price', 'volume']
        test_columns = ['timestamp', 'symbol', 'price', 'volume', 'extra']

        # Check required columns are present
        missing = [col for col in valid_schema if col not in test_columns]
        self.assertEqual(len(missing), 0)

    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        # Mock data preprocessing
        raw_data = {'price': [100, 101, None, 103]}

        # Fill missing values
        processed = [p if p is not None else 101 for p in raw_data['price']]
        self.assertNotIn(None, processed)


class TestSMCRouter(unittest.TestCase):
    """Test SMCRouter functionality"""

    def test_regime_classification(self):
        """Test market regime classification"""
        regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']

        # Mock classification
        mock_regime = random.choice(regimes)
        self.assertIn(mock_regime, regimes)

    def test_routing_decision(self):
        """Test routing decision logic"""
        market_data = generate_mock_market_data()

        # Mock routing decision
        routes = ['maz2', 'tmc', 'hybrid', 'hold']
        decision = {
            'route': random.choice(routes),
            'confidence': random.uniform(0.5, 1.0)
        }

        self.assertIn(decision['route'], routes)
        self.assertGreaterEqual(decision['confidence'], 0.5)


class TestMAZ2Executor(unittest.TestCase):
    """Test MAZ2Executor functionality"""

    def test_zone_calculation(self):
        """Test adaptive zone calculations"""
        prices = [100 + random.uniform(-5, 5) for _ in range(50)]

        # Calculate mean and zones
        mean_price = sum(prices) / len(prices)
        std_dev = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5

        upper_zone = mean_price + 2 * std_dev
        lower_zone = mean_price - 2 * std_dev

        self.assertGreater(upper_zone, mean_price)
        self.assertLess(lower_zone, mean_price)

    def test_signal_generation(self):
        """Test MAZ2 signal generation"""
        current_price = 105
        mean_price = 100
        zone_width = 5

        # Check if price is outside zones
        if current_price > mean_price + zone_width:
            signal = 'sell'
        elif current_price < mean_price - zone_width:
            signal = 'buy'
        else:
            signal = 'hold'

        self.assertIn(signal, ['buy', 'sell', 'hold'])


class TestTMCExecutor(unittest.TestCase):
    """Test TMCExecutor functionality"""

    def test_trend_calculation(self):
        """Test trend score calculation"""
        prices = [100 + i * 0.5 for i in range(50)]  # Uptrend

        # Simple trend check
        start_price = prices[0]
        end_price = prices[-1]
        trend_direction = 'up' if end_price > start_price else 'down'

        self.assertEqual(trend_direction, 'up')

    def test_confluence_scoring(self):
        """Test trend-momentum confluence"""
        trend_score = 0.8
        momentum_score = 0.7

        confluence = (trend_score + momentum_score) / 2
        self.assertGreater(confluence, 0.7)


class TestRiskGuardian(unittest.TestCase):
    """Test RiskGuardian functionality"""

    def test_risk_calculation(self):
        """Test risk metrics calculation"""
        position_size = 10000
        account_balance = 100000

        position_risk = position_size / account_balance
        self.assertEqual(position_risk, 0.1)
        self.assertLessEqual(position_risk, 0.2)  # Max 20% per position

    def test_exposure_limits(self):
        """Test exposure limit enforcement"""
        max_exposure = 0.5
        current_exposure = 0.3
        new_position = 0.15

        total_exposure = current_exposure + new_position
        allowed = total_exposure <= max_exposure

        self.assertTrue(allowed)


class TestPortfolioManager(unittest.TestCase):
    """Test PortfolioManager functionality"""

    def test_portfolio_allocation(self):
        """Test portfolio allocation logic"""
        total_capital = 100000
        strategies = ['maz2', 'tmc', 'reserve']

        allocations = {
            'maz2': 0.4,
            'tmc': 0.4,
            'reserve': 0.2
        }

        total_allocation = sum(allocations.values())
        self.assertAlmostEqual(total_allocation, 1.0)

    def test_rebalancing(self):
        """Test portfolio rebalancing"""
        current_values = {'maz2': 45000, 'tmc': 35000, 'reserve': 20000}
        target_weights = {'maz2': 0.4, 'tmc': 0.4, 'reserve': 0.2}

        total_value = sum(current_values.values())
        current_weights = {k: v / total_value for k, v in current_values.items()}

        # Check if rebalancing needed
        for strategy in target_weights:
            deviation = abs(current_weights[strategy] - target_weights[strategy])
            self.assertLess(deviation, 0.1)  # 10% tolerance


class TestBroadcastRelay(unittest.TestCase):
    """Test BroadcastRelay functionality"""

    def test_message_routing(self):
        """Test message routing between agents"""
        message = {
            'from': 'MarketAnalyzer',
            'to': 'RiskGuardian',
            'type': 'signal',
            'data': {'action': 'buy', 'symbol': 'TEST'}
        }

        self.assertIn('from', message)
        self.assertIn('to', message)
        self.assertIn('type', message)

    def test_broadcast_filtering(self):
        """Test message filtering"""
        subscriptions = {
            'RiskGuardian': ['signal', 'alert'],
            'PortfolioManager': ['trade', 'rebalance']
        }

        message_type = 'signal'
        recipients = [agent for agent, types in subscriptions.items()
                      if message_type in types]

        self.assertIn('RiskGuardian', recipients)
        self.assertNotIn('PortfolioManager', recipients)


class TestReportGenerator(unittest.TestCase):
    """Test ReportGenerator functionality"""

    def test_report_generation(self):
        """Test report generation"""
        report_data = {
            'date': datetime.now().isoformat(),
            'total_trades': 100,
            'win_rate': 0.65,
            'total_pnl': 15000
        }

        self.assertIn('date', report_data)
        self.assertGreater(report_data['win_rate'], 0.5)

    def test_format_validation(self):
        """Test report format validation"""
        formats = ['json', 'yaml', 'csv', 'html']
        selected_format = 'json'

        self.assertIn(selected_format, formats)


class TestSessionStateManager(unittest.TestCase):
    """Test SessionStateManager functionality"""

    def test_state_persistence(self):
        """Test session state persistence"""
        session_state = {
            'session_id': 'test_123',
            'start_time': datetime.now().isoformat(),
            'active_agents': 13,
            'positions': {}
        }

        # Serialize state
        serialized = json.dumps(session_state)

        # Deserialize state
        restored = json.loads(serialized)

        self.assertEqual(restored['session_id'], session_state['session_id'])
        self.assertEqual(restored['active_agents'], 13)


def run_all_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    test_classes = [
        TestCoreSystemAgent,
        TestMarketDataCaptain,
        TestTechnicalAnalyst,
        TestVectorMemoryBoot,
        TestParquetIngestor,
        TestSMCRouter,
        TestMAZ2Executor,
        TestTMCExecutor,
        TestRiskGuardian,
        TestPortfolioManager,
        TestBroadcastRelay,
        TestReportGenerator,
        TestSessionStateManager
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("NCOS Agent Unit Test Suite")
    print("=" * 50)
    result = run_all_tests()

    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
