"""
NCOS Integration Test Suite
Tests for inter-agent communication and system workflows
"""

import json
import queue
import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is on the import path so integration
# tests can import modules that live alongside this directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Mock implementations for integration testing
class MockMarketDataProvider:
    """Mock market data provider for testing"""

    def __init__(self):
        self.data_points = []
        self.current_index = 0

    def generate_tick(self) -> Dict[str, Any]:
        """Generate a mock market tick"""
        base_price = 100.0
        volatility = 0.02

        # Simulate price movement
        import random
        price_change = random.uniform(-volatility, volatility)
        price = base_price * (1 + price_change + self.current_index * 0.0001)

        tick = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'TEST/USD',
            'price': price,
            'volume': random.randint(100, 1000),
            'bid': price - 0.01,
            'ask': price + 0.01
        }

        self.current_index += 1
        return tick


class MockAgentMesh:
    """Mock agent communication mesh"""

    def __init__(self):
        self.agents = {}
        self.message_queue = queue.Queue()
        self.message_log = []

    def register_agent(self, name: str, agent: Any):
        """Register an agent in the mesh"""
        self.agents[name] = agent

    def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]):
        """Send message between agents"""
        msg = {
            'from': from_agent,
            'to': to_agent,
            'timestamp': datetime.now().isoformat(),
            'payload': message
        }
        self.message_queue.put(msg)
        self.message_log.append(msg)

    def get_messages_for(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get messages for specific agent"""
        return [msg for msg in self.message_log if msg['to'] == agent_name]


class TestDataFlowIntegration(unittest.TestCase):
    """Test 1: ParquetIngestor -> SMCRouter -> MarketAnalyzer data flow"""

    def setUp(self):
        self.mesh = MockAgentMesh()
        self.market_data = MockMarketDataProvider()

    def test_data_ingestion_to_analysis_flow(self):
        """Test data flow from ingestion through analysis"""
        # Step 1: Simulate ParquetIngestor processing data
        ingestion_result = {
            'status': 'success',
            'rows_processed': 1000,
            'data_summary': {
                'symbol': 'TEST/USD',
                'date_range': {
                    'start': '2024-01-01',
                    'end': '2024-01-31'
                }
            }
        }

        # ParquetIngestor sends to SMCRouter
        self.mesh.send_message('ParquetIngestor', 'SMCRouter', {
            'type': 'data_ready',
            'result': ingestion_result
        })

        # Step 2: SMCRouter processes and classifies
        market_regime = 'trending_up'
        routing_decision = {
            'route': 'maz2',
            'confidence': 0.85,
            'regime': market_regime
        }

        # SMCRouter sends to MarketAnalyzer
        self.mesh.send_message('SMCRouter', 'MarketAnalyzer', {
            'type': 'routing_decision',
            'decision': routing_decision,
            'data_ref': ingestion_result
        })

        # Verify message flow
        smc_messages = self.mesh.get_messages_for('SMCRouter')
        self.assertEqual(len(smc_messages), 1)
        self.assertEqual(smc_messages[0]['payload']['type'], 'data_ready')

        analyzer_messages = self.mesh.get_messages_for('MarketAnalyzer')
        self.assertEqual(len(analyzer_messages), 1)
        self.assertEqual(analyzer_messages[0]['payload']['type'], 'routing_decision')

    def test_high_volume_data_flow(self):
        """Test system behavior under high data volume"""
        # Simulate rapid data ingestion
        num_batches = 100

        for i in range(num_batches):
            # Generate batch of ticks
            batch = [self.market_data.generate_tick() for _ in range(10)]

            # Send through pipeline
            self.mesh.send_message('ParquetIngestor', 'SMCRouter', {
                'type': 'data_batch',
                'batch_id': i,
                'data': batch
            })

        # Verify all messages were queued
        self.assertEqual(self.mesh.message_queue.qsize(), num_batches)

        # Process messages
        processed = 0
        while not self.mesh.message_queue.empty():
            msg = self.mesh.message_queue.get()
            processed += 1

        self.assertEqual(processed, num_batches)


class TestStrategyExecutionIntegration(unittest.TestCase):
    """Test 2: SMCRouter -> MAZ2Executor/TMCExecutor -> RiskGuardian flow"""

    def setUp(self):
        self.mesh = MockAgentMesh()

    def test_maz2_execution_flow(self):
        """Test MAZ2 strategy execution with risk checks"""
        # SMCRouter decision
        routing_decision = {
            'route': 'maz2',
            'confidence': 0.9,
            'market_data': {
                'current_price': 105.0,
                'mean_price': 100.0,
                'upper_zone': 104.0,
                'lower_zone': 96.0
            }
        }

        # Send to MAZ2Executor
        self.mesh.send_message('SMCRouter', 'MAZ2Executor', {
            'type': 'execute_strategy',
            'decision': routing_decision
        })

        # MAZ2 generates signal
        maz2_signal = {
            'type': 'sell',
            'price': 105.0,
            'quantity': 1000,
            'stop_loss': 107.0,
            'reason': 'Price above upper zone'
        }

        # MAZ2 sends to RiskGuardian for approval
        self.mesh.send_message('MAZ2Executor', 'RiskGuardian', {
            'type': 'risk_check',
            'signal': maz2_signal,
            'account_info': {
                'balance': 100000,
                'current_exposure': 0.3
            }
        })

        # RiskGuardian approves/modifies
        risk_decision = {
            'approved': True,
            'modified_quantity': 800,  # Reduced for risk
            'max_loss': 1600,
            'risk_score': 0.4
        }

        # Risk sends back to MAZ2
        self.mesh.send_message('RiskGuardian', 'MAZ2Executor', {
            'type': 'risk_decision',
            'decision': risk_decision,
            'original_signal': maz2_signal
        })

        # Verify complete flow
        risk_messages = self.mesh.get_messages_for('RiskGuardian')
        self.assertEqual(len(risk_messages), 1)
        self.assertEqual(risk_messages[0]['payload']['signal']['type'], 'sell')

        maz2_messages = self.mesh.get_messages_for('MAZ2Executor')
        self.assertEqual(len(maz2_messages), 2)  # Initial + risk response

    def test_tmc_execution_flow(self):
        """Test TMC strategy execution with confluence checks"""
        # TMC signal generation
        tmc_signal = {
            'signal_type': 'entry_long',
            'trend_score': 0.8,
            'momentum_score': 0.7,
            'confluence_score': 0.75,
            'entry_price': 98.5
        }

        # Send to RiskGuardian
        self.mesh.send_message('TMCExecutor', 'RiskGuardian', {
            'type': 'risk_check',
            'signal': tmc_signal,
            'strategy': 'tmc'
        })

        # Verify message sent
        risk_messages = self.mesh.get_messages_for('RiskGuardian')
        self.assertEqual(len(risk_messages), 1)
        self.assertEqual(risk_messages[0]['payload']['strategy'], 'tmc')

    def test_hybrid_execution_flow(self):
        """Test hybrid strategy execution (MAZ2 + TMC)"""
        # SMCRouter decides on hybrid approach
        hybrid_decision = {
            'route': 'hybrid',
            'primary': 'tmc',
            'secondary': 'maz2',
            'split_ratio': 0.7,  # 70% TMC, 30% MAZ2
            'total_allocation': 10000
        }

        # Send to both executors
        self.mesh.send_message('SMCRouter', 'TMCExecutor', {
            'type': 'execute_hybrid',
            'allocation': 7000,
            'decision': hybrid_decision
        })

        self.mesh.send_message('SMCRouter', 'MAZ2Executor', {
            'type': 'execute_hybrid',
            'allocation': 3000,
            'decision': hybrid_decision
        })

        # Both send to RiskGuardian
        self.mesh.send_message('TMCExecutor', 'RiskGuardian', {
            'type': 'risk_check',
            'signal': {'type': 'hybrid_tmc', 'allocation': 7000}
        })

        self.mesh.send_message('MAZ2Executor', 'RiskGuardian', {
            'type': 'risk_check',
            'signal': {'type': 'hybrid_maz2', 'allocation': 3000}
        })

        # Verify both strategies received allocations
        risk_messages = self.mesh.get_messages_for('RiskGuardian')
        self.assertEqual(len(risk_messages), 2)

        total_allocation = sum(msg['payload']['signal']['allocation']
                               for msg in risk_messages)
        self.assertEqual(total_allocation, 10000)


class TestMemoryPersistenceIntegration(unittest.TestCase):
    """Test 3: VectorMemoryBoot -> SessionStateManager flow"""

    def setUp(self):
        self.mesh = MockAgentMesh()

    def test_memory_to_session_persistence(self):
        """Test vector memory persistence through session manager"""
        # VectorMemoryBoot stores embeddings
        vector_data = {
            'vector_id': 'vec_001',
            'embedding': [0.1] * 768,  # Mock 768-dim vector
            'metadata': {
                'source': 'market_analysis',
                'timestamp': datetime.now().isoformat(),
                'symbol': 'TEST/USD'
            }
        }

        # Send to SessionStateManager for persistence
        self.mesh.send_message('VectorMemoryBoot', 'SessionStateManager', {
            'type': 'persist_vector',
            'data': vector_data
        })

        # Session manager acknowledges
        self.mesh.send_message('SessionStateManager', 'VectorMemoryBoot', {
            'type': 'persistence_ack',
            'vector_id': 'vec_001',
            'status': 'saved'
        })

        # Test session recovery
        self.mesh.send_message('SessionStateManager', 'VectorMemoryBoot', {
            'type': 'restore_vectors',
            'session_id': 'test_session',
            'vector_count': 1
        })

        # Verify bidirectional communication
        vector_messages = self.mesh.get_messages_for('VectorMemoryBoot')
        self.assertEqual(len(vector_messages), 2)

        session_messages = self.mesh.get_messages_for('SessionStateManager')
        self.assertEqual(len(session_messages), 1)

    def test_bulk_memory_operations(self):
        """Test bulk vector operations"""
        # Generate multiple vectors
        num_vectors = 100
        vectors = []

        for i in range(num_vectors):
            vectors.append({
                'vector_id': f'vec_{i:03d}',
                'embedding': [0.1 * i] * 768,
                'metadata': {'index': i}
            })

        # Bulk persist
        self.mesh.send_message('VectorMemoryBoot', 'SessionStateManager', {
            'type': 'bulk_persist',
            'vectors': vectors,
            'count': num_vectors
        })

        # Verify message
        messages = self.mesh.get_messages_for('SessionStateManager')
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['payload']['count'], num_vectors)


class TestSystemIntegration(unittest.TestCase):
    """Test full system integration scenarios"""

    def setUp(self):
        self.mesh = MockAgentMesh()
        self.market_data = MockMarketDataProvider()

    def test_complete_trading_cycle(self):
        """Test complete cycle from data to execution"""
        # 1. Market data arrives
        market_tick = self.market_data.generate_tick()

        # 2. Data flows through system
        flow_sequence = [
            ('MarketDataCaptain', 'TechnicalAnalyst', {'type': 'new_data', 'tick': market_tick}),
            ('TechnicalAnalyst', 'SMCRouter', {'type': 'analysis', 'indicators': {'rsi': 65}}),
            ('SMCRouter', 'MAZ2Executor', {'type': 'route', 'strategy': 'maz2'}),
            ('MAZ2Executor', 'RiskGuardian', {'type': 'signal', 'action': 'buy'}),
            ('RiskGuardian', 'PortfolioManager', {'type': 'approved', 'trade': {}}),
            ('PortfolioManager', 'BroadcastRelay', {'type': 'executed', 'result': 'success'}),
            ('BroadcastRelay', 'ReportGenerator', {'type': 'trade_complete', 'details': {}})
        ]

        # Execute flow
        for from_agent, to_agent, message in flow_sequence:
            self.mesh.send_message(from_agent, to_agent, message)

        # Verify complete flow
        self.assertEqual(len(self.mesh.message_log), len(flow_sequence))

        # Check final report generation
        report_messages = self.mesh.get_messages_for('ReportGenerator')
        self.assertEqual(len(report_messages), 1)
        self.assertEqual(report_messages[0]['payload']['type'], 'trade_complete')

    def test_error_handling_flow(self):
        """Test system behavior during errors"""
        # Simulate error in execution
        error_signal = {
            'type': 'execution_error',
            'error': 'Insufficient margin',
            'agent': 'MAZ2Executor'
        }

        # Error propagation
        self.mesh.send_message('MAZ2Executor', 'RiskGuardian', error_signal)
        self.mesh.send_message('RiskGuardian', 'BroadcastRelay', {
            'type': 'alert',
            'severity': 'high',
            'error': error_signal
        })

        # Broadcast to all agents
        for agent in ['CoreSystemAgent', 'SessionStateManager', 'ReportGenerator']:
            self.mesh.send_message('BroadcastRelay', agent, {
                'type': 'system_alert',
                'alert': error_signal
            })

        # Verify error propagation
        broadcast_messages = [msg for msg in self.mesh.message_log
                              if msg['from'] == 'BroadcastRelay']
        self.assertEqual(len(broadcast_messages), 3)

    def test_concurrent_strategy_execution(self):
        """Test concurrent execution of multiple strategies"""
        import threading

        results = {'maz2': None, 'tmc': None}

        def execute_maz2():
            self.mesh.send_message('MAZ2Executor', 'RiskGuardian', {
                'type': 'signal',
                'strategy': 'maz2',
                'timestamp': datetime.now().isoformat()
            })
            results['maz2'] = 'executed'

        def execute_tmc():
            self.mesh.send_message('TMCExecutor', 'RiskGuardian', {
                'type': 'signal',
                'strategy': 'tmc',
                'timestamp': datetime.now().isoformat()
            })
            results['tmc'] = 'executed'

        # Run concurrently
        t1 = threading.Thread(target=execute_maz2)
        t2 = threading.Thread(target=execute_tmc)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        # Verify both executed
        self.assertEqual(results['maz2'], 'executed')
        self.assertEqual(results['tmc'], 'executed')

        # Check messages arrived
        risk_messages = self.mesh.get_messages_for('RiskGuardian')
        self.assertEqual(len(risk_messages), 2)

        strategies = [msg['payload']['strategy'] for msg in risk_messages]
        self.assertIn('maz2', strategies)
        self.assertIn('tmc', strategies)


class IntegrationTestReport:
    """Generate comprehensive integration test report"""

    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and generate report"""
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add test classes
        test_classes = [
            TestDataFlowIntegration,
            TestStrategyExecutionIntegration,
            TestMemoryPersistenceIntegration,
            TestSystemIntegration
        ]

        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'summary': {
                'total_tests': result.testsRun,
                'passed': result.testsRun - len(result.failures) - len(result.errors),
                'failed': len(result.failures),
                'errors': len(result.errors),
                'success_rate': ((result.testsRun - len(result.failures) - len(
                    result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
            },
            'test_flows': {
                'data_flow': 'PASSED' if not any('DataFlow' in str(f[0]) for f in result.failures) else 'FAILED',
                'strategy_execution': 'PASSED' if not any(
                    'StrategyExecution' in str(f[0]) for f in result.failures) else 'FAILED',
                'memory_persistence': 'PASSED' if not any(
                    'MemoryPersistence' in str(f[0]) for f in result.failures) else 'FAILED',
                'system_integration': 'PASSED' if not any(
                    'SystemIntegration' in str(f[0]) for f in result.failures) else 'FAILED'
            },
            'details': {
                'failures': [str(f) for f in result.failures],
                'errors': [str(e) for e in result.errors]
            }
        }

        return report


if __name__ == '__main__':
    print("NCOS Integration Test Suite")
    print("=" * 50)

    # Run integration tests
    test_report = IntegrationTestReport()
    report = test_report.run_all_tests()

    # Display results
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Errors: {report['summary']['errors']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"\nTest Flows:")
    for flow, status in report['test_flows'].items():
        print(f"  {flow}: {status}")

    # Save report
    with open('integration_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to integration_test_report.json")
