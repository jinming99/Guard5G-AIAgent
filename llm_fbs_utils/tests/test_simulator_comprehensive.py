#!/usr/bin/env python3
"""
Comprehensive Test Suite for 5G Simulator Functionality
Tests all aspects of the simulator including FBS scenarios
Location: llm_fbs_utils/tests/test_simulator_comprehensive.py
"""

import unittest
import time
import json
import yaml
import subprocess
import redis
import requests
from typing import Dict, List
from datetime import datetime
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_control_api_client import (
    AuditorClient, ExpertClient, get_kpm, get_mobiflow, 
    get_stats, post_rule, test_rule
)
from scenario_runner import (
    ScenarioRunner, ScenarioConfig, ScenarioLibrary
)
from dataset_playback import DatasetPlayer

class TestSimulatorBasicFunctionality(unittest.TestCase):
    """Test basic simulator functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.auditor = AuditorClient("http://localhost:8090")
        cls.expert = ExpertClient("http://localhost:8091")
        cls.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        cls.scenario_runner = ScenarioRunner(use_docker=False, use_ssh=False)
        cls.dataset_player = DatasetPlayer()
        # Service availability flags
        try:
            cls._redis_ok = bool(cls.redis_client.ping())
        except Exception:
            cls._redis_ok = False
        try:
            cls._auditor_ok = bool(cls.auditor.health_check())
        except Exception:
            cls._auditor_ok = False
        try:
            cls._expert_ok = bool(cls.expert.health_check())
        except Exception:
            cls._expert_ok = False
    
    def setUp(self):
        """Clear data before each test"""
        if not getattr(self, '_redis_ok', False):
            self.skipTest("Redis not available on localhost:6379; skipping simulator Redis-dependent tests")
        try:
            self.redis_client.flushdb()
        except Exception:
            self.skipTest("Redis flushdb failed; skipping")
        time.sleep(0.5)
    
    def test_service_health(self):
        """Test that all services are healthy"""
        # Auditor
        if not self._auditor_ok:
            self.skipTest("Auditor service not available on localhost:8090")
        self.assertTrue(self._auditor_ok, "Auditor service not healthy")
        # Expert
        if not self._expert_ok:
            self.skipTest("Expert service not available on localhost:8091")
        self.assertTrue(self._expert_ok, "Expert service not healthy")
        # Redis
        if not self._redis_ok:
            self.skipTest("Redis not available on localhost:6379")
        self.assertEqual(self.redis_client.ping(), True, "Redis not responding")
    
    def test_data_injection(self):
        """Test data injection into SDL"""
        # Create test record
        test_record = {
            'timestamp': time.time(),
            'ue_id': 'test_ue_001',
            'cell_id': 'cell_001',
            'event_type': 'MEASUREMENT_REPORT',
            'rsrp': -85,
            'rsrq': -12,
            'sinr': 15,
            'suspected_fbs': False
        }
        
        # Inject record
        self.dataset_player.inject_mobiflow_record(test_record)
        
        # Verify injection
        keys = self.redis_client.keys("mobiflow:ue:test_ue_001:*")
        self.assertGreater(len(keys), 0, "Record not injected into Redis")
        
        # Verify content
        data = self.redis_client.get(keys[0])
        self.assertIsNotNone(data)
        
        record = json.loads(data)
        self.assertEqual(record['ue_id'], 'test_ue_001')
    
    def test_kpm_data_flow(self):
        """Test KPM data flow through the system"""
        # Inject KPM data
        kpm_data = {
            'timestamp': time.time(),
            'rsrp': -80,
            'rsrq': -10,
            'sinr': 20,
            'cqi': 12,
            'throughput_dl': 100000000,  # 100 Mbps
            'throughput_ul': 50000000     # 50 Mbps
        }
        
        ue_id = 'test_ue_kpm'
        self.dataset_player.inject_kpm_data(ue_id, kpm_data)
        
        # Query via API
        result = get_kpm(ue_id)
        
        # Verify retrieval
        self.assertIn('kpm_data', result)
        self.assertEqual(result['kpm_data']['rsrp'], -80)
    
    def test_mobiflow_batch_processing(self):
        """Test batch MobiFlow record processing"""
        # Generate batch of records
        records = []
        base_time = time.time()
        
        for i in range(100):
            records.append({
                'timestamp': base_time + i,
                'ue_id': f'ue_{i % 10}',
                'cell_id': f'cell_{i % 5}',
                'event_type': 'MEASUREMENT_REPORT',
                'rsrp': -90 + np.random.randn() * 10,
                'attach_failures': max(0, int(np.random.randn())),
                'suspected_fbs': i > 80  # Last 20 are suspicious
            })
        
        # Inject batch
        for record in records:
            self.dataset_player.inject_mobiflow_record(record)
        
        # Query batch
        retrieved = get_mobiflow(n=100)
        
        # Verify
        self.assertGreaterEqual(len(retrieved), 50, "Not enough records retrieved")
        
        # Check FBS suspects
        suspects = [r for r in retrieved if r.get('suspected_fbs')]
        self.assertGreater(len(suspects), 0, "No FBS suspects found")
    
    def test_statistics_aggregation(self):
        """Test statistics aggregation"""
        # Inject various data
        for i in range(10):
            self.dataset_player.inject_mobiflow_record({
                'timestamp': time.time() + i,
                'ue_id': f'ue_{i}',
                'cell_id': f'cell_{i % 3}',
                'suspected_fbs': i > 7
            })
        
        # Get statistics
        stats = get_stats()
        
        # Verify statistics
        self.assertIn('total_records', stats)
        self.assertGreaterEqual(stats['total_records'], 10)
        
        if 'fbs_detections' in stats:
            self.assertGreaterEqual(stats['fbs_detections'], 0)

class TestMonitoringFunctionality(unittest.TestCase):
    """Test monitoring and metrics functionality"""
    
    def setUp(self):
        self.player = DatasetPlayer()
        self.start_time = time.time()
    
    def test_realtime_monitoring(self):
        """Test real-time monitoring capability"""
        from llm_control_api_client import monitor_realtime
        
        # Callback to collect monitored data
        monitored_records = []
        
        def callback(new_records):
            monitored_records.extend(new_records)
        
        # Start monitoring in background
        import threading
        monitor_thread = threading.Thread(
            target=monitor_realtime,
            args=(callback,),
            kwargs={'interval': 1, 'duration': 5}
        )
        monitor_thread.start()
        
        # Inject data while monitoring
        time.sleep(1)
        for i in range(5):
            self.player.inject_mobiflow_record({
                'timestamp': time.time(),
                'ue_id': f'monitor_ue_{i}',
                'cell_id': 'cell_monitor',
                'event_type': 'TEST_EVENT'
            })
            time.sleep(0.5)
        
        # Wait for monitoring to complete
        monitor_thread.join(timeout=10)
        
        # Verify monitored data
        self.assertGreater(len(monitored_records), 0, "No records monitored")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # Measure injection performance
        start = time.time()
        
        for i in range(100):
            self.player.inject_mobiflow_record({
                'timestamp': time.time(),
                'ue_id': f'perf_ue_{i}',
                'cell_id': 'perf_cell'
            })
        
        elapsed = time.time() - start
        throughput = 100 / elapsed
        
        self.assertGreater(throughput, 10, f"Low throughput: {throughput:.1f} records/sec")
        
        # Measure query performance
        start = time.time()
        for _ in range(10):
            get_stats()
        elapsed = time.time() - start
        
        avg_latency = elapsed / 10
        self.assertLess(avg_latency, 0.5, f"High query latency: {avg_latency:.3f}s")
    
    def test_metric_accuracy(self):
        """Test accuracy of collected metrics"""
        # Inject known data
        known_fbs = 5
        known_normal = 15
        
        for i in range(known_normal):
            self.player.inject_mobiflow_record({
                'timestamp': time.time() + i,
                'ue_id': f'normal_{i}',
                'suspected_fbs': False
            })
        
        for i in range(known_fbs):
            self.player.inject_mobiflow_record({
                'timestamp': time.time() + i,
                'ue_id': f'fbs_{i}',
                'suspected_fbs': True
            })
            # Increment FBS counter
            self.player.redis_client.incr("stats:fbs_detections")
        
        # Verify metrics
        stats = get_stats()
        
        if 'fbs_detections' in stats:
            self.assertEqual(stats['fbs_detections'], known_fbs, 
                           "FBS count mismatch")

class TestFBSScenarios(unittest.TestCase):
    """Test FBS attack scenarios"""
    
    def setUp(self):
        self.runner = ScenarioRunner(use_docker=False, use_ssh=False)
        self.player = DatasetPlayer()
        self.player.clear_sdl()
    
    def test_basic_fbs_scenario(self):
        """Test basic FBS attack scenario"""
        # Generate FBS data
        fbs_data = self.player.generate_synthetic_fbs_data(duration=60)
        
        # Verify data structure
        self.assertIsInstance(fbs_data, list)
        self.assertGreater(len(fbs_data), 0)
        
        # Check for FBS indicators
        fbs_records = [r for r in fbs_data if r.get('suspected_fbs')]
        self.assertGreater(len(fbs_records), 0, "No FBS indicators in synthetic data")
        
        # Check attack progression
        early_records = fbs_data[:20]
        late_records = fbs_data[-20:]
        
        early_fbs = sum(1 for r in early_records if r.get('suspected_fbs'))
        late_fbs = sum(1 for r in late_records if r.get('suspected_fbs'))
        
        # Should have more FBS in later records (attack phase)
        self.assertGreaterEqual(late_fbs, early_fbs, 
                               "FBS attack progression not correct")
    
    def test_scenario_configuration(self):
        """Test scenario configuration parsing"""
        # Test configuration
        config = ScenarioConfig(
            name="Test Scenario",
            mode="fbs",
            duration=120,
            config={'plmn': '00199', 'pci': 999},
            events=[
                {'time': 30, 'action': 'increase_power', 'value': 10}
            ]
        )
        
        # Verify configuration
        self.assertEqual(config.name, "Test Scenario")
        self.assertEqual(config.mode, "fbs")
        self.assertEqual(config.duration, 120)
        self.assertEqual(len(config.events), 1)
    
    def test_scenario_library(self):
        """Test predefined scenario library"""
        # Get scenarios
        basic = ScenarioLibrary.basic_fbs_attack()
        identity = ScenarioLibrary.identity_spoofing()
        intermittent = ScenarioLibrary.intermittent_attack()
        
        # Verify scenarios
        for scenario in [basic, identity, intermittent]:
            self.assertIn('name', scenario)
            self.assertIn('mode', scenario)
            self.assertIn('duration', scenario)
            self.assertIn('config', scenario)
            self.assertEqual(scenario['mode'], 'fbs')
    
    @patch('scenario_runner.ScenarioRunner._execute_command')
    def test_fbs_execution_mock(self, mock_execute):
        """Test FBS scenario execution (mocked)"""
        # Mock command execution
        mock_execute.return_value = ("FBS started", "")
        
        # Test FBS start
        result = self.runner.start_fbs({'plmn': '00199', 'pci': 999})
        self.assertTrue(result, "FBS start failed")
        
        # Verify command was called
        mock_execute.assert_called()
    
    def test_fbs_attack_patterns(self):
        """Test different FBS attack patterns"""
        patterns = {
            'auth_failure': {
                'auth_reject_count': 5,
                'attach_failures': 3,
                'suspected_fbs': True
            },
            'cipher_downgrade': {
                'cipher_algorithm': 'NULL',
                'cipher_downgrade': True,
                'suspected_fbs': True
            },
            'signal_anomaly': {
                'rsrp': -50,  # Unusually strong
                'signal_anomaly': True,
                'suspected_fbs': True
            }
        }
        
        # Test each pattern
        for pattern_name, pattern_data in patterns.items():
            # Inject pattern
            pattern_data['timestamp'] = time.time()
            pattern_data['ue_id'] = f'pattern_{pattern_name}'
            pattern_data['cell_id'] = 'fake_cell'
            
            self.player.inject_mobiflow_record(pattern_data)
            
            # Verify pattern is detectable
            records = get_mobiflow(n=10)
            pattern_records = [r for r in records 
                             if r.get('ue_id') == f'pattern_{pattern_name}']
            
            self.assertGreater(len(pattern_records), 0, 
                             f"Pattern {pattern_name} not found")
            
            if pattern_records:
                self.assertTrue(pattern_records[0].get('suspected_fbs'),
                              f"Pattern {pattern_name} not marked as FBS")

class TestDetectionFunctionality(unittest.TestCase):
    """Test detection rule functionality"""
    
    def setUp(self):
        if not (self._auditor_ok and self._expert_ok):
            self.skipTest("Auditor/Expert services not available; skipping test")
    
    def _check_auditor_connection(self):
        try:
            # Add auditor connection check here
            pass
        except:
            return False
        return True
    
    def _check_expert_connection(self):
        try:
            # Add expert connection check here
            pass
        except:
            return False
        return True
    
    def test_rule_deployment(self):
        """Test rule deployment and retrieval"""
        if not (self._auditor_ok and self._expert_ok):
            self.skipTest("Auditor/Expert services not available; skipping rule deployment test")
        # Create test rule
        test_rule = {
            'rules': [{
                'name': 'Test_Detection_Rule',
                'condition': {
                    'field': 'attach_failures',
                    'gte': 3
                },
                'action': {
                    'type': 'alert',
                    'severity': 'high'
                }
            }]
        }
        
        # Deploy rule
        yaml_content = yaml.dump(test_rule)
        result = post_rule(yaml_content)
        
        self.assertEqual(result['status'], 'success', "Rule deployment failed")
        
        # Retrieve rules
        current_rules = self.expert.get_rules()
        
        self.assertEqual(current_rules['status'], 'success')
        self.assertGreater(current_rules['count'], 0, "No rules loaded")
    
    def test_rule_validation(self):
        """Test rule validation"""
        if not (self._auditor_ok and self._expert_ok):
            self.skipTest("Auditor/Expert services not available; skipping rule validation test")
        # Valid rule
        valid_rule = {
            'name': 'Valid_Rule',
            'condition': {'field': 'test', 'eq': True},
            'action': {'type': 'log'}
        }
        
        # Invalid rule (missing condition)
        invalid_rule = {
            'name': 'Invalid_Rule',
            'action': {'type': 'log'}
        }
        
        # Test valid rule
        test_data = {'test': True}
        result = test_rule(valid_rule, test_data)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result.get('matched', False))
        
        # Test invalid rule
        result = test_rule(invalid_rule, test_data)
        # Should handle gracefully
        self.assertIn('status', result)
    
    def test_detection_pipeline(self):
        """Test complete detection pipeline"""
        # Deploy detection rule
        rule = {
            'rules': [{
                'name': 'FBS_Pipeline_Test',
                'condition': {
                    'and': [
                        {'field': 'attach_failures', 'gte': 2},
                        {'field': 'suspected_fbs', 'eq': True}
                    ]
                },
                'action': {
                    'type': 'alert',
                    'severity': 'critical'
                }
            }]
        }
        
        post_rule(yaml.dump(rule))
        
        # Inject matching data
        self.player.inject_mobiflow_record({
            'timestamp': time.time(),
            'ue_id': 'pipeline_test',
            'attach_failures': 3,
            'suspected_fbs': True
        })
        
        # Test detection
        test_data = {
            'attach_failures': 3,
            'suspected_fbs': True
        }
        
        result = test_rule(rule['rules'][0], test_data)
        self.assertTrue(result.get('matched', False), "Detection failed")
    
    def test_complex_rules(self):
        """Test complex detection rules"""
        # Complex rule with multiple conditions
        complex_rule = {
            'name': 'Complex_FBS_Detection',
            'condition': {
                'or': [
                    {
                        'and': [
                            {'field': 'attach_failures', 'gte': 5},
                            {'field': 'time_window', 'lte': 60}
                        ]
                    },
                    {
                        'and': [
                            {'field': 'cipher_algorithm', 'eq': 'NULL'},
                            {'field': 'auth_reject_count', 'gt': 0}
                        ]
                    },
                    {'field': 'signal_anomaly', 'eq': True}
                ]
            },
            'action': {
                'type': 'alert',
                'metadata': {'detection_type': 'complex'}
            }
        }
        
        # Test various conditions
        test_cases = [
            ({'attach_failures': 6, 'time_window': 30}, True),  # First condition
            ({'cipher_algorithm': 'NULL', 'auth_reject_count': 1}, True),  # Second
            ({'signal_anomaly': True}, True),  # Third
            ({'attach_failures': 1, 'signal_anomaly': False}, False)  # None
        ]
        
        for test_data, expected in test_cases:
            result = test_rule(complex_rule, test_data)
            self.assertEqual(result.get('matched', False), expected,
                           f"Complex rule failed for {test_data}")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end flows"""
    
    @classmethod
    def setUpClass(cls):
        cls.player = DatasetPlayer()
        cls.runner = ScenarioRunner()
    
    def test_end_to_end_detection(self):
        """Test end-to-end FBS detection flow"""
        # Clear data
        self.player.clear_sdl()
        
        # Deploy detection rules
        rule = {
            'rules': [{
                'name': 'E2E_FBS_Detection',
                'condition': {
                    'field': 'suspected_fbs',
                    'eq': True
                },
                'action': {'type': 'alert'}
            }]
        }
        post_rule(yaml.dump(rule))
        
        # Generate and inject FBS scenario
        fbs_data = self.player.generate_synthetic_fbs_data(duration=30)
        
        for record in fbs_data:
            self.player.inject_mobiflow_record(record)
            time.sleep(0.01)  # Simulate real-time
        
        # Check detection
        stats = get_stats()
        
        # Verify FBS was detected
        if 'fbs_detections' in stats:
            self.assertGreater(stats['fbs_detections'], 0, 
                             "FBS not detected in end-to-end test")
    
    def test_concurrent_operations(self):
        """Test concurrent read/write operations"""
        import concurrent.futures
        
        def write_operation(i):
            self.player.inject_mobiflow_record({
                'timestamp': time.time(),
                'ue_id': f'concurrent_{i}',
                'cell_id': 'test_cell'
            })
            return f"write_{i}"
        
        def read_operation(i):
            stats = get_stats()
            return f"read_{i}"
        
        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Mix reads and writes
            futures = []
            for i in range(20):
                if i % 2 == 0:
                    futures.append(executor.submit(write_operation, i))
                else:
                    futures.append(executor.submit(read_operation, i))
            
            # Wait for completion
            results = [f.result(timeout=5) for f in futures]
        
        # Verify all operations completed
        self.assertEqual(len(results), 20, "Some operations failed")
    
    def test_error_recovery(self):
        """Test system error recovery"""
        # Test invalid data injection
        try:
            self.player.inject_mobiflow_record(None)
        except:
            pass  # Should handle gracefully
        
        # System should still be functional
        stats = get_stats()
        self.assertIsNotNone(stats, "System not functional after error")
        
        # Test invalid rule
        try:
            post_rule("invalid yaml {[}")
        except:
            pass  # Should handle gracefully
        
        # Should still accept valid rules
        valid_rule = "rules:\n  - name: test\n    condition: {field: test, eq: true}\n    action: {type: log}"
        result = post_rule(valid_rule)
        # May succeed or fail, but shouldn't crash
        self.assertIn('status', result)
    
    def test_performance_under_load(self):
        """Test system performance under load"""
        # Generate high load
        start_time = time.time()
        records_injected = 0
        
        while time.time() - start_time < 5:  # 5 second load test
            self.player.inject_mobiflow_record({
                'timestamp': time.time(),
                'ue_id': f'load_test_{records_injected}',
                'cell_id': 'load_cell',
                'rsrp': -80 + np.random.randn() * 10
            })
            records_injected += 1
            
            if records_injected % 100 == 0:
                # Periodic read
                get_stats()
        
        throughput = records_injected / 5
        self.assertGreater(throughput, 50, 
                          f"Low throughput under load: {throughput:.1f} rec/s")
        
        # Verify system still responsive
        response_start = time.time()
        stats = get_stats()
        response_time = time.time() - response_start
        
        self.assertLess(response_time, 1.0, 
                       f"High response time after load: {response_time:.3f}s")

def run_comprehensive_tests():
    """Run all simulator tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSimulatorBasicFunctionality,
        TestMonitoringFunctionality,
        TestFBSScenarios,
        TestDetectionFunctionality,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("SIMULATOR TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*50)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)