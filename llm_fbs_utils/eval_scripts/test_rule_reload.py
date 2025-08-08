#!/usr/bin/env python3
"""
Test Rule Reload Functionality
Tests MobieXpert rule hot-reload capability
Location: llm_fbs_utils/eval_scripts/test_rule_reload.py
"""

import unittest
import yaml
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_control_api_client import (
    ExpertClient, post_rule, get_rules, test_rule
)

class TestRuleReload(unittest.TestCase):
    """Test MobieXpert rule reload functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.api_url = "http://localhost:8091"
        cls.client = ExpertClient(cls.api_url)
        
        # Test rule template
        cls.test_rule = {
            'rules': [
                {
                    'name': 'Test_FBS_Rule',
                    'id': 'test_001',
                    'priority': 5,
                    'condition': {
                        'field': 'attach_failures',
                        'gte': 3
                    },
                    'action': {
                        'type': 'alert',
                        'severity': 'medium',
                        'message': 'Test rule triggered'
                    }
                }
            ]
        }
    
    def test_post_rule(self):
        """Test posting new rules"""
        yaml_content = yaml.dump(self.test_rule)
        result = post_rule(yaml_content)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('rules_count', result)
        self.assertGreater(result['rules_count'], 0)
    
    def test_get_rules(self):
        """Test retrieving current rules"""
        result = get_rules()
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('rules', result)
        self.assertIn('count', result)
        
        # Check if our test rule is loaded
        rules = result['rules']
        rule_names = [r.get('name') for r in rules]
        self.assertIn('Test_FBS_Rule', rule_names)
    
    def test_rule_validation(self):
        """Test rule validation"""
        # Invalid rule (missing required fields)
        invalid_rule = {
            'rules': [
                {
                    'name': 'Invalid_Rule'
                    # Missing condition and action
                }
            ]
        }
        
        yaml_content = yaml.dump(invalid_rule)
        result = post_rule(yaml_content)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
    
    def test_rule_testing(self):
        """Test rule against sample data"""
        # Test data that should trigger the rule
        test_data_match = {
            'ue_id': 'test_ue',
            'attach_failures': 5,
            'timestamp': time.time()
        }
        
        # Test data that should NOT trigger the rule
        test_data_no_match = {
            'ue_id': 'test_ue',
            'attach_failures': 1,
            'timestamp': time.time()
        }
        
        # Test matching case
        result = test_rule(self.test_rule['rules'][0], test_data_match)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['matched'])
        
        # Test non-matching case
        result = test_rule(self.test_rule['rules'][0], test_data_no_match)
        self.assertEqual(result['status'], 'success')
        self.assertFalse(result['matched'])
    
    def test_hot_reload(self):
        """Test hot reload without restart"""
        # Get initial rule count
        initial = get_rules()
        initial_count = initial['count']
        
        # Add new rule
        new_rule = {
            'rules': [
                {
                    'name': f'Hot_Reload_Test_{int(time.time())}',
                    'condition': {'field': 'test', 'eq': True},
                    'action': {'type': 'log'}
                }
            ]
        }
        
        yaml_content = yaml.dump(new_rule)
        result = post_rule(yaml_content)
        
        self.assertEqual(result['status'], 'success')
        
        # Verify rule is loaded immediately
        current = get_rules()
        self.assertGreaterEqual(current['count'], initial_count)
    
    def test_complex_rule(self):
        """Test complex rule with multiple conditions"""
        complex_rule = {
            'rules': [
                {
                    'name': 'Complex_FBS_Detection',
                    'priority': 10,
                    'condition': {
                        'and': [
                            {'field': 'attach_failures', 'gte': 2},
                            {'field': 'auth_reject_count', 'gt': 0},
                            {'or': [
                                {'field': 'cipher_downgrade', 'eq': True},
                                {'field': 'signal_anomaly', 'eq': True}
                            ]}
                        ]
                    },
                    'action': {
                        'type': 'alert',
                        'severity': 'critical',
                        'metadata': {
                            'detection_type': 'complex_pattern'
                        }
                    }
                }
            ]
        }
        
        yaml_content = yaml.dump(complex_rule)
        result = post_rule(yaml_content)
        
        self.assertEqual(result['status'], 'success')
        
        # Test with matching data
        test_data = {
            'attach_failures': 3,
            'auth_reject_count': 1,
            'cipher_downgrade': True,
            'signal_anomaly': False
        }
        
        result = test_rule(complex_rule['rules'][0], test_data)
        self.assertTrue(result['matched'])
    
    def test_rule_persistence(self):
        """Test if rules persist across API calls"""
        # Post a rule with unique ID
        unique_id = f"persist_test_{int(time.time())}"
        rule = {
            'rules': [
                {
                    'name': unique_id,
                    'id': unique_id,
                    'condition': {'field': 'test', 'eq': True},
                    'action': {'type': 'log'}
                }
            ]
        }
        
        yaml_content = yaml.dump(rule)
        post_rule(yaml_content)
        
        # Wait a bit
        time.sleep(1)
        
        # Check if rule still exists
        current = get_rules()
        rule_ids = [r.get('id') for r in current['rules']]
        self.assertIn(unique_id, rule_ids)
    
    def test_concurrent_updates(self):
        """Test concurrent rule updates"""
        import concurrent.futures
        
        def post_test_rule(index):
            rule = {
                'rules': [
                    {
                        'name': f'Concurrent_Test_{index}',
                        'condition': {'field': f'test_{index}', 'eq': True},
                        'action': {'type': 'log'}
                    }
                ]
            }
            yaml_content = yaml.dump(rule)
            return post_rule(yaml_content)
        
        # Post 10 rules concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(post_test_rule, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All should succeed
        for result in results:
            self.assertEqual(result['status'], 'success')

class TestRulePerformance(unittest.TestCase):
    """Test rule engine performance"""
    
    def test_rule_load_time(self):
        """Test time to load new rules"""
        rule = {
            'rules': [
                {
                    'name': 'Performance_Test',
                    'condition': {'field': 'test', 'eq': True},
                    'action': {'type': 'log'}
                }
            ]
        }
        
        yaml_content = yaml.dump(rule)
        
        # Measure load time
        start = time.time()
        result = post_rule(yaml_content)
        elapsed = time.time() - start
        
        self.assertEqual(result['status'], 'success')
        self.assertLess(elapsed, 1.0, f"Rule load took {elapsed:.2f}s (>1s)")
        
        print(f"Rule load time: {elapsed*1000:.2f}ms")
    
    def test_rule_evaluation_speed(self):
        """Test rule evaluation performance"""
        rule = {
            'name': 'Speed_Test',
            'condition': {
                'and': [
                    {'field': 'value1', 'gt': 10},
                    {'field': 'value2', 'lt': 100},
                    {'field': 'value3', 'eq': 'test'}
                ]
            },
            'action': {'type': 'log'}
        }
        
        test_data = {
            'value1': 50,
            'value2': 75,
            'value3': 'test'
        }
        
        # Measure evaluation time (100 iterations)
        start = time.time()
        for _ in range(100):
            test_rule(rule, test_data)
        elapsed = time.time() - start
        
        avg_time = elapsed / 100
        print(f"Average rule evaluation time: {avg_time*1000:.2f}ms")
        
        self.assertLess(avg_time, 0.01, f"Evaluation too slow: {avg_time:.3f}s")

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestRuleReload))
    suite.addTests(loader.loadTestsFromTestCase(TestRulePerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)