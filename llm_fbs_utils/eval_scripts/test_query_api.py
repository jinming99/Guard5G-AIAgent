#!/usr/bin/env python3
"""
Test Query API Functionality
Tests MobiFlow-Auditor REST endpoints
Location: llm_fbs_utils/eval_scripts/test_query_api.py
"""

import unittest
import requests
import json
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_control_api_client import (
    AuditorClient, get_kpm, get_mobiflow, get_stats
)

class TestQueryAPI(unittest.TestCase):
    """Test MobiFlow-Auditor API functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.api_url = "http://localhost:8090"
        cls.client = AuditorClient(cls.api_url)
        cls.test_ue_id = "001010123456789"
        
        # Wait for service
        cls._wait_for_service()
    
    @classmethod
    def _wait_for_service(cls, timeout=30):
        """Wait for API to be ready"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if cls.client.health_check():
                    return
            except:
                pass
            time.sleep(1)
        
        raise Exception("API service not available")
    
    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{self.api_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('service', data)
        self.assertIn('timestamp', data)
    
    def test_get_kpm(self):
        """Test KPM data retrieval"""
        result = get_kpm(self.test_ue_id)
        
        # Check structure
        if 'error' not in result:
            self.assertIn('ue_id', result)
            self.assertIn('timestamp', result)
            self.assertIn('kpm_data', result)
            
            # Check KPM fields
            kpm = result['kpm_data']
            expected_fields = [
                'rsrp', 'rsrq', 'sinr', 'cqi',
                'throughput_dl', 'throughput_ul'
            ]
            
            for field in expected_fields:
                self.assertIn(field, kpm, f"Missing KPM field: {field}")
    
    def test_get_mobiflow_last(self):
        """Test MobiFlow record retrieval"""
        result = self.client.get_mobiflow_last(n=10)
        
        self.assertIsInstance(result, list)
        
        if result:
            # Check first record structure
            record = result[0]
            self.assertIn('timestamp', record)
            self.assertIn('ue_id', record)
            self.assertIn('cell_id', record)
            self.assertIn('event_type', record)
            
            # Check FBS fields
            fbs_fields = [
                'suspected_fbs', 'attach_failures',
                'auth_reject_count', 'cipher_downgrade'
            ]
            
            for field in fbs_fields:
                self.assertIn(field, record, f"Missing FBS field: {field}")
    
    def test_get_ue_mobiflow(self):
        """Test UE-specific MobiFlow retrieval"""
        result = self.client.get_ue_mobiflow(self.test_ue_id)
        
        self.assertIsInstance(result, list)
        
        # All records should be for the requested UE
        for record in result:
            self.assertEqual(record.get('ue_id'), self.test_ue_id)
    
    def test_get_stats(self):
        """Test statistics endpoint"""
        stats = get_stats()
        
        # Check expected fields
        expected_fields = [
            'total_ues', 'active_cells',
            'total_records', 'fbs_detections'
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats, f"Missing stats field: {field}")
        
        # Check types
        self.assertIsInstance(stats['total_ues'], int)
        self.assertIsInstance(stats['active_cells'], (list, int))
        self.assertIsInstance(stats['total_records'], int)
        self.assertIsInstance(stats['fbs_detections'], int)
    
    def test_pagination(self):
        """Test pagination with different n values"""
        for n in [1, 10, 50, 100]:
            result = self.client.get_mobiflow_last(n=n)
            self.assertLessEqual(len(result), n)
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Non-existent UE
        result = get_kpm("nonexistent_ue_999")
        
        if 'error' in result:
            self.assertIn('error', result)
        
        # Invalid parameters
        response = requests.get(f"{self.api_url}/mobiflow/last?n=invalid")
        self.assertIn(response.status_code, [400, 500])
    
    def test_response_time(self):
        """Test API response times"""
        # Measure response times
        times = []
        
        for _ in range(10):
            start = time.time()
            get_stats()
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"Average response time: {avg_time*1000:.2f}ms")
        print(f"Max response time: {max_time*1000:.2f}ms")
        
        # Assert reasonable response times
        self.assertLess(avg_time, 0.5, "Average response time > 500ms")
        self.assertLess(max_time, 1.0, "Max response time > 1s")
    
    def test_concurrent_requests(self):
        """Test concurrent API requests"""
        import concurrent.futures
        
        def make_request():
            return get_stats()
        
        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in futures]
        
        # All should succeed
        for result in results:
            self.assertNotIn('error', result)

class TestDataValidation(unittest.TestCase):
    """Test data validation and integrity"""
    
    def test_timestamp_format(self):
        """Test timestamp formatting"""
        records = get_mobiflow(n=10)
        
        for record in records:
            timestamp = record.get('timestamp')
            self.assertIsNotNone(timestamp)
            
            # Should be Unix timestamp
            self.assertIsInstance(timestamp, (int, float))
            self.assertGreater(timestamp, 1000000000)  # After year 2001
            self.assertLess(timestamp, 2000000000)  # Before year 2033
    
    def test_fbs_indicator_consistency(self):
        """Test FBS indicator consistency"""
        records = get_mobiflow(n=100)
        
        for record in records:
            if record.get('suspected_fbs'):
                # If FBS suspected, should have some indicators
                indicators = [
                    record.get('attach_failures', 0) > 0,
                    record.get('auth_reject_count', 0) > 0,
                    record.get('cipher_downgrade', False),
                    record.get('signal_anomaly', False)
                ]
                
                # At least one indicator should be present
                self.assertTrue(any(indicators),
                    "FBS suspected but no indicators present")
    
    def test_cell_id_format(self):
        """Test cell ID formatting"""
        records = get_mobiflow(n=50)
        
        for record in records:
            cell_id = record.get('cell_id')
            if cell_id:
                # Cell ID should be string or integer
                self.assertIsInstance(cell_id, (str, int))
                
                # If string, should be numeric or hex
                if isinstance(cell_id, str):
                    self.assertTrue(
                        cell_id.isdigit() or 
                        all(c in '0123456789abcdefABCDEF' for c in cell_id),
                        f"Invalid cell ID format: {cell_id}"
                    )

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQueryAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)