#!/usr/bin/env python3
"""
API Client for MobiFlow-Auditor and MobieXpert
Provides Python interface to REST endpoints
Location: llm_fbs_utils/llm_control_api_client.py
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import time

logger = logging.getLogger(__name__)

# Default API endpoints
AUDITOR_API = "http://localhost:8090"
EXPERT_API = "http://localhost:8091"

class APIClient:
    """Base API client with common functionality"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with error handling"""
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {'status': 'success'}
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {url}")
            return {'status': 'error', 'message': 'Request timeout'}
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: {url}")
            return {'status': 'error', 'message': 'Connection failed'}
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return {'status': 'error', 'message': str(e)}
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON response")
            return {'status': 'error', 'message': 'Invalid response'}
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {'status': 'error', 'message': str(e)}
    

    def health_check(self) -> bool:
        """Check if API is accessible"""
        result = self._request('GET', '/health')
        status = str(result.get('status', '')).lower()
        return status in ('healthy', 'ok', 'success')

# ============================================================================
# MobiFlow-Auditor API Client
# ============================================================================

class AuditorClient(APIClient):
    """Client for MobiFlow-Auditor API"""
    
    def get_kpm(self, ue_id: str) -> Dict:
        """Get KPM data for specific UE"""
        return self._request('GET', f'/kpm/{ue_id}')
    
    def get_mobiflow_last(self, n: int = 100) -> List[Dict]:
        """Get last N MobiFlow records"""
        result = self._request('GET', f'/mobiflow/last?n={n}')
        return result.get('records', [])
    
    def get_ue_mobiflow(self, ue_id: str) -> List[Dict]:
        """Get MobiFlow records for specific UE"""
        result = self._request('GET', f'/mobiflow/ue/{ue_id}')
        return result.get('mobiflow_data', [])
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        return self._request('GET', '/stats')

# ============================================================================
# MobieXpert API Client
# ============================================================================

class ExpertClient(APIClient):
    """Client for MobieXpert API"""
    
    def post_rule(self, yaml_content: str) -> Dict:
        """Post new detection rules"""
        return self._request(
            'POST', '/rules',
            data=yaml_content,
            headers={'Content-Type': 'application/yaml'}
        )
    
    def get_rules(self) -> Dict:
        """Get current rules"""
        return self._request('GET', '/rules')
    
    def test_rule(self, rule: Dict, test_data: Dict) -> Dict:
        """Test a rule against sample data"""
        return self._request(
            'POST', '/rules/test',
            json={'rule': rule, 'test_data': test_data}
        )
    
    def reload_rules(self, filename: Optional[str] = None) -> Dict:
        """Reload rules from file"""
        payload = {}
        if filename:
            payload['filename'] = filename
        
        return self._request('POST', '/rules/reload', json=payload)

# ============================================================================
# Global Client Instances
# ============================================================================

# Initialize default clients
auditor = AuditorClient(AUDITOR_API)
expert = ExpertClient(EXPERT_API)

# ============================================================================
# Convenience Functions
# ============================================================================

def get_kpm(ue_id: str) -> Dict:
    """Get KPM data for UE"""
    return auditor.get_kpm(ue_id)

def get_mobiflow(ue_id: Optional[str] = None, n: int = 100) -> Any:
    """Get MobiFlow records"""
    if ue_id:
        return auditor.get_ue_mobiflow(ue_id)
    else:
        return auditor.get_mobiflow_last(n)

def get_stats() -> Dict:
    """Get statistics"""
    return auditor.get_stats()

def post_rule(yaml_content: str) -> Dict:
    """Post detection rule"""
    return expert.post_rule(yaml_content)

def get_rules() -> Dict:
    """Get current rules"""
    return expert.get_rules()

def test_rule(rule: Dict, test_data: Dict) -> Dict:
    """Test a rule"""
    return expert.test_rule(rule, test_data)

def wait_for_services(timeout: int = 60, poll: int = 5) -> bool:
    """Wait for both services to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if auditor.health_check() and expert.health_check():
                logger.info("All services are ready")
                return True
        except Exception as e:
            logger.debug("Service check failed temporarily: %s", e)
        logger.info("Waiting for services...")
        time.sleep(poll)
    logger.error("Services did not become ready within %ss", timeout)
    return False


# ============================================================================
# Advanced Query Functions
# ============================================================================

def get_fbs_suspects(confidence_threshold: float = 0.5) -> List[Dict]:
    """Get UEs suspected of FBS attack"""
    suspects = []
    
    # Get recent MobiFlow records
    records = get_mobiflow(n=200)
    
    # Group by UE
    ue_records = {}
    for record in records:
        ue_id = record.get('ue_id')
        if ue_id:
            if ue_id not in ue_records:
                ue_records[ue_id] = []
            ue_records[ue_id].append(record)
    
    # Analyze each UE
    for ue_id, ue_data in ue_records.items():
        # Count FBS indicators
        indicators = 0
        
        for record in ue_data:
            if record.get('suspected_fbs'):
                indicators += 3
            if record.get('attach_failures', 0) > 2:
                indicators += 2
            if record.get('auth_reject_count', 0) > 0:
                indicators += 2
            if record.get('cipher_downgrade'):
                indicators += 3
        
        # Calculate confidence
        confidence = min(indicators / 10.0, 1.0)
        
        if confidence >= confidence_threshold:
            suspects.append({
                'ue_id': ue_id,
                'confidence': confidence,
                'indicators': indicators,
                'record_count': len(ue_data)
            })
    
    return sorted(suspects, key=lambda x: x['confidence'], reverse=True)

def get_cell_statistics() -> Dict[str, Dict]:
    """Get statistics grouped by cell"""
    records = get_mobiflow(n=500)
    
    cells = {}
    for record in records:
        cell_id = record.get('cell_id')
        if cell_id:
            if cell_id not in cells:
                cells[cell_id] = {
                    'total_records': 0,
                    'fbs_suspects': 0,
                    'attach_failures': 0,
                    'auth_failures': 0,
                    'unique_ues': set()
                }
            
            cells[cell_id]['total_records'] += 1
            cells[cell_id]['unique_ues'].add(record.get('ue_id'))
            
            if record.get('suspected_fbs'):
                cells[cell_id]['fbs_suspects'] += 1
            if record.get('attach_failures', 0) > 0:
                cells[cell_id]['attach_failures'] += 1
            if record.get('auth_reject_count', 0) > 0:
                cells[cell_id]['auth_failures'] += 1
    
    # Convert sets to counts
    for cell_id in cells:
        cells[cell_id]['unique_ues'] = len(cells[cell_id]['unique_ues'])
    
    return cells

def monitor_realtime(callback, interval: int = 5, duration: int = 60):
    """Monitor network in real-time"""
    start_time = time.time()
    last_records = []
    
    while time.time() - start_time < duration:
        # Get latest records
        current_records = get_mobiflow(n=50)
        
        # Find new records
        new_records = []
        for record in current_records:
            if record not in last_records:
                new_records.append(record)
        
        if new_records:
            # Call callback with new records
            callback(new_records)
        
        last_records = current_records
        time.sleep(interval)

# ============================================================================
# Testing Functions
# ============================================================================

def test_connectivity():
    """Test API connectivity"""
    print("Testing API connectivity...")
    
    # Test Auditor
    print("\nMobiFlow-Auditor:")
    if auditor.health_check():
        print("✓ Connected")
        stats = get_stats()
        print(f"  Total UEs: {stats.get('total_ues', 0)}")
        print(f"  Active cells: {stats.get('active_cells', [])}")
    else:
        print("✗ Connection failed")
    
    # Test Expert
    print("\nMobieXpert:")
    if expert.health_check():
        print("✓ Connected")
        rules = get_rules()
        print(f"  Loaded rules: {rules.get('count', 0)}")
    else:
        print("✗ Connection failed")

if __name__ == "__main__":
    # Test connectivity when run directly
    test_connectivity()
    
    # Example: Get FBS suspects
    print("\nChecking for FBS suspects...")
    suspects = get_fbs_suspects()
    if suspects:
        print(f"Found {len(suspects)} suspected UEs:")
        for suspect in suspects:
            print(f"  UE {suspect['ue_id']}: "
                  f"confidence={suspect['confidence']:.2f}")
    else:
        print("No FBS suspects detected")
    
    # Example: Get cell statistics
    print("\nCell statistics:")
    cells = get_cell_statistics()
    for cell_id, stats in cells.items():
        print(f"  Cell {cell_id}: {stats['total_records']} records, "
              f"{stats['unique_ues']} UEs")