"""
LLM Rule Patcher for MobieXpert
Enables hot-reloading of P-BEST rules without restart
Location: MobieXpert/src/pypbest/llm_rule_patch.py
"""

import yaml
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Global reference to P-BEST engine (set by xapp.py)
pbest_engine = None

def set_pbest_engine(engine):
    """Set the P-BEST engine instance"""
    global pbest_engine
    pbest_engine = engine

def patch_rules(yaml_string: str) -> Dict[str, Any]:
    """
    Hot-reload P-BEST rules from YAML string
    
    Args:
        yaml_string: YAML formatted rule definition
    
    Returns:
        Dictionary with status and details
    """
    try:
        # Parse YAML
        new_rules = yaml.safe_load(yaml_string)
        
        if not new_rules:
            return {
                'status': 'error',
                'message': 'Empty or invalid YAML'
            }
        
        # Validate rule structure
        validation = validate_rules(new_rules)
        if not validation['valid']:
            return {
                'status': 'error',
                'message': f"Rule validation failed: {validation['errors']}"
            }
        
        # Generate rule ID if not present
        for rule in new_rules.get('rules', []):
            if 'id' not in rule:
                rule['id'] = generate_rule_id(rule)
            
            # Add metadata
            rule['loaded_at'] = datetime.now().isoformat()
            rule['source'] = 'llm_generated'
        
        # Backup current rules
        backup = backup_current_rules()
        
        # Apply new rules to P-BEST engine
        if pbest_engine:
            result = apply_rules_to_engine(new_rules)
            
            if result['success']:
                # Save rules to file for persistence
                save_rules_to_file(new_rules)
                
                return {
                    'status': 'success',
                    'message': 'Rules successfully loaded',
                    'rules_count': len(new_rules.get('rules', [])),
                    'backup_id': backup['id'],
                    'details': result
                }
            else:
                # Restore from backup on failure
                restore_rules_from_backup(backup)
                return {
                    'status': 'error',
                    'message': f"Failed to apply rules: {result['error']}",
                    'backup_restored': True
                }
        else:
            # Dry-run success when engine not initialized (for unit tests)
            return {
                'status': 'success',
                'message': 'Validation OK (engine not initialized; dry-run)',
                'rules_count': len(new_rules.get('rules', [])),
            }
            
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        return {
            'status': 'error',
            'message': f"YAML parsing error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in patch_rules: {e}")
        return {
            'status': 'error',
            'message': f"Unexpected error: {str(e)}"
        }

def validate_rules(rules: Dict) -> Dict[str, Any]:
    """
    Validate P-BEST rule structure
    
    Returns:
        Dictionary with 'valid' boolean and 'errors' list
    """
    errors = []
    
    # Check for required top-level structure
    if 'rules' not in rules:
        errors.append("Missing 'rules' section")
        return {'valid': False, 'errors': errors}
    
    if not isinstance(rules['rules'], list):
        errors.append("'rules' must be a list")
        return {'valid': False, 'errors': errors}
    
    # Validate each rule
    for idx, rule in enumerate(rules['rules']):
        rule_errors = validate_single_rule(rule, idx)
        errors.extend(rule_errors)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_single_rule(rule: Dict, idx: int) -> List[str]:
    """Validate a single P-BEST rule"""
    errors = []
    prefix = f"Rule {idx}"
    
    # Required fields
    required_fields = ['name', 'condition', 'action']
    for field in required_fields:
        if field not in rule:
            errors.append(f"{prefix}: Missing required field '{field}'")
    
    # Validate condition structure
    if 'condition' in rule:
        if not isinstance(rule['condition'], dict):
            errors.append(f"{prefix}: 'condition' must be a dictionary")
        else:
            # Check for valid condition operators
            valid_operators = ['and', 'or', 'not', 'eq', 'neq', 'gt', 'lt', 
                             'gte', 'lte', 'contains', 'matches', 'in','not_in']
            condition_ops = set(rule['condition'].keys())
            invalid_ops = condition_ops - set(valid_operators + ['field', 'value'])
            
            if invalid_ops:
                errors.append(f"{prefix}: Invalid condition operators: {invalid_ops}")
    
    # Validate action
    if 'action' in rule:
        if not isinstance(rule['action'], dict):
            errors.append(f"{prefix}: 'action' must be a dictionary")
        elif 'type' not in rule['action']:
            errors.append(f"{prefix}: Action missing 'type' field")
    
    return errors

def apply_rules_to_engine(rules: Dict) -> Dict[str, Any]:
    """
    Apply rules to the P-BEST engine
    
    Returns:
        Dictionary with 'success' boolean and details
    """
    try:
        # Clear existing LLM-generated rules (keep system rules)
        if hasattr(pbest_engine, 'clear_llm_rules'):
            pbest_engine.clear_llm_rules()
        
        # Load new rules
        loaded_count = 0
        failed_rules = []
        
        for rule in rules.get('rules', []):
            try:
                if hasattr(pbest_engine, 'add_rule'):
                    pbest_engine.add_rule(rule)
                    loaded_count += 1
                else:
                    # Fallback: directly modify rule list
                    pbest_engine.rules.append(rule)
                    loaded_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to load rule {rule.get('name', 'unknown')}: {e}")
                failed_rules.append({
                    'name': rule.get('name', 'unknown'),
                    'error': str(e)
                })
        
        # Recompile rules if needed
        if hasattr(pbest_engine, 'compile_rules'):
            pbest_engine.compile_rules()
        
        return {
            'success': len(failed_rules) == 0,
            'loaded': loaded_count,
            'failed': failed_rules,
            'error': None if len(failed_rules) == 0 else f"{len(failed_rules)} rules failed"
        }
        
    except Exception as e:
        logger.error(f"Error applying rules to engine: {e}")
        return {
            'success': False,
            'loaded': 0,
            'failed': [],
            'error': str(e)
        }

def generate_rule_id(rule: Dict) -> str:
    """Generate unique ID for a rule"""
    # Create hash from rule content
    rule_str = json.dumps(rule, sort_keys=True)
    hash_obj = hashlib.md5(rule_str.encode())
    return f"llm_{hash_obj.hexdigest()[:8]}_{int(time.time())}"

def backup_current_rules() -> Dict[str, Any]:
    """Backup current rules before applying new ones"""
    try:
        backup_id = f"backup_{int(time.time())}"
        
        if pbest_engine and hasattr(pbest_engine, 'rules'):
            # Create backup
            backup_data = {
                'id': backup_id,
                'timestamp': datetime.now().isoformat(),
                'rules': pbest_engine.rules.copy() if hasattr(pbest_engine.rules, 'copy') else list(pbest_engine.rules)
            }
            
            # Save to file
            backup_path = f"/tmp/pbest_backup_{backup_id}.json"
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Created rule backup: {backup_id}")
            return backup_data
        
        return {'id': None, 'rules': []}
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return {'id': None, 'rules': []}

def restore_rules_from_backup(backup: Dict) -> bool:
    """Restore rules from backup"""
    try:
        if not backup.get('rules'):
            logger.warning("No rules in backup to restore")
            return False
        
        if pbest_engine:
            pbest_engine.rules = backup['rules']
            if hasattr(pbest_engine, 'compile_rules'):
                pbest_engine.compile_rules()
            
            logger.info(f"Restored rules from backup {backup.get('id', 'unknown')}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to restore from backup: {e}")
        return False

def save_rules_to_file(rules: Dict):
    """Save rules to file for persistence"""
    try:
        # Save to experiments directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.environ.get("PBEST_RULES_DIR", "experiments/generated_rules")
        filename = os.path.join(base_dir, f"rules_{timestamp}.yaml")

        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)
        
        # Also save as 'latest'
        latest_file = "experiments/generated_rules/latest.yaml"
        with open(latest_file, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)
        
        logger.info(f"Saved rules to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save rules to file: {e}")

def get_current_rules() -> Dict[str, Any]:
    """Get currently loaded rules"""
    try:
        if pbest_engine and hasattr(pbest_engine, 'rules'):
            return {
                'status': 'success',
                'rules': pbest_engine.rules,
                'count': len(pbest_engine.rules)
            }
        else:
            return {
                'status': 'error',
                'message': 'P-BEST engine not initialized or no rules loaded'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def test_rule(rule: Dict, test_data: Dict) -> Dict[str, Any]:
    """
    Test a single rule against test data
    
    Args:
        rule: P-BEST rule dictionary
        test_data: MobiFlow record or test data
    
    Returns:
        Test result with match status and details
    """
    try:
        # Create temporary engine instance for testing
        from pypbest.PBest import PBestEngine
        
        test_engine = PBestEngine()
        test_engine.add_rule(rule)
        
        # Run test
        result = test_engine.evaluate(test_data)
        
        return {
            'status': 'success',
            'matched': result.get('matched', False),
            'actions_triggered': result.get('actions', []),
            'details': result
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }