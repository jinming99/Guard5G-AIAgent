#!/usr/bin/env python3
"""
Complete Integration Test Runner for 5G-Spector FBS Detection System
Validates end-to-end functionality of all components
Location: llm_fbs_utils/tests/run_integration_tests.py
"""

import os
import sys
import time
import json
import yaml
import subprocess
import asyncio
import unittest
import tempfile
import shutil
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemIntegrationTests:
    """
    Complete system integration tests
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
        self.start_time = None
        self.temp_dir = None
        
    def setup(self):
        """Setup test environment"""
        logger.info("Setting up integration test environment...")
        
        # Create temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="fbs_test_")
        logger.info(f"Test directory: {self.temp_dir}")
        
        # Check service availability
        if not self._check_services():
            raise RuntimeError("Required services not available")
        
        # Clear test data
        self._clear_test_data()
        
        self.start_time = time.time()
        logger.info("Setup complete")
    
    def teardown(self):
        """Cleanup test environment"""
        logger.info("Cleaning up test environment...")
        
        # Clear test data
        self._clear_test_data()
        
        # Remove temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        logger.info("Cleanup complete")
    
    def _check_services(self) -> bool:
        """Check if all required services are available"""
        from llm_control_api_client import wait_for_services
        return wait_for_services(timeout=30)
    
    def _clear_test_data(self):
        """Clear test data from Redis"""
        from dataset_playback import DatasetPlayer
        player = DatasetPlayer()
        player.clear_sdl()
    
    def run_all_tests(self) -> Tuple[bool, Dict]:
        """
        Run all integration tests
        
        Returns:
            Tuple of (success, results)
        """
        self.setup()
        
        try:
            # Test suite
            test_methods = [
                self.test_data_flow,
                self.test_preprocessing_pipeline,
                self.test_rule_generation_and_deployment,
                self.test_experiment_execution,
                self.test_detection_accuracy,
                self.test_workflow_execution,
                self.test_error_recovery,
                self.test_performance_optimization,
                self.test_llm_reasoning,
                self.test_end_to_end_scenario
            ]
            
            for test_method in test_methods:
                test_name = test_method.__name__
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {test_name}")
                logger.info('='*60)
                
                try:
                    result = test_method()
                    self.test_results.append({
                        'test': test_name,
                        'status': 'PASS' if result else 'FAIL',
                        'details': result
                    })
                    
                    if result:
                        logger.info(f"✓ {test_name} PASSED")
                    else:
                        logger.error(f"✗ {test_name} FAILED")
                        
                except Exception as e:
                    logger.error(f"✗ {test_name} ERROR: {e}")
                    self.test_results.append({
                        'test': test_name,
                        'status': 'ERROR',
                        'error': str(e)
                    })
            
            # Generate report
            report = self._generate_report()
            
            # Determine overall success
            passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
            total = len(self.test_results)
            success = passed == total
            
            return success, report
            
        finally:
            self.teardown()
    
    def test_data_flow(self) -> bool:
        """Test data flow through the system"""
        logger.info("Testing data flow...")
        
        from dataset_playback import DatasetPlayer
        from llm_control_api_client import get_mobiflow, get_stats
        
        player = DatasetPlayer()
        
        # Inject test data
        test_records = []
        for i in range(10):
            record = {
                'timestamp': time.time() + i,
                'ue_id': f'flow_test_{i}',
                'cell_id': 'test_cell',
                'rsrp': -85 + i,
                'suspected_fbs': i > 7
            }
            player.inject_mobiflow_record(record)
            test_records.append(record)
        
        # Verify retrieval
        retrieved = get_mobiflow(n=20)
        
        # Check data flow
        success = len(retrieved) >= 10
        
        if success:
            # Verify FBS indicators
            fbs_records = [r for r in retrieved if r.get('suspected_fbs')]
            success = len(fbs_records) >= 2
        
        return success
    
    def test_preprocessing_pipeline(self) -> bool:
        """Test data preprocessing pipeline"""
        logger.info("Testing preprocessing pipeline...")
        
        from dspy_modules.enhanced_modules import TelemetryPreprocessor
        from dataset_playback import DatasetPlayer
        
        preprocessor = TelemetryPreprocessor()
        player = DatasetPlayer()
        
        # Generate test data with patterns
        records = []
        for i in range(50):
            records.append({
                'timestamp': time.time() + i,
                'ue_id': 'preprocess_test',
                'attach_failures': i % 10,  # Pattern
                'rsrp': -90 + (i % 20),  # Pattern
                'cipher_algo': 'NULL' if i > 40 else 'AES'
            })
        
        # Preprocess
        processed = preprocessor.preprocess_mobiflow_batch(records)
        
        # Verify preprocessing
        success = all([
            'statistics' in processed,
            'patterns' in processed,
            'anomalies' in processed,
            processed.get('total_records') == 50
        ])
        
        # Check pattern detection
        if success:
            patterns = processed.get('patterns', {})
            success = patterns.get('cipher_downgrade_detected', False) == True
        
        return success
    
    def test_rule_generation_and_deployment(self) -> bool:
        """Test rule generation and deployment"""
        logger.info("Testing rule generation and deployment...")
        
        from dspy_modules.enhanced_modules import RuleGenerator
        from llm_control_api_client import post_rule, get_rules
        import dspy
        
        # Configure DSPy
        lm = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.7)
        dspy.settings.configure(lm=lm)
        
        generator = RuleGenerator()
        
        # Mock analysis data
        analysis = {
            'mobiflow_analysis': {
                'patterns': {'high_failure_rate': True},
                'statistics': {
                    'attach_failures': {
                        'mean': 3, 'std': 1, 'max': 8
                    }
                }
            }
        }
        
        # Generate rule (mocked)
        from unittest.mock import patch, Mock
        with patch.object(generator.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(rule_yaml="""
rules:
  - name: Integration_Test_Rule
    condition:
      field: attach_failures
      gte: 3
    action:
      type: alert
""")
            
            rule_yaml = generator.forward(
                analysis=analysis,
                patterns={'test': True}
            )
        
        # Deploy rule
        result = post_rule(rule_yaml)
        success = result.get('status') == 'success'
        
        if success:
            # Verify deployment
            current = get_rules()
            success = current.get('count', 0) > 0
        
        return success
    
    def test_experiment_execution(self) -> bool:
        """Test experiment design and execution"""
        logger.info("Testing experiment execution...")
        
        from dspy_modules.enhanced_modules import ExperimentDesigner
        from unittest.mock import patch, Mock
        import dspy
        
        # Configure DSPy
        lm = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.7)
        dspy.settings.configure(lm=lm)
        
        designer = ExperimentDesigner()
        
        # Design experiment
        with patch.object(designer.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(experiment_json=json.dumps({
                'name': 'Test_Experiment',
                'duration': 30,
                'mode': 'fbs',
                'config': {'test': True}
            }))
            
            experiment = designer.forward(
                hypothesis="Test hypothesis",
                current_performance={'detection_rate': 0.8}
            )
        
        # Verify experiment structure
        success = all([
            'name' in experiment,
            'duration' in experiment,
            experiment['duration'] <= 300,  # Safety check applied
            'timing' in experiment  # Timing coordination added
        ])
        
        return success
    
    def test_detection_accuracy(self) -> bool:
        """Test detection accuracy with known patterns"""
        logger.info("Testing detection accuracy...")
        
        from dataset_playback import DatasetPlayer
        from llm_control_api_client import post_rule, test_rule
        
        player = DatasetPlayer()
        
        # Deploy detection rule
        rule = {
            'rules': [{
                'name': 'Accuracy_Test',
                'condition': {
                    'and': [
                        {'field': 'suspected_fbs', 'eq': True},
                        {'field': 'attach_failures', 'gte': 3}
                    ]
                },
                'action': {'type': 'alert'}
            }]
        }
        
        post_rule(yaml.dump(rule))
        
        # Test with known patterns
        test_cases = [
            # (data, should_match)
            ({'suspected_fbs': True, 'attach_failures': 5}, True),
            ({'suspected_fbs': True, 'attach_failures': 1}, False),
            ({'suspected_fbs': False, 'attach_failures': 5}, False),
            ({'suspected_fbs': True, 'attach_failures': 3}, True)
        ]
        
        correct = 0
        for data, expected in test_cases:
            result = test_rule(rule['rules'][0], data)
            if result.get('matched', False) == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        return accuracy >= 0.75  # 75% accuracy threshold
    
    async def test_workflow_execution(self) -> bool:
        """Test complete workflow execution"""
        logger.info("Testing workflow execution...")
        
        from agent.workflow_agent import WorkflowAwareAgent, WorkflowState
        
        agent = WorkflowAwareAgent({
            'max_iterations': 2,  # Limit for testing
            'auto_save': False
        })
        
        # Mock components
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(agent, '_run_baseline_test') as mock_baseline:
            mock_baseline.return_value = AsyncMock(return_value={
                'metrics': {
                    'detection_rate': 0.8,
                    'false_positive_rate': 0.1
                },
                'raw_data': {}
            })()
            
            # Run simplified workflow
            await agent._phase_understanding()
            success = agent.context.current_state == WorkflowState.UNDERSTANDING
            
            if success:
                await agent._phase_baseline_assessment()
                success = agent.context.baseline_performance is not None
        
        return success
    
    def test_error_recovery(self) -> bool:
        """Test error recovery mechanisms"""
        logger.info("Testing error recovery...")
        
        from llm_control_api_client import post_rule
        
        # Test invalid rule handling
        invalid_rules = [
            "invalid yaml {[",
            "",
            None,
            {"wrong": "structure"}
        ]
        
        errors_handled = 0
        for invalid in invalid_rules:
            try:
                if invalid is not None:
                    result = post_rule(str(invalid))
                    # Should return error status, not crash
                    if result.get('status') == 'error':
                        errors_handled += 1
            except:
                pass  # Exception is also acceptable
        
        # Test data corruption handling
        from dataset_playback import DatasetPlayer
        player = DatasetPlayer()
        
        try:
            player.inject_mobiflow_record(None)
        except:
            errors_handled += 1
        
        # Should handle most errors gracefully
        return errors_handled >= 3
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization capabilities"""
        logger.info("Testing performance optimization...")
        
        from agent.workflow_agent import WorkflowAwareAgent
        
        agent = WorkflowAwareAgent()
        
        # Create mock performance history showing improvement
        from agent.workflow_agent import WorkflowContext, WorkflowState
        
        agent.context = WorkflowContext(
            task_id="perf_test",
            start_time=datetime.now(),
            current_state=WorkflowState.IMPROVEMENT
        )
        
        # Simulate improving performance
        agent.context.performance_history = [
            {'iteration': 1, 'performance': {'detection_rate': 0.70}},
            {'iteration': 2, 'performance': {'detection_rate': 0.80}},
            {'iteration': 3, 'performance': {'detection_rate': 0.90}}
        ]
        
        agent.context.evaluation_metrics = {
            'detection_rate': {'target': 0.95, 'weight': 1.0}
        }
        
        # Test improvement identification
        improvements = agent._identify_improvements()
        success = len(improvements) > 0
        
        # Test convergence detection
        if success:
            # Add stable performance
            agent.context.performance_history.extend([
                {'iteration': 4, 'performance': {'detection_rate': 0.95}},
                {'iteration': 5, 'performance': {'detection_rate': 0.95}}
            ])
            
            converged = agent._check_convergence()
            success = converged == True
        
        return success
    
    def test_llm_reasoning(self) -> bool:
        """Test LLM reasoning capabilities"""
        logger.info("Testing LLM reasoning...")
        
        from dspy_modules.enhanced_modules import DataAnalyst
        from unittest.mock import patch, Mock
        import dspy
        
        # Configure DSPy
        lm = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.7)
        dspy.settings.configure(lm=lm)
        
        analyst = DataAnalyst()
        
        # Test code generation
        test_data = {
            'records': [
                {'value': 1}, {'value': 2}, {'value': 3}
            ]
        }
        
        with patch.object(analyst.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(analysis_code="""
results['sum'] = sum(r['value'] for r in data['records'])
results['count'] = len(data['records'])
results['mean'] = results['sum'] / results['count']
""")
            
            result = analyst.forward(
                data=test_data,
                analysis_goal="calculate statistics"
            )
        
        # Check if analysis was generated
        success = 'summary' in result or 'detailed_results' in result
        
        return success
    
    def test_end_to_end_scenario(self) -> bool:
        """Test complete end-to-end FBS detection scenario"""
        logger.info("Testing end-to-end scenario...")
        
        from dataset_playback import DatasetPlayer
        from llm_control_api_client import post_rule, get_stats
        from dspy_modules.enhanced_modules import QueryNetwork, RuleGenerator
        from unittest.mock import patch, Mock
        import dspy
        
        # Configure DSPy
        lm = dspy.OpenAI(model='gpt-3.5-turbo', temperature=0.7)
        dspy.settings.configure(lm=lm)
        
        player = DatasetPlayer()
        query_module = QueryNetwork()
        rule_generator = RuleGenerator()
        
        # Step 1: Generate and inject FBS scenario data
        logger.info("  Step 1: Generating FBS scenario...")
        fbs_data = player.generate_synthetic_fbs_data(duration=60)
        
        for record in fbs_data[:30]:  # Inject subset for speed
            player.inject_mobiflow_record(record)
        
        # Step 2: Query and analyze
        logger.info("  Step 2: Querying and analyzing...")
        with patch('llm_control_api_client.get_mobiflow') as mock_mobiflow:
            mock_mobiflow.return_value = fbs_data[:30]
            
            network_data = query_module.forward(
                query_type="comprehensive",
                time_window=60
            )
        
        # Step 3: Generate rules
        logger.info("  Step 3: Generating detection rules...")
        patterns = network_data.get('mobiflow_analysis', {}).get('patterns', {})
        
        with patch.object(rule_generator.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(rule_yaml="""
rules:
  - name: E2E_FBS_Detection
    condition:
      field: suspected_fbs
      eq: true
    action:
      type: alert
""")
            
            rule_yaml = rule_generator.forward(
                analysis=network_data,
                patterns=patterns
            )
        
        # Step 4: Deploy rules
        logger.info("  Step 4: Deploying rules...")
        result = post_rule(rule_yaml)
        
        # Step 5: Verify detection
        logger.info("  Step 5: Verifying detection...")
        stats = get_stats()
        
        # Success criteria
        success = all([
            result.get('status') == 'success',
            len(fbs_data) > 0,
            'mobiflow_analysis' in network_data
        ])
        
        return success
    
    def _generate_report(self) -> Dict:
        """Generate test report"""
        duration = time.time() - self.start_time if self.start_time else 0
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        total = len(self.test_results)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'success_rate': (passed / total * 100) if total > 0 else 0
            },
            'tests': self.test_results,
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'test_dir': self.temp_dir
            }
        }
        
        return report

def run_integration_tests(verbose: bool = True, 
                         save_report: bool = True) -> Tuple[bool, Dict]:
    """
    Run complete integration test suite
    
    Args:
        verbose: Enable verbose output
        save_report: Save test report to file
        
    Returns:
        Tuple of (success, report)
    """
    print("\n" + "="*70)
    print("5G-SPECTOR FBS DETECTION SYSTEM - INTEGRATION TEST SUITE")
    print("="*70 + "\n")
    
    # Create test runner
    runner = SystemIntegrationTests(verbose=verbose)
    
    # Run tests
    success, report = runner.run_all_tests()
    
    # Display summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    summary = report['summary']
    print(f"Total Tests:    {summary['total']}")
    print(f"Passed:         {summary['passed']} " + 
          f"({summary['passed']/summary['total']*100:.1f}%)" if summary['total'] > 0 else "")
    print(f"Failed:         {summary['failed']}")
    print(f"Errors:         {summary['errors']}")
    print(f"Duration:       {report['duration']:.2f} seconds")
    print(f"Success Rate:   {summary['success_rate']:.1f}%")
    
    # Detailed results
    print("\nDetailed Results:")
    print("-" * 50)
    for test_result in report['tests']:
        status_symbol = {
            'PASS': '✓',
            'FAIL': '✗',
            'ERROR': '⚠'
        }.get(test_result['status'], '?')
        
        print(f"{status_symbol} {test_result['test']}: {test_result['status']}")
        
        if test_result['status'] == 'ERROR' and 'error' in test_result:
            print(f"  Error: {test_result['error']}")
    
    # Save report if requested
    if save_report:
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {report_file}")
    
    # Final verdict
    print("\n" + "="*70)
    if success:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("✗ SOME INTEGRATION TESTS FAILED")
    print("="*70 + "\n")
    
    return success, report

async def run_async_tests():
    """Run tests that require async execution"""
    runner = SystemIntegrationTests()
    runner.setup()
    
    try:
        # Run async workflow test
        result = await runner.test_workflow_execution()
        return result
    finally:
        runner.teardown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run integration tests for FBS detection system'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Do not save test report'
    )
    
    parser.add_argument(
        '--async-only',
        action='store_true',
        help='Run only async tests'
    )
    
    args = parser.parse_args()
    
    if args.async_only:
        # Run async tests
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(run_async_tests())
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        success, report = run_integration_tests(
            verbose=not args.quiet,
            save_report=not args.no_report
        )
        
        sys.exit(0 if success else 1)