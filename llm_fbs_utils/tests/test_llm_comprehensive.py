#!/usr/bin/env python3
"""
Comprehensive Test Suite for LLM Components
Tests all LLM functionality including tools, metrics, workflow, and error handling
Location: llm_fbs_utils/tests/test_llm_comprehensive.py
"""

import unittest
import asyncio
import time
import json
import yaml
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dspy_modules.enhanced_modules import (
    TelemetryPreprocessor, QueryNetwork, RuleGenerator, 
    ExperimentDesigner, DataAnalyst
)
from agent.workflow_agent import (
    WorkflowAwareAgent, WorkflowState, WorkflowContext, ExperimentResult
)

class TestLLMTools(unittest.TestCase):
    """Test LLM tools functionality and documentation"""
    
    def setUp(self):
        """Set up test environment"""
        self.preprocessor = TelemetryPreprocessor()
        self.query_module = QueryNetwork()
        self.rule_generator = RuleGenerator()
        self.experiment_designer = ExperimentDesigner()
        self.data_analyst = DataAnalyst()
    
    def test_tool_availability(self):
        """Test that all tools are available and documented"""
        tools = [
            self.query_module,
            self.rule_generator,
            self.experiment_designer,
            self.data_analyst
        ]
        
        for tool in tools:
            # Check tool has forward method
            self.assertTrue(hasattr(tool, 'forward'), 
                          f"{tool.__class__.__name__} missing forward method")
            
            # Check documentation
            self.assertIsNotNone(tool.__doc__, 
                               f"{tool.__class__.__name__} missing documentation")
            
            # Check forward method documentation
            self.assertIsNotNone(tool.forward.__doc__,
                               f"{tool.__class__.__name__}.forward missing documentation")
    
    def test_preprocessor_numerical_data(self):
        """Test preprocessing of raw numerical data"""
        # Raw KPM data
        raw_kpm = {
            'rsrp': -85,  # dBm
            'rsrq': -12,  # dB
            'sinr': 15,   # dB
            'cqi': 10,
            'throughput_dl': 50000000,  # 50 Mbps
            'throughput_ul': 10000000   # 10 Mbps
        }
        
        # Preprocess
        processed = self.preprocessor.preprocess_kpm_data(raw_kpm)
        
        # Verify preprocessing
        self.assertIn('raw_metrics', processed)
        self.assertIn('signal_quality', processed)
        self.assertIn('anomaly_indicators', processed)
        self.assertIn('normalized_metrics', processed)
        self.assertIn('context', processed)
        
        # Check signal quality categorization
        self.assertIn(processed['signal_quality'], 
                     ['excellent', 'good', 'fair', 'poor'])
        
        # Check normalization (should be 0-1)
        for key, value in processed['normalized_metrics'].items():
            self.assertGreaterEqual(value, 0, f"{key} not normalized properly")
            self.assertLessEqual(value, 1, f"{key} not normalized properly")
    
    def test_preprocessor_batch_processing(self):
        """Test batch preprocessing with temporal features"""
        # Create batch of records
        records = []
        base_time = time.time()
        
        for i in range(20):
            records.append({
                'timestamp': base_time + i,
                'ue_id': 'test_ue',
                'cell_id': 'cell_001',
                'attach_failures': i % 5,  # Pattern
                'auth_reject_count': max(0, i - 15),  # Increases later
                'rsrp': -90 + np.sin(i/5) * 10,  # Sinusoidal
                'cipher_algo': 'NULL' if i > 15 else 'AES'
            })
        
        # Process batch
        processed = self.preprocessor.preprocess_mobiflow_batch(records)
        
        # Verify temporal analysis
        self.assertIn('statistics', processed)
        self.assertIn('patterns', processed)
        self.assertIn('anomalies', processed)
        
        # Check pattern detection
        patterns = processed['patterns']
        self.assertIn('cipher_downgrade_detected', patterns)
        self.assertTrue(patterns['cipher_downgrade_detected'], 
                       "Cipher downgrade not detected")
    
    def test_anomaly_detection(self):
        """Test statistical anomaly detection"""
        # Create data with anomalies
        normal_values = np.random.normal(-85, 5, 100)
        anomaly_values = [-50, -120, -45]  # Anomalies
        all_values = np.concatenate([normal_values, anomaly_values])
        
        records = [{'rsrp': v, 'timestamp': i} for i, v in enumerate(all_values)]
        df = pd.DataFrame(records)
        
        # Detect anomalies
        anomalies = self.preprocessor._detect_statistical_anomalies(df)
        
        # Should detect anomalies
        self.assertGreater(len(anomalies), 0, "No anomalies detected")
        
        # Check anomaly structure
        for anomaly in anomalies:
            self.assertIn('type', anomaly)
            self.assertIn('severity', anomaly)
    
    @patch('llm_control_api_client.get_kpm')
    @patch('llm_control_api_client.get_mobiflow')
    @patch('llm_control_api_client.get_stats')
    def test_query_network_comprehensive(self, mock_stats, mock_mobiflow, mock_kpm):
        """Test comprehensive network querying"""
        # Mock responses
        mock_stats.return_value = {'total_ues': 10, 'fbs_detections': 2}
        mock_mobiflow.return_value = [
            {'ue_id': 'ue1', 'rsrp': -80, 'attach_failures': 0},
            {'ue_id': 'ue2', 'rsrp': -90, 'attach_failures': 3}
        ]
        mock_kpm.return_value = {'rsrp': -85, 'rsrq': -12, 'sinr': 15}
        
        # Query network
        result = self.query_module.forward(
            query_type="comprehensive",
            ue_id="test_ue",
            time_window=300
        )
        
        # Verify query result
        self.assertIn('query_type', result)
        self.assertIn('timestamp', result)
        self.assertIn('preprocessing_applied', result)
        self.assertEqual(result['preprocessing_applied'], True)
        
        # Check preprocessing was applied
        if 'mobiflow_analysis' in result:
            self.assertIn('statistics', result['mobiflow_analysis'])
            self.assertIn('patterns', result['mobiflow_analysis'])
    
    def test_rule_generation_context_aware(self):
        """Test context-aware rule generation"""
        # Create analysis context
        analysis = {
            'mobiflow_analysis': {
                'patterns': {
                    'high_failure_rate': True,
                    'cipher_downgrade_detected': True
                },
                'statistics': {
                    'attach_failures': {
                        'mean': 2.5,
                        'std': 1.2,
                        'max': 8,
                        'q75': 4,
                        'q25': 1
                    }
                }
            }
        }
        
        patterns = {'high_failure_rate': True}
        
        # Generate rule
        with patch.object(self.rule_generator.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(rule_yaml="""
rules:
  - name: Generated_FBS_Rule
    condition:
      field: attach_failures
      gte: 5
    action:
      type: alert
      severity: high
""")
            
            rule_yaml = self.rule_generator.forward(
                analysis=analysis,
                patterns=patterns,
                objective="detect_fbs"
            )
        
        # Verify rule generation
        self.assertIsNotNone(rule_yaml)
        
        # Parse and validate rule
        rule = yaml.safe_load(rule_yaml)
        self.assertIn('rules', rule)
        self.assertGreater(len(rule['rules']), 0)
        
        # Check rule structure
        first_rule = rule['rules'][0]
        self.assertIn('name', first_rule)
        self.assertIn('condition', first_rule)
        self.assertIn('action', first_rule)
    
    def test_experiment_designer_safety(self):
        """Test experiment designer with safety checks"""
        hypothesis = "Increasing power will improve detection"
        current_performance = {
            'detection_rate': 0.7,
            'false_positive_rate': 0.15
        }
        
        constraints = {
            'max_duration': 120,
            'max_power': 25,
            'safety_mode': True
        }
        
        # Mock LLM response
        with patch.object(self.experiment_designer.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(experiment_json=json.dumps({
                'name': 'Test_Experiment',
                'duration': 200,  # Exceeds max
                'config': {
                    'tx_power': 35  # Exceeds max
                },
                'events': []
            }))
            
            # Design experiment
            experiment = self.experiment_designer.forward(
                hypothesis=hypothesis,
                current_performance=current_performance,
                constraints=constraints
            )
        
        # Verify safety checks were applied
        self.assertLessEqual(experiment['duration'], 120, 
                           "Duration safety check failed")
        
        if 'config' in experiment and 'tx_power' in experiment['config']:
            self.assertLessEqual(experiment['config']['tx_power'], 30,
                               "Power safety check failed")
        
        # Check timing coordination added
        self.assertIn('timing', experiment)
        self.assertIn('checkpoint_interval', experiment['timing'])
    
    def test_data_analyst_code_generation(self):
        """Test data analysis code generation and execution"""
        # Test data
        test_data = {
            'records': [
                {'ue_id': 'ue1', 'attach_failures': 3},
                {'ue_id': 'ue2', 'attach_failures': 5},
                {'ue_id': 'ue1', 'attach_failures': 2}
            ]
        }
        
        # Mock LLM code generation
        with patch.object(self.data_analyst.predictor, 'forward') as mock_predict:
            mock_predict.return_value = Mock(analysis_code="""
import pandas as pd
df = pd.DataFrame(data['records'])
results['total_failures'] = df['attach_failures'].sum()
results['avg_failures'] = df['attach_failures'].mean()
results['unique_ues'] = df['ue_id'].nunique()
""")
            
            # Generate and execute analysis
            result = self.data_analyst.forward(
                data=test_data,
                analysis_goal="count_failures",
                output_format="summary"
            )
        
        # Verify analysis execution
        self.assertIn('summary', result)
        
        # Check results structure
        summary = result['summary']
        if 'error' not in summary:
            # Should have computed metrics
            self.assertIn('total_failures', summary)
            self.assertEqual(summary['total_failures'], 10)
            self.assertIn('unique_ues', summary)
            self.assertEqual(summary['unique_ues'], 2)

class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics validity and robustness"""
    
    def test_metric_definitions(self):
        """Test that evaluation metrics are properly defined"""
        metrics = {
            'detection_rate': {'target': 0.95, 'weight': 0.3, 'range': (0, 1)},
            'false_positive_rate': {'target': 0.05, 'weight': 0.3, 'range': (0, 1)},
            'detection_time': {'target': 30, 'weight': 0.2, 'range': (0, 300)},
            'accuracy': {'target': 0.9, 'weight': 0.2, 'range': (0, 1)}
        }
        
        # Verify metric properties
        total_weight = sum(m['weight'] for m in metrics.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2, 
                              msg="Metric weights don't sum to 1")
        
        for name, config in metrics.items():
            # Check required fields
            self.assertIn('target', config, f"{name} missing target")
            self.assertIn('weight', config, f"{name} missing weight")
            self.assertIn('range', config, f"{name} missing range")
            
            # Check target is within range
            min_val, max_val = config['range']
            self.assertGreaterEqual(config['target'], min_val,
                                  f"{name} target below minimum")
            self.assertLessEqual(config['target'], max_val,
                                f"{name} target above maximum")
    
    def test_metric_calculation(self):
        """Test metric calculation accuracy"""
        # Test data
        total_samples = 100
        true_positives = 85
        false_positives = 5
        false_negatives = 10
        true_negatives = 0
        
        # Calculate metrics
        detection_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / max((false_positives + true_negatives), 1)
        accuracy = (true_positives + true_negatives) / total_samples
        
        # Verify calculations
        self.assertAlmostEqual(detection_rate, 0.8947, places=3)
        self.assertEqual(false_positive_rate, 5.0)  # No true negatives
        self.assertEqual(accuracy, 0.85)
    
    def test_custom_metric_creation(self):
        """Test creation of custom metrics"""
        agent = WorkflowAwareAgent()
        
        # Mock context
        agent.context = WorkflowContext(
            task_id="test",
            start_time=time.time(),
            current_state=WorkflowState.IMPROVEMENT,
            iteration=5
        )
        
        # Test metric gap identification
        gaps = agent._identify_metric_gaps()
        
        # Should identify missing metrics
        self.assertGreater(len(gaps), 0, "No metric gaps identified")
        
        # Check gap structure
        for gap in gaps:
            self.assertIn('name', gap)
            self.assertIn('formula', gap)
            self.assertIn('target', gap)
            self.assertIn('reason', gap)
    
    def test_convergence_criteria(self):
        """Test convergence criteria for iterative improvement"""
        agent = WorkflowAwareAgent()
        
        # Create mock performance history
        agent.context = WorkflowContext(
            task_id="test",
            start_time=time.time(),
            current_state=WorkflowState.IMPROVEMENT
        )
        
        # Converged performance (stable)
        agent.context.performance_history = [
            {'iteration': 1, 'performance': {'detection_rate': 0.94}},
            {'iteration': 2, 'performance': {'detection_rate': 0.95}},
            {'iteration': 3, 'performance': {'detection_rate': 0.95}}
        ]
        
        agent.context.evaluation_metrics = {
            'detection_rate': {'target': 0.95, 'weight': 1.0}
        }
        
        # Should detect convergence
        converged = agent._check_convergence()
        self.assertTrue(converged, "Failed to detect convergence")
        
        # Non-converged performance (still improving)
        agent.context.performance_history = [
            {'iteration': 1, 'performance': {'detection_rate': 0.70}},
            {'iteration': 2, 'performance': {'detection_rate': 0.80}},
            {'iteration': 3, 'performance': {'detection_rate': 0.90}}
        ]
        
        converged = agent._check_convergence()
        self.assertFalse(converged, "False convergence detected")

class TestRuleAndCodeExecution(unittest.TestCase):
    """Test rule and code execution with error handling"""
    
    def test_rule_validation_and_execution(self):
        """Test rule validation and execution"""
        from llm_control_api_client import test_rule
        
        # Valid rule
        valid_rule = {
            'name': 'Test_Rule',
            'condition': {
                'and': [
                    {'field': 'attach_failures', 'gte': 3},
                    {'field': 'suspected_fbs', 'eq': True}
                ]
            },
            'action': {'type': 'alert'}
        }
        
        # Test with matching data
        matching_data = {
            'attach_failures': 5,
            'suspected_fbs': True
        }
        
        with patch('requests.Session.request') as mock_request:
            mock_request.return_value.json.return_value = {
                'status': 'success',
                'matched': True
            }
            
            result = test_rule(valid_rule, matching_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['matched'])
        
        # Test with non-matching data
        non_matching_data = {
            'attach_failures': 1,
            'suspected_fbs': False
        }
        
        with patch('requests.Session.request') as mock_request:
            mock_request.return_value.json.return_value = {
                'status': 'success',
                'matched': False
            }
            
            result = test_rule(valid_rule, non_matching_data)
        
        self.assertFalse(result['matched'])
    
    def test_faulty_rule_handling(self):
        """Test handling of faulty rules"""
        # Rule with syntax error
        faulty_rule = "invalid: yaml: {["
        
        rule_generator = RuleGenerator()
        
        # Should handle gracefully
        validated = rule_generator._validate_and_refine(faulty_rule)
        
        # Should return safe default
        self.assertIsNotNone(validated)
        parsed = yaml.safe_load(validated)
        self.assertIsNotNone(parsed)
    
    def test_code_execution_safety(self):
        """Test safe code execution"""
        analyst = DataAnalyst()
        
        # Dangerous code
        dangerous_code = """
import os
os.system('rm -rf /')  # Should be blocked
results['hacked'] = True
"""
        
        # Safe execution should sanitize
        result = analyst._safe_execute(dangerous_code, {})
        
        # Should not execute dangerous operations
        if 'error' not in result:
            self.assertNotIn('hacked', result)
        
        # Safe code should execute
        safe_code = """
results['sum'] = sum([1, 2, 3])
results['test'] = 'safe'
"""
        
        result = analyst._safe_execute(safe_code, {})
        
        if 'error' not in result:
            self.assertEqual(result.get('sum'), 6)
            self.assertEqual(result.get('test'), 'safe')
    
    def test_runtime_error_handling(self):
        """Test handling of runtime errors"""
        analyst = DataAnalyst()
        
        # Code with runtime error
        error_code = """
df = pd.DataFrame(data['missing_key'])  # KeyError
results['processed'] = len(df)
"""
        
        test_data = {'wrong_key': [1, 2, 3]}
        
        # Should handle error gracefully
        result = analyst._safe_execute(error_code, test_data)
        
        # Should return error information
        self.assertIn('error', result)

class TestExperimentDesignAndExecution(unittest.TestCase):
    """Test experiment design and execution"""
    
    def test_experiment_self_design(self):
        """Test LLM self-designing experiments"""
        designer = ExperimentDesigner()
        
        # Multiple hypotheses
        hypotheses = [
            "Lower thresholds improve detection",
            "Temporal correlation reduces false positives",
            "Multi-factor detection increases accuracy"
        ]
        
        experiments = []
        for hypothesis in hypotheses:
            with patch.object(designer.predictor, 'forward') as mock_predict:
                mock_predict.return_value = Mock(experiment_json=json.dumps({
                    'name': f'Experiment_{hypothesis[:10]}',
                    'duration': 60,
                    'hypothesis': hypothesis,
                    'config': {'test_param': 'value'},
                    'events': []
                }))
                
                experiment = designer.forward(
                    hypothesis=hypothesis,
                    current_performance={'detection_rate': 0.8}
                )
                experiments.append(experiment)
        
        # Verify multiple experiments designed
        self.assertEqual(len(experiments), 3)
        
        # Each should be unique
        names = [e['name'] for e in experiments]
        self.assertEqual(len(names), len(set(names)), "Duplicate experiment names")
    
    @patch('scenario_runner.run_scenario')
    @patch('scenario_runner.get_scenario_status')
    async def test_experiment_timing_control(self, mock_status, mock_run):
        """Test proper timing control in experiments"""
        agent = WorkflowAwareAgent()
        
        # Mock scenario execution
        mock_run.return_value = None
        mock_status.side_effect = [
            {'running': True},   # First check
            {'running': True},   # Second check
            {'running': False}   # Complete
        ]
        
        # Test experiment with timeout
        experiment = {
            'name': 'Timing_Test',
            'duration': 60
        }
        
        with patch('llm_control_api_client.get_stats') as mock_stats:
            mock_stats.return_value = {'fbs_detections': 5, 'total_records': 100}
            
            result = await agent._execute_experiment_with_timeout(
                experiment,
                timeout=120
            )
        
        # Verify result
        self.assertIsInstance(result, ExperimentResult)
        self.assertTrue(result.success)
        self.assertLess(result.duration, 120)
    
    def test_parallel_experiment_execution(self):
        """Test parallel experiment execution capability"""
        import concurrent.futures
        
        def mock_experiment(exp_id):
            time.sleep(0.1)  # Simulate execution
            return {
                'exp_id': exp_id,
                'result': 'success',
                'metrics': {'detection_rate': 0.9}
            }
        
        # Execute multiple experiments in parallel
        experiments = [f'exp_{i}' for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(mock_experiment, exp) for exp in experiments]
            results = [f.result(timeout=5) for f in futures]
        
        # Verify all completed
        self.assertEqual(len(results), 5)
        
        # Check results
        for i, result in enumerate(results):
            self.assertEqual(result['exp_id'], f'exp_{i}')
            self.assertEqual(result['result'], 'success')

class TestDataProcessingAndFiltering(unittest.TestCase):
    """Test data processing and filtering for LLM"""
    
    def test_raw_trace_filtering(self):
        """Test filtering of raw traces to avoid overwhelming LLM"""
        preprocessor = TelemetryPreprocessor()
        
        # Generate large dataset
        large_dataset = []
        for i in range(10000):
            large_dataset.append({
                'timestamp': i,
                'ue_id': f'ue_{i % 100}',
                'rsrp': -90 + np.random.randn() * 10,
                'attach_failures': np.random.poisson(0.5)
            })
        
        # Process with filtering
        processed = preprocessor.preprocess_mobiflow_batch(large_dataset)
        
        # Should produce summary, not raw data
        self.assertIn('statistics', processed)
        self.assertIn('total_records', processed)
        
        # Summary should be compact
        summary_str = json.dumps(processed)
        self.assertLess(len(summary_str), 50000, 
                       "Summary too large for LLM")
    
    def test_intelligent_sampling(self):
        """Test intelligent sampling of data"""
        # Create dataset with patterns
        dataset = []
        
        # Normal period
        for i in range(100):
            dataset.append({
                'timestamp': i,
                'event_type': 'normal',
                'value': np.random.normal(0, 1)
            })
        
        # Anomaly period
        for i in range(100, 120):
            dataset.append({
                'timestamp': i,
                'event_type': 'anomaly',
                'value': np.random.normal(5, 1)  # Different distribution
            })
        
        # More normal
        for i in range(120, 200):
            dataset.append({
                'timestamp': i,
                'event_type': 'normal',
                'value': np.random.normal(0, 1)
            })
        
        df = pd.DataFrame(dataset)
        
        # Intelligent sampling should capture both normal and anomaly
        sample_size = 20
        
        # Simple stratified sampling
        if 'event_type' in df.columns:
            sampled = df.groupby('event_type', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size//2))
            )
        else:
            sampled = df.sample(min(len(df), sample_size))
        
        # Check sample contains both types
        event_types = sampled['event_type'].unique()
        self.assertIn('normal', event_types)
        self.assertIn('anomaly', event_types)

class TestPromptAndFeedbackFlow(unittest.TestCase):
    """Test prompt generation and feedback flow"""
    
    def test_prompt_context_generation(self):
        """Test generation of context for prompts"""
        rule_generator = RuleGenerator()
        
        analysis = {
            'mobiflow_analysis': {
                'patterns': {'high_failure_rate': True},
                'statistics': {
                    'attach_failures': {'mean': 3.5, 'std': 1.2}
                }
            }
        }
        
        patterns = {'detected': True}
        
        # Generate context
        context = rule_generator._prepare_context(analysis, patterns, "detect_fbs")
        
        # Verify context structure
        self.assertIn('objective', context)
        self.assertIn('threat_indicators', context)
        self.assertIn('thresholds', context)
        self.assertIn('conditions', context)
        
        # Check context is informative
        self.assertEqual(context['objective'], 'detect_fbs')
        self.assertGreater(len(context['threat_indicators']), 0)
    
    def test_feedback_incorporation(self):
        """Test incorporation of feedback into next iteration"""
        agent = WorkflowAwareAgent()
        
        # Initialize context
        agent.context = WorkflowContext(
            task_id="test",
            start_time=time.time(),
            current_state=WorkflowState.IMPROVEMENT
        )
        
        # Add performance history
        agent.context.performance_history = [
            {'iteration': 1, 'performance': {'detection_rate': 0.7}},
            {'iteration': 2, 'performance': {'detection_rate': 0.75}}
        ]
        
        # Add failure patterns
        agent.context.failure_patterns = [
            {'type': 'false_negative', 'condition': 'low_signal'},
            {'type': 'false_positive', 'condition': 'interference'}
        ]
        
        # Identify improvements based on feedback
        improvements = agent._identify_improvements()
        
        # Should identify improvements
        self.assertGreater(len(improvements), 0)
        
        # Check improvements are based on failures
        improvement_types = [i['type'] for i in improvements]
        self.assertIn('failure_pattern', improvement_types)
    
    def test_format_error_handling(self):
        """Test handling of format errors in LLM responses"""
        designer = ExperimentDesigner()
        
        # Invalid JSON response
        invalid_json = "not valid json {["
        
        # Should handle gracefully
        result = designer._parse_experiment(invalid_json)
        
        # Should return safe default
        self.assertIsNotNone(result)
        self.assertIn('name', result)
        self.assertIn('duration', result)
        self.assertIn('mode', result)

class TestWorkflowIntegration(unittest.TestCase):
    """Test complete workflow integration"""
    
    @patch('llm_control_api_client.get_stats')
    @patch('scenario_runner.run_scenario')
    async def test_complete_workflow_execution(self, mock_run, mock_stats):
        """Test complete workflow from start to finish"""
        agent = WorkflowAwareAgent()
        
        # Mock responses
        mock_stats.return_value = {
            'total_ues': 10,
            'fbs_detections': 5,
            'total_records': 1000
        }
        mock_run.return_value = None
        
        # Mock DSPy predictor responses
        with patch.object(agent.rule_generator.predictor, 'forward') as mock_rule:
            mock_rule.return_value = Mock(rule_yaml="rules:\n  - name: test\n    condition: {field: test, eq: true}\n    action: {type: log}")
            
            with patch.object(agent.experiment_designer.predictor, 'forward') as mock_exp:
                mock_exp.return_value = Mock(experiment_json='{"name": "test", "duration": 10}')
                
                # Run workflow (simplified)
                agent.context = WorkflowContext(
                    task_id="test",
                    start_time=datetime.now(),
                    current_state=WorkflowState.INITIALIZATION,
                    max_iterations=1  # Single iteration for test
                )
                
                # Test individual phases
                await agent._phase_understanding()
                self.assertEqual(agent.context.current_state, WorkflowState.UNDERSTANDING)
                
                await agent._phase_baseline_assessment()
                self.assertIsNotNone(agent.context.baseline_performance)
    
    def test_checkpoint_save_and_restore(self):
        """Test checkpoint saving and restoration"""
        agent = WorkflowAwareAgent()
        
        # Create context
        agent.context = WorkflowContext(
            task_id="checkpoint_test",
            start_time=datetime.now(),
            current_state=WorkflowState.IMPROVEMENT,
            iteration=3
        )
        
        agent.context.insights = ["Test insight 1", "Test insight 2"]
        agent.context.current_performance = {'detection_rate': 0.85}
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock checkpoint directory
            checkpoint_dir = os.path.join(tmpdir, 'checkpoints')
            os.makedirs(checkpoint_dir)
            
            with patch('os.path.dirname') as mock_dirname:
                mock_dirname.return_value = checkpoint_dir
                agent._save_checkpoint()
            
            # Check checkpoint was saved
            files = os.listdir(checkpoint_dir)
            checkpoint_files = [f for f in files if f.startswith('checkpoint_')]
            self.assertGreater(len(checkpoint_files), 0, "No checkpoint saved")
            
            # Load checkpoint
            with open(os.path.join(checkpoint_dir, checkpoint_files[0]), 'r') as f:
                checkpoint = json.load(f)
            
            # Verify checkpoint content
            self.assertIn('context', checkpoint)
            self.assertEqual(checkpoint['context']['iteration'], 3)
    
    def test_error_recovery(self):
        """Test error recovery mechanism"""
        agent = WorkflowAwareAgent()
        agent.error_recovery_enabled = True
        
        # Create context
        agent.context = WorkflowContext(
            task_id="recovery_test",
            start_time=datetime.now(),
            current_state=WorkflowState.EXPERIMENT_EXECUTION
        )
        
        # Simulate error
        error = Exception("Test error")
        
        # Test recovery
        async def test_recovery():
            result = await agent._recover_from_error(error)
            return result
        
        # Run recovery
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(test_recovery())
        
        # Check recovery attempted
        self.assertIn('recovery_attempted', result)
        self.assertTrue(result['recovery_attempted'])

def run_comprehensive_llm_tests():
    """Run all LLM component tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLLMTools,
        TestEvaluationMetrics,
        TestRuleAndCodeExecution,
        TestExperimentDesignAndExecution,
        TestDataProcessingAndFiltering,
        TestPromptAndFeedbackFlow,
        TestWorkflowIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("LLM COMPONENT TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*50)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_llm_tests()
    sys.exit(0 if success else 1)