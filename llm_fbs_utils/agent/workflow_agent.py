#!/usr/bin/env python3
"""
Workflow-Aware LLM Agent for FBS Detection
Implements complete iterative learning and experimentation workflow
Location: llm_fbs_utils/agent/workflow_agent.py
"""

import os
import json
import yaml
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

try:
    import dspy
except ImportError:
    dspy = None
try:
    from dspy_modules.enhanced_modules import QueryNetwork, RuleGenerator, ExperimentDesigner, DataAnalyst
except ImportError:
    from enhanced_modules import QueryNetwork, RuleGenerator, ExperimentDesigner, DataAnalyst


logger = logging.getLogger(__name__)

# ============================================================================
# Workflow States and Data Structures
# ============================================================================

class WorkflowState(Enum):
    """States in the detection workflow"""
    INITIALIZATION = "initialization"
    UNDERSTANDING = "understanding"
    BASELINE_ASSESSMENT = "baseline_assessment"
    RULE_GENERATION = "rule_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULT_ANALYSIS = "result_analysis"
    IMPROVEMENT = "improvement"
    VALIDATION = "validation"
    COMPLETION = "completion"
    ERROR = "error"

@dataclass
class WorkflowContext:
    """Context maintained throughout workflow"""
    task_id: str
    start_time: datetime
    current_state: WorkflowState
    iteration: int = 0
    max_iterations: int = 10
    
    # Task understanding
    task_description: str = ""
    available_tools: List[str] = field(default_factory=list)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    current_performance: Dict[str, float] = field(default_factory=dict)
    performance_history: List[Dict] = field(default_factory=list)
    
    # Generated artifacts
    generated_rules: List[Dict] = field(default_factory=list)
    experiments_run: List[Dict] = field(default_factory=list)
    analysis_results: List[Dict] = field(default_factory=list)
    
    # Learning insights
    insights: List[str] = field(default_factory=list)
    failure_patterns: List[Dict] = field(default_factory=list)
    success_patterns: List[Dict] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert context to dictionary"""
        return {
            'task_id': self.task_id,
            'iteration': self.iteration,
            'state': self.current_state.value,
            'performance': self.current_performance,
            'insights': self.insights
        }

@dataclass
class ExperimentResult:
    """Results from an experiment"""
    experiment_id: str
    hypothesis: str
    duration: float
    success: bool
    metrics: Dict[str, float]
    raw_data: Dict
    errors: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

# ============================================================================
# Workflow-Aware Agent
# ============================================================================

class WorkflowAwareAgent:
    """
    Intelligent agent that understands and executes the complete workflow
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize DSPy modules
        self.query_module = QueryNetwork()
        self.rule_generator = RuleGenerator()
        self.experiment_designer = ExperimentDesigner()
        self.data_analyst = DataAnalyst()
        
        # Workflow management
        self.context: Optional[WorkflowContext] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Error handling
        self.max_retries = 3
        self.error_recovery_enabled = True
        
        # Timing control
        self.experiment_timeout = 600  # 10 minutes max per experiment
        self.checkpoint_interval = 30  # Save progress every 30s
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_iterations': 10,
            'convergence_threshold': 0.95,
            'experiment_budget': 20,  # Max experiments
            'parallel_experiments': False,
            'auto_save': True,
            'debug_mode': False
        }
    
    async def run_complete_workflow(self, task_description: str) -> Dict:
        """
        Run the complete workflow from task understanding to validation
        
        Args:
            task_description: High-level task description
            
        Returns:
            Final results and learnings
        """
        # Initialize context
        self.context = WorkflowContext(
            task_id=f"task_{int(time.time())}",
            start_time=datetime.now(),
            current_state=WorkflowState.INITIALIZATION,
            task_description=task_description,
            max_iterations=self.config['max_iterations']
        )
        
        try:
            # Phase 1: Understanding
            await self._phase_understanding()
            
            # Phase 2: Baseline Assessment
            await self._phase_baseline_assessment()
            
            # Phase 3: Iterative Improvement Loop
            while self.context.iteration < self.context.max_iterations:
                self.context.iteration += 1
                logger.info(f"Starting iteration {self.context.iteration}")
                
                # Generate rules
                await self._phase_rule_generation()
                
                # Design experiments
                await self._phase_experiment_design()
                
                # Execute experiments with proper timing
                await self._phase_experiment_execution()
                
                # Analyze results
                await self._phase_result_analysis()
                
                # Improve based on learnings
                improvement_needed = await self._phase_improvement()
                
                if not improvement_needed:
                    logger.info("Convergence achieved")
                    break
                
                # Save checkpoint
                if self.config['auto_save']:
                    self._save_checkpoint()
            
            # Phase 4: Validation
            await self._phase_validation()
            
            # Phase 5: Completion
            return await self._phase_completion()
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.context.current_state = WorkflowState.ERROR
            
            if self.error_recovery_enabled:
                return await self._recover_from_error(e)
            else:
                raise
    
    async def _phase_understanding(self):
        """Phase 1: Understand the task and available tools"""
        self.context.current_state = WorkflowState.UNDERSTANDING
        logger.info("Phase 1: Understanding task and tools")
        
        # Analyze task description
        task_analysis = await self._analyze_task(self.context.task_description)
        
        # Discover available tools
        self.context.available_tools = self._discover_tools()
        
        # Define evaluation metrics
        self.context.evaluation_metrics = {
            'detection_rate': {'target': 0.95, 'weight': 0.3},
            'false_positive_rate': {'target': 0.05, 'weight': 0.3},
            'detection_time': {'target': 30, 'weight': 0.2},
            'accuracy': {'target': 0.9, 'weight': 0.2}
        }
        
        # Set constraints
        self.context.constraints = {
            'max_false_positives': 5,
            'max_detection_time': 60,
            'min_accuracy': 0.8
        }
        
        # Log understanding
        self.context.insights.append(
            f"Task understood: Detect FBS with metrics {list(self.context.evaluation_metrics.keys())}"
        )
    
    async def _phase_baseline_assessment(self):
        """Phase 2: Assess baseline performance"""
        self.context.current_state = WorkflowState.BASELINE_ASSESSMENT
        logger.info("Phase 2: Assessing baseline performance")
        
        # Query current network state
        network_state = await self._query_network_comprehensive()
        
        # Run baseline detection test
        baseline_results = await self._run_baseline_test()
        
        # Store baseline performance
        self.context.baseline_performance = baseline_results['metrics']
        self.context.current_performance = baseline_results['metrics'].copy()
        
        # Analyze baseline
        analysis = self.data_analyst.forward(
            data=baseline_results,
            analysis_goal="identify_weaknesses",
            output_format="summary"
        )
        
        # Extract insights
        self.context.insights.append(
            f"Baseline: Detection rate {baseline_results['metrics'].get('detection_rate', 0):.2%}"
        )
    
    async def _phase_rule_generation(self):
        """Phase 3: Generate detection rules"""
        self.context.current_state = WorkflowState.RULE_GENERATION
        logger.info(f"Phase 3: Generating rules (iteration {self.context.iteration})")
        
        # Query current state
        network_data = self.query_module.forward(
            query_type="comprehensive",
            time_window=300
        )
        
        # Identify patterns
        patterns = self._identify_patterns(network_data)
        
        # Generate rules based on patterns and previous learnings
        rule_context = {
            'patterns': patterns,
            'previous_rules': self.context.generated_rules,
            'failure_patterns': self.context.failure_patterns,
            'performance': self.context.current_performance
        }
        
        rule_yaml = self.rule_generator.forward(
            analysis=network_data,
            patterns=patterns,
            objective=f"improve_detection_iteration_{self.context.iteration}"
        )
        
        # Validate and deploy rule
        deployed = await self._deploy_rule_with_validation(rule_yaml)
        
        if deployed:
            self.context.generated_rules.append({
                'iteration': self.context.iteration,
                'rule': rule_yaml,
                'timestamp': datetime.now().isoformat()
            })
    
    async def _phase_experiment_design(self):
        """Phase 4: Design experiments"""
        self.context.current_state = WorkflowState.EXPERIMENT_DESIGN
        logger.info("Phase 4: Designing experiments")
        
        # Generate hypotheses based on current performance
        hypotheses = self._generate_hypotheses()
        
        # Design experiments for each hypothesis
        experiments = []
        for hypothesis in hypotheses[:3]:  # Limit to 3 experiments per iteration
            experiment = self.experiment_designer.forward(
                hypothesis=hypothesis,
                current_performance=self.context.current_performance,
                constraints=self.context.constraints
            )
            
            experiments.append({
                'hypothesis': hypothesis,
                'design': experiment
            })
        
        self.context.experiments_run.extend(experiments)
        
        return experiments
    
    async def _phase_experiment_execution(self):
        """Phase 5: Execute experiments with proper timing"""
        self.context.current_state = WorkflowState.EXPERIMENT_EXECUTION
        logger.info("Phase 5: Executing experiments")
        
        # Get experiments for this iteration
        experiments = [e for e in self.context.experiments_run 
                      if e.get('iteration', self.context.iteration) == self.context.iteration]
        
        results = []
        for exp in experiments:
            try:
                # Execute with timeout
                result = await self._execute_experiment_with_timeout(
                    exp['design'],
                    timeout=self.experiment_timeout
                )
                results.append(result)
                
                # Wait between experiments
                await asyncio.sleep(10)
                
            except TimeoutError:
                logger.error(f"Experiment timed out: {exp['hypothesis']}")
                results.append(ExperimentResult(
                    experiment_id=exp['design'].get('name', 'unknown'),
                    hypothesis=exp['hypothesis'],
                    duration=self.experiment_timeout,
                    success=False,
                    metrics={},
                    raw_data={},
                    errors=['Timeout']
                ))
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                results.append(ExperimentResult(
                    experiment_id=exp['design'].get('name', 'unknown'),
                    hypothesis=exp['hypothesis'],
                    duration=0,
                    success=False,
                    metrics={},
                    raw_data={},
                    errors=[str(e)]
                ))
        
        # Store results
        self.context.analysis_results.extend([r.__dict__ for r in results])
        
        return results
    
    async def _phase_result_analysis(self):
        """Phase 6: Analyze experiment results"""
        self.context.current_state = WorkflowState.RESULT_ANALYSIS
        logger.info("Phase 6: Analyzing results")
        
        # Get latest results
        latest_results = self.context.analysis_results[-3:] if self.context.analysis_results else []
        
        # Comprehensive analysis
        analysis = self.data_analyst.forward(
            data={'results': latest_results},
            analysis_goal="identify_success_and_failure_patterns",
            output_format="detailed"
        )
        
        # Extract patterns
        self._extract_patterns_from_analysis(analysis)
        
        # Update performance metrics
        self._update_performance_metrics(latest_results)
        
        # Generate insights
        insights = self._generate_insights_from_analysis(analysis)
        self.context.insights.extend(insights)
    
    async def _phase_improvement(self) -> bool:
        """Phase 7: Improve based on learnings"""
        self.context.current_state = WorkflowState.IMPROVEMENT
        logger.info("Phase 7: Improving based on learnings")
        
        # Check if improvement is needed
        if self._check_convergence():
            return False  # No improvement needed
        
        # Identify improvement opportunities
        improvements = self._identify_improvements()
        
        # Create custom metrics if needed
        if self.context.iteration > 3:  # After initial iterations
            custom_metrics = await self._create_custom_metrics()
            self.context.custom_metrics.update(custom_metrics)
        
        # Apply improvements
        for improvement in improvements:
            await self._apply_improvement(improvement)
        
        return True  # Continue iterations
    
    async def _phase_validation(self):
        """Phase 8: Validate final solution"""
        self.context.current_state = WorkflowState.VALIDATION
        logger.info("Phase 8: Validating solution")
        
        # Run comprehensive validation tests
        validation_results = await self._run_validation_suite()
        
        # Check against all metrics
        validation_passed = all(
            validation_results.get(metric, 0) >= target['target'] * 0.95
            for metric, target in self.context.evaluation_metrics.items()
        )
        
        if not validation_passed:
            logger.warning("Validation failed, may need additional iterations")
    
    async def _phase_completion(self) -> Dict:
        """Phase 9: Complete workflow and prepare results"""
        self.context.current_state = WorkflowState.COMPLETION
        logger.info("Phase 9: Completing workflow")
        
        # Prepare final report
        report = {
            'task_id': self.context.task_id,
            'duration': (datetime.now() - self.context.start_time).total_seconds(),
            'iterations': self.context.iteration,
            'final_performance': self.context.current_performance,
            'improvement': {
                metric: (
                    self.context.current_performance.get(metric, 0) - 
                    self.context.baseline_performance.get(metric, 0)
                )
                for metric in self.context.evaluation_metrics.keys()
            },
            'generated_rules': len(self.context.generated_rules),
            'experiments_run': len(self.context.experiments_run),
            'insights': self.context.insights,
            'best_rule': self._get_best_rule(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save final results
        self._save_final_results(report)
        
        return report
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    async def _analyze_task(self, description: str) -> Dict:
        """Analyze task description to understand requirements"""
        # In practice, use LLM to analyze
        return {
            'objective': 'detect_fbs',
            'requirements': ['high_accuracy', 'low_latency', 'robust'],
            'challenges': ['false_positives', 'evolving_threats']
        }
    
    def _discover_tools(self) -> List[str]:
        """Discover available tools and APIs"""
        tools = []
        
        # Check API availability
        try:
            from llm_control_api_client import get_stats
            if get_stats():
                tools.append('network_query_api')
        except:
            pass
        
        tools.extend([
            'rule_generator',
            'experiment_designer',
            'data_analyst',
            'scenario_runner'
        ])
        
        return tools
    
    async def _query_network_comprehensive(self) -> Dict:
        """Comprehensive network query"""
        return self.query_module.forward(
            query_type="comprehensive",
            time_window=600
        )
    
    async def _run_baseline_test(self) -> Dict:
        """Run baseline detection test"""
        from scenario_runner import ScenarioLibrary
        from llm_control_api_client import get_stats
        
        # Run basic scenario
        scenario = ScenarioLibrary.basic_fbs_attack()
        
        # Mock execution for now
        await asyncio.sleep(2)
        
        # Get metrics
        stats = get_stats()
        
        return {
            'metrics': {
                'detection_rate': 0.7,  # Starting baseline
                'false_positive_rate': 0.1,
                'detection_time': 45,
                'accuracy': 0.8
            },
            'raw_data': stats
        }
    
    def _identify_patterns(self, data: Dict) -> Dict:
        """Identify patterns in network data"""
        patterns = {}
        
        if 'mobiflow_analysis' in data:
            analysis = data['mobiflow_analysis']
            if 'patterns' in analysis:
                patterns.update(analysis['patterns'])
        
        # Add domain-specific pattern detection
        patterns['time_based'] = self._detect_temporal_patterns(data)
        patterns['spatial'] = self._detect_spatial_patterns(data)
        
        return patterns
    
    def _detect_temporal_patterns(self, data: Dict) -> Dict:
        """Detect temporal patterns"""
        return {
            'peak_hours': 'detected' if data.get('statistics', {}).get('time_variance', 0) > 0.5 else 'not_detected',
            'periodicity': 'detected' if data.get('statistics', {}).get('autocorrelation', 0) > 0.7 else 'not_detected'
        }
    
    def _detect_spatial_patterns(self, data: Dict) -> Dict:
        """Detect spatial patterns"""
        return {
            'clustering': 'detected' if data.get('statistics', {}).get('unique_cells', 0) < 3 else 'not_detected',
            'mobility': 'high' if data.get('statistics', {}).get('cell_changes', 0) > 10 else 'low'
        }
    
    async def _deploy_rule_with_validation(self, rule_yaml: str) -> bool:
        """Deploy rule with validation"""
        from llm_control_api_client import post_rule, test_rule
        
        try:
            # Parse rule
            rule = yaml.safe_load(rule_yaml)
            
            # Test rule first
            test_data = {
                'attach_failures': 5,
                'suspected_fbs': True
            }
            
            test_result = test_rule(rule['rules'][0] if 'rules' in rule else rule, test_data)
            
            if test_result.get('matched'):
                # Deploy rule
                result = post_rule(rule_yaml)
                return result.get('status') == 'success'
            
            return False
            
        except Exception as e:
            logger.error(f"Rule deployment failed: {e}")
            return False
    
    def _generate_hypotheses(self) -> List[str]:
        """Generate hypotheses for experiments"""
        hypotheses = []
        
        # Based on current performance
        if self.context.current_performance.get('detection_rate', 0) < 0.9:
            hypotheses.append("Lowering detection thresholds will improve detection rate")
        
        if self.context.current_performance.get('false_positive_rate', 1) > 0.1:
            hypotheses.append("Adding temporal correlation will reduce false positives")
        
        # Based on patterns
        if self.context.failure_patterns:
            hypotheses.append("Addressing failure patterns will improve robustness")
        
        # Generic hypotheses
        hypotheses.extend([
            "Multi-factor detection improves accuracy",
            "Adaptive thresholds handle varying conditions better",
            "Combining rules reduces both false positives and negatives"
        ])
        
        return hypotheses
    
    async def _execute_experiment_with_timeout(self, design: Dict, 
                                              timeout: int) -> ExperimentResult:
        """Execute experiment with timeout control"""
        from scenario_runner import run_scenario, get_scenario_status
        
        start_time = time.time()
        
        # Start experiment
        run_scenario(design)
        
        # Monitor with timeout
        while time.time() - start_time < timeout:
            status = get_scenario_status()
            
            if not status['running']:
                # Experiment completed
                break
            
            # Checkpoint
            await asyncio.sleep(self.checkpoint_interval)
        
        # Collect results
        from llm_control_api_client import get_stats
        
        stats = get_stats()
        
        return ExperimentResult(
            experiment_id=design.get('name', 'unknown'),
            hypothesis=design.get('hypothesis', ''),
            duration=time.time() - start_time,
            success=True,
            metrics={
                'detection_rate': stats.get('fbs_detections', 0) / max(stats.get('total_records', 1), 1),
                'false_positives': stats.get('false_positives', 0)
            },
            raw_data=stats
        )
    
    def _extract_patterns_from_analysis(self, analysis: Dict):
        """Extract success and failure patterns"""
        if 'detailed_results' in analysis:
            results = analysis['detailed_results']
            
            # Success patterns
            if 'success_patterns' in results:
                self.context.success_patterns.extend(results['success_patterns'])
            
            # Failure patterns
            if 'failure_patterns' in results:
                self.context.failure_patterns.extend(results['failure_patterns'])
    
    def _update_performance_metrics(self, results: List[Dict]):
        """Update performance metrics from results"""
        if not results:
            return
        
        # Average metrics from results
        metrics = {}
        for result in results:
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Calculate averages
        for key, values in metrics.items():
            self.context.current_performance[key] = np.mean(values)
        
        # Track history
        self.context.performance_history.append({
            'iteration': self.context.iteration,
            'performance': self.context.current_performance.copy(),
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_insights_from_analysis(self, analysis: Dict) -> List[str]:
        """Generate insights from analysis"""
        insights = []
        
        if 'summary' in analysis:
            summary = analysis['summary']
            
            # Extract key findings
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if 'improvement' in key.lower():
                        insights.append(f"Found improvement opportunity: {value}")
                    elif 'pattern' in key.lower():
                        insights.append(f"Detected pattern: {value}")
        
        return insights
    
    def _check_convergence(self) -> bool:
        """Check if performance has converged"""
        if len(self.context.performance_history) < 3:
            return False
        
        # Check if performance is stable
        recent = self.context.performance_history[-3:]
        
        for metric in self.context.evaluation_metrics.keys():
            values = [h['performance'].get(metric, 0) for h in recent]
            
            if len(values) >= 3:
                variance = np.var(values)
                if variance > 0.01:  # Still changing
                    return False
                
                # Check if target is met
                target = self.context.evaluation_metrics[metric]['target']
                if values[-1] < target * self.config['convergence_threshold']:
                    return False
        
        return True
    
    def _identify_improvements(self) -> List[Dict]:
        """Identify improvement opportunities"""
        improvements = []
        
        # Based on performance gaps
        for metric, config in self.context.evaluation_metrics.items():
            current = self.context.current_performance.get(metric, 0)
            target = config['target']
            
            if current < target:
                gap = target - current
                improvements.append({
                    'type': 'performance_gap',
                    'metric': metric,
                    'gap': gap,
                    'priority': config['weight']
                })
        
        # Based on failure patterns
        for pattern in self.context.failure_patterns[-5:]:  # Recent failures
            improvements.append({
                'type': 'failure_pattern',
                'pattern': pattern,
                'priority': 0.8
            })
        
        # Sort by priority
        improvements.sort(key=lambda x: x['priority'], reverse=True)
        
        return improvements[:3]  # Top 3 improvements
    
    async def _create_custom_metrics(self) -> Dict:
        """Create custom metrics for improvement"""
        custom = {}
        
        # Analyze what's missing
        gaps = self._identify_metric_gaps()
        
        for gap in gaps:
            # Generate metric definition
            metric_def = {
                'name': gap['name'],
                'formula': gap['formula'],
                'target': gap['target'],
                'weight': 0.1
            }
            
            custom[gap['name']] = metric_def
        
        logger.info(f"Created {len(custom)} custom metrics")
        
        return custom
    
    def _identify_metric_gaps(self) -> List[Dict]:
        """Identify gaps in current metrics"""
        gaps = []
        
        # Check for time-based metrics
        if 'detection_variance' not in self.context.evaluation_metrics:
            gaps.append({
                'name': 'detection_variance',
                'formula': 'std(detection_times)',
                'target': 5,
                'reason': 'Ensure consistent detection times'
            })
        
        # Check for robustness metrics
        if 'robustness_score' not in self.context.custom_metrics:
            gaps.append({
                'name': 'robustness_score',
                'formula': 'success_rate_across_scenarios',
                'target': 0.9,
                'reason': 'Ensure detection works across scenarios'
            })
        
        return gaps
    
    async def _apply_improvement(self, improvement: Dict):
        """Apply an improvement"""
        if improvement['type'] == 'performance_gap':
            # Adjust rules for this metric
            logger.info(f"Adjusting for {improvement['metric']} gap of {improvement['gap']}")
            
        elif improvement['type'] == 'failure_pattern':
            # Address specific failure pattern
            logger.info(f"Addressing failure pattern: {improvement['pattern']}")
    
    async def _run_validation_suite(self) -> Dict:
        """Run comprehensive validation"""
        # Run multiple test scenarios
        from scenario_runner import ScenarioLibrary
        
        scenarios = [
            ScenarioLibrary.basic_fbs_attack(),
            ScenarioLibrary.identity_spoofing(),
            ScenarioLibrary.intermittent_attack()
        ]
        
        results = {}
        for scenario in scenarios:
            # Run scenario (mocked for now)
            await asyncio.sleep(1)
            
            # Collect metrics
            results[scenario['name']] = {
                'detection_rate': 0.92,
                'false_positive_rate': 0.06,
                'detection_time': 35,
                'accuracy': 0.89
            }
        
        # Average across scenarios
        averaged = {}
        for metric in self.context.evaluation_metrics.keys():
            values = [r.get(metric, 0) for r in results.values()]
            averaged[metric] = np.mean(values)
        
        return averaged
    
    def _get_best_rule(self) -> Dict:
        """Get the best performing rule"""
        if not self.context.generated_rules:
            return {}
        
        # For now, return the latest
        return self.context.generated_rules[-1]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on learnings"""
        recommendations = []
        
        # Based on performance
        if self.context.current_performance.get('detection_rate', 0) < 0.95:
            recommendations.append("Consider ensemble detection methods for higher detection rate")
        
        if self.context.current_performance.get('false_positive_rate', 1) > 0.05:
            recommendations.append("Implement adaptive thresholding to reduce false positives")
        
        # Based on insights
        if len(self.context.insights) > 10:
            recommendations.append("Document key insights for future reference")
        
        # Based on patterns
        if self.context.success_patterns:
            recommendations.append(f"Leverage success patterns: {self.context.success_patterns[0]}")
        
        return recommendations
    
    def _save_checkpoint(self):
        """Save workflow checkpoint"""
        checkpoint = {
            'context': self.context.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"checkpoints/checkpoint_{self.context.task_id}_{self.context.iteration}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {filename}")
    
    def _save_final_results(self, report: Dict):
        """Save final results"""
        filename = f"results/final_{self.context.task_id}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Final results saved: {filename}")
    
    async def _recover_from_error(self, error: Exception) -> Dict:
        """Attempt to recover from error"""
        logger.info(f"Attempting recovery from: {error}")
        
        # Try to load last checkpoint
        checkpoint_files = [f for f in os.listdir('checkpoints') 
                          if f.startswith(f'checkpoint_{self.context.task_id}')]
        
        if checkpoint_files:
            # Load most recent
            latest = sorted(checkpoint_files)[-1]
            with open(f'checkpoints/{latest}', 'r') as f:
                checkpoint = json.load(f)
            
            logger.info(f"Recovered from checkpoint: {latest}")
            
            # Continue from checkpoint
            # ... recovery logic ...
        
        # Return partial results
        return {
            'error': str(error),
            'partial_results': self.context.to_dict() if self.context else {},
            'recovery_attempted': True
        }