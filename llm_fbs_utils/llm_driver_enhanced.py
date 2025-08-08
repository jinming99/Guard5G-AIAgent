#!/usr/bin/env python3
"""
Enhanced LLM Driver for 5G-Spector FBS Detection System
Integrates enhanced DSPy modules and workflow-aware agent
Location: llm_fbs_utils/llm_driver_enhanced.py
"""

import os
import sys
import json
import yaml
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhanced modules
from dspy_modules.enhanced_modules import (
    TelemetryPreprocessor, QueryNetwork, RuleGenerator,
    ExperimentDesigner, DataAnalyst
)
from agent.workflow_agent import WorkflowAwareAgent, WorkflowState
from llm_control_api_client import wait_for_services, get_stats
from scenario_runner import get_runner, ScenarioLibrary
from dataset_playback import DatasetPlayer

import dspy

class EnhancedLLMOrchestrator:
    """
    Enhanced orchestrator with full workflow support and preprocessing
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize enhanced orchestrator"""
        self.config = self._load_config(config_file)
        self.setup_dspy()
        
        # Initialize components
        self.preprocessor = TelemetryPreprocessor()
        self.query_module = QueryNetwork()
        self.rule_generator = RuleGenerator()
        self.experiment_designer = ExperimentDesigner()
        self.data_analyst = DataAnalyst()
        self.workflow_agent = WorkflowAwareAgent(self.config.get('workflow', {}))
        
        # Data management
        self.dataset_player = DatasetPlayer()
        self.scenario_runner = get_runner()
        
        # State tracking
        self.current_task = None
        self.results_history = []
        self.active_experiments = []
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Enhanced default configuration
        return {
            'llm': {
                'model': 'gpt-4',  # Use GPT-4 for better reasoning
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'apis': {
                'auditor': 'http://localhost:8090',
                'expert': 'http://localhost:8091'
            },
            'workflow': {
                'max_iterations': 10,
                'convergence_threshold': 0.95,
                'experiment_budget': 20,
                'parallel_experiments': True,
                'auto_save': True,
                'debug_mode': False
            },
            'preprocessing': {
                'enable': True,
                'window_size': 10,
                'anomaly_threshold': 3.0
            },
            'experiment': {
                'max_duration': 300,
                'safety_checks': True,
                'checkpoint_interval': 30
            }
        }
    
    def setup_dspy(self):
        """Configure DSPy with enhanced settings"""
        model = self.config['llm']['model']
        
        if 'gpt' in model:
            lm = dspy.OpenAI(
                model=model,
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens']
            )
        elif 'claude' in model:
            lm = dspy.Claude(
                model=model,
                temperature=self.config['llm']['temperature']
            )
        else:
            # Default to GPT-3.5
            lm = dspy.OpenAI(
                model='gpt-3.5-turbo',
                temperature=0.7
            )
        
        dspy.settings.configure(lm=lm)
        logger.info(f"DSPy configured with {model}")
    
    async def run_complete_detection_workflow(self, 
                                             task_description: Optional[str] = None) -> Dict:
        """
        Run the complete FBS detection workflow with all enhancements
        
        Args:
            task_description: High-level task description
            
        Returns:
            Comprehensive results and learnings
        """
        if not task_description:
            task_description = """
            Detect fake base station attacks in 5G network with:
            - Detection rate > 95%
            - False positive rate < 5%
            - Detection time < 30 seconds
            - Robust across different attack patterns
            """
        
        logger.info("Starting enhanced FBS detection workflow")
        logger.info(f"Task: {task_description}")
        
        # Ensure services are ready
        if not wait_for_services():
            logger.error("Services not ready")
            return {'error': 'Services not available'}
        
        # Run workflow through agent
        try:
            results = await self.workflow_agent.run_complete_workflow(task_description)
            
            # Store results
            self.results_history.append(results)
            
            # Generate enhanced report
            report = self._generate_enhanced_report(results)
            
            return report
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                'error': str(e),
                'partial_results': self.workflow_agent.context.to_dict() if self.workflow_agent.context else {}
            }
    
    def run_interactive_session(self):
        """
        Run interactive session with enhanced capabilities
        """
        print("\n" + "="*60)
        print("5G-Spector Enhanced LLM Detection System")
        print("="*60)
        print("\nAvailable commands:")
        print("  detect    - Run detection cycle with preprocessing")
        print("  workflow  - Run complete workflow")
        print("  analyze   - Analyze current data")
        print("  experiment - Design and run experiment")
        print("  generate  - Generate detection rules")
        print("  status    - Show current status")
        print("  help      - Show detailed help")
        print("  quit      - Exit")
        print()
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'detect':
                    self._interactive_detect()
                elif cmd == 'workflow':
                    self._interactive_workflow()
                elif cmd == 'analyze':
                    self._interactive_analyze()
                elif cmd == 'experiment':
                    self._interactive_experiment()
                elif cmd == 'generate':
                    self._interactive_generate()
                elif cmd == 'status':
                    self._show_status()
                elif cmd == 'help':
                    self._show_help()
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                logger.error(f"Command failed: {e}")
                print(f"Error: {e}")
    
    def _interactive_detect(self):
        """Interactive detection with preprocessing"""
        print("\n--- Detection Cycle ---")
        
        # Query network with preprocessing
        print("Querying network state...")
        network_data = self.query_module.forward(
            query_type="comprehensive",
            time_window=300
        )
        
        # Show preprocessed insights
        if 'mobiflow_analysis' in network_data:
            analysis = network_data['mobiflow_analysis']
            
            print(f"\nNetwork Analysis:")
            print(f"  Total records: {analysis.get('total_records', 0)}")
            print(f"  Time span: {analysis.get('time_span', 0):.1f} seconds")
            print(f"  Unique UEs: {analysis.get('unique_ues', 0)}")
            
            if 'patterns' in analysis:
                print(f"\nDetected Patterns:")
                for pattern, detected in analysis['patterns'].items():
                    if detected:
                        print(f"  ✓ {pattern}")
            
            if 'anomalies' in analysis:
                print(f"\nAnomalies: {len(analysis['anomalies'])} detected")
        
        # Generate detection recommendation
        print("\nGenerating detection rules...")
        patterns = network_data.get('mobiflow_analysis', {}).get('patterns', {})
        
        rule_yaml = self.rule_generator.forward(
            analysis=network_data,
            patterns=patterns,
            objective="detect_fbs"
        )
        
        print("\nGenerated Rule:")
        print(rule_yaml[:500] + "..." if len(rule_yaml) > 500 else rule_yaml)
        
        # Ask to deploy
        deploy = input("\nDeploy this rule? (y/n): ").lower() == 'y'
        if deploy:
            from llm_control_api_client import post_rule
            result = post_rule(rule_yaml)
            if result['status'] == 'success':
                print("✓ Rule deployed successfully")
            else:
                print(f"✗ Deployment failed: {result.get('message', 'Unknown error')}")
    
    def _interactive_workflow(self):
        """Run complete workflow interactively"""
        print("\n--- Complete Workflow ---")
        
        # Get task description
        print("Enter task description (or press Enter for default):")
        task = input("> ").strip()
        
        if not task:
            task = "Detect FBS attacks with high accuracy and low false positives"
        
        print(f"\nStarting workflow for: {task}")
        print("This may take several minutes...")
        
        # Run workflow
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.run_complete_detection_workflow(task)
        )
        
        # Show results
        self._display_workflow_results(results)
    
    def _interactive_analyze(self):
        """Interactive data analysis"""
        print("\n--- Data Analysis ---")
        
        # Get recent data
        from llm_control_api_client import get_mobiflow
        records = get_mobiflow(n=100)
        
        if not records:
            print("No data available")
            return
        
        # Preprocess
        print(f"Analyzing {len(records)} records...")
        analysis = self.preprocessor.preprocess_mobiflow_batch(records)
        
        # Generate analysis code
        print("\nWhat would you like to analyze?")
        print("1. Failure patterns")
        print("2. Temporal trends")
        print("3. Anomaly detection")
        print("4. Custom analysis")
        
        choice = input("Choice (1-4): ").strip()
        
        goals = {
            '1': "Analyze failure patterns and identify root causes",
            '2': "Analyze temporal trends and periodicity",
            '3': "Detect and characterize anomalies",
            '4': input("Enter analysis goal: ")
        }
        
        goal = goals.get(choice, goals['1'])
        
        # Generate and execute analysis
        print(f"\nAnalyzing: {goal}")
        
        result = self.data_analyst.forward(
            data={'records': records, 'preprocessing': analysis},
            analysis_goal=goal,
            output_format="detailed"
        )
        
        # Display results
        if 'detailed_results' in result:
            print("\nAnalysis Results:")
            for key, value in result['detailed_results'].items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
    
    def _interactive_experiment(self):
        """Design and run experiment interactively"""
        print("\n--- Experiment Design ---")
        
        # Get hypothesis
        print("Enter hypothesis to test:")
        hypothesis = input("> ").strip()
        
        if not hypothesis:
            hypothesis = "Lowering detection thresholds will improve detection rate"
        
        # Get current performance
        stats = get_stats()
        current_performance = {
            'detection_rate': stats.get('fbs_detections', 0) / max(stats.get('total_records', 1), 1),
            'false_positives': stats.get('false_positives', 0)
        }
        
        print(f"\nCurrent Performance:")
        print(f"  Detection rate: {current_performance['detection_rate']:.2%}")
        print(f"  False positives: {current_performance['false_positives']}")
        
        # Design experiment
        print(f"\nDesigning experiment for: {hypothesis}")
        
        experiment = self.experiment_designer.forward(
            hypothesis=hypothesis,
            current_performance=current_performance
        )
        
        print(f"\nExperiment Design:")
        print(f"  Name: {experiment.get('name', 'Unknown')}")
        print(f"  Duration: {experiment.get('duration', 0)} seconds")
        print(f"  Mode: {experiment.get('mode', 'unknown')}")
        
        if 'events' in experiment:
            print(f"  Events: {len(experiment['events'])}")
        
        # Ask to run
        run = input("\nRun this experiment? (y/n): ").lower() == 'y'
        if run:
            print("Running experiment...")
            
            # Save experiment
            exp_file = f"experiments/exp_{int(time.time())}.json"
            os.makedirs(os.path.dirname(exp_file), exist_ok=True)
            with open(exp_file, 'w') as f:
                json.dump(experiment, f, indent=2)
            
            # Run scenario
            from scenario_runner import run_scenario
            run_scenario(exp_file)
            
            print(f"✓ Experiment started. Duration: {experiment['duration']}s")
            print("Monitor progress with 'status' command")
    
    def _interactive_generate(self):
        """Generate rules interactively"""
        print("\n--- Rule Generation ---")
        
        # Query current state
        network_data = self.query_module.forward(query_type="targeted")
        
        # Get patterns
        print("Analyzing patterns...")
        patterns = {}
        
        if 'mobiflow_analysis' in network_data:
            patterns = network_data['mobiflow_analysis'].get('patterns', {})
        
        print(f"\nDetected patterns: {list(patterns.keys())}")
        
        # Generate rules
        print("\nGenerating optimized rules...")
        
        rule_yaml = self.rule_generator.forward(
            analysis=network_data,
            patterns=patterns,
            objective="optimize_detection"
        )
        
        # Parse and display
        try:
            rules = yaml.safe_load(rule_yaml)
            if 'rules' in rules:
                print(f"\nGenerated {len(rules['rules'])} rules:")
                for i, rule in enumerate(rules['rules'], 1):
                    print(f"\n{i}. {rule.get('name', 'Unnamed')}")
                    print(f"   Priority: {rule.get('priority', 'N/A')}")
                    print(f"   Condition: {rule.get('condition', {})}")
                    print(f"   Action: {rule.get('action', {}).get('type', 'unknown')}")
        except:
            print("\nGenerated rules (raw):")
            print(rule_yaml)
    
    def _show_status(self):
        """Show current system status"""
        print("\n--- System Status ---")
        
        # Get statistics
        stats = get_stats()
        
        print(f"\nNetwork Statistics:")
        print(f"  Total UEs: {stats.get('total_ues', 0)}")
        print(f"  Active cells: {len(stats.get('active_cells', []))}")
        print(f"  Total records: {stats.get('total_records', 0)}")
        print(f"  FBS detections: {stats.get('fbs_detections', 0)}")
        
        # Workflow status
        if self.workflow_agent.context:
            ctx = self.workflow_agent.context
            print(f"\nWorkflow Status:")
            print(f"  Task ID: {ctx.task_id}")
            print(f"  State: {ctx.current_state.value}")
            print(f"  Iteration: {ctx.iteration}/{ctx.max_iterations}")
            
            if ctx.current_performance:
                print(f"\nCurrent Performance:")
                for metric, value in ctx.current_performance.items():
                    print(f"  {metric}: {value:.3f}")
        
        # Active experiments
        if self.active_experiments:
            print(f"\nActive Experiments: {len(self.active_experiments)}")
            for exp in self.active_experiments[-3:]:
                print(f"  - {exp.get('name', 'Unknown')}")
    
    def _show_help(self):
        """Show detailed help"""
        print("\n--- Help ---")
        print("""
Enhanced LLM Detection System Commands:

detect
  Run a detection cycle with intelligent preprocessing.
  - Queries network state
  - Applies statistical analysis and anomaly detection
  - Generates optimized detection rules
  - Optionally deploys rules to the system

workflow
  Execute the complete detection workflow.
  - Understands task requirements
  - Assesses baseline performance
  - Iteratively improves through experiments
  - Validates final solution
  - Generates comprehensive report

analyze
  Perform deep analysis on network data.
  - Failure pattern analysis
  - Temporal trend detection
  - Anomaly characterization
  - Custom analysis with LLM-generated code

experiment
  Design and execute experiments.
  - Hypothesis-driven testing
  - Automated experiment design
  - Safety checks and timing control
  - Performance comparison

generate
  Generate optimized detection rules.
  - Pattern-based rule generation
  - Context-aware thresholds
  - Multi-condition rules
  - Validation and refinement

status
  Display current system status.
  - Network statistics
  - Workflow progress
  - Performance metrics
  - Active experiments

Tips:
- Start with 'workflow' for automated optimization
- Use 'analyze' to understand your data
- Use 'experiment' to test specific hypotheses
- Monitor progress with 'status'
""")
    
    def _display_workflow_results(self, results: Dict):
        """Display workflow results in readable format"""
        print("\n" + "="*60)
        print("WORKFLOW RESULTS")
        print("="*60)
        
        if 'error' in results:
            print(f"\n✗ Error: {results['error']}")
            return
        
        # Summary
        print(f"\nTask ID: {results.get('task_id', 'Unknown')}")
        print(f"Duration: {results.get('duration', 0):.1f} seconds")
        print(f"Iterations: {results.get('iterations', 0)}")
        print(f"Rules generated: {results.get('generated_rules', 0)}")
        print(f"Experiments run: {results.get('experiments_run', 0)}")
        
        # Performance improvement
        if 'improvement' in results:
            print("\nPerformance Improvement:")
            for metric, change in results['improvement'].items():
                symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"  {metric}: {symbol} {abs(change):.3f}")
        
        # Final performance
        if 'final_performance' in results:
            print("\nFinal Performance:")
            for metric, value in results['final_performance'].items():
                print(f"  {metric}: {value:.3f}")
        
        # Key insights
        if 'insights' in results and results['insights']:
            print(f"\nKey Insights ({len(results['insights'])} total):")
            for insight in results['insights'][-5:]:  # Last 5
                print(f"  • {insight}")
        
        # Recommendations
        if 'recommendations' in results:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  → {rec}")
        
        print("\n" + "="*60)
    
    def _generate_enhanced_report(self, results: Dict) -> Dict:
        """Generate enhanced report with visualizations and insights"""
        report = results.copy()
        
        # Add analysis summary
        report['analysis_summary'] = self._generate_analysis_summary(results)
        
        # Add performance trends
        if self.workflow_agent.context and self.workflow_agent.context.performance_history:
            report['performance_trends'] = self._analyze_performance_trends(
                self.workflow_agent.context.performance_history
            )
        
        # Add pattern summary
        if self.workflow_agent.context:
            report['pattern_summary'] = {
                'success_patterns': len(self.workflow_agent.context.success_patterns),
                'failure_patterns': len(self.workflow_agent.context.failure_patterns),
                'insights_generated': len(self.workflow_agent.context.insights)
            }
        
        return report
    
    def _generate_analysis_summary(self, results: Dict) -> Dict:
        """Generate analysis summary"""
        summary = {
            'success': 'error' not in results,
            'convergence_achieved': False,
            'targets_met': {}
        }
        
        # Check target achievement
        if 'final_performance' in results:
            targets = {
                'detection_rate': 0.95,
                'false_positive_rate': 0.05,
                'detection_time': 30,
                'accuracy': 0.9
            }
            
            for metric, target in targets.items():
                if metric in results['final_performance']:
                    value = results['final_performance'][metric]
                    if metric == 'false_positive_rate' or metric == 'detection_time':
                        met = value <= target
                    else:
                        met = value >= target
                    summary['targets_met'][metric] = met
        
        # Check convergence
        if self.workflow_agent.context:
            summary['convergence_achieved'] = self.workflow_agent._check_convergence()
        
        return summary
    
    def _analyze_performance_trends(self, history: List[Dict]) -> Dict:
        """Analyze performance trends over iterations"""
        if not history:
            return {}
        
        df = pd.DataFrame([h['performance'] for h in history])
        
        trends = {}
        for col in df.columns:
            if len(df[col]) > 1:
                # Calculate trend
                x = np.arange(len(df))
                y = df[col].values
                
                # Simple linear regression
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                
                trends[col] = {
                    'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
                    'rate': float(slope),
                    'final_value': float(y[-1]),
                    'best_value': float(np.max(y)) if slope > 0 else float(np.min(y))
                }
        
        return trends

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced LLM Driver for 5G-Spector FBS Detection'
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['detect', 'workflow', 'analyze', 'experiment', 'interactive'],
        default='interactive',
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--task',
        default='Detect FBS attacks with high accuracy',
        help='Task description for workflow'
    )
    
    parser.add_argument(
        '--hypothesis',
        help='Hypothesis for experiment'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize orchestrator
    orchestrator = EnhancedLLMOrchestrator(args.config)
    
    # Execute command
    if args.command == 'interactive':
        orchestrator.run_interactive_session()
    
    elif args.command == 'workflow':
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            orchestrator.run_complete_detection_workflow(args.task)
        )
        print(json.dumps(results, indent=2, default=str))
    
    elif args.command == 'detect':
        # Run single detection cycle
        network_data = orchestrator.query_module.forward(
            query_type="comprehensive"
        )
        
        patterns = network_data.get('mobiflow_analysis', {}).get('patterns', {})
        
        rule = orchestrator.rule_generator.forward(
            analysis=network_data,
            patterns=patterns
        )
        
        print(rule)
    
    elif args.command == 'analyze':
        # Run analysis
        from llm_control_api_client import get_mobiflow
        records = get_mobiflow(n=100)
        
        analysis = orchestrator.preprocessor.preprocess_mobiflow_batch(records)
        print(json.dumps(analysis, indent=2, default=str))
    
    elif args.command == 'experiment':
        # Design experiment
        hypothesis = args.hypothesis or "Test detection improvement"
        
        experiment = orchestrator.experiment_designer.forward(
            hypothesis=hypothesis,
            current_performance={}
        )
        
        print(json.dumps(experiment, indent=2))

if __name__ == "__main__":
    main()