#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for FBS Detection System
Runs multiple scenarios and collects comprehensive metrics
Location: llm_fbs_utils/eval_scripts/evaluate_pipeline.py
"""

import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_control_api_client import get_mobiflow, get_stats, post_rule

from scenario_runner import ScenarioRunner, ScenarioLibrary, ScenarioConfig

from dataset_playback import DatasetPlayer
from llm_driver import LLMOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Complete evaluation pipeline for FBS detection system"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scenario_runner = ScenarioRunner()
        self.dataset_player = DatasetPlayer()
        self.llm_orchestrator = None
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': [],
            'metrics': {},
            'analysis': {}
        }
    
    def run_component_tests(self) -> Dict:
        """Test individual components"""
        logger.info("Running component tests...")
        
        results = {}
        
        # Test Query API
        logger.info("Testing Query API...")
        result = subprocess.run(
            ["python", "test_query_api.py"],
            capture_output=True,
            text=True
        )
        results['query_api'] = {
            'passed': result.returncode == 0,
            'output': result.stdout
        }
        
        # Test Rule Reload
        logger.info("Testing Rule Reload...")
        result = subprocess.run(
            ["python", "test_rule_reload.py"],
            capture_output=True,
            text=True
        )
        results['rule_reload'] = {
            'passed': result.returncode == 0,
            'output': result.stdout
        }
        
        return results
    
    def run_scenario(self, scenario_name: str, scenario_config: Dict) -> Dict:
        """Run a single scenario and collect metrics"""
        logger.info(f"Running scenario: {scenario_name}")
        
        start_time = time.time()
        
        # Clear previous data
        self.dataset_player.clear_sdl()
        
        # Inject baseline normal traffic
        normal_data = self.dataset_player.generate_synthetic_fbs_data(60)[:30]
        for record in normal_data:
            self.dataset_player.inject_mobiflow_record(record)
        
        time.sleep(2)
        
        # Get baseline stats
        baseline_stats = get_stats()
        
        # Run FBS scenario
        if scenario_name == 'synthetic':
            # Use synthetic data
            fbs_data = self.dataset_player.generate_synthetic_fbs_data(
                scenario_config['duration']
            )
            for record in fbs_data:
                self.dataset_player.inject_mobiflow_record(record)
                time.sleep(0.1)  # Simulate real-time
        else:

            # Run actual scenario (convert dict -> dataclass if needed)
            if isinstance(scenario_config, dict):
                scenario_config = ScenarioConfig.from_dict(scenario_config)
            self.scenario_runner.run_scenario(scenario_config)
        
        # Monitor detection
        detections = self.monitor_detection(scenario_config['duration'])
        
        # Collect final stats
        final_stats = get_stats()
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            baseline_stats,
            final_stats,
            detections,
            time.time() - start_time
        )
        
        return {
            'scenario': scenario_name,
            'config': scenario_config,
            'baseline_stats': baseline_stats,
            'final_stats': final_stats,
            'detections': detections,
            'metrics': metrics,
            'duration': time.time() - start_time
        }
    
    def monitor_detection(self, duration: int) -> List[Dict]:
        """Monitor for FBS detections during scenario"""
        detections = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get recent MobiFlow records
            records = get_mobiflow(n=50)
            
            # Check for FBS detections
            for record in records:
                if record.get('suspected_fbs'):
                    detection = {
                        'timestamp': record.get('timestamp'),
                        'time_offset': time.time() - start_time,
                        'ue_id': record.get('ue_id'),
                        'confidence': self.calculate_confidence(record),
                        'indicators': self.extract_indicators(record)
                    }
                    
                    # Avoid duplicates
                    if not any(d['timestamp'] == detection['timestamp'] 
                             for d in detections):
                        detections.append(detection)
                        logger.info(f"FBS detected at {detection['time_offset']:.1f}s")
            
            time.sleep(1)
        
        return detections
    
    def calculate_confidence(self, record: Dict) -> float:
        """Calculate detection confidence from record"""
        score = 0
        
        if record.get('attach_failures', 0) > 3:
            score += 0.3
        if record.get('auth_reject_count', 0) > 0:
            score += 0.3
        if record.get('cipher_downgrade'):
            score += 0.3
        if record.get('signal_anomaly'):
            score += 0.1
        
        return min(score, 1.0)
    
    def extract_indicators(self, record: Dict) -> List[str]:
        """Extract FBS indicators from record"""
        indicators = []
        
        if record.get('attach_failures', 0) > 0:
            indicators.append('attach_failures')
        if record.get('auth_reject_count', 0) > 0:
            indicators.append('auth_failures')
        if record.get('cipher_downgrade'):
            indicators.append('cipher_downgrade')
        if record.get('signal_anomaly'):
            indicators.append('signal_anomaly')
        if record.get('cell_reselection_count', 0) > 5:
            indicators.append('rapid_reselection')
        
        return indicators
    
    def calculate_metrics(self, baseline: Dict, final: Dict, 
                         detections: List, duration: float) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Detection metrics
        metrics['detection_count'] = len(detections)
        metrics['detection_rate'] = len(detections) > 0
        
        if detections:
            metrics['first_detection_time'] = min(d['time_offset'] for d in detections)
            metrics['avg_confidence'] = np.mean([d['confidence'] for d in detections])
        else:
            metrics['first_detection_time'] = None
            metrics['avg_confidence'] = 0
        
        # Performance metrics
        metrics['total_duration'] = duration
        metrics['records_processed'] = final.get('total_records', 0) - baseline.get('total_records', 0)
        
        # Calculate false positives (simplified)
        normal_period_detections = [d for d in detections if d['time_offset'] < 30]
        metrics['false_positives'] = len(normal_period_detections)
        
        # Detection accuracy (simplified)
        if detections:
            metrics['accuracy'] = 1.0 - (metrics['false_positives'] / len(detections))
        else:
            metrics['accuracy'] = 0
        
        return metrics
    
    def run_all_scenarios(self) -> None:
        """Run all predefined scenarios"""
        scenarios = [
            ('basic_fbs', ScenarioLibrary.basic_fbs_attack()),
            ('identity_spoofing', ScenarioLibrary.identity_spoofing()),
            ('intermittent', ScenarioLibrary.intermittent_attack()),
            ('synthetic', {'duration': 120})
        ]
        
        for name, config in scenarios:
            try:
                result = self.run_scenario(name, config)
                self.results['scenarios'].append(result)
                logger.info(f"Scenario {name} completed")
            except Exception as e:
                logger.error(f"Scenario {name} failed: {e}")
                self.results['scenarios'].append({
                    'scenario': name,
                    'error': str(e)
                })
    
    def test_rule_generation(self) -> Dict:
        """Test LLM rule generation"""
        logger.info("Testing LLM rule generation...")
        
        try:
            self.llm_orchestrator = LLMOrchestrator()
            
            # Generate rules from patterns
            patterns = {
                'indicators': ['attach_failures', 'cipher_downgrade'],
                'confidence': 0.8,
                'telemetry_summary': {'fbs_detections': 5}
            }
            
            rule_yaml = self.llm_orchestrator.generate.forward(patterns)
            
            # Test generated rule
            result = post_rule(rule_yaml)
            
            return {
                'success': result['status'] == 'success',
                'rule': rule_yaml,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Rule generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_results(self) -> None:
        """Analyze evaluation results"""
        logger.info("Analyzing results...")
        
        if not self.results['scenarios']:
            logger.warning("No scenarios to analyze")
            return
        
        # Extract metrics
        metrics_df = pd.DataFrame([
            s['metrics'] for s in self.results['scenarios'] 
            if 'metrics' in s
        ])
        
        if not metrics_df.empty:
            # Calculate summary statistics
            self.results['analysis']['summary'] = {
                'avg_detection_time': metrics_df['first_detection_time'].mean(),
                'detection_success_rate': metrics_df['detection_rate'].mean(),
                'avg_accuracy': metrics_df['accuracy'].mean(),
                'avg_false_positives': metrics_df['false_positives'].mean(),
                'total_scenarios': len(metrics_df)
            }
            
            # Performance analysis
            self.results['analysis']['performance'] = {
                'avg_duration': metrics_df['total_duration'].mean(),
                'total_records': metrics_df['records_processed'].sum(),
                'records_per_second': metrics_df['records_processed'].sum() / 
                                     metrics_df['total_duration'].sum()
            }
    
    def generate_plots(self) -> None:
        """Generate visualization plots"""
        logger.info("Generating plots...")
        
        if not self.results['scenarios']:
            return
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Detection times
        scenarios = [s['scenario'] for s in self.results['scenarios'] if 'metrics' in s]
        detection_times = [s['metrics'].get('first_detection_time', 0) 
                          for s in self.results['scenarios'] if 'metrics' in s]
        
        axes[0, 0].bar(scenarios, detection_times)
        axes[0, 0].set_title('Detection Time by Scenario')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Accuracy vs False Positives
        accuracies = [s['metrics'].get('accuracy', 0) 
                     for s in self.results['scenarios'] if 'metrics' in s]
        false_positives = [s['metrics'].get('false_positives', 0) 
                          for s in self.results['scenarios'] if 'metrics' in s]
        
        axes[0, 1].scatter(false_positives, accuracies, s=100)
        axes[0, 1].set_title('Accuracy vs False Positives')
        axes[0, 1].set_xlabel('False Positives')
        axes[0, 1].set_ylabel('Accuracy')
        
        # Plot 3: Detection confidence distribution
        all_confidences = []
        for s in self.results['scenarios']:
            if 'detections' in s:
                all_confidences.extend([d['confidence'] for d in s['detections']])
        
        if all_confidences:
            axes[1, 0].hist(all_confidences, bins=20, edgecolor='black')
            axes[1, 0].set_title('Detection Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Performance metrics
        durations = [s['metrics'].get('total_duration', 0) 
                    for s in self.results['scenarios'] if 'metrics' in s]
        records = [s['metrics'].get('records_processed', 0) 
                  for s in self.results['scenarios'] if 'metrics' in s]
        
        axes[1, 1].bar(scenarios, records)
        axes[1, 1].set_title('Records Processed by Scenario')
        axes[1, 1].set_ylabel('Records')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        
        plt.close()
    
    def plot_results(self) -> None:
        """Compatibility shim for tests expecting plot_results().
        Runs analysis, generates plots, and saves results."""
        try:
            self.analyze_results()
        except Exception as e:
            logger.warning("analyze_results failed: %s", e)
        try:
            self.generate_plots()
        except Exception as e:
            logger.warning("generate_plots failed: %s", e)
        try:
            self.save_results()
        except Exception as e:
            logger.warning("save_results failed: %s", e)
    
    def save_results(self) -> None:
        """Save evaluation results"""
        # Save JSON results
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {json_path}")
        
        # Generate Markdown report
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate evaluation report"""
        report_path = self.output_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# FBS Detection System Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            if 'summary' in self.results['analysis']:
                summary = self.results['analysis']['summary']
                f.write(f"- **Total Scenarios**: {summary['total_scenarios']}\n")
                f.write(f"- **Detection Success Rate**: {summary['detection_success_rate']:.1%}\n")
                f.write(f"- **Average Detection Time**: {summary['avg_detection_time']:.1f}s\n")
                f.write(f"- **Average Accuracy**: {summary['avg_accuracy']:.1%}\n")
                f.write(f"- **Average False Positives**: {summary['avg_false_positives']:.1f}\n\n")
            
            # Scenario Results
            f.write("## Scenario Results\n\n")
            for scenario in self.results['scenarios']:
                if 'error' in scenario:
                    f.write(f"### {scenario['scenario']} - FAILED\n")
                    f.write(f"Error: {scenario['error']}\n\n")
                elif 'metrics' in scenario:
                    metrics = scenario['metrics']
                    f.write(f"### {scenario['scenario']}\n")
                    f.write(f"- Detection Count: {metrics['detection_count']}\n")
                    f.write(f"- First Detection: {metrics.get('first_detection_time', 'N/A')}s\n")
                    f.write(f"- Accuracy: {metrics['accuracy']:.1%}\n")
                    f.write(f"- False Positives: {metrics['false_positives']}\n\n")
            
            # Performance
            f.write("## Performance\n\n")
            if 'performance' in self.results['analysis']:
                perf = self.results['analysis']['performance']
                f.write(f"- **Average Duration**: {perf['avg_duration']:.1f}s\n")
                f.write(f"- **Total Records**: {perf['total_records']}\n")
                f.write(f"- **Throughput**: {perf['records_per_second']:.1f} records/s\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            self.generate_recommendations(f)
        
        logger.info(f"Report saved to {report_path}")
    
    def generate_recommendations(self, f) -> None:
        """Generate recommendations based on results"""
        if 'summary' in self.results['analysis']:
            summary = self.results['analysis']['summary']
            
            if summary['avg_detection_time'] > 60:
                f.write("- ⚠️ Detection time is high. Consider optimizing rules or increasing monitoring frequency.\n")
            
            if summary['avg_accuracy'] < 0.8:
                f.write("- ⚠️ Accuracy is below 80%. Review and refine detection rules.\n")
            
            if summary['avg_false_positives'] > 2:
                f.write("- ⚠️ High false positive rate. Adjust rule thresholds.\n")
            
            if summary['detection_success_rate'] < 0.9:
                f.write("- ⚠️ Some scenarios not detected. Add more comprehensive rules.\n")
            
            if summary['detection_success_rate'] == 1.0 and summary['avg_accuracy'] > 0.9:
                f.write("- ✅ Excellent detection performance! System is working well.\n")

def main():
    parser = argparse.ArgumentParser(description='FBS Detection Evaluation Pipeline')
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--scenarios', nargs='+',
                       help='Specific scenarios to run')
    parser.add_argument('--skip-component-tests', action='store_true',
                       help='Skip component tests')
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM tests')
    parser.add_argument('--parallel', action='store_true',
                       help='Run scenarios in parallel')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EvaluationPipeline(args.output_dir)
    
    try:
        # Run component tests
        if not args.skip_component_tests:
            component_results = pipeline.run_component_tests()
            pipeline.results['components'] = component_results
        
        # Run scenarios
        if args.scenarios:
            # Run specific scenarios
            for scenario in args.scenarios:
                if hasattr(ScenarioLibrary, scenario):
                    config = getattr(ScenarioLibrary, scenario)()
                    result = pipeline.run_scenario(scenario, config)
                    pipeline.results['scenarios'].append(result)
        else:
            # Run all scenarios
            pipeline.run_all_scenarios()
        
        # Test LLM integration
        if not args.skip_llm:
            llm_result = pipeline.test_rule_generation()
            pipeline.results['llm_test'] = llm_result
        
        # Analyze results
        pipeline.analyze_results()
        
        # Generate visualizations
        pipeline.generate_plots()
        
        # Save results
        pipeline.save_results()
        
        logger.info("Evaluation complete!")
        
        # Print summary
        if 'summary' in pipeline.results['analysis']:
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            summary = pipeline.results['analysis']['summary']
            print(f"Detection Success Rate: {summary['detection_success_rate']:.1%}")
            print(f"Average Detection Time: {summary['avg_detection_time']:.1f}s")
            print(f"Average Accuracy: {summary['avg_accuracy']:.1%}")
            print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()