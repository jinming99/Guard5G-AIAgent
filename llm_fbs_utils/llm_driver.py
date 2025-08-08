#!/usr/bin/env python3
"""
LLM Driver for 5G-Spector FBS Detection System
Bridge version that maintains backward compatibility while using enhanced modules
Location: llm_fbs_utils/llm_driver.py
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Import Strategy: Try enhanced modules first, fallback to basic
# ============================================================================

try:
    # Try to import enhanced modules


    try:
        from dspy_modules.enhanced_modules import (
            QueryNetwork, RuleGenerator, ExperimentDesigner, DataAnalyst
        )
    except ImportError:
        # fallback import path for flat layouts
        from enhanced_modules import (
            QueryNetwork, RuleGenerator, ExperimentDesigner, DataAnalyst
        )
    try:
        from agent.workflow_agent import WorkflowAwareAgent
    except ImportError:
        from workflow_agent import WorkflowAwareAgent
    ENHANCED_MODE = True


    logger.info("Using enhanced DSPy modules with preprocessing")
except ImportError:
    # Fallback to basic implementation
    logger.warning("Enhanced modules not found, using basic implementation")
    ENHANCED_MODE = False
    
    # Import DSPy
    try:
        import dspy
        from dspy import Module, ChainOfThought, Signature
    except ImportError:
        print("Please install DSPy: pip install dspy-ai")
        sys.exit(1)
    
    # Use basic implementations from original
    from llm_control_api_client import get_kpm, get_mobiflow, post_rule, get_stats
    from scenario_runner import run_scenario, stop_scenario

# ============================================================================
# Backward Compatible DSPy Modules (if enhanced not available)
# ============================================================================

if not ENHANCED_MODE:
    class QueryNetwork(Module):
        """Basic network query module for backward compatibility"""
        def forward(self, ue_id: Optional[str] = None) -> Dict:
            try:
                if ue_id:
                    kpm = get_kpm(ue_id)
                    mobiflow = get_mobiflow(ue_id)
                    return {
                        'ue_id': ue_id,
                        'kpm': kpm,
                        'mobiflow': mobiflow,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    stats = get_stats()
                    recent = get_mobiflow(n=50)
                    return {
                        'stats': stats,
                        'recent_flows': recent,
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Failed to query network: {e}")
                return {'error': str(e)}

    class RuleGenerator(Module):
        """Basic rule generation module"""
        def forward(self, patterns: Dict) -> str:
            # Simple rule generation
            rule = {
                'rules': [{
                    'name': f'Generated_Rule_{int(time.time())}',
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
            return yaml.dump(rule)

    class ExperimentDesigner(Module):
        """Basic experiment designer"""
        def forward(self, hypothesis: str) -> Dict:
            return {
                'name': f'Experiment_{int(time.time())}',
                'mode': 'fbs',
                'duration': 60,
                'config': {
                    'plmn': '00199',
                    'pci': 999,
                    'tac': 999
                }
            }

    class DataAnalyst(Module):
        """Basic data analyst"""
        def forward(self, goal: str) -> str:
            return f"# Analysis for: {goal}\nimport pandas as pd\n# Analysis code here"

# ============================================================================
# Main Orchestrator (Compatible with both modes)
# ============================================================================

class LLMOrchestrator:
    """
    Main orchestration class that works with both basic and enhanced modules
    Maintains backward compatibility with existing scripts
    """
    
    def __init__(self, config_file: Optional[str] = None, use_enhanced: Optional[bool] = None):
        self.config = self._load_config(config_file)
        
        # Determine which mode to use
        if use_enhanced is None:
            self.use_enhanced = ENHANCED_MODE
        else:
            self.use_enhanced = use_enhanced and ENHANCED_MODE
        
        if self.use_enhanced:
            logger.info("Initializing with enhanced workflow-aware agent")
            self.workflow_agent = WorkflowAwareAgent(self.config)
            
        # Always initialize basic modules for compatibility
        self.query = QueryNetwork()
        self.generate = RuleGenerator() 
        self.experiment = ExperimentDesigner()
        self.data_analysis = DataAnalyst()
        
        # Setup DSPy (used by both flows when available)
        self.setup_dspy()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        return {
            'llm': {
                'model': 'gpt-3.5-turbo',
                'temperature': 0.7
            },
            'apis': {
                'auditor': 'http://localhost:8090',
                'expert': 'http://localhost:8091'
            },
            'experiment': {
                'max_duration': 300,
                'safety_checks': True
            }
        }
    
    def setup_dspy(self):
        """Configure DSPy with LLM"""
        # Configure DSPy only if available and enabled in config
        if not self.config.get('llm', {}).get('enable_dspy', True):
            return
        try:
            import dspy
        except ImportError:
            logger.warning("DSPy not installed; skipping DSPy configuration")
            return
        
        model = self.config['llm']['model']
        
        if 'gpt' in model:
            lm = dspy.OpenAI(
                model=model,
                temperature=self.config['llm']['temperature']
            )
        else:
            lm = dspy.OpenAI(model='gpt-3.5-turbo')
        
        dspy.settings.configure(lm=lm)
        logger.info(f"DSPy configured with {model}")
    
    def run_detection_cycle(self, ue_id: Optional[str] = None):
        """
        Run detection cycle - compatible with existing scripts
        Uses enhanced mode if available, otherwise basic
        """
        if self.use_enhanced:
            # Use enhanced workflow
            logger.info("Running enhanced detection cycle with preprocessing...")
            
            # Query with preprocessing
            state = self.query.forward(
                query_type="comprehensive",
                ue_id=ue_id,
                time_window=300
            )
            
            # Analyze with enhanced capabilities
            if 'mobiflow_analysis' in state:
                analysis = {
                    'fbs_detected': state['mobiflow_analysis'].get('patterns', {}).get('cipher_downgrade_detected', False),
                    'indicators': list(state['mobiflow_analysis'].get('patterns', {}).keys()),
                    'confidence': 0.8 if state['mobiflow_analysis'].get('anomalies') else 0.3
                }
            else:
                analysis = {'fbs_detected': False, 'indicators': [], 'confidence': 0}
            
            # Generate rules with context
            if analysis['fbs_detected']:
                patterns = state.get('mobiflow_analysis', {}).get('patterns', {})
                rule_yaml = self.generate.forward(
                    analysis=state,
                    patterns=patterns,
                    objective="detect_fbs"
                )
                
                # Deploy rules
                from llm_control_api_client import post_rule
                result = post_rule(rule_yaml)
                logger.info(f"Rule deployment: {result.get('status')}")
            
        else:
            # Use basic implementation
            logger.info("Running basic detection cycle...")
            
            # Query network state
            state = self.query(ue_id)
            
            if 'error' in state:
                logger.error(f"Failed to query network: {state['error']}")
                return {'error': state['error']}
            
            # Basic analysis
            analysis = self._basic_analyze(state)
            
            # Generate rules if needed
            if analysis['fbs_detected']:
                patterns = {'basic_detection': True}
                rule_yaml = self.generate(patterns)
                
                from llm_control_api_client import post_rule
                result = post_rule(rule_yaml)
                logger.info(f"Rules deployed: {result.get('status')}")
        
        return analysis
    
    def _basic_analyze(self, state: Dict) -> Dict:
        """Basic analysis for backward compatibility"""
        analysis = {
            'fbs_detected': False,
            'indicators': [],
            'confidence': 0.0
        }
        
        # Simple detection logic
        if 'stats' in state:
            if state['stats'].get('fbs_detections', 0) > 0:
                analysis['fbs_detected'] = True
                analysis['confidence'] = 0.8
                analysis['indicators'].append('fbs_detections')
        
        if 'recent_flows' in state:
            fbs_records = [r for r in state['recent_flows'] 
                          if r.get('suspected_fbs')]
            if fbs_records:
                analysis['fbs_detected'] = True
                analysis['confidence'] = len(fbs_records) / len(state['recent_flows'])
                analysis['indicators'].append('suspected_fbs_records')
        
        return analysis
    
    def run_experiment(self, hypothesis: str):
        """
        Run experiment - compatible with existing scripts
        """
        if self.use_enhanced:
            logger.info("Designing experiment with safety checks and timing coordination...")
            scenario = self.experiment.forward(
                hypothesis=hypothesis,
                current_performance={'detection_rate': 0.7},  # Mock for now
                constraints={
                    'max_duration': self.config['experiment']['max_duration'],
                    'safety_mode': self.config['experiment']['safety_checks']
                }
            )
        else:
            logger.info("Designing basic experiment...")
            scenario = self.experiment(hypothesis)
        
        # Save scenario
        scenario_file = f"experiments/scenario_{int(time.time())}.json"
        os.makedirs(os.path.dirname(scenario_file), exist_ok=True)
        with open(scenario_file, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        logger.info(f"Starting experiment: {scenario.get('name')}")
        
        # Run scenario
        from scenario_runner import run_scenario, stop_scenario, get_scenario_status
        run_scenario(scenario_file)
        
        # Monitor execution
        duration = scenario.get('duration', 60)
        start_time = time.time()
        detections = []
        
        while time.time() - start_time < duration:
            time.sleep(10)
            
            # Check status
            status = get_scenario_status()
            if not status['running']:
                break
            
            # Query and analyze
            state = self.query()
            if self.use_enhanced:
                # Enhanced analysis
                if hasattr(state, 'get') and 'mobiflow_analysis' in state:
                    if state['mobiflow_analysis'].get('patterns', {}).get('suspected_fbs'):
                        detections.append(time.time() - start_time)
            else:
                # Basic analysis
                analysis = self._basic_analyze(state)
                if analysis['fbs_detected']:
                    detections.append(time.time() - start_time)
        
        # Stop experiment
        stop_scenario()
        
        return {
            'scenario': scenario,
            'duration': time.time() - start_time,
            'detections': detections,
            'detection_rate': len(detections) > 0
        }
    
    def generate_analysis_code(self, goal: str):
        """Generate analysis code - compatible with existing scripts"""
        if self.use_enhanced:
            logger.info("Generating sophisticated analysis code...")
            
            # Get current data
            state = self.query.forward(query_type="diagnostic")
            
            # Generate code with context
            code = self.data_analysis.forward(
                data=state,
                analysis_goal=goal,
                output_format="summary"
            )
            
            if isinstance(code, dict) and 'summary' in code:
                # Extract code from result
                return code.get('summary', {}).get('code', '# Analysis code')
            return code
        else:
            logger.info("Generating basic analysis code...")
            return self.data_analysis(goal)
    
    async def run_complete_workflow(self, task_description: str):
        """
        Run complete enhanced workflow if available
        This is the new advanced feature
        """
        if not self.use_enhanced:
            logger.warning("Enhanced workflow not available, running basic detection")
            # Fallback to basic detection
            result = self.run_detection_cycle()
            return {
                'mode': 'basic',
                'result': result,
                'message': 'Enhanced workflow not available'
            }
        
        logger.info("Starting complete workflow with iterative improvement...")
        return await self.workflow_agent.run_complete_workflow(task_description)

# ============================================================================
# CLI Interface (Backward Compatible)
# ============================================================================

def main():
    """Main CLI interface - maintains backward compatibility"""
    parser = argparse.ArgumentParser(
        description='LLM Driver for 5G-Spector FBS Detection'
    )
    
    parser.add_argument(
        'command',
        choices=['detect', 'experiment', 'analyze', 'interactive', 'workflow'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--ue-id',
        help='UE identifier for targeted detection'
    )
    
    parser.add_argument(
        '--hypothesis',
        default='FBS causes authentication failures',
        help='Hypothesis for experiment'
    )
    
    parser.add_argument(
        '--goal',
        default='Count attach failures per UE',
        help='Analysis goal'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file'
    )
    
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Force use of enhanced modules (if available)'
    )
    
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Force use of basic modules'
    )
    
    parser.add_argument(
        '--task',
        default='Detect fake base stations in 5G network',
        help='Task description for workflow mode'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    use_enhanced = None
    if args.enhanced:
        use_enhanced = True
    elif args.basic:
        use_enhanced = False
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator(args.config, use_enhanced)
    
    # Execute command (backward compatible)
    if args.command == 'detect':
        result = orchestrator.run_detection_cycle(args.ue_id)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'experiment':
        result = orchestrator.run_experiment(args.hypothesis)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == 'analyze':
        code = orchestrator.generate_analysis_code(args.goal)
        print(code)
    
    elif args.command == 'workflow':
        # New enhanced workflow command
        if orchestrator.use_enhanced:
            # Run async workflow
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                orchestrator.run_complete_workflow(args.task)
            )
            print(json.dumps(result, indent=2, default=str))
        else:
            print("Enhanced workflow not available. Please ensure enhanced modules are installed.")
            print("Run: pip install -r requirements.txt")
            sys.exit(1)
    
    elif args.command == 'interactive':
        # Interactive mode (backward compatible)
        print("5G-Spector LLM Driver - Interactive Mode")
        print(f"Mode: {'Enhanced' if orchestrator.use_enhanced else 'Basic'}")
        print("Commands: detect, experiment, analyze, workflow, quit")
        
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'detect':
                result = orchestrator.run_detection_cycle()
                print(json.dumps(result, indent=2))
            elif cmd == 'experiment':
                hyp = input("Hypothesis: ")
                result = orchestrator.run_experiment(hyp)
                print(json.dumps(result, indent=2, default=str))
            elif cmd == 'analyze':
                goal = input("Analysis goal: ")
                code = orchestrator.generate_analysis_code(goal)
                print(code)
            elif cmd == 'workflow':
                if orchestrator.use_enhanced:
                    task = input("Task description: ")
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        orchestrator.run_complete_workflow(task)
                    )
                    print(json.dumps(result, indent=2, default=str))
                else:
                    print("Enhanced workflow not available in basic mode")
            else:
                print("Unknown command")

if __name__ == "__main__":
    main()