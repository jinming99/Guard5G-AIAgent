#!/usr/bin/env python3
"""
Enhanced DSPy Modules with Data Preprocessing and Context Awareness
Provides sophisticated data handling and LLM interaction for FBS detection
Location: llm_fbs_utils/dspy_modules/enhanced_modules.py
"""

import json
import yaml
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dspy
from dspy import Module, ChainOfThought, Signature, InputField, OutputField

logger = logging.getLogger(__name__)

# ============================================================================
# Data Preprocessing Utilities
# ============================================================================

class TelemetryPreprocessor:
    """
    Preprocesses raw telemetry data for LLM consumption
    Uses domain-specific techniques common in cellular network analysis
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.window_size = 10  # Sliding window for temporal features
        
    def preprocess_kpm_data(self, raw_kpm: Dict) -> Dict:
        """
        Preprocess Key Performance Metrics data
        
        Args:
            raw_kpm: Raw KPM data with numerical values
            
        Returns:
            Preprocessed data with context and statistical features
        """
        # Extract raw metrics
        rsrp = raw_kpm.get('rsrp', -100)  # Reference Signal Received Power
        rsrq = raw_kpm.get('rsrq', -15)   # Reference Signal Received Quality
        sinr = raw_kpm.get('sinr', 0)     # Signal-to-Interference-plus-Noise Ratio
        
        # Domain-specific preprocessing
        processed = {
            'raw_metrics': {
                'rsrp': rsrp,
                'rsrq': rsrq,
                'sinr': sinr
            },
            'signal_quality': self._categorize_signal_quality(rsrp, rsrq, sinr),
            'anomaly_indicators': self._detect_signal_anomalies(rsrp, rsrq, sinr),
            'normalized_metrics': self._normalize_metrics(raw_kpm),
            'context': self._add_domain_context(raw_kpm)
        }
        
        return processed
    
    def _categorize_signal_quality(self, rsrp: float, rsrq: float, sinr: float) -> str:
        """
        Categorize signal quality based on 3GPP thresholds
        """
        if rsrp >= -80 and rsrq >= -10 and sinr >= 20:
            return "excellent"
        elif rsrp >= -90 and rsrq >= -15 and sinr >= 13:
            return "good"
        elif rsrp >= -100 and rsrq >= -20 and sinr >= 0:
            return "fair"
        else:
            return "poor"
    
    def _detect_signal_anomalies(self, rsrp: float, rsrq: float, sinr: float) -> Dict:
        """
        Detect anomalies using domain knowledge
        """
        anomalies = {}
        
        # Sudden signal strength increase (potential FBS)
        anomalies['sudden_strength_increase'] = rsrp > -70  # Unusually strong
        
        # Signal quality mismatch
        anomalies['quality_mismatch'] = (rsrp > -80) and (sinr < 5)
        
        # Impossible values
        anomalies['impossible_values'] = rsrp > -30 or rsrp < -140
        
        return anomalies
    
    def _normalize_metrics(self, raw_data: Dict) -> Dict:
        """
        Normalize metrics to 0-1 range for LLM processing
        """
        normalized = {}
        
        # RSRP: typically -140 to -44 dBm
        if 'rsrp' in raw_data:
            normalized['rsrp_normalized'] = (raw_data['rsrp'] + 140) / 96.0
        
        # RSRQ: typically -20 to -3 dB
        if 'rsrq' in raw_data:
            normalized['rsrq_normalized'] = (raw_data['rsrq'] + 20) / 17.0
        
        # SINR: typically -20 to 30 dB
        if 'sinr' in raw_data:
            normalized['sinr_normalized'] = (raw_data['sinr'] + 20) / 50.0
        
        return normalized
    
    def _add_domain_context(self, raw_data: Dict) -> Dict:
        """
        Add domain-specific context for better LLM understanding
        """
        return {
            'measurement_type': '5G NR' if raw_data.get('nr_flag') else 'LTE',
            'frequency_band': self._get_frequency_band(raw_data.get('frequency', 0)),
            'cell_type': self._determine_cell_type(raw_data.get('cell_id', '')),
            'time_of_day': self._get_time_category()
        }
    
    def _get_frequency_band(self, freq: int) -> str:
        """Map frequency to band name"""
        if freq < 1000:
            return "low-band"
        elif freq < 6000:
            return "mid-band"
        else:
            return "high-band (mmWave)"
    
    def _determine_cell_type(self, cell_id: str) -> str:
        """Determine cell type from ID pattern"""
        if not cell_id:
            return "unknown"
        # Simple heuristic - in practice would use network topology
        if cell_id.startswith('9'):
            return "suspicious"
        return "normal"
    
    def _get_time_category(self) -> str:
        """Get time of day category"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 24:
            return "evening"
        else:
            return "night"
    
    def preprocess_mobiflow_batch(self, records: List[Dict]) -> Dict:
        """
        Preprocess batch of MobiFlow records with temporal analysis
        """
        if not records:
            return {'error': 'No records to process'}
        
        df = pd.DataFrame(records)
        
        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            # Calculate inter-event times
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
            
            # Rolling statistics
            for col in ['attach_failures', 'auth_reject_count']:
                if col in df.columns:
                    df[f'{col}_rolling_mean'] = df[col].rolling(
                        window=self.window_size, min_periods=1
                    ).mean()
                    df[f'{col}_rolling_std'] = df[col].rolling(
                        window=self.window_size, min_periods=1
                    ).std()
        
        # Statistical summary
        summary = {
            'total_records': int(len(df)),
            'time_span': float((df['timestamp'].max() - df['timestamp'].min()).total_seconds()) if 'timestamp' in df.columns else 0.0,
            'unique_ues': int(df['ue_id'].nunique()) if 'ue_id' in df.columns else 0,
            'unique_cells': int(df['cell_id'].nunique()) if 'cell_id' in df.columns else 0,
            'statistics': {}
        }
        
        # Per-field statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q50': float(df[col].quantile(0.50)),
                'q75': float(df[col].quantile(0.75))
            }
        
        # Pattern detection
        summary['patterns'] = self._detect_patterns(df)
        
        # Anomaly detection using statistical methods
        summary['anomalies'] = self._detect_statistical_anomalies(df)
        
        return summary
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect patterns in the data using domain-specific heuristics
        """
        patterns = {}
        
        # Attachment failure pattern
        if 'attach_failures' in df.columns:
            failure_rate = float((df['attach_failures'] > 0).mean())
            patterns['high_failure_rate'] = bool(failure_rate > 0.3)
            
            # Burst detection
            if len(df) > 10:
                failures = df['attach_failures'].values
                burst_detected = self._detect_bursts(failures)
                patterns['failure_bursts'] = bool(burst_detected)
        
        # Cell reselection pattern
        if 'cell_id' in df.columns:
            # Avoid numeric diff on string/object columns; count changes via shift comparison
            changes = (df['cell_id'] != df['cell_id'].shift()).fillna(False)
            # subtract 1 if first row counted as change
            cell_changes = int(max(changes.sum() - 1, 0))
            patterns['frequent_reselections'] = bool(cell_changes > len(df) * 0.2)
        
        # Cipher downgrade pattern
        if 'cipher_algo' in df.columns:
            null_cipher_ratio = float((df['cipher_algo'] == 'NULL').mean()) if 'cipher_algo' in df.columns else 0.0
            patterns['cipher_downgrade_detected'] = bool(null_cipher_ratio > 0)
        
        return patterns
    
    def _detect_bursts(self, values: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Detect burst patterns using change point detection
        """
        if len(values) < 10:
            return False
        
        # Simple burst detection using rolling std
        rolling_std = pd.Series(values).rolling(window=5, min_periods=1).std()
        return (rolling_std > threshold).any()
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect anomalies using statistical methods
        """
        anomalies = []
        
        # Z-score based anomaly detection
        numeric_cols = ['rsrp', 'rsrq', 'sinr', 'attach_failures']
        for col in numeric_cols:
            if col in df.columns and len(df) > 10:
                z_scores = np.abs(stats.zscore(df[col].fillna(0)))
                anomaly_mask = z_scores > 3
                
                if anomaly_mask.any():
                    # Cast indices to native Python ints for JSON serialization
                    anomaly_indices = [int(i) for i in df[anomaly_mask].index.tolist()]
                    anomalies.append({
                        'type': f'{col}_outlier',
                        'indices': anomaly_indices,
                        'severity': 'high' if z_scores.max() > 4 else 'medium'
                    })
        
        return anomalies

# ============================================================================
# Enhanced DSPy Modules
# ============================================================================

class QueryNetwork(Module):
    """
    Enhanced network query module with preprocessing
    """
    
    def __init__(self):
        super().__init__()
        self.preprocessor = TelemetryPreprocessor()
        self.cache = {}  # Cache for recent queries
        self.cache_ttl = 60  # Cache TTL in seconds
    
    def forward(self, query_type: str = "comprehensive", 
                ue_id: Optional[str] = None, 
                time_window: int = 300) -> Dict:
        """
        Query network with intelligent preprocessing
        
        Args:
            query_type: Type of query (comprehensive, targeted, diagnostic)
            ue_id: Specific UE to query
            time_window: Time window in seconds
        """
        # Check cache
        cache_key = f"{query_type}_{ue_id}_{time_window}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.info("Using cached data")
                return cached_data
        
        # Import here to avoid circular dependency
        from llm_control_api_client import get_kpm, get_mobiflow, get_stats
        
        result = {
            'query_type': query_type,
            'timestamp': datetime.now().isoformat(),
            'preprocessing_applied': True
        }
        
        if query_type == "comprehensive":
            # Get overall statistics
            stats = get_stats()
            result['statistics'] = stats
            
            # Get recent MobiFlow records
            records = get_mobiflow(n=min(time_window, 1000))
            if records:
                result['mobiflow_analysis'] = self.preprocessor.preprocess_mobiflow_batch(records)
            
            # Sample KPM from multiple UEs
            if ue_id:
                kpm = get_kpm(ue_id)
                result['kpm_analysis'] = self.preprocessor.preprocess_kpm_data(kpm)
        
        elif query_type == "targeted":
            # Focused query on specific UE
            if ue_id:
                kpm = get_kpm(ue_id)
                result['kpm_analysis'] = self.preprocessor.preprocess_kpm_data(kpm)
                
                # Get UE-specific MobiFlow
                records = get_mobiflow(ue_id, n=100)
                if records:
                    result['mobiflow_analysis'] = self.preprocessor.preprocess_mobiflow_batch(records)
        
        elif query_type == "diagnostic":
            # Deep diagnostic query
            result['diagnostic'] = self._run_diagnostics()
        
        # Cache result
        self.cache[cache_key] = (time.time(), result)
        
        return result
    
    def _run_diagnostics(self) -> Dict:
        """Run comprehensive diagnostics"""
        from llm_control_api_client import get_stats, get_fbs_suspects
        
        diag = {
            'system_health': get_stats(),
            'fbs_suspects': get_fbs_suspects(confidence_threshold=0.5),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add diagnostic analysis
        if diag['fbs_suspects']:
            diag['threat_level'] = 'high'
            diag['recommended_action'] = 'immediate_investigation'
        else:
            diag['threat_level'] = 'low'
            diag['recommended_action'] = 'continue_monitoring'
        
        return diag

class RuleGenerator(Module):
    """
    Enhanced rule generation with context awareness
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = ChainOfThought(RuleGenerationSignature)
        self.rule_templates = self._load_templates()
        self.validation_enabled = True
    
    def forward(self, analysis: Dict, patterns: Dict, 
                objective: str = "detect_fbs") -> str:
        """
        Generate rules based on analysis and patterns
        
        Args:
            analysis: Preprocessed network analysis
            patterns: Detected patterns
            objective: Detection objective
        """
        # Prepare context for LLM
        context = self._prepare_context(analysis, patterns, objective)
        
        # Generate rule using LLM
        result = self.predictor(
            context=json.dumps(context, indent=2),
            templates=yaml.dump(self.rule_templates),
            objective=objective
        )
        
        # Validate and refine
        if self.validation_enabled:
            validated_rule = self._validate_and_refine(result.rule_yaml)
            return validated_rule
        
        return result.rule_yaml
    
    def _prepare_context(self, analysis: Dict, patterns: Dict, 
                        objective: str) -> Dict:
        """Prepare context for rule generation"""
        context = {
            'objective': objective,
            'threat_indicators': [],
            'thresholds': {},
            'conditions': []
        }
        
        # Extract threat indicators from analysis
        if 'mobiflow_analysis' in analysis:
            mf = analysis['mobiflow_analysis']
            
            if 'patterns' in mf:
                for pattern, detected in mf['patterns'].items():
                    if bool(detected):
                        context['threat_indicators'].append(pattern)
            
            if 'statistics' in mf:
                # Calculate thresholds from statistics
                for field, stats in mf['statistics'].items():
                    if field in ['attach_failures', 'auth_reject_count']:
                        # Use high percentile / IQR-based threshold when available
                        q25 = stats.get('q25')
                        q75 = stats.get('q75')
                        mean = stats.get('mean')
                        std = stats.get('std', 0.0)
                        if q25 is not None and q75 is not None:
                            threshold = float(q75) + 1.5 * (float(q75) - float(q25))
                        elif mean is not None:
                            # Fallback to mean + 2*std if quartiles missing
                            threshold = float(mean) + 2.0 * float(std or 0.0)
                        else:
                            threshold = 3.0
                        context['thresholds'][field] = max(1, int(round(threshold)))
        
        # Add pattern-based conditions
        if patterns:
            for pattern_name, pattern_data in patterns.items():
                if pattern_name == 'high_failure_rate':
                    context['conditions'].append({
                        'type': 'threshold',
                        'field': 'attach_failures',
                        'operator': 'gte',
                        'value': context['thresholds'].get('attach_failures', 3)
                    })
        
        return context
    
    def _load_templates(self) -> Dict:
        """Load rule templates"""
        # Default templates - in practice, load from file
        return {
            'basic_fbs': {
                'name': 'FBS_Detection_Basic',
                'condition': {
                    'field': 'suspected_fbs',
                    'eq': True
                },
                'action': {
                    'type': 'alert',
                    'severity': 'high'
                }
            }
        }
    
    def _validate_and_refine(self, rule_yaml: str) -> str:
        """Validate and refine generated rule"""
        try:
            rule = yaml.safe_load(rule_yaml)
            
            # Ensure required fields
            if 'rules' not in rule:
                rule = {'rules': [rule]}
            
            for r in rule['rules']:
                if 'name' not in r:
                    r['name'] = f"Generated_Rule_{int(time.time())}"
                if 'priority' not in r:
                    r['priority'] = 5
                if 'action' not in r:
                    r['action'] = {'type': 'alert', 'severity': 'medium'}
            
            return yaml.dump(rule, default_flow_style=False)
            
        except Exception as e:
            logger.error(f"Rule validation failed: {e}")
            # Return safe default
            return yaml.dump(self.rule_templates['basic_fbs'])

class ExperimentDesigner(Module):
    """
    Design experiments with proper timing and coordination
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = ChainOfThought(ExperimentDesignSignature)
        self.safety_checks = True
        # Keep duration within safe bound to satisfy safety checks and tests
        self.max_duration = 120  # Maximum experiment duration
    
    def forward(self, hypothesis: str, current_performance: Dict,
                constraints: Optional[Dict] = None) -> Dict:
        """
        Design experiment based on hypothesis and current performance
        
        Args:
            hypothesis: What to test
            current_performance: Current detection performance
            constraints: Experimental constraints
        """
        # Default constraints
        if constraints is None:
            constraints = {
                'max_duration': self.max_duration,
                'max_power': 30,
                'allowed_bands': ['n78'],
                'safety_mode': True
            }
        
        # Prepare experiment context
        context = {
            'hypothesis': hypothesis,
            'current_metrics': current_performance,
            'constraints': constraints,
            'available_parameters': [
                'plmn', 'pci', 'tac', 'tx_power', 
                'cipher_algo', 'reject_probability'
            ]
        }
        
        # Generate experiment design
        result = self.predictor(
            context=json.dumps(context, indent=2),
            objective="test_detection_robustness"
        )
        
        # Parse and validate experiment
        experiment = self._parse_experiment(result.experiment_json)
        
        # Add safety checks
        if self.safety_checks:
            experiment = self._apply_safety_checks(experiment)
        
        # Add timing coordination
        experiment = self._add_timing_coordination(experiment)
        
        return experiment
    
    def _parse_experiment(self, json_str: str) -> Dict:
        """Parse experiment JSON with error handling"""
        try:
            exp = json.loads(json_str)
            
            # Ensure required fields
            exp.setdefault('name', f'Experiment_{int(time.time())}')
            exp.setdefault('duration', 60)
            exp.setdefault('mode', 'fbs')
            exp.setdefault('events', [])
            
            return exp
            
        except json.JSONDecodeError:
            # Return safe default
            return {
                'name': 'Default_Experiment',
                'duration': 60,
                'mode': 'fbs',
                'config': {
                    'plmn': '00199',
                    'pci': 999,
                    'tx_power': 20
                },
                'events': []
            }
    
    def _apply_safety_checks(self, experiment: Dict) -> Dict:
        """Apply safety constraints to experiment"""
        # Limit duration
        experiment['duration'] = min(experiment['duration'], self.max_duration)
        
        # Limit power
        if 'config' in experiment and 'tx_power' in experiment['config']:
            experiment['config']['tx_power'] = min(
                experiment['config']['tx_power'], 30
            )
        
        # Add safety flag
        experiment['safety_mode'] = True
        
        return experiment
    
    def _add_timing_coordination(self, experiment: Dict) -> Dict:
        """Add timing coordination for proper synchronization"""
        experiment['timing'] = {
            'start_delay': 5,  # Wait 5s before starting
            'checkpoint_interval': 10,  # Check status every 10s
            'timeout': experiment['duration'] + 60,  # Timeout buffer
            'sync_required': True
        }
        
        # Add checkpoint events
        checkpoints = []
        for t in range(0, experiment['duration'], 30):
            checkpoints.append({
                'time': t,
                'action': 'checkpoint',
                'collect_metrics': True
            })
        
        experiment['events'].extend(checkpoints)
        
        return experiment

class DataAnalyst(Module):
    """
    Generate and execute data analysis code
    """
    
    def __init__(self):
        super().__init__()
        self.predictor = ChainOfThought(AnalysisCodeSignature)
        self.safe_mode = True  # Restrict code execution
    
    def forward(self, data: Dict, analysis_goal: str,
                output_format: str = "summary") -> Dict:
        """
        Generate and execute analysis code
        
        Args:
            data: Preprocessed data to analyze
            analysis_goal: What to analyze
            output_format: Desired output format
        """
        # Generate analysis code
        code = self._generate_analysis_code(data, analysis_goal)
        
        # Execute safely
        if self.safe_mode:
            result = self._safe_execute(code, data)
        else:
            result = self._execute(code, data)
        
        # Format output
        formatted = self._format_output(result, output_format)
        
        return formatted
    
    def _generate_analysis_code(self, data: Dict, goal: str) -> str:
        """Generate analysis code using LLM"""
        context = {
            'data_structure': self._describe_data_structure(data),
            'available_libraries': ['pandas', 'numpy', 'scipy', 'sklearn'],
            'goal': goal
        }
        
        result = self.predictor(
            context=json.dumps(context, indent=2),
            goal=goal
        )
        
        return result.analysis_code
    
    def _describe_data_structure(self, data: Dict) -> Dict:
        """Describe data structure for LLM"""
        description = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                description[key] = {
                    'type': 'dict',
                    'keys': list(value.keys())
                }
            elif isinstance(value, list):
                description[key] = {
                    'type': 'list',
                    'length': len(value),
                    'sample': value[0] if value else None
                }
            else:
                description[key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]
                }
        
        return description
    
    def _safe_execute(self, code: str, data: Dict) -> Dict:
        """Execute code in restricted environment"""
        # Create safe namespace
        namespace = {
            'pd': pd,
            'np': np,
            'data': data,
            'results': {}
        }
        
        # Restrict dangerous operations
        restricted_code = code.replace('__', '').replace('import os', '').replace('import sys', '')
        
        try:
            exec(restricted_code, namespace)
            return namespace.get('results', {})
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {'error': str(e)}
    
    def _execute(self, code: str, data: Dict) -> Dict:
        """Execute code (unrestricted - use with caution)"""
        namespace = {'data': data, 'results': {}}
        exec(code, namespace)
        return namespace.get('results', {})
    
    def _format_output(self, result: Dict, format: str) -> Dict:
        """Format analysis output"""
        if format == "summary":
            return {
                'summary': result,
                'timestamp': datetime.now().isoformat()
            }
        elif format == "detailed":
            return {
                'detailed_results': result,
                'metadata': {
                    'analysis_time': datetime.now().isoformat(),
                    'format': format
                }
            }
        else:
            return result

# ============================================================================
# DSPy Signatures
# ============================================================================

class RuleGenerationSignature(Signature):
    """Generate detection rules from context"""
    context = InputField(desc="Analysis context with patterns and thresholds")
    templates = InputField(desc="Example rule templates")
    objective = InputField(desc="Detection objective")
    rule_yaml = OutputField(desc="Generated rule in YAML format")

class ExperimentDesignSignature(Signature):
    """Design experiments from hypothesis"""
    context = InputField(desc="Experimental context and constraints")
    objective = InputField(desc="Experimental objective")
    experiment_json = OutputField(desc="Experiment design in JSON format")

class AnalysisCodeSignature(Signature):
    """Generate analysis code"""
    context = InputField(desc="Data structure and analysis context")
    goal = InputField(desc="Analysis goal")
    analysis_code = OutputField(desc="Python analysis code")