#!/usr/bin/env python3
"""
Scenario Runner for FBS Experiments
Interfaces with OAI-5G to run attack scenarios
Location: llm_fbs_utils/scenario_runner.py
"""

import os
import json
import subprocess
import time
import logging
import threading
import paramiko
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration
OAI_HOST = os.getenv('OAI_HOST', 'localhost')
OAI_USER = os.getenv('OAI_USER', 'root')
OAI_PATH = os.getenv('OAI_PATH', '/opt/oai-5g')
DOCKER_CONTAINER = os.getenv('OAI_CONTAINER', 'oai-gnb')

@dataclass
class ScenarioConfig:
    """Scenario configuration"""
    name: str
    mode: str
    duration: int
    config: Dict[str, Any]
    events: List[Dict[str, Any]]
    
    @classmethod
    def from_file(cls, filepath: str):
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        return cls(
            name=data.get('name', 'Unnamed Scenario'),
            mode=data.get('mode', 'fbs'),
            duration=data.get('duration', 60),
            config=data.get('config', {}),
            events=data.get('events', [])
        )

class ScenarioRunner:
    """Manages FBS scenario execution"""
    
    def __init__(self, use_docker: bool = True, use_ssh: bool = False):
        self.use_docker = use_docker
        self.use_ssh = use_ssh
        self.running_scenario = None
        self.scenario_thread = None
        self.stop_event = threading.Event()
        
        # SSH client for remote execution
        self.ssh_client = None
        if use_ssh:
            self._setup_ssh()
    
    def _setup_ssh(self):
        """Setup SSH connection"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(OAI_HOST, username=OAI_USER)
            logger.info(f"SSH connected to {OAI_HOST}")
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            self.use_ssh = False
    
    def _execute_command(self, command: str) -> tuple:
        """Execute command locally, via Docker, or SSH"""
        try:
            if self.use_ssh and self.ssh_client:
                # Execute via SSH
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                return stdout.read().decode(), stderr.read().decode()
            
            elif self.use_docker:
                # Execute in Docker container
                docker_cmd = f"docker exec {DOCKER_CONTAINER} {command}"

                result = subprocess.run(
                    ["docker", "exec", DOCKER_CONTAINER] + command if isinstance(command, list) else ["bash","-lc", command],
                    shell=False, capture_output=True, text=True
                )

                return result.stdout, result.stderr
            
            else:
                # Execute locally

                cmd = command if isinstance(command, list) else ["bash","-lc", command]
                result = subprocess.run(cmd, shell=False, capture_output=True, text=True)

                return result.stdout, result.stderr
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return "", str(e)
    
    def start_fbs(self, config: Dict[str, Any]) -> bool:
        """Start fake base station"""
        logger.info("Starting FBS with config:")
        logger.info(json.dumps(config, indent=2))
        
        # Build launch command
        script_path = f"{OAI_PATH}/tools/fbs_scenarios/launch_fbs.sh"
        
        # Create temporary config file
        config_file = f"/tmp/fbs_config_{int(time.time())}.json"
        config_json = json.dumps(config)
        
        if self.use_ssh:
            # Write config via SSH
            sftp = self.ssh_client.open_sftp()
            with sftp.file(config_file, 'w') as fp:
                fp.write(config_json)
            sftp.close()
        else:
            with open(config_file, 'w') as f:
                json.dump(config, f)
        
        # Launch FBS
        command = f"{script_path} start {config_file}"
        stdout, stderr = self._execute_command(command)
        
        if "FBS started" in stdout:
            logger.info("FBS started successfully")
            return True
        else:
            logger.error(f"FBS start failed: {stderr}")
            return False
    
    def stop_fbs(self) -> bool:
        """Stop fake base station"""
        logger.info("Stopping FBS...")
        
        script_path = f"{OAI_PATH}/tools/fbs_scenarios/launch_fbs.sh"
        command = f"{script_path} stop"
        
        stdout, stderr = self._execute_command(command)
        
        if "FBS stopped" in stdout or "not running" in stdout:
            logger.info("FBS stopped")
            return True
        else:
            logger.error(f"FBS stop failed: {stderr}")
            return False
    
    def configure_fbs(self, param: str, value: Any) -> bool:
        """Configure FBS parameter"""
        logger.info(f"Configuring FBS: {param} = {value}")
        
        script_path = f"{OAI_PATH}/tools/fbs_scenarios/launch_fbs.sh"
        command = [script_path, "configure", str(param), str(value)]

        
        stdout, stderr = self._execute_command(command)
        
        if "Configuration updated" in stdout:
            logger.info(f"FBS configured: {param} = {value}")
            return True
        else:
            logger.error(f"Configuration failed: {stderr}")
            return False
    
    def run_scenario(self, scenario: ScenarioConfig):
        """Run a complete scenario"""
        logger.info(f"Running scenario: {scenario.name}")
        self.running_scenario = scenario
        self.stop_event.clear()
        
        # Start scenario in thread
        self.scenario_thread = threading.Thread(
            target=self._scenario_worker,
            args=(scenario,)
        )
        self.scenario_thread.start()
    
    def _scenario_worker(self, scenario: ScenarioConfig):
        """Worker thread for scenario execution"""
        start_time = time.time()
        
        try:
            # Apply initial configuration
            for param, value in scenario.config.items():
                self.configure_fbs(param, value)
            
            # Start FBS if in FBS mode
            if scenario.mode == 'fbs':
                if not self.start_fbs(scenario.config):
                    logger.error("Failed to start FBS")
                    return
            
            # Process scheduled events
            for event in scenario.events:
                if self.stop_event.is_set():
                    break
                
                # Wait for event time
                event_time = event.get('time', 0)
                while time.time() - start_time < event_time:
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)
                
                if self.stop_event.is_set():
                    break
                
                # Execute event action
                self._execute_event(event)
            
            # Wait for scenario duration
            while time.time() - start_time < scenario.duration:
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)
            
        finally:
            # Clean up
            if scenario.mode == 'fbs':
                self.stop_fbs()
            
            self.running_scenario = None
            logger.info(f"Scenario '{scenario.name}' completed")
    
    def _execute_event(self, event: Dict[str, Any]):
        """Execute a scenario event"""
        action = event.get('action', '')
        logger.info(f"Executing event: {action}")
        
        if action == 'start_fbs':
            self.start_fbs(event.get('config', {}))
        
        elif action == 'stop_fbs':
            self.stop_fbs()
        
        elif action == 'configure':
            param = event.get('param')
            value = event.get('value')
            if param and value is not None:
                self.configure_fbs(param, value)
        
        elif action == 'increase_power':
            delta = event.get('value', 10)
            logger.info(f"Increasing TX power by {delta} dB")
            # Get current power and increase
            self.configure_fbs('power', f"+{delta}")
        
        elif action == 'change_identity':
            new_pci = event.get('pci')
            if new_pci:
                self.configure_fbs('pci', new_pci)
        
        elif action == 'jam_frequency':
            freq = event.get('frequency')
            logger.info(f"Jamming frequency: {freq} MHz")
            # Implementation specific
        
        else:
            logger.warning(f"Unknown event action: {action}")
    
    def stop_scenario(self):
        """Stop running scenario"""
        if self.running_scenario:
            logger.info(f"Stopping scenario: {self.running_scenario.name}")
            self.stop_event.set()
            
            if self.scenario_thread:
                self.scenario_thread.join(timeout=10)
            
            self.stop_fbs()
            self.running_scenario = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get scenario status"""
        if self.running_scenario:
            return {
                'running': True,
                'scenario': self.running_scenario.name,
                'mode': self.running_scenario.mode,
                'duration': self.running_scenario.duration
            }
        else:
            return {'running': False}
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_scenario()
        
        if self.ssh_client:
            self.ssh_client.close()

# ============================================================================
# Convenience Functions
# ============================================================================

# Global runner instance
_runner = None

def get_runner() -> ScenarioRunner:
    """Get or create runner instance"""
    global _runner
    if _runner is None:
        _runner = ScenarioRunner()
    return _runner

def run_scenario(scenario_file: str):
    """Run scenario from file"""
    runner = get_runner()
    
    if isinstance(scenario_file, str):
        scenario = ScenarioConfig.from_file(scenario_file)
    else:
        scenario = ScenarioConfig.from_dict(scenario_file)
    
    runner.run_scenario(scenario)

def stop_scenario():
    """Stop running scenario"""
    runner = get_runner()
    runner.stop_scenario()

def get_scenario_status() -> Dict:
    """Get scenario status"""
    runner = get_runner()
    return runner.get_status()

# ============================================================================
# Predefined Scenarios
# ============================================================================

class ScenarioLibrary:
    """Library of predefined scenarios"""
    
    @staticmethod
    def basic_fbs_attack() -> Dict:
        """Basic FBS attack scenario"""
        return {
            'name': 'Basic FBS Attack',
            'mode': 'fbs',
            'duration': 120,
            'config': {
                'plmn': '00101',
                'pci': 999,
                'tac': 999,
                'tx_power': 30
            },
            'events': [
                {
                    'time': 10,
                    'action': 'increase_power',
                    'value': 10
                },
                {
                    'time': 30,
                    'action': 'configure',
                    'param': 'cipher',
                    'value': 'null'
                }
            ]
        }
    
    @staticmethod
    def identity_spoofing() -> Dict:
        """Identity spoofing scenario"""
        return {
            'name': 'Identity Spoofing',
            'mode': 'fbs',
            'duration': 180,
            'config': {
                'plmn': '00101',  # Same as legitimate
                'pci': 1,  # Same as legitimate
                'tac': 1,  # Same as legitimate
                'tx_power': 20
            },
            'events': [
                {
                    'time': 20,
                    'action': 'increase_power',
                    'value': 15
                },
                {
                    'time': 60,
                    'action': 'change_identity',
                    'pci': 999
                },
                {
                    'time': 120,
                    'action': 'change_identity',
                    'pci': 1
                }
            ]
        }
    
    @staticmethod
    def intermittent_attack() -> Dict:
        """Intermittent FBS attack"""
        return {
            'name': 'Intermittent FBS',
            'mode': 'fbs',
            'duration': 300,
            'config': {
                'plmn': '00199',
                'pci': 888,
                'tac': 888,
                'tx_power': 25
            },
            'events': [
                {'time': 30, 'action': 'stop_fbs'},
                {'time': 60, 'action': 'start_fbs'},
                {'time': 90, 'action': 'stop_fbs'},
                {'time': 120, 'action': 'start_fbs'},
                {'time': 150, 'action': 'stop_fbs'},
                {'time': 180, 'action': 'start_fbs'}
            ]
        }

# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FBS Scenario Runner')
    parser.add_argument('action', choices=['run', 'stop', 'status', 'library'])
    parser.add_argument('--scenario', help='Scenario file or name')
    parser.add_argument('--docker', action='store_true', help='Use Docker')
    parser.add_argument('--ssh', action='store_true', help='Use SSH')
    
    args = parser.parse_args()
    
    if args.action == 'run':
        if args.scenario:
            if os.path.exists(args.scenario):
                # Run from file
                run_scenario(args.scenario)
            else:
                # Check library
                lib = ScenarioLibrary()
                if hasattr(lib, args.scenario):
                    scenario = getattr(lib, args.scenario)()
                    run_scenario(scenario)
                else:
                    print(f"Scenario not found: {args.scenario}")
        else:
            print("Please specify a scenario")
    
    elif args.action == 'stop':
        stop_scenario()
        print("Scenario stopped")
    
    elif args.action == 'status':
        status = get_scenario_status()
        print(json.dumps(status, indent=2))
    
    elif args.action == 'library':
        print("Available scenarios:")
        lib = ScenarioLibrary()
        for name in dir(lib):
            if not name.startswith('_'):
                print(f"  - {name}")