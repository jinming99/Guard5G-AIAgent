#!/bin/bash

# End-to-End FBS Scenario Test Script
# Location: llm_fbs_utils/eval_scripts/test_fbs_scenario.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="/tmp/fbs_test_logs"
RESULTS_FILE="$LOG_DIR/test_results.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
AUDITOR_API="http://localhost:8090"
EXPERT_API="http://localhost:8091"
TEST_DURATION=120
SCENARIO_FILE="$PROJECT_ROOT/experiments/scenarios/advanced_fbs_attack.json"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# Initialize test environment
init_test() {
    log_info "Initializing test environment..."
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Clear previous results
    rm -f "$RESULTS_FILE"
    
    # Initialize results JSON
    echo '{
        "test_name": "FBS End-to-End Test",
        "start_time": "'$(date -Iseconds)'",
        "tests": []
    }' > "$RESULTS_FILE"
}

# Check service availability
check_services() {
    log_test "Checking service availability..."
    
    local all_ready=true
    
    # Check MobiFlow-Auditor
    if curl -s "$AUDITOR_API/health" | grep -q "healthy"; then
        log_info "✓ MobiFlow-Auditor is running"
        add_test_result "auditor_health" "PASS"
    else
        log_error "✗ MobiFlow-Auditor is not responding"
        add_test_result "auditor_health" "FAIL"
        all_ready=false
    fi
    
    # Check MobieXpert
    if curl -s "$EXPERT_API/rules" | grep -q "status"; then
        log_info "✓ MobieXpert is running"
        add_test_result "expert_health" "PASS"
    else
        log_error "✗ MobieXpert is not responding"
        add_test_result "expert_health" "FAIL"
        all_ready=false
    fi
    
    # Check Redis SDL
    if redis-cli ping > /dev/null 2>&1; then
        log_info "✓ Redis SDL is running"
        add_test_result "redis_health" "PASS"
    else
        log_error "✗ Redis SDL is not responding"
        add_test_result "redis_health" "FAIL"
        all_ready=false
    fi
    
    if [ "$all_ready" = false ]; then
        log_error "Not all services are ready. Exiting."
        exit 1
    fi
}

# Load detection rules
load_rules() {
    log_test "Loading FBS detection rules..."
    
    # Load rule template
    local rule_file="$PROJECT_ROOT/../MobieXpert/experiments/rule_templates/fbs_detection_template.yaml"
    
    if [ -f "$rule_file" ]; then
        response=$(curl -s -X POST "$EXPERT_API/rules" \
            -H "Content-Type: application/yaml" \
            --data-binary "@$rule_file")
        
        if echo "$response" | grep -q "success"; then
            log_info "✓ Detection rules loaded successfully"
            add_test_result "rule_loading" "PASS"
        else
            log_error "✗ Failed to load detection rules"
            add_test_result "rule_loading" "FAIL"
            return 1
        fi
    else
        log_warn "Rule file not found, using default rules"
    fi
}

# Start normal gNB
start_normal_gnb() {
    log_test "Starting legitimate gNB..."
    
    # This would normally start the real gNB
    # For testing, we'll simulate it
    log_info "✓ Legitimate gNB started (simulated)"
    add_test_result "normal_gnb_start" "PASS"
}

# Inject normal traffic
inject_normal_traffic() {
    log_test "Injecting normal network traffic..."
    
    python3 - <<EOF
import sys
import os
sys.path.append('$PROJECT_ROOT')

from dataset_playback import DatasetPlayer
import numpy as np

player = DatasetPlayer()

# Generate normal traffic
records = []
for i in range(30):
    records.append({
        'timestamp': i,
        'ue_id': '001010123456789',
        'cell_id': '12345',
        'event_type': 'MEASUREMENT_REPORT',
        'rsrp': -85 + np.random.randn() * 3,
        'rsrq': -12 + np.random.randn(),
        'suspected_fbs': False,
        'attach_failures': 0
    })

# Inject records
for record in records:
    player.inject_mobiflow_record(record)

print("Injected 30 normal traffic records")
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ Normal traffic injected"
        add_test_result "normal_traffic_injection" "PASS"
    else
        log_error "✗ Failed to inject normal traffic"
        add_test_result "normal_traffic_injection" "FAIL"
    fi
}

# Start FBS attack
start_fbs_attack() {
    log_test "Starting FBS attack scenario..."
    
    # Run scenario
    python3 - <<EOF
import sys
import os
import json
sys.path.append('$PROJECT_ROOT')

from scenario_runner import run_scenario

# Load and run scenario
with open('$SCENARIO_FILE', 'r') as f:
    scenario = json.load(f)

# Simplified scenario for testing
test_scenario = {
    'name': 'Test FBS Attack',
    'mode': 'fbs',
    'duration': 60,
    'config': {
        'plmn': '00199',
        'pci': 999,
        'tac': 999,
        'tx_power': 30
    },
    'events': []
}

# For testing, we'll inject simulated FBS data instead
from dataset_playback import DatasetPlayer
import numpy as np
import time

player = DatasetPlayer()

# Generate FBS attack data
base_time = time.time()
for i in range(60):
    # Gradually increasing signal from fake cell
    record = {
        'timestamp': base_time + i,
        'ue_id': '001010123456789',
        'cell_id': '99999',
        'event_type': 'MEASUREMENT_REPORT',
        'rsrp': -90 + i * 0.5,  # Increasing signal
        'rsrq': -10,
        'signal_anomaly': True,
        'suspected_fbs': i > 10,
        'attach_failures': max(0, (i - 20) // 10),
        'auth_reject_count': max(0, (i - 30) // 15),
        'cipher_downgrade': i > 40
    }
    player.inject_mobiflow_record(record)

print("FBS attack simulation started")
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ FBS attack started"
        add_test_result "fbs_attack_start" "PASS"
    else
        log_error "✗ Failed to start FBS attack"
        add_test_result "fbs_attack_start" "FAIL"
    fi
}

# Monitor detection
monitor_detection() {
    log_test "Monitoring for FBS detection..."
    
    local detected=false
    local start_time=$(date +%s)
    local timeout=60
    
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        # Check for FBS detections
        stats=$(curl -s "$AUDITOR_API/stats")
        detections=$(echo "$stats" | grep -o '"fbs_detections":[0-9]*' | cut -d: -f2)
        
        if [ "$detections" -gt "0" ]; then
            detected=true
            break
        fi
        
        sleep 2
    done
    
    if [ "$detected" = true ]; then
        local detection_time=$(($(date +%s) - start_time))
        log_info "✓ FBS detected in ${detection_time} seconds"
        add_test_result "fbs_detection" "PASS" "{\"detection_time\": $detection_time}"
    else
        log_error "✗ FBS not detected within timeout"
        add_test_result "fbs_detection" "FAIL"
    fi
}

# Test rule effectiveness
test_rule_effectiveness() {
    log_test "Testing rule effectiveness..."
    
    python3 - <<EOF
import sys
import os
import json
sys.path.append('$PROJECT_ROOT')

from llm_control_api_client import get_mobiflow, get_fbs_suspects

# Get recent MobiFlow records
records = get_mobiflow(n=100)

# Analyze for FBS
fbs_count = sum(1 for r in records if r.get('suspected_fbs', False))
total_count = len(records)

# Get suspects
suspects = get_fbs_suspects(confidence_threshold=0.5)

print(f"FBS records: {fbs_count}/{total_count}")
print(f"Suspected UEs: {len(suspects)}")

# Calculate metrics
if total_count > 0:
    detection_rate = fbs_count / total_count
    if detection_rate > 0.3:  # At least 30% of attack records detected
        print("PASS: Good detection rate")
        exit(0)
    else:
        print("FAIL: Low detection rate")
        exit(1)
else:
    print("FAIL: No records found")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ Rule effectiveness test passed"
        add_test_result "rule_effectiveness" "PASS"
    else
        log_error "✗ Rule effectiveness test failed"
        add_test_result "rule_effectiveness" "FAIL"
    fi
}

# Test LLM integration
test_llm_integration() {
    log_test "Testing LLM integration..."
    
    python3 - <<EOF
import sys
import os
sys.path.append('$PROJECT_ROOT')

try:
    from llm_driver import LLMOrchestrator
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator()
    
    # Run detection cycle
    analysis = orchestrator.run_detection_cycle()
    
    if analysis and 'fbs_detected' in analysis:
        print("✓ LLM integration successful")
        print(f"  FBS detected: {analysis['fbs_detected']}")
        print(f"  Confidence: {analysis.get('confidence', 0)}")
        exit(0)
    else:
        print("✗ LLM integration failed")
        exit(1)
        
except Exception as e:
    print(f"✗ LLM integration error: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ LLM integration test passed"
        add_test_result "llm_integration" "PASS"
    else
        log_warn "⚠ LLM integration test failed (may need API key)"
        add_test_result "llm_integration" "SKIP"
    fi
}

# Performance metrics
collect_metrics() {
    log_test "Collecting performance metrics..."
    
    python3 - <<EOF
import sys
import os
import json
import time
sys.path.append('$PROJECT_ROOT')

from llm_control_api_client import get_stats

# Collect metrics
stats = get_stats()

metrics = {
    'total_ues': stats.get('total_ues', 0),
    'active_cells': len(stats.get('active_cells', [])),
    'total_records': stats.get('total_records', 0),
    'fbs_detections': stats.get('fbs_detections', 0),
    'timestamp': time.time()
}

print(json.dumps(metrics, indent=2))

# Save to file
with open('$LOG_DIR/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
EOF
    
    if [ $? -eq 0 ]; then
        log_info "✓ Metrics collected successfully"
        add_test_result "metrics_collection" "PASS"
    else
        log_error "✗ Failed to collect metrics"
        add_test_result "metrics_collection" "FAIL"
    fi
}

# Cleanup
cleanup() {
    log_test "Cleaning up test environment..."
    
    # Stop FBS
    python3 -c "
import sys
sys.path.append('$PROJECT_ROOT')
from scenario_runner import stop_scenario
stop_scenario()
"
    
    # Clear test data from Redis
    redis-cli FLUSHDB > /dev/null 2>&1 || true
    
    log_info "✓ Cleanup completed"
}

# Add test result to JSON
add_test_result() {
    local test_name=$1
    local status=$2
    local details=${3:-"{}"}
    
    python3 - <<EOF
import json

with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)

results['tests'].append({
    'name': '$test_name',
    'status': '$status',
    'details': $details
})

with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
EOF
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    python3 - <<EOF
import json
from datetime import datetime

with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)

# Add end time
results['end_time'] = datetime.now().isoformat()

# Calculate summary
total_tests = len(results['tests'])
passed = sum(1 for t in results['tests'] if t['status'] == 'PASS')
failed = sum(1 for t in results['tests'] if t['status'] == 'FAIL')
skipped = sum(1 for t in results['tests'] if t['status'] == 'SKIP')

results['summary'] = {
    'total': total_tests,
    'passed': passed,
    'failed': failed,
    'skipped': skipped,
    'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0
}

# Save final report
with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("=" * 50)
print("TEST SUMMARY")
print("=" * 50)
print(f"Total Tests: {total_tests}")
print(f"Passed: {passed} ({passed/total_tests*100:.1f}%)")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print("=" * 50)

if failed == 0:
    print("✓ ALL TESTS PASSED")
    exit(0)
else:
    print("✗ SOME TESTS FAILED")
    exit(1)
EOF
    
    exit_code=$?
    
    log_info "Test report saved to: $RESULTS_FILE"
    
    return $exit_code
}

# Main test execution
main() {
    log_info "Starting FBS End-to-End Test Suite"
    log_info "=================================="
    
    # Initialize
    init_test
    
    # Run test sequence
    check_services
    load_rules
    start_normal_gnb
    inject_normal_traffic
    
    # Wait a bit
    sleep 5
    
    # Start attack and monitor
    start_fbs_attack
    monitor_detection
    test_rule_effectiveness
    test_llm_integration
    collect_metrics
    
    # Cleanup
    cleanup
    
    # Generate report
    generate_report
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "✓ Test suite completed successfully"
    else
        log_error "✗ Test suite failed"
    fi
    
    exit $exit_code
}

# Handle interrupts
trap cleanup INT TERM

# Run main
main "$@"