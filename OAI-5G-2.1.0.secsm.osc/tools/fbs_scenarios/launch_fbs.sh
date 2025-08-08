#!/bin/bash

# Launch script for Fake Base Station (FBS) scenarios
# Location: OAI-5G/tools/fbs_scenarios/launch_fbs.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
FBS_CONFIG="${FBS_CONFIG:-$SCRIPT_DIR/fake_gnb.cfg}"
LOG_DIR="/tmp/fbs_logs"
PID_FILE="/tmp/fbs_gnb.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

check_requirements() {
    # Check if nr-softmodem exists
    if [ ! -f "$OAI_DIR/cmake_targets/ran_build/build/nr-softmodem" ]; then
        log_error "nr-softmodem not found. Please build OAI first."
        exit 1
    fi
    
    # Check if config file exists
    if [ ! -f "$FBS_CONFIG" ]; then
        log_error "FBS config file not found: $FBS_CONFIG"
        exit 1
    fi
    
    # Create log directory
    mkdir -p "$LOG_DIR"
}

start_fbs() {
    log_info "Starting Fake Base Station..."
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            log_warn "FBS already running with PID $OLD_PID"
            return 1
        fi
    fi
    
    # Parse scenario parameters
    local SCENARIO_FILE="${1:-}"
    local EXTRA_ARGS=""
    
    if [ -n "$SCENARIO_FILE" ] && [ -f "$SCENARIO_FILE" ]; then
        log_info "Loading scenario from: $SCENARIO_FILE"
        EXTRA_ARGS="--scenario-file $SCENARIO_FILE"
    fi
    
    # Launch the fake gNB
    # Using different ports and RF settings to avoid interference
    cd "$OAI_DIR"
    
    nohup ./cmake_targets/ran_build/build/nr-softmodem \
        -O "$FBS_CONFIG" \
        --sa \
        --rfsim \
        --log_config.global_log_level info \
        --log_config.global_log_options nocolor,level,thread_id,time \
        $EXTRA_ARGS \
        > "$LOG_DIR/fbs_gnb.log" 2>&1 &
    
    local FBS_PID=$!
    echo $FBS_PID > "$PID_FILE"
    
    log_info "FBS started with PID $FBS_PID"
    log_info "Logs: $LOG_DIR/fbs_gnb.log"
    
    # Wait a bit and check if it's still running
    sleep 3
    if kill -0 "$FBS_PID" 2>/dev/null; then
        log_info "FBS is running successfully"
        return 0
    else
        log_error "FBS failed to start. Check logs at $LOG_DIR/fbs_gnb.log"
        return 1
    fi
}

stop_fbs() {
    log_info "Stopping Fake Base Station..."
    
    if [ ! -f "$PID_FILE" ]; then
        log_warn "PID file not found. FBS might not be running."
        return 1
    fi
    
    local FBS_PID=$(cat "$PID_FILE")
    
    if kill -0 "$FBS_PID" 2>/dev/null; then
        kill -TERM "$FBS_PID"
        sleep 2
        
        # Force kill if still running
        if kill -0 "$FBS_PID" 2>/dev/null; then
            log_warn "Forcing FBS termination..."
            kill -KILL "$FBS_PID"
        fi
        
        rm -f "$PID_FILE"
        log_info "FBS stopped"
    else
        log_warn "FBS process $FBS_PID not found"
        rm -f "$PID_FILE"
    fi
}

status_fbs() {
    if [ -f "$PID_FILE" ]; then
        local FBS_PID=$(cat "$PID_FILE")
        if kill -0 "$FBS_PID" 2>/dev/null; then
            log_info "FBS is running (PID: $FBS_PID)"
            
            # Show some stats
            if [ -f "$LOG_DIR/fbs_gnb.log" ]; then
                local LINE_COUNT=$(wc -l < "$LOG_DIR/fbs_gnb.log")
                log_info "Log lines: $LINE_COUNT"
                
                # Check for errors in last 100 lines
                local ERROR_COUNT=$(tail -n 100 "$LOG_DIR/fbs_gnb.log" | grep -c "ERROR" || true)
                if [ "$ERROR_COUNT" -gt 0 ]; then
                    log_warn "Recent errors detected: $ERROR_COUNT"
                fi
            fi
            return 0
        else
            log_warn "FBS PID file exists but process is not running"
            return 1
        fi
    else
        log_info "FBS is not running"
        return 1
    fi
}

restart_fbs() {
    stop_fbs
    sleep 2
    start_fbs "$@"
}

configure_fbs() {
    local PARAM="$1"
    local VALUE="$2"
    
    log_info "Configuring FBS: $PARAM = $VALUE"
    
    # Modify configuration file
    case "$PARAM" in
        plmn)
            sed -i "s/mcc = .*/mcc = ${VALUE:0:3};/" "$FBS_CONFIG"
            sed -i "s/mnc = .*/mnc = ${VALUE:3};/" "$FBS_CONFIG"
            ;;
        pci)
            sed -i "s/physCellId = .*/physCellId = $VALUE;/" "$FBS_CONFIG"
            ;;
        power)
            sed -i "s/tx_gain = .*/tx_gain = $VALUE;/" "$FBS_CONFIG"
            ;;
        tac)
            sed -i "s/tracking_area_code = .*/tracking_area_code = $VALUE;/" "$FBS_CONFIG"
            ;;
        *)
            log_error "Unknown parameter: $PARAM"
            return 1
            ;;
    esac
    
    log_info "Configuration updated. Restart FBS to apply changes."
}

run_scenario() {
    local SCENARIO_FILE="$1"
    
    if [ ! -f "$SCENARIO_FILE" ]; then
        log_error "Scenario file not found: $SCENARIO_FILE"
        return 1
    fi
    
    log_info "Running scenario: $SCENARIO_FILE"
    
    # Parse JSON scenario using Python
    python3 - <<EOF
import json
import subprocess
import time

with open('$SCENARIO_FILE', 'r') as f:
    scenario = json.load(f)

print(f"Scenario: {scenario.get('name', 'unnamed')}")
print(f"Duration: {scenario.get('duration', 60)} seconds")

# Configure FBS parameters
if 'config' in scenario:
    for param, value in scenario['config'].items():
        subprocess.run(['$0', 'configure', param, str(value)])

# Start FBS
if scenario.get('start_fbs', True):
    subprocess.run(['$0', 'start'])
    time.sleep(5)  # Wait for FBS to initialize

# Run scenario steps
if 'steps' in scenario:
    for step in scenario['steps']:
        print(f"Step: {step.get('name', 'unnamed')}")
        if 'delay' in step:
            time.sleep(step['delay'])
        if 'action' in step:
            if step['action'] == 'stop':
                subprocess.run(['$0', 'stop'])
            elif step['action'] == 'restart':
                subprocess.run(['$0', 'restart'])
            elif step['action'] == 'configure':
                subprocess.run(['$0', 'configure', step['param'], str(step['value'])])

# Wait for scenario duration
time.sleep(scenario.get('duration', 60))

# Stop FBS if requested
if scenario.get('stop_fbs', True):
    subprocess.run(['$0', 'stop'])

print("Scenario completed")
EOF
}

# Main script logic
case "${1:-}" in
    start)
        check_requirements
        start_fbs "${2:-}"
        ;;
    stop)
        stop_fbs
        ;;
    restart)
        check_requirements
        restart_fbs "${2:-}"
        ;;
    status)
        status_fbs
        ;;
    configure)
        if [ $# -lt 3 ]; then
            log_error "Usage: $0 configure <param> <value>"
            exit 1
        fi
        configure_fbs "$2" "$3"
        ;;
    scenario)
        if [ $# -lt 2 ]; then
            log_error "Usage: $0 scenario <scenario_file>"
            exit 1
        fi
        check_requirements
        run_scenario "$2"
        ;;
    logs)
        if [ -f "$LOG_DIR/fbs_gnb.log" ]; then
            tail -f "$LOG_DIR/fbs_gnb.log"
        else
            log_error "Log file not found"
            exit 1
        fi
        ;;
    clean)
        log_info "Cleaning up FBS files..."
        stop_fbs
        rm -rf "$LOG_DIR"
        rm -f "$PID_FILE"
        log_info "Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|configure|scenario|logs|clean} [options]"
        echo ""
        echo "Commands:"
        echo "  start [scenario_file]  - Start the fake base station"
        echo "  stop                   - Stop the fake base station"
        echo "  restart [scenario_file]- Restart the fake base station"
        echo "  status                 - Check FBS status"
        echo "  configure <param> <val>- Configure FBS parameter"
        echo "  scenario <file>        - Run a scenario from JSON file"
        echo "  logs                   - Tail FBS logs"
        echo "  clean                  - Clean up all FBS files"
        echo ""
        echo "Configuration parameters:"
        echo "  plmn <mccmnc>  - Set PLMN (e.g., 00101)"
        echo "  pci <id>       - Set Physical Cell ID"
        echo "  power <dB>     - Set transmission power"
        echo "  tac <code>     - Set Tracking Area Code"
        exit 1
        ;;
esac

exit $?