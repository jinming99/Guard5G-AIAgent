#!/bin/bash

# Setup script for 5G-Spector FBS Detection System
# This script handles initial configuration and dependency installation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
WORKSPACE_DIR="$(pwd)"
REQUIRED_PYTHON="3.8"
REQUIRED_DOCKER="20.10"

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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ $(echo "$PYTHON_VERSION >= $REQUIRED_PYTHON" | bc -l) -eq 1 ]]; then
            log_info "✓ Python $PYTHON_VERSION found"
        else
            log_error "Python $REQUIRED_PYTHON or higher required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
        log_info "✓ Docker $DOCKER_VERSION found"
    else
        log_error "Docker not found"
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        log_info "✓ Docker Compose found"
    else
        log_warn "Docker Compose not found, installing..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    # Check Redis
    if command -v redis-cli &> /dev/null; then
        log_info "✓ Redis CLI found"
    else
        log_warn "Redis CLI not found, will use Docker version"
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        log_info "✓ Git found"
    else
        log_error "Git not found"
        exit 1
    fi
}

# Clone repositories
clone_repos() {
    log_step "Cloning repositories..."
    
    if [ ! -d "MobiFlow-Auditor" ]; then
        log_info "Cloning MobiFlow-Auditor..."
        git clone https://github.com/5GSEC/MobiFlow-Auditor.git
    else
        log_info "MobiFlow-Auditor already exists"
    fi
    
    if [ ! -d "MobieXpert" ]; then
        log_info "Cloning MobieXpert..."
        git clone https://github.com/5GSEC/MobieXpert.git
    else
        log_info "MobieXpert already exists"
    fi
    
    if [ ! -d "OAI-5G" ]; then
        log_info "Cloning OAI-5G..."
        git clone https://github.com/5GSEC/OAI-5G.git
        cd OAI-5G
        git checkout fhi72_security || git checkout main
        cd ..
    else
        log_info "OAI-5G already exists"
    fi
}

# Apply patches and modifications
apply_modifications() {
    log_step "Applying modifications to repositories..."
    
    # Copy new files to MobiFlow-Auditor
    log_info "Modifying MobiFlow-Auditor..."
    if [ -f "llm_fbs_utils/patches/mobiflow_auditor/llm_control_api.py" ]; then
        cp llm_fbs_utils/patches/mobiflow_auditor/llm_control_api.py MobiFlow-Auditor/src/
        log_info "✓ Added llm_control_api.py"
    fi
    
    # Copy new files to MobieXpert
    log_info "Modifying MobieXpert..."
    if [ -f "llm_fbs_utils/patches/mobiexpert/llm_rule_patch.py" ]; then
        cp llm_fbs_utils/patches/mobiexpert/llm_rule_patch.py MobieXpert/src/pypbest/
        log_info "✓ Added llm_rule_patch.py"
    fi
    
    # Create experiments directory
    mkdir -p MobieXpert/experiments/rule_templates
    mkdir -p MobieXpert/experiments/generated_rules
    
    # Copy FBS scenarios to OAI-5G
    log_info "Modifying OAI-5G..."
    mkdir -p OAI-5G/tools/fbs_scenarios
    if [ -f "llm_fbs_utils/patches/oai/launch_fbs.sh" ]; then
        cp llm_fbs_utils/patches/oai/launch_fbs.sh OAI-5G/tools/fbs_scenarios/
        chmod +x OAI-5G/tools/fbs_scenarios/launch_fbs.sh
        log_info "✓ Added launch_fbs.sh"
    fi
    if [ -f "llm_fbs_utils/patches/oai/fake_gnb.cfg" ]; then
        cp llm_fbs_utils/patches/oai/fake_gnb.cfg OAI-5G/tools/fbs_scenarios/
        log_info "✓ Added fake_gnb.cfg"
    fi
}

# Install Python dependencies
install_python_deps() {
    log_step "Installing Python dependencies..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "llm_fbs_utils/requirements.txt" ]; then
        pip install -r llm_fbs_utils/requirements.txt
        log_info "✓ Python dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi
}

# Create configuration files
create_configs() {
    log_step "Creating configuration files..."
    
    # Create .env file
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Environment Configuration for 5G-Spector FBS Detection

# API Keys
OPENAI_API_KEY=your-openai-api-key-here

# Service URLs
AUDITOR_API=http://localhost:8090
EXPERT_API=http://localhost:8091

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# OAI Configuration
OAI_HOST=localhost
OAI_USER=root
OAI_PATH=/opt/oai-5g
OAI_CONTAINER=oai-gnb

# Docker Configuration
DOCKER_NETWORK=fbs-network
COMPOSE_PROJECT_NAME=fbs-detection

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
EOF
        log_info "✓ Created .env file"
        log_warn "Please edit .env file with your API keys"
    else
        log_info ".env file already exists"
    fi
    
    # Create config.yaml
    if [ ! -f "llm_fbs_utils/config.yaml" ]; then
        cat > llm_fbs_utils/config.yaml << EOF
# Configuration for LLM FBS Detection System

llm:
  model: gpt-3.5-turbo
  temperature: 0.7
  max_tokens: 2000
  
apis:
  auditor: http://localhost:8090
  expert: http://localhost:8091
  
redis:
  host: localhost
  port: 6379
  db: 0
  
experiment:
  max_duration: 300
  safety_checks: true
  auto_stop: true
  
detection:
  confidence_threshold: 0.7
  max_false_positives: 2
  detection_window: 60
  
monitoring:
  interval: 5
  metrics_enabled: true
  prometheus_port: 9090
  
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/fbs_detection.log
EOF
        log_info "✓ Created config.yaml"
    else
        log_info "config.yaml already exists"
    fi
}

# Create directory structure
create_directories() {
    log_step "Creating directory structure..."
    
    directories=(
        "llm_fbs_utils/experiments/scenarios"
        "llm_fbs_utils/experiments/generated_rules"
        "llm_fbs_utils/experiments/results"
        "llm_fbs_utils/logs"
        "llm_fbs_utils/data"
        "llm_fbs_utils/monitoring/prometheus"
        "llm_fbs_utils/monitoring/grafana/dashboards"
        "llm_fbs_utils/monitoring/grafana/datasources"
        "evaluation_results"
        "docker/volumes"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "✓ Created $dir"
    done
}

# Build Docker images
build_docker() {
    log_step "Building Docker images..."
    
    cd llm_fbs_utils
    
    # Create Dockerfile for LLM orchestrator if not exists
    if [ ! -f "Dockerfile.llm" ]; then
        cat > Dockerfile.llm << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "llm_driver.py", "interactive"]
EOF
        log_info "✓ Created Dockerfile.llm"
    fi
    
    # Build images
    docker-compose build
    
    cd ..
}

# Test installation
test_installation() {
    log_step "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "import dspy; import flask; import redis; print('✓ Python imports successful')"
    
    # Test Docker
    docker ps > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "✓ Docker is running"
    else
        log_warn "Docker daemon is not running"
    fi
    
    # Test connectivity (if services are running)
    if curl -s http://localhost:8090/health > /dev/null 2>&1; then
        log_info "✓ MobiFlow-Auditor is accessible"
    else
        log_warn "MobiFlow-Auditor is not running (this is normal if services haven't been started)"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          Setup Complete! Next Steps:                      ║"
    echo "╠════════════════════════════════════════════════════════════╣"
    echo "║ 1. Edit .env file with your API keys                      ║"
    echo "║    nano .env                                              ║"
    echo "║                                                            ║"
    echo "║ 2. Start services:                                        ║"
    echo "║    make docker-up                                         ║"
    echo "║                                                            ║"
    echo "║ 3. Run tests:                                             ║"
    echo "║    make test                                              ║"
    echo "║                                                            ║"
    echo "║ 4. Start detection:                                       ║"
    echo "║    make run-detection                                     ║"
    echo "║                                                            ║"
    echo "║ 5. View logs:                                             ║"
    echo "║    make docker-logs                                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "For more information, see README.md"
}

# Main setup flow
main() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     5G-Spector FBS Detection System Setup                 ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    check_requirements
    clone_repos
    create_directories
    apply_modifications
    install_python_deps
    create_configs
    
    # Optional: build Docker images
    read -p "Do you want to build Docker images now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_docker
    fi
    
    test_installation
    print_next_steps
    
    log_info "Setup completed successfully!"
}

# Handle errors
trap 'log_error "Setup failed! Check the error messages above."' ERR

# Run main
main "$@"