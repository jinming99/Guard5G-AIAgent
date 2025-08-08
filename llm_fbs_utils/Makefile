# Makefile for 5G-Spector FBS Detection System
# Location: llm_fbs_utils/Makefile

.PHONY: help install test run clean docker-up docker-down evaluate

# Variables
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := fbs-detection

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)5G-Spector FBS Detection System$(NC)"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install all dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Creating necessary directories...$(NC)"
	mkdir -p experiments/scenarios
	mkdir -p experiments/generated_rules
	mkdir -p experiments/results
	mkdir -p logs
	mkdir -p data
	@echo "$(GREEN)Installation complete!$(NC)"

setup: install ## Initial setup and configuration
	@echo "$(GREEN)Running initial setup...$(NC)"
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		echo "OPENAI_API_KEY=your-key-here" > .env; \
		echo "REDIS_HOST=localhost" >> .env; \
		echo "REDIS_PORT=6379" >> .env; \
		echo "AUDITOR_API=http://localhost:8090" >> .env; \
		echo "EXPERT_API=http://localhost:8091" >> .env; \
		echo "OAI_HOST=localhost" >> .env; \
		echo "$(YELLOW)Please edit .env file with your configuration$(NC)"; \
	fi
	@echo "$(GREEN)Setup complete!$(NC)"

test-unit: ## Run unit tests
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTHON) -m pytest eval_scripts/test_query_api.py -v
	$(PYTHON) -m pytest eval_scripts/test_rule_reload.py -v

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	bash eval_scripts/test_fbs_scenario.sh

test: test-unit test-integration ## Run all tests

docker-build: ## Build Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

docker-up: ## Start all services with Docker Compose
	@echo "$(GREEN)Starting services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Waiting for services to be ready...$(NC)"
	@sleep 10
	@$(MAKE) check-services

docker-down: ## Stop all Docker services
	@echo "$(YELLOW)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-restart: docker-down docker-up ## Restart all Docker services

check-services: ## Check if all services are running
	@echo "$(GREEN)Checking service health...$(NC)"
	@curl -s http://localhost:8090/health > /dev/null && \
		echo "✓ MobiFlow-Auditor is running" || \
		echo "✗ MobiFlow-Auditor is not responding"
	@curl -s http://localhost:8091/rules > /dev/null && \
		echo "✓ MobieXpert is running" || \
		echo "✗ MobieXpert is not responding"
	@redis-cli ping > /dev/null 2>&1 && \
		echo "✓ Redis SDL is running" || \
		echo "✗ Redis SDL is not responding"

run-detection: ## Run FBS detection cycle
	@echo "$(GREEN)Running FBS detection...$(NC)"
	$(PYTHON) llm_driver.py detect

run-experiment: ## Run FBS experiment
	@echo "$(GREEN)Running FBS experiment...$(NC)"
	$(PYTHON) llm_driver.py experiment --hypothesis "FBS causes authentication failures"

run-analysis: ## Run data analysis
	@echo "$(GREEN)Running data analysis...$(NC)"
	$(PYTHON) llm_driver.py analyze --goal "Count attach failures per UE"

run-interactive: ## Start interactive mode
	@echo "$(GREEN)Starting interactive mode...$(NC)"
	$(PYTHON) llm_driver.py interactive

generate-data: ## Generate synthetic FBS data
	@echo "$(GREEN)Generating synthetic data...$(NC)"
	$(PYTHON) dataset_playback.py generate --duration 300 --output data/synthetic_fbs.json

playback-data: ## Playback dataset
	@echo "$(GREEN)Playing back dataset...$(NC)"
	$(PYTHON) dataset_playback.py play --file data/synthetic_fbs.json --realtime

evaluate: ## Run complete evaluation pipeline
	@echo "$(GREEN)Running evaluation pipeline...$(NC)"
	$(PYTHON) eval_scripts/evaluate_pipeline.py --output-dir evaluation_results

evaluate-parallel: ## Run evaluation with parallel scenarios
	@echo "$(GREEN)Running parallel evaluation...$(NC)"
	$(PYTHON) eval_scripts/evaluate_pipeline.py --parallel --output-dir evaluation_results

clean-redis: ## Clear Redis data
	@echo "$(YELLOW)Clearing Redis data...$(NC)"
	redis-cli FLUSHDB

clean-logs: ## Clean log files
	@echo "$(YELLOW)Cleaning log files...$(NC)"
	rm -rf logs/*
	rm -rf /tmp/fbs_logs
	rm -rf /tmp/fbs_test_logs

clean-results: ## Clean result files
	@echo "$(YELLOW)Cleaning result files...$(NC)"
	rm -rf evaluation_results
	rm -rf experiments/results/*

clean: clean-logs clean-results ## Clean all temporary files
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

reset: clean clean-redis ## Reset everything to clean state
	@echo "$(GREEN)System reset complete!$(NC)"

monitor: ## Monitor system in real-time
	@echo "$(GREEN)Starting system monitor...$(NC)"
	$(PYTHON) -c "from llm_control_api_client import monitor_realtime; \
		monitor_realtime(lambda x: print(f'New records: {len(x)}'), duration=300)"

stats: ## Show current statistics
	@echo "$(GREEN)Current System Statistics:$(NC)"
	@$(PYTHON) -c "from llm_control_api_client import get_stats; \
		import json; \
		stats = get_stats(); \
		print(json.dumps(stats, indent=2))"

lint: ## Run code linting
	@echo "$(GREEN)Running linters...$(NC)"
	flake8 . --max-line-length=100 --exclude=.git,__pycache__
	black --check .

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	black .

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	$(PYTHON) -m sphinx -b html docs/source docs/build

deploy-rules: ## Deploy default FBS detection rules
	@echo "$(GREEN)Deploying detection rules...$(NC)"
	$(PYTHON) -c "from llm_control_api_client import post_rule; \
		import yaml; \
		with open('../MobieXpert/experiments/rule_templates/fbs_detection_template.yaml') as f: \
			rules = f.read(); \
		result = post_rule(rules); \
		print(f'Result: {result}')"

benchmark: ## Run performance benchmark
	@echo "$(GREEN)Running performance benchmark...$(NC)"
	$(PYTHON) -m timeit -n 100 "from llm_control_api_client import get_stats; get_stats()"
	$(PYTHON) -m timeit -n 100 "from llm_control_api_client import get_mobiflow; get_mobiflow(n=10)"

version: ## Show component versions
	@echo "$(GREEN)Component Versions:$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$($(DOCKER_COMPOSE) --version)"
	@echo "Redis: $$(redis-cli --version)"

.DEFAULT_GOAL := help