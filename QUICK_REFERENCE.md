# Quick Reference Guide - 5G-Spector FBS Detection

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Clone and setup
git clone <your-repo-url> fbs-detection
cd fbs-detection
./setup.sh

# 2. Configure API key
echo "OPENAI_API_KEY=sk-..." >> .env

# 3. Start everything
make docker-up

# 4. Run detection
make run-detection

# 5. View results
make stats
```

## 📋 Essential Commands

### Service Management

| Command | Description |
|---------|-------------|
| `make docker-up` | Start all services |
| `make docker-down` | Stop all services |
| `make docker-restart` | Restart services |
| `make docker-logs` | View logs |
| `make check-services` | Health check |

### Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests |
| `make evaluate` | Full evaluation |

### Operations

| Command | Description |
|---------|-------------|
| `make run-detection` | Run FBS detection |
| `make run-experiment` | Run experiment |
| `make run-interactive` | Interactive mode |
| `make monitor` | Real-time monitoring |
| `make stats` | Show statistics |

### Maintenance

| Command | Description |
|---------|-------------|
| `make clean` | Clean temp files |
| `make clean-redis` | Clear Redis data |
| `make reset` | Full system reset |
| `make backup` | Backup data |

## 🔧 API Endpoints

### MobiFlow-Auditor (Port 8090)

```bash
# Health check
curl http://localhost:8090/health

# Get KPM data
curl http://localhost:8090/kpm/<ue_id>

# Get MobiFlow records
curl http://localhost:8090/mobiflow/last?n=100

# Get statistics
curl http://localhost:8090/stats
```

### MobieXpert (Port 8091)

```bash
# Get current rules
curl http://localhost:8091/rules

# Deploy new rules
curl -X POST http://localhost:8091/rules \
  -H "Content-Type: application/yaml" \
  -d @rule.yaml

# Test rule
curl -X POST http://localhost:8091/rules/test \
  -H "Content-Type: application/json" \
  -d '{"rule": {...}, "test_data": {...}}'
```

## 🐍 Python Scripts

### Detection Operations

```python
# Query network state
from llm_control_api_client import get_kpm, get_mobiflow, get_stats

kpm = get_kpm("001010123456789")
records = get_mobiflow(n=100)
stats = get_stats()

# Run detection
from llm_driver import LLMOrchestrator

orchestrator = LLMOrchestrator()
analysis = orchestrator.run_detection_cycle()
```

### Rule Management

```python
# Deploy rules
from llm_control_api_client import post_rule, get_rules

yaml_rule = """
rules:
  - name: "FBS_Detection"
    condition:
      field: "attach_failures"
      gte: 3
    action:
      type: "alert"
"""

result = post_rule(yaml_rule)
current = get_rules()
```

### Scenario Execution

```python
# Run FBS scenario
from scenario_runner import run_scenario, ScenarioLibrary

scenario = ScenarioLibrary.basic_fbs_attack()
run_scenario(scenario)

# Custom scenario
custom = {
    'name': 'Test',
    'mode': 'fbs',
    'duration': 60,
    'config': {'plmn': '00199', 'pci': 999}
}
run_scenario(custom)
```

### Data Playback

```python
# Generate synthetic data
from dataset_playback import DatasetPlayer

player = DatasetPlayer()
data = player.generate_synthetic_fbs_data(300)

# Playback data
player.play_dataset(data, realtime=True)
```

## 🔍 Debugging Commands

### Check Logs

```bash
# All logs
docker-compose logs

# Specific service
docker-compose logs mobiflow-auditor

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### Redis Operations

```bash
# Connect to Redis
redis-cli

# Common Redis commands
KEYS *                    # List all keys
GET key                   # Get value
DEL key                   # Delete key
FLUSHDB                   # Clear database
MONITOR                   # Monitor commands
INFO                      # Server info
```

### Docker Debugging

```bash
# List containers
docker ps -a

# Execute command in container
docker exec -it fbs-auditor bash

# Inspect container
docker inspect fbs-auditor

# View resource usage
docker stats

# Clean up
docker system prune -a
```

## 📊 Monitoring

### Prometheus Queries

```promql
# FBS detection rate
rate(fbs_detections_total[5m])

# API latency (99th percentile)
http_request_duration_seconds{quantile="0.99"}

# Service uptime
up{job="mobiflow-auditor"}

# Memory usage
container_memory_usage_bytes{name="fbs-auditor"}

# CPU usage
rate(container_cpu_usage_seconds_total{name="fbs-auditor"}[5m])
```

### Grafana Dashboards

- **Default**: http://localhost:3000
- **Username**: admin
- **Password**: admin

## 🚨 Common Fixes

### Service Won't Start

```bash
# Check ports
sudo netstat -tulpn | grep -E '(8090|8091|6379)'

# Kill conflicting process
sudo kill -9 <PID>

# Restart Docker
sudo systemctl restart docker
```

### No Data/Detections

```bash
# Check data flow
curl http://localhost:8090/stats

# Inject test data
python dataset_playback.py generate --duration 60

# Check rules
curl http://localhost:8091/rules
```

### API Connection Failed

```bash
# Check service health
make check-services

# Restart specific service
docker-compose restart mobiflow-auditor

# Check network
docker network inspect fbs-network
```

### LLM Errors

```bash
# Check API key
echo $OPENAI_API_KEY

# Test LLM directly
python -c "import openai; openai.api_key='...'; print(openai.Model.list())"
```

## 📁 Important Files

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `config.yaml` | LLM configuration |
| `docker-compose.yml` | Service orchestration |
| `prometheus.yml` | Monitoring config |

### Code Files

| File | Purpose |
|------|---------|
| `llm_driver.py` | Main orchestrator |
| `llm_control_api_client.py` | API client |
| `scenario_runner.py` | Scenario execution |
| `dataset_playback.py` | Data replay |

### Test Files

| File | Purpose |
|------|---------|
| `test_query_api.py` | API tests |
| `test_rule_reload.py` | Rule tests |
| `test_fbs_scenario.sh` | E2E test |
| `evaluate_pipeline.py` | Full evaluation |

## 🔗 Useful URLs

- **Auditor API**: http://localhost:8090
- **Expert API**: http://localhost:8091
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Redis Commander**: http://localhost:8081 (if installed)

## 💡 Pro Tips

1. **Always check service health first**
   ```bash
   make check-services
   ```

2. **Monitor in real-time during experiments**
   ```bash
   make monitor  # In one terminal
   make run-experiment  # In another
   ```

3. **Use interactive mode for testing**
   ```bash
   make run-interactive
   ```

4. **Keep logs for debugging**
   ```bash
   docker-compose logs > debug_$(date +%Y%m%d).log
   ```

5. **Test rules before deployment**
   ```python
   from llm_control_api_client import test_rule
   test_rule(rule, test_data)
   ```

## 📈 Performance Tuning

### Redis Optimization
```bash
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Docker Resources
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
```

### Python Optimization
```python
# Use connection pooling
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
r = redis.Redis(connection_pool=pool)
```

## 🔄 Workflow Examples

### Complete Detection Cycle
```bash
# 1. Start services
make docker-up

# 2. Deploy rules
make deploy-rules

# 3. Generate attack data
python dataset_playback.py generate --duration 300

# 4. Run detection
make run-detection

# 5. Analyze results
make evaluate
```

### Experiment Workflow
```bash
# 1. Design scenario
cat > scenario.json << EOF
{
  "name": "Test",
  "mode": "fbs",
  "duration": 120,
  "config": {...}
}
EOF

# 2. Run scenario
python scenario_runner.py run scenario.json

# 3. Monitor detection
make monitor

# 4. Collect metrics
make stats > results.txt
```

## 📝 Notes

- Default passwords: Change them in production!
- API keys: Never commit to repository
- Logs: Rotate regularly to save space
- Backups: Automate with cron
- Updates: Check for security patches

---

**Quick Help**: `make help` | **Full Docs**: See README.md | **Issues**: GitHub Issues