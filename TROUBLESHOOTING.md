# Troubleshooting Guide for 5G-Spector FBS Detection System

## Table of Contents

1. [Common Issues](#common-issues)
2. [Service-Specific Problems](#service-specific-problems)
3. [Performance Issues](#performance-issues)
4. [Detection Problems](#detection-problems)
5. [LLM Integration Issues](#llm-integration-issues)
6. [Docker and Networking](#docker-and-networking)
7. [Debugging Tools](#debugging-tools)
8. [Log Analysis](#log-analysis)
9. [Recovery Procedures](#recovery-procedures)

## Common Issues

### Services Not Starting

#### Symptom
```bash
$ make docker-up
# Services fail to start or immediately exit
```

#### Solutions

1. **Check Docker daemon**
```bash
# Verify Docker is running
sudo systemctl status docker

# Restart Docker if needed
sudo systemctl restart docker
```

2. **Check port conflicts**
```bash
# Check if ports are already in use
sudo netstat -tulpn | grep -E '(8090|8091|6379)'

# Kill conflicting processes
sudo kill -9 <PID>
```

3. **Verify Docker Compose**
```bash
# Validate compose file
docker-compose config

# Check for syntax errors
docker-compose -f docker-compose.yml config --quiet
```

4. **Resource constraints**
```bash
# Check available resources
docker system df
docker system prune -a  # Clean up if needed

# Check memory
free -h
```

### API Connection Failures

#### Symptom
```
Error: Connection refused to http://localhost:8090
```

#### Solutions

1. **Check service status**
```bash
# Check if containers are running
docker ps

# Check specific service
docker-compose ps mobiflow-auditor

# Check service logs
docker-compose logs mobiflow-auditor
```

2. **Network connectivity**
```bash
# Test from host
curl http://localhost:8090/health

# Test from within container
docker exec -it fbs-auditor curl http://localhost:8090/health

# Check Docker network
docker network inspect fbs-network
```

3. **Firewall issues**
```bash
# Check firewall rules
sudo iptables -L -n

# Temporarily disable firewall (for testing)
sudo systemctl stop firewalld  # or ufw disable
```

## Service-Specific Problems

### MobiFlow-Auditor Issues

#### LLM API not accessible

**Symptom**: Port 8090 not responding

**Debug steps**:
```bash
# Check if API is enabled
docker exec -it fbs-auditor env | grep ENABLE_LLM_API

# Check Python process
docker exec -it fbs-auditor ps aux | grep python

# Manual start
docker exec -it fbs-auditor python src/llm_control_api.py
```

**Fix**: Ensure environment variable is set
```yaml
# In docker-compose.yml
environment:
  - ENABLE_LLM_API=true
```

#### No telemetry data

**Symptom**: Empty responses from `/kpm` endpoints

**Debug**:
```bash
# Check Redis for data
redis-cli
> KEYS kpm:*
> KEYS mobiflow:*

# Check SDL connection
docker exec -it fbs-auditor python -c "import redis; r=redis.Redis('redis-sdl'); print(r.ping())"
```

### MobieXpert Issues

#### Rule reload failures

**Symptom**: Rules not updating after POST

**Debug**:
```bash
# Check current rules
curl http://localhost:8091/rules

# Test rule syntax
python -c "import yaml; yaml.safe_load(open('rule.yaml'))"

# Check P-BEST engine logs
docker exec -it fbs-expert tail -f /app/logs/pbest.log
```

**Fix**:
```python
# Validate rule structure
from llm_control_api_client import test_rule

rule = {...}  # Your rule
test_data = {...}  # Test data
result = test_rule(rule, test_data)
print(result)
```

### OAI-5G Issues

#### FBS not starting

**Symptom**: `launch_fbs.sh` fails

**Debug**:
```bash
# Check script permissions
ls -la OAI-5G/tools/fbs_scenarios/launch_fbs.sh

# Check configuration
cat OAI-5G/tools/fbs_scenarios/fake_gnb.cfg | grep -E '(plmn|pci|tac)'

# Manual test
cd OAI-5G
./cmake_targets/ran_build/build/nr-softmodem --help
```

**Fix**:
```bash
# Set executable permission
chmod +x OAI-5G/tools/fbs_scenarios/launch_fbs.sh

# Fix path issues
export OAI_PATH=/path/to/OAI-5G
```

## Performance Issues

### High Latency

#### Diagnosis
```python
# Measure API response times
import time
import requests

for _ in range(10):
    start = time.time()
    requests.get("http://localhost:8090/stats")
    print(f"Response time: {(time.time() - start)*1000:.2f}ms")
```

#### Solutions

1. **Optimize Redis**
```bash
# Check Redis performance
redis-cli --latency
redis-cli --latency-history

# Optimize Redis config
redis-cli CONFIG SET save ""  # Disable persistence for testing
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

2. **Scale services**
```yaml
# In docker-compose.yml
services:
  mobiflow-auditor:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### Memory Leaks

#### Detection
```bash
# Monitor memory usage
docker stats --no-stream

# Check for leaks
docker exec -it fbs-auditor python -m tracemalloc
```

#### Fix
```python
# Add memory profiling
import tracemalloc
tracemalloc.start()

# ... your code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Detection Problems

### No FBS Detections

#### Diagnosis Checklist

1. **Rules loaded?**
```bash
curl http://localhost:8091/rules | jq '.count'
```

2. **Data flowing?**
```bash
# Check record count
curl http://localhost:8090/stats | jq '.total_records'

# Monitor in real-time
watch -n 1 'curl -s http://localhost:8090/stats | jq .total_records'
```

3. **Thresholds correct?**
```python
# Test with lower thresholds
rule = {
    'condition': {
        'field': 'attach_failures',
        'gte': 1  # Lower threshold
    }
}
```

### False Positives

#### Analysis
```python
# Analyze false positives
from llm_control_api_client import get_mobiflow

records = get_mobiflow(n=1000)
false_positives = [r for r in records 
                   if r['suspected_fbs'] and not r['actual_fbs']]

print(f"False positive rate: {len(false_positives)/len(records)*100:.1f}%")
```

#### Tuning
```yaml
# Adjust detection thresholds
rules:
  - name: "FBS_Detection"
    condition:
      and:
        - field: "attach_failures"
          gte: 5  # Increase threshold
        - field: "confidence"
          gt: 0.8  # Require higher confidence
```

## LLM Integration Issues

### API Key Problems

#### Symptom
```
OpenAI API error: Invalid API key
```

#### Fix
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=sk-..." >> .env

# Export for current session
export OPENAI_API_KEY="sk-..."

# Verify in Python
python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10])"
```

### Rate Limiting

#### Symptom
```
Error: Rate limit exceeded
```

#### Solutions

1. **Implement retry logic**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def call_llm():
    # Your LLM call
    pass
```

2. **Use caching**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_rule(pattern):
    # LLM rule generation
    pass
```

### Model Errors

#### Debug LLM responses
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test LLM directly
import dspy
lm = dspy.OpenAI(model="gpt-3.5-turbo", temperature=0.7)
dspy.settings.configure(lm=lm)

# Simple test
response = lm("Test prompt")
print(response)
```

## Docker and Networking

### Container Communication Issues

#### Debug inter-container networking
```bash
# Test from one container to another
docker exec -it fbs-auditor ping redis-sdl
docker exec -it fbs-auditor nc -zv mobiexpert 8091

# Check DNS resolution
docker exec -it fbs-auditor nslookup redis-sdl

# Inspect network
docker network inspect fbs-network | jq '.[0].Containers'
```

### Volume Mount Problems

#### Symptom
```
Error: Configuration file not found
```

#### Fix
```bash
# Check mounts
docker inspect fbs-auditor | jq '.[0].Mounts'

# Fix permissions
sudo chown -R $USER:$USER ./llm_fbs_utils

# Verify files exist
ls -la MobiFlow-Auditor/src/llm_control_api.py
```

## Debugging Tools

### Interactive Debugging

```python
# Start interactive Python session
docker exec -it fbs-auditor python

>>> from llm_control_api import app, sdl_manager
>>> sdl_manager.get_ue_count()
>>> 
>>> # Test API endpoints
>>> with app.test_client() as client:
...     response = client.get('/health')
...     print(response.json)
```

### Network Traffic Analysis

```bash
# Monitor Redis commands
redis-cli monitor

# Capture network traffic
docker exec -it fbs-auditor tcpdump -i eth0 -w /tmp/capture.pcap

# Analyze HTTP traffic
docker exec -it fbs-auditor tcpdump -i any -A 'tcp port 8090'
```

### Performance Profiling

```python
# Profile Python code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
run_detection_cycle()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Log Analysis

### Centralized Log Collection

```bash
# Collect all logs
mkdir -p logs/$(date +%Y%m%d)
docker-compose logs > logs/$(date +%Y%m%d)/all.log

# Service-specific logs
docker-compose logs mobiflow-auditor > logs/$(date +%Y%m%d)/auditor.log
docker-compose logs mobiexpert > logs/$(date +%Y%m%d)/expert.log
```

### Log Parsing

```python
# Parse logs for errors
import re

with open('logs/all.log') as f:
    errors = []
    for line in f:
        if re.search(r'ERROR|CRITICAL|Exception', line):
            errors.append(line.strip())

print(f"Found {len(errors)} errors")
for error in errors[:10]:
    print(error)
```

### Real-time Monitoring

```bash
# Monitor specific patterns
docker-compose logs -f | grep -E "(ERROR|WARNING|FBS)"

# Monitor with highlighting
docker-compose logs -f | grep --color=always -E "(ERROR|WARNING|FBS|detected)"
```

## Recovery Procedures

### Complete System Reset

```bash
#!/bin/bash
# Full system reset

# Stop all services
make docker-down

# Clear all data
make clean-redis
make clean-logs

# Rebuild
make docker-build

# Start fresh
make docker-up

# Verify
make check-services
```

### Backup and Restore

```bash
# Backup Redis data
redis-cli --rdb backup.rdb

# Backup configurations
tar -czf configs_backup.tar.gz *.yaml *.yml .env

# Restore Redis
redis-cli --rdb backup.rdb restore

# Restore configs
tar -xzf configs_backup.tar.gz
```

### Emergency Procedures

#### System Overload
```bash
# Kill resource-intensive processes
docker-compose stop llm-orchestrator
docker-compose restart mobiflow-auditor

# Reduce load
redis-cli CONFIG SET maxclients 100
```

#### Data Corruption
```bash
# Clear corrupted data
redis-cli FLUSHDB

# Reimport clean dataset
python dataset_playback.py play --file data/clean_baseline.json
```

## Getting Help

### Diagnostic Information Collection

```bash
# Generate diagnostic report
cat > diagnostic_report.sh << 'EOF'
#!/bin/bash
echo "=== System Information ==="
uname -a
docker --version
docker-compose --version
python3 --version

echo "=== Service Status ==="
docker-compose ps

echo "=== Recent Logs ==="
docker-compose logs --tail=50

echo "=== Network Status ==="
docker network ls
netstat -tulpn | grep -E '(8090|8091|6379)'

echo "=== Resource Usage ==="
docker stats --no-stream
df -h
free -h
EOF

chmod +x diagnostic_report.sh
./diagnostic_report.sh > diagnostic_$(date +%Y%m%d_%H%M%S).txt
```

### Support Resources

- **GitHub Issues**: Open an issue with diagnostic report
- **Documentation**: Check README.md and inline documentation
- **Logs**: Always include relevant log excerpts
- **Community**: Stack Overflow tags: `5g-security`, `oran`, `fbs-detection`

## Prevention Tips

1. **Regular Maintenance**
   - Weekly: `make clean-logs`
   - Monthly: `make reset`
   - Always: Keep backups

2. **Monitoring**
   - Set up Prometheus/Grafana dashboards
   - Configure alerts for critical metrics
   - Review logs daily

3. **Testing**
   - Run tests before deployment: `make test`
   - Validate configurations: `docker-compose config`
   - Test in isolation first

4. **Documentation**
   - Document any custom configurations
   - Keep change log
   - Note any workarounds applied